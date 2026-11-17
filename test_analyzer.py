import json

import pytest

from analyzer import (AnalyzerError, OfferAnalyzer, combine_scores, evidence_in_text, verdict_for)
from samples import SAMPLE_LEGIT, SAMPLE_SCAM


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        return FakeResponse(self.texts.pop(0))


class FakeClient:
    def __init__(self, texts):
        self.models = FakeModels(texts)


def llm_json(**overrides):
    base = {
        "company": "Global Tech Solutions", "role": "Data Entry Intern", "stipend": "Rs. 3,000 per day",
        "duration": "Not stated", "work_mode": "Work from home", "llm_risk_score": 95,
        "summary": "Almost certainly a fee scam.",
        "red_flags": [], "green_flags": [], "missing_information": ["Duration"],
        "questions_to_ask": ["What is your official website?"], "skills_you_match": [], "skill_gaps": [],
    }
    base.update(overrides)
    return json.dumps(base)


def test_verdict_thresholds():
    assert verdict_for(0) == "Looks legitimate" and verdict_for(34) == "Looks legitimate"
    assert verdict_for(35) == "Proceed with caution" and verdict_for(64) == "Proceed with caution"
    assert verdict_for(65) == "High risk" and verdict_for(100) == "High risk"


def test_combine_scores_floors():
    assert combine_scores(0, 0, False, False) == 0
    assert combine_scores(10, 10, False, False) == 10
    assert combine_scores(45, 0, True, False) == 80     # critical rule floor
    assert combine_scores(0, 20, False, True) == 65     # verified critical AI floor


def test_evidence_matching_is_normalised_and_supports_ellipsis():
    text = "To confirm your seat you must pay   a refundable registration fee of Rs. 2,999 via UPI."
    assert evidence_in_text("you must pay a refundable registration fee", text)
    assert evidence_in_text("To confirm your seat ... via UPI", text)
    assert not evidence_in_text("send us your bank password immediately", text)
    assert not evidence_in_text("fee", text)  # too short to be meaningful evidence


def test_scam_report_grounding_dedupe_and_floor():
    flags = [
        {"title": "Pay to confirm", "severity": "critical", "evidence": "you must pay a refundable registration fee",
         "why_it_matters": "dup of rule"},
        {"title": "Fake claim", "severity": "high", "evidence": "we are listed on the stock exchange", "why_it_matters": "x"},
        {"title": "Work from home data entry", "severity": "medium", "evidence": "Data Entry Internship at Global Tech Solutions",
         "why_it_matters": "vague role"},
    ]
    client = FakeClient([llm_json(red_flags=flags)])
    report = OfferAnalyzer(client=client).analyze(SAMPLE_SCAM)
    assert report.verdict == "High risk" and report.final_risk_score >= 80
    ai = [f for f in report.red_flags if f.source == "ai"]
    assert {f.title for f in ai} == {"Fake claim", "Work from home data entry"}  # duplicate of a rule hit dropped
    assert next(f for f in ai if f.title == "Fake claim").evidence_verified is False
    assert next(f for f in ai if f.title == "Work from home data entry").evidence_verified is True
    assert report.schema_attempts == 1 and not report.degraded


def test_legit_offer_scores_low():
    client = FakeClient([llm_json(llm_risk_score=8, company="Brightwave", summary="Looks fine.",
                                  green_flags=["Named manager", "Company domain email"])])
    report = OfferAnalyzer(client=client).analyze(SAMPLE_LEGIT)
    assert report.verdict == "Looks legitimate" and report.final_risk_score < 35


def test_schema_failure_retries_once():
    client = FakeClient(["not json at all", llm_json(llm_risk_score=90)])
    report = OfferAnalyzer(client=client).analyze(SAMPLE_SCAM)
    assert report.schema_attempts == 2 and not report.degraded and client.models.calls == 2


def test_two_schema_failures_degrade_to_rules_only():
    client = FakeClient(["garbage", llm_json(llm_risk_score=500)])  # 500 violates the 0-100 validator
    report = OfferAnalyzer(client=client).analyze(SAMPLE_SCAM)
    assert report.degraded and report.verdict == "High risk" and report.final_risk_score == report.rule_score


def test_degraded_never_declares_offer_legitimate():
    client = FakeClient(["garbage", "garbage"])
    report = OfferAnalyzer(client=client).analyze(SAMPLE_LEGIT)
    assert report.degraded and report.verdict == "Inconclusive"


def test_short_input_rejected_and_missing_key(monkeypatch):
    with pytest.raises(AnalyzerError):
        OfferAnalyzer(client=FakeClient([])).analyze("too short")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(AnalyzerError):
        OfferAnalyzer()
