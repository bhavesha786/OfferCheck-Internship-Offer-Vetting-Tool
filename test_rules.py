from rules import has_severity, rule_score, run_rules
from samples import SAMPLE_LEGIT, SAMPLE_SCAM


def ids(text):
    return {h.rule_id for h in run_rules(text)}


def test_scam_sample_triggers_critical_rules():
    found = ids(SAMPLE_SCAM)
    assert {"FEE_REQUEST", "PAYMENT_APP", "GUARANTEED_OUTCOME", "EARNINGS_CLAIM", "FREE_EMAIL_SENDER", "NO_INTERVIEW", "URGENCY_PRESSURE", "DOC_REQUEST"} <= found
    assert has_severity(run_rules(SAMPLE_SCAM), "critical")
    assert rule_score(run_rules(SAMPLE_SCAM)) == 100


def test_legit_sample_is_clean():
    hits = run_rules(SAMPLE_LEGIT)
    assert not has_severity(hits, "critical")
    assert not has_severity(hits, "high")
    assert rule_score(hits) <= 12


def test_negated_fee_statements_do_not_trigger():
    assert "FEE_REQUEST" not in ids("There is no registration fee for this internship.")
    assert "FEE_REQUEST" not in ids("We never ask candidates to pay a security deposit.")
    assert "FEE_REQUEST" not in ids("A security deposit of any kind is not required.")


def test_company_paying_stipend_via_upi_is_fine():
    assert "PAYMENT_APP" not in ids("Your stipend will be transferred to your account via UPI every month.")


def test_pay_to_confirm_seat_triggers_fee():
    assert "FEE_REQUEST" in ids("Please pay Rs. 999 to confirm your seat today.")


def test_free_email_detected_but_company_domain_is_not():
    assert "FREE_EMAIL_SENDER" in ids("Write to hr.team@gmail.com for details.")
    assert "FREE_EMAIL_SENDER" not in ids("Write to hr@acme-corp.com for details.")


def test_unrealistic_pay_requires_no_experience_claim():
    assert "UNREALISTIC_PAY" in ids("Earn Rs. 1,20,000 per month. No experience required.")
    assert "UNREALISTIC_PAY" not in ids("Stipend Rs. 1,20,000 per month for senior research interns with a PhD.")


def test_sensitive_credentials_critical():
    hits = run_rules("To verify your identity please share your OTP with our HR executive.")
    assert any(h.rule_id == "SENSITIVE_CREDENTIALS" and h.severity == "critical" for h in hits)


def test_hits_sorted_by_severity():
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    ranks = [order[h.severity] for h in run_rules(SAMPLE_SCAM)]
    assert ranks == sorted(ranks, reverse=True)
