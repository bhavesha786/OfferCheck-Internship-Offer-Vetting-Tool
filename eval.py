"""Evaluate OfferCheck on labelled cases (eval_cases.json).

    python eval.py              # rules-only baseline, no API key needed
    python eval.py --llm        # full pipeline (rules + Gemini); needs GEMINI_API_KEY
    python eval.py --llm --delay 8

The shipped cases are SYNTHETIC. Add your own anonymised real offers to eval_cases.json
(label "safe" or "risky") to make the numbers meaningful.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from rules import rule_score, run_rules

CASES_PATH = Path(__file__).with_name("eval_cases.json")
RULE_THRESHOLD = 25    # rules-only: one high-severity hit (or two medium) counts as risky
FINAL_THRESHOLD = 50   # full pipeline: midpoint between 'caution' (35) and 'high risk' (65)


def load_cases() -> list:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    for case in cases:
        if case["label"] not in ("safe", "risky"):
            raise ValueError(f"Case {case['id']}: label must be 'safe' or 'risky'")
    return cases


def report_metrics(title: str, rows: list) -> dict:
    """rows: (case_id, label, score, predicted_risky)."""
    tp = sum(1 for _, l, _, p in rows if l == "risky" and p)
    fp = sum(1 for _, l, _, p in rows if l == "safe" and p)
    tn = sum(1 for _, l, _, p in rows if l == "safe" and not p)
    fn = sum(1 for _, l, _, p in rows if l == "risky" and not p)
    total = len(rows)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\n{title}")
    print("-" * len(title))
    for case_id, label, score, pred in rows:
        mark = "ok " if (label == "risky") == pred else "MISS"
        print(f"  [{mark}] {case_id:<4} label={label:<5} score={score:>3} predicted={'risky' if pred else 'safe'}")
    print(f"  accuracy={accuracy:.2f} precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}   (TP={tp} FP={fp} TN={tn} FN={fn})")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def eval_rules(cases: list) -> dict:
    rows = []
    for case in cases:
        score = rule_score(run_rules(case["text"]))
        rows.append((case["id"], case["label"], score, score >= RULE_THRESHOLD))
    return report_metrics(f"RULES-ONLY BASELINE (risky if rule score >= {RULE_THRESHOLD})", rows)


def eval_llm(cases: list, delay: float) -> dict:
    from analyzer import AnalyzerError, OfferAnalyzer

    analyzer = OfferAnalyzer()
    rows, degraded, first_try, ai_flags, ai_verified = [], 0, 0, 0, 0
    for n, case in enumerate(cases):
        if n:
            time.sleep(delay)
        try:
            report = analyzer.analyze(case["text"])
        except AnalyzerError as exc:
            print(f"  [{case['id']}] ERROR: {exc}")
            degraded += 1
            rows.append((case["id"], case["label"], 0, False))
            continue
        degraded += report.degraded
        first_try += (not report.degraded) and report.schema_attempts == 1
        for flag in report.red_flags:
            if flag.source == "ai":
                ai_flags += 1
                ai_verified += flag.evidence_verified
        rows.append((case["id"], case["label"], report.final_risk_score, report.final_risk_score >= FINAL_THRESHOLD))

    metrics = report_metrics(f"FULL PIPELINE (rules + Gemini; risky if final score >= {FINAL_THRESHOLD})", rows)
    total = len(cases)
    print(f"\n  structured-output pass rate : {(total - degraded)}/{total} = {(total - degraded) / total:.0%}")
    print(f"    ...on first attempt       : {first_try}/{total} = {first_try / total:.0%}")
    if ai_flags:
        print(f"  AI evidence verified in text: {ai_verified}/{ai_flags} = {ai_verified / ai_flags:.0%}")
    else:
        print("  AI evidence verified in text: n/a (no AI-only flags raised)")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OfferCheck.")
    parser.add_argument("--llm", action="store_true", help="Also run the full Gemini pipeline.")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds between API calls (default 5).")
    args = parser.parse_args()

    cases = load_cases()
    print(f"Loaded {len(cases)} labelled cases "
          f"({sum(c['label'] == 'risky' for c in cases)} risky, {sum(c['label'] == 'safe' for c in cases)} safe).")
    eval_rules(cases)
    if args.llm:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            print("\nGEMINI_API_KEY not set; cannot run --llm.")
            return 1
        eval_llm(cases, args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
