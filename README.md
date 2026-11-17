# 🛡️ OfferCheck

Paste an internship offer (email, WhatsApp message or posting) and get a structured risk analysis: **explainable red flags with quoted evidence, missing information, questions to ask the sender, and a skill-fit check against your resume**. Save analysed offers to a built-in tracker.

## Why I built this

>While applying for off-campus summer and winter internships, I noticed how common fake offer letters, pay-to-work training traps, and suspicious WhatsApp recruiter messages are in student groups. I wanted a quick way to sanity-check an offer before replying or sharing personal details. Instead of relying solely on an LLM—which can sometimes miss obvious red flags or hallucinate—I built a deterministic rule engine for hard signals (like fee requests or free email domains) paired with Gemini for context, requiring every AI-flagged warning to quote the exact text from the offer.


## How it works

```
offer text ──► rules.py (deterministic checks, explainable, unit-tested)
        │                         │
        │                         ▼ rule hits
        └──────────► analyzer.py ──► Gemini 3.8 Flash (Pydantic structured output)
                           │            • contextual red flags (must quote the offer)
                           │            • missing info, questions to ask, resume fit
                           ▼
                verify quotes verbatim ─► combine scores ─► deterministic verdict ─► SQLite tracker + Streamlit UI
```

**Design decisions**

- **Rules + LLM, not LLM alone.** Rules catch well-known patterns (registration fees, UPI payment requests, free-email recruiters, pressure tactics) deterministically and handle negation ("there is *no* registration fee"). The LLM handles context the rules cannot, like a mandatory laptop purchase or a pyramid-style task.
- **Grounded evidence.** Every AI red flag must quote the offer text. Quotes are verified verbatim, and unverifiable ones are labelled in the UI.
- **Transparent scoring.** `final = 0.4 × rule score + 0.6 × AI score`, with floors: a critical rule hit (fee, UPI payment, OTP request) forces at least 80, and a verified critical AI flag forces at least 65. The verdict comes from fixed thresholds (<35 *Looks legitimate*, <65 *Proceed with caution*, otherwise *High risk*), never from free-form model text.
- **Graceful degradation.** If the AI output fails schema validation twice, the app returns a rules-only result marked *Inconclusive* instead of crashing, and never labels an offer "legitimate" without AI analysis.
- **Prompt-injection aware.** Offer text is treated as untrusted data in the prompt.

## Setup

Requires Python 3.10+. There is no heavy ML dependency, so installs take about a minute.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"                       # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
# optional: export GEMINI_MODEL="gemini-3.8-flash"
```

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).

## Run

```bash
streamlit run app.py          # the web app
python check_setup.py          # verifies your API key and model (run this first, and after any 404 error)
pytest                        # unit tests (no API key needed)
python eval.py                # rules-only baseline on labelled cases (no API key needed)
python eval.py --llm          # full pipeline: precision/recall, schema pass rate, evidence verification
```

## Evaluation

### Results (rules-only baseline, 12 synthetic cases)
Accuracy 0.83 · Precision 1.00 · Recall 0.67 · F1 0.80
The two misses are subtle cases (a forced laptop purchase and a pyramid-style task) that the Gemini layer is designed to catch.

## Project structure

```
├── app.py            Streamlit UI (analyze tab, tracker tab)
├── analyzer.py       Gemini structured analysis, verification, score combination, degradation
├── rules.py          Deterministic red-flag rules
├── db.py             SQLite tracker
├── samples.py        Synthetic example offers
├── check_setup.py    Verifies API key and lists usable Gemini models
├── eval.py           Evaluation harness
├── eval_cases.json   Labelled cases
└── tests/            pytest suite for rules, db and analyzer (Gemini mocked)
```

## Limitations

- Heuristic guidance, not proof. A clean result does not guarantee an offer is genuine; always verify the company's website, official email domain and registration independently.
- Rules are tuned for common Indian internship scams and English text.
- The eval set is small; metrics are indicative, not statistically strong.
- The tracker stores data in a local SQLite file, which is ephemeral on most free hosting.
- Offer text is sent to the Gemini API. Redact personal details first.

If you are targeted by a scam in India, you can report it at cybercrime.gov.in or call 1930.
