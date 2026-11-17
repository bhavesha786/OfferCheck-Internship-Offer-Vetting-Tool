"""Deterministic red-flag rules for internship offers.

No network and no LLM: fast, explainable and unit-testable. The LLM layer (analyzer.py) adds
contextual judgement on top of these hits.
"""
import re
from dataclasses import dataclass
from typing import Optional

SEVERITIES = ("low", "medium", "high", "critical")
SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITIES)}
SEVERITY_WEIGHT = {"low": 5, "medium": 12, "high": 25, "critical": 45}

_I = re.IGNORECASE
_CUR = r"(?:₹|\brs\.?|\binr\b)"

# Context checks so that "There is no registration fee" does not trigger the fee rule.
_NEG_BEFORE = re.compile(
    r"\b(?:no|not|without|zero|never|nor|neither|free\s+of|free\s+from|don'?t|do\s+not|doesn'?t|does\s+not|isn'?t|is\s+not)\b",
    _I,
)
_NEG_AFTER = re.compile(r"^[^.\n]{0,25}?\b(?:not|never)\s+(?:required|needed|charged|applicable|collected|payable|applied)\b", _I)
# "We will transfer your stipend via UPI" is the company paying YOU, not a payment request.
_COMPANY_PAYS = re.compile(r"\b(?:stipend|salary|ctc|reimburs\w*|we\s+will|company\s+will|will\s+be\s+paid|paid\s+to\s+you|credited)\b", _I)
_CLAUSE_BREAK = re.compile(r"[.!?;:\n–—]|\s-\s")  # sentence/clause boundaries, incl. spaced hyphens


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    severity: str
    title: str
    evidence: str
    why_it_matters: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: str
    title: str
    why: str
    patterns: tuple
    skip_negated: bool = True
    skip_company_pays: bool = False


RULES = (
    Rule(
        "FEE_REQUEST", "critical", "Asks you to pay money",
        "Legitimate employers do not charge interns for registration, training, kits or deposits. "
        "Paying to 'confirm a seat' is the most common internship scam.",
        (
            re.compile(r"\b(?:registration|security|refundable|training|onboarding|joining|processing|verification|certification|certificate|kit|laptop|admission|enrol+ment|course|id\s*card)\s+(?:fee|fees|deposit|charges?|amount)\b", _I),
            re.compile(r"\byou\s+(?:must|need\s+to|have\s+to|should|are\s+required\s+to|will\s+need\s+to)\s+(?:pay|deposit|transfer|send)\b[^\n]{0,80}", _I),
            re.compile(r"\b(?:pay|payable|deposit|transfer)\b[^\n]{0,50}?" + _CUR + r"\s*\d[\d,]*[^\n]{0,40}?\b(?:to\s+(?:confirm|secure|book|reserve|activate|complete)|for\s+(?:registration|confirmation|training|kit|verification|seat|slot))\b", _I),
        ),
        skip_company_pays=True,
    ),
    Rule(
        "PAYMENT_APP", "critical", "Payment requested through UPI / wallet / QR code",
        "Requests to pay through personal UPI IDs, wallets or QR codes are a strong scam signal; companies collect no money from interns.",
        (re.compile(r"\b(?:pay|payable|send|transfer|deposit|remit)\b[^\n]{0,70}?\b(?:upi|paytm|phone\s*pe|gpay|google\s*pay|bhim|qr\s*code)\b", _I),),
        skip_company_pays=True,
    ),
    Rule(
        "SENSITIVE_CREDENTIALS", "critical", "Asks for OTP, PIN, password or card details",
        "Nobody legitimate needs your OTP, PIN, banking password or card number to hire you.",
        (re.compile(r"\b(?:share|send|give|tell|provide|enter|submit)\b[^\n]{0,40}?\b(?:otp|cvv|(?:atm|upi)\s*pin|password|card\s*number|net\s*banking|internet\s*banking)\b", _I),),
    ),
    Rule(
        "GUARANTEED_OUTCOME", "high", "Guarantees a job or placement",
        "Real internships cannot guarantee jobs or conversions; guarantees are used to lower your guard.",
        (re.compile(r"\b(?:100\s*%|guaranteed|assured)\s+(?:job|placement|ppo|conversion|selection|internship|stipend)\b|\bguarantee[sd]?\s+(?:you\s+)?(?:a\s+)?(?:job|placement|ppo|full[- ]time)\b", _I),),
    ),
    Rule(
        "EARNINGS_CLAIM", "high", "Promises high daily or weekly earnings",
        "'Earn Rs X per day from home' is typical of task, typing and data-entry scams, not genuine internships.",
        (re.compile(r"\b(?:earn|earning|income|make)\b[^\n]{0,40}?" + _CUR + r"\s*\d[\d,]*\s*(?:/|per|a)\s*(?:day|week|daily|weekly|hour)\b", _I),),
    ),
    Rule(
        "DOC_REQUEST", "medium", "Mentions Aadhaar / PAN / bank details",
        "Employers collect ID and bank details at onboarding, after a signed offer, through an HR portal. "
        "Asking for them up front, over chat or email, is risky.",
        (re.compile(r"\b(?:aadhaar|aadhar|pan\s*card|pan\s*number|bank\s*account|account\s*number|passbook|ifsc)\b", _I),),
    ),
    Rule(
        "FREE_EMAIL_SENDER", "medium", "Recruiter uses a free personal email address",
        "Genuine companies usually email from their own domain, not Gmail, Yahoo or Outlook.",
        (re.compile(r"[\w.+-]+@(?:gmail|googlemail|yahoo|ymail|outlook|hotmail|live|rediffmail|protonmail|proton)\.[a-z.]{2,}", _I),),
        skip_negated=False,
    ),
    Rule(
        "MESSAGING_INTERVIEW", "medium", "Interview or selection handled over WhatsApp / Telegram",
        "Scammers favour chat apps because they are hard to trace. Real hiring processes use calls, video or company email.",
        (re.compile(r"\b(?:interview|selection|screening|task|test|hr|joining)\b[^\n]{0,50}?\b(?:whatsapp|telegram|signal)\b|\b(?:whatsapp|telegram)\b[^\n]{0,40}?\b(?:interview|hr|selection)\b", _I),),
    ),
    Rule(
        "NO_INTERVIEW", "medium", "Selected without any interview",
        "Being 'selected' with no interview or screening suggests the goal is not to evaluate you.",
        (re.compile(r"\bno\s+interview\b|\bwithout\s+(?:any\s+)?interview\b|\b(?:direct|instant)\s+(?:selection|joining|offer)\b|\bselected\s+without\b|\binterview\s+(?:is\s+)?not\s+required\b", _I),),
        skip_negated=False,
    ),
    Rule(
        "URGENCY_PRESSURE", "medium", "Pressure tactics (limited seats, hurry)",
        "Artificial scarcity and urgency push you to act before you can verify anything.",
        (re.compile(r"\bonly\s+\d+\s+(?:seats?|slots?|vacanc(?:y|ies)|positions?)\b|\b(?:hurry|act\s+now|limited\s+(?:seats?|slots?|vacanc(?:y|ies))|last\s+date\s+(?:is\s+)?today|offer\s+expires\s+(?:today|tonight|soon)|confirm\s+immediately)\b", _I),),
        skip_negated=False,
    ),
    Rule(
        "SHORT_DEADLINE", "low", "Very short reply deadline",
        "Short deadlines are common in real offers too, so this is weak evidence on its own; just do not let it stop you verifying.",
        (re.compile(r"\b(?:within|in|before)\s+(?:the\s+next\s+)?(?:12|24|48|2|3)\s*(?:hours?|hrs?)\b", _I),),
        skip_negated=False,
    ),
    Rule(
        "MESSAGING_CONTACT", "low", "Contact only through WhatsApp / Telegram",
        "Chat-only contact makes the sender hard to verify. Ask for an official email and company website.",
        (re.compile(r"\b(?:contact|message|reach|send|write|text|whatsapp)\b[^\n]{0,60}?\b(?:whatsapp|telegram)\b", _I),),
        skip_negated=False,
    ),
)

_NO_EXPERIENCE = re.compile(r"\bno\s+(?:prior\s+)?(?:experience|skills?|qualification|degree)\s+(?:is\s+)?(?:required|needed|necessary)\b|\banyone\s+can\s+(?:apply|join)\b", _I)
_AMOUNT = re.compile(_CUR + r"\s*(\d[\d,]*(?:\.\d+)?)\s*(k|lakhs?|lpa)?", _I)


def _snippet(text: str, start: int, end: int, pad: int = 20) -> str:
    return re.sub(r"\s+", " ", text[max(0, start - pad): min(len(text), end + pad)]).strip()


def _clause_before(text: str, start: int) -> str:
    return _CLAUSE_BREAK.split(text[max(0, start - 60): start])[-1]


def _first_match(rule: Rule, text: str) -> Optional[RuleHit]:
    for pattern in rule.patterns:
        for m in pattern.finditer(text):
            clause = _clause_before(text, m.start())
            if rule.skip_negated and (_NEG_BEFORE.search(clause) or _NEG_AFTER.match(text[m.end(): m.end() + 40])):
                continue
            if rule.skip_company_pays and _COMPANY_PAYS.search(clause + " " + m.group(0)):
                continue
            return RuleHit(rule.rule_id, rule.severity, rule.title, _snippet(text, m.start(), m.end()), rule.why)
    return None


def _unrealistic_pay(text: str) -> Optional[RuleHit]:
    if not _NO_EXPERIENCE.search(text):
        return None
    for m in _AMOUNT.finditer(text):
        value = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit == "k":
            value *= 1_000
        elif unit.startswith("lakh") or unit == "lpa":
            value *= 100_000
        context = _snippet(text, m.start(), m.end(), 30)
        if value >= 50_000 and re.search(r"month|stipend|salary|ctc|pm\b|earn", context, _I):
            return RuleHit(
                "UNREALISTIC_PAY", "medium", "Very high pay for 'no experience required'", context,
                "Very high pay combined with no skills or experience needed is a classic bait pattern.",
            )
    return None


def _no_verifiable_contact(text: str) -> Optional[RuleHit]:
    has_url_or_domain = re.search(r"https?://|www\.|\b[a-z0-9-]{2,}\.[a-z]{2,}\b", text, _I)
    has_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    if has_url_or_domain or has_email:
        return None
    return RuleHit(
        "NO_VERIFIABLE_CONTACT", "low", "No website or email to verify the sender",
        "(no company website or email address found in the text)",
        "Without a website or official email you cannot verify who is making the offer.",
    )


def run_rules(text: str) -> list:
    """Return rule hits, most severe first. One hit per rule."""
    hits = [h for h in (_first_match(rule, text) for rule in RULES) if h]
    for detector in (_unrealistic_pay, _no_verifiable_contact):
        hit = detector(text)
        if hit:
            hits.append(hit)
    return sorted(hits, key=lambda h: -SEVERITY_RANK[h.severity])


def rule_score(hits: list) -> int:
    """0-100: sum of severity weights, capped."""
    return min(100, sum(SEVERITY_WEIGHT[h.severity] for h in hits))


def has_severity(hits: list, severity: str) -> bool:
    return any(h.severity == severity for h in hits)
