"""OfferCheck - Streamlit UI."""
import html
import os

import streamlit as st

import db
from analyzer import MODEL_NAME, AnalyzerError, OfferAnalyzer
from samples import SAMPLE_LEGIT, SAMPLE_SCAM

st.set_page_config(page_title="OfferCheck", page_icon="🛡️", layout="wide")
db.init_db()

st.markdown(
    """
<style>
.badge {display:inline-block; padding:3px 10px; margin:2px 4px 2px 0; border-radius:999px;
        font-size:0.78rem; font-weight:600; border:1px solid transparent;}
.badge-ok   {background:rgba(34,197,94,0.14);  color:#16a34a; border-color:rgba(34,197,94,0.35);}
.badge-gap  {background:rgba(239,68,68,0.12);  color:#dc2626; border-color:rgba(239,68,68,0.35);}
.sev {display:inline-block; padding:2px 9px; border-radius:6px; font-size:0.72rem; font-weight:700;
      letter-spacing:0.04em; text-transform:uppercase; margin-right:8px;}
.sev-critical {background:#7f1d1d; color:#fff;}
.sev-high     {background:#dc2626; color:#fff;}
.sev-medium   {background:#f59e0b; color:#1f2937;}
.sev-low      {background:#94a3b8; color:#1f2937;}
.section-label {font-size:0.8rem; font-weight:700; letter-spacing:0.04em; text-transform:uppercase;
                color:#64748b; margin:12px 0 4px 0;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_analyzer(api_key: str) -> OfferAnalyzer:
    return OfferAnalyzer(api_key=api_key)


def badges(items: list, kind: str) -> str:
    return "".join(f'<span class="badge badge-{kind}">{html.escape(str(i))}</span>' for i in items)


def render_report(report: dict, allow_save: bool = True) -> None:
    verdict = report["verdict"]
    banner = f"**{verdict}**: {report['summary']}"
    if verdict == "Looks legitimate":
        st.success(banner)
    elif verdict in ("Proceed with caution", "Inconclusive"):
        st.warning(banner)
    else:
        st.error(banner)
    if report.get("degraded"):
        st.caption(f"⚠️ AI analysis unavailable ({report.get('degraded_reason', '')}). Showing rule-based checks only.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall risk", f"{report['final_risk_score']}/100")
    c2.metric("Rule-based score", f"{report['rule_score']}/100")
    c3.metric("AI score", "n/a" if report.get("degraded") else f"{report['llm_risk_score']}/100")
    st.progress(report["final_risk_score"] / 100)
    st.caption(
        f"**{report['company']}** · {report['role']} · Stipend: {report['stipend']} · "
        f"Duration: {report['duration']} · {report['work_mode']}"
    )

    st.markdown('<div class="section-label">Red flags</div>', unsafe_allow_html=True)
    if not report["red_flags"]:
        st.write("No red flags found in the text.")
    for flag in report["red_flags"]:
        with st.container(border=True):
            st.markdown(
                f'<span class="sev sev-{flag["severity"]}">{flag["severity"]}</span>**{html.escape(flag["title"])}**',
                unsafe_allow_html=True,
            )
            st.markdown(f"> {flag['evidence']}")
            st.write(flag["why_it_matters"])
            origin = "rule-based check" if flag["source"] == "rule" else "AI"
            note = "" if flag["evidence_verified"] else " · ⚠️ quote not found verbatim in the offer"
            st.caption(f"Source: {origin}{note}")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-label">Reassuring signs</div>', unsafe_allow_html=True)
        if report["green_flags"]:
            st.markdown(badges(report["green_flags"], "ok"), unsafe_allow_html=True)
        else:
            st.write("None clearly stated.")
    with right:
        st.markdown('<div class="section-label">Missing information</div>', unsafe_allow_html=True)
        if report["missing_information"]:
            st.markdown(badges(report["missing_information"], "gap"), unsafe_allow_html=True)
        else:
            st.write("Nothing important missing.")

    if report["questions_to_ask"]:
        st.markdown('<div class="section-label">Ask the sender before accepting</div>', unsafe_allow_html=True)
        for i, question in enumerate(report["questions_to_ask"], 1):
            st.markdown(f"{i}. {question}")

    if report["skills_you_match"] or report["skill_gaps"]:
        st.markdown('<div class="section-label">Fit with your resume</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.markdown("**You match**")
        a.markdown(badges(report["skills_you_match"], "ok") or "n/a", unsafe_allow_html=True)
        b.markdown("**Gaps**")
        b.markdown(badges(report["skill_gaps"], "gap") or "n/a", unsafe_allow_html=True)

    if allow_save:
        if st.session_state.get("saved_id"):
            st.success(f"Saved to tracker (#{st.session_state.saved_id}).")
        elif st.button("💾 Save to tracker", key="save_btn"):
            st.session_state.saved_id = db.save_offer(report, st.session_state.get("report_text", ""))
            st.rerun()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛡️ OfferCheck")
    st.caption("Vet an internship offer before you accept it.")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password", help="Or export GEMINI_API_KEY before launching.")
    st.subheader("Your resume (optional)")
    st.text_area("Paste to get a skill-fit check", key="resume", height=200)
    st.subheader("Try an example")
    st.button("Legitimate-looking offer", on_click=lambda: st.session_state.update(offer_text=SAMPLE_LEGIT), use_container_width=True)
    st.button("Suspicious offer", on_click=lambda: st.session_state.update(offer_text=SAMPLE_SCAM), use_container_width=True)
    st.caption("Examples are synthetic.")
    st.divider()
    st.caption(f"Model: {MODEL_NAME}")
    st.caption("Offer text is sent to the Gemini API. Remove personal details first. OfferCheck gives guidance, "
               "not proof: always verify the company independently.")

# --------------------------------------------------------------------------------- main
st.title("OfferCheck")
st.caption("Rule-based checks + AI analysis. Paste an internship offer, email or message.")
tab_analyze, tab_tracker = st.tabs(["Analyze an offer", "Tracker"])

with tab_analyze:
    st.text_area("Offer text", key="offer_text", height=260, placeholder="Paste the offer email, message or posting here...")
    if st.button("Analyze offer", type="primary"):
        if not api_key:
            st.warning("Add your Gemini API key in the sidebar (or set GEMINI_API_KEY) to continue.")
        else:
            try:
                with st.spinner("Running rule checks and AI analysis..."):
                    result = get_analyzer(api_key).analyze(st.session_state.offer_text, st.session_state.get("resume", ""))
                st.session_state.report = result.to_dict()
                st.session_state.report_text = st.session_state.offer_text
                st.session_state.saved_id = None
            except AnalyzerError as exc:
                st.error(str(exc))
            except Exception as exc:  # keep the app alive on unexpected failures
                st.error(f"Something went wrong: {exc}")
    if st.session_state.get("report"):
        render_report(st.session_state.report)

with tab_tracker:
    offers = db.list_offers()
    if not offers:
        st.info("No saved offers yet. Analyze an offer and click 'Save to tracker'.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Saved offers", len(offers))
        m2.metric("High risk", sum(o["verdict"] == "High risk" for o in offers))
        m3.metric("Accepted", sum(o["status"] == "Accepted" for o in offers))
        st.dataframe(
            [{k: o[k] for k in ("id", "created_at", "company", "role", "final_risk_score", "verdict", "status")} for o in offers],
            use_container_width=True, hide_index=True,
        )
        selected = st.selectbox("Open offer", [o["id"] for o in offers], format_func=lambda i: next(
            f"#{o['id']} · {o['company']} · {o['role']}" for o in offers if o["id"] == i))
        current = db.get_offer(selected)
        status = st.selectbox("Status", db.STATUSES, index=db.STATUSES.index(current["status"]), key=f"status_{selected}")
        notes = st.text_area("Notes", value=current["notes"], key=f"notes_{selected}")
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Update"):
            db.update_offer(selected, status, notes)
            st.rerun()
        if b2.button("Delete"):
            db.delete_offer(selected)
            st.rerun()
        with st.expander("Saved analysis"):
            render_report(current["report"], allow_save=False)
