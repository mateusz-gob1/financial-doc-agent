import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import DocumentState
from schemas.financial import FinancialMetrics
from tools.llm_client import get_llm, get_langfuse_handler
from tools.cost_tracker import CostTracker
from tools.pdf_parser import parse_document
from tools.vector_store import retrieve_red_flags

load_dotenv()

EXTRACTION_MODEL = "google/gemini-2.5-flash"
CRITIQUE_MODEL = "google/gemini-2.5-flash-lite"


def node_parse_document(state: DocumentState) -> dict:
    text, page_count = parse_document(state["pdf_path"])
    return {"document_text": text, "page_count": page_count}


def node_extract_metrics(state: DocumentState) -> dict:
    llm = get_llm(EXTRACTION_MODEL)
    langfuse = get_langfuse_handler()
    tracker = CostTracker(model=EXTRACTION_MODEL)

    structured_llm = llm.with_structured_output(FinancialMetrics)

    system = SystemMessage(content=(
        "You are a financial analyst. Extract structured financial data from the document. "
        "Return only what is explicitly stated — use null for missing values. "
        "For revenue and monetary values, convert to USD millions. "
        "For key_risks, extract up to 5 specific risks mentioned in the document."
    ))
    human = HumanMessage(content=(
        f"Extract financial metrics from this document:\n\n{state['document_text']}"
    ))

    metrics = structured_llm.invoke(
        [system, human],
        config={"callbacks": [langfuse, tracker]},
    )
    return {
        "metrics": metrics,
        "cost_usd": state.get("cost_usd", 0.0) + tracker.cost_usd,
    }


def node_classify_risks(state: DocumentState) -> dict:
    m = state["metrics"]
    risk_text = " ".join(m.key_risks)
    financial_signals = (
        f"Net income: {m.net_income_usd_m}M. "
        f"EBITDA: {m.ebitda_usd_m}M. "
        f"YoY growth: {m.yoy_revenue_growth_pct}%. "
        f"Management tone: {m.management_tone}. "
        f"Risks: {risk_text}"
    )
    red_flags = retrieve_red_flags(query=financial_signals, k=5)
    return {"red_flags_context": red_flags}


def node_generate_report(state: DocumentState) -> dict:
    llm = get_llm(EXTRACTION_MODEL)
    langfuse = get_langfuse_handler()
    tracker = CostTracker(model=EXTRACTION_MODEL)

    m = state["metrics"]
    critique_section = (
        f"\n\nPrevious attempt feedback (retry {state['retry_count']}):\n{state['critique_feedback']}"
        if state.get("retry_count", 0) > 0 else ""
    )

    red_flags_section = ""
    if state.get("red_flags_context"):
        flags = "\n".join(f"- {rf}" for rf in state["red_flags_context"])
        red_flags_section = f"\nMatched risk patterns from red flags library:\n{flags}\n"

    prompt = f"""You are writing a financial intelligence brief for an investor or analyst.

Company: {m.company_name} | Period: {m.period} | Document: {m.document_type}

Key metrics:
- Revenue: {m.revenue_usd_m}M USD | YoY growth: {m.yoy_revenue_growth_pct}%
- Gross margin: {m.gross_margin_pct}% | EBITDA: {m.ebitda_usd_m}M USD
- Net income: {m.net_income_usd_m}M USD
- Cash: {m.cash_usd_m}M USD | Debt: {m.debt_usd_m}M USD

Risk score: {m.risk_score}
Risk justification: {m.risk_justification}

Top risks:
{chr(10).join(f'- {r}' for r in m.key_risks)}
{red_flags_section}
Guidance: {m.guidance_summary}
Management tone: {m.management_tone}
{critique_section}

Write a concise 3-paragraph intelligence brief:
1. Financial performance summary (what happened, key numbers)
2. Risk assessment (what concerns, why the risk score)
3. Outlook (guidance, what to watch)

Be specific with numbers. No filler sentences."""

    result = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [langfuse, tracker]},
    )
    return {
        "report": result.content,
        "cost_usd": state.get("cost_usd", 0.0) + tracker.cost_usd,
    }


def node_critique_report(state: DocumentState) -> dict:
    llm = get_llm(CRITIQUE_MODEL)
    langfuse = get_langfuse_handler()
    tracker = CostTracker(model=CRITIQUE_MODEL)

    prompt = f"""Score this financial intelligence brief on 3 criteria (1–3 each, max 9):

1. GROUNDED — all claims supported by the metrics provided (no hallucinated numbers)
2. ACTIONABLE — gives the reader clear takeaways, not just a summary
3. CONCISE — 3 paragraphs, no filler, specific numbers used

Brief to score:
{state['report']}

Respond in this exact format:
GROUNDED: <1-3>
ACTIONABLE: <1-3>
CONCISE: <1-3>
TOTAL: <sum>
FEEDBACK: <one sentence on what to improve, or "none" if all good>"""

    result = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [langfuse, tracker]},
    )

    score = 0
    feedback = ""
    for line in result.content.strip().splitlines():
        if line.startswith("TOTAL:"):
            try:
                score = int(line.split(":")[1].strip())
            except ValueError:
                score = 0
        if line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return {
        "critique_score": score,
        "critique_feedback": feedback,
        "cost_usd": state.get("cost_usd", 0.0) + tracker.cost_usd,
    }


def should_retry(state: DocumentState) -> str:
    if state.get("critique_score", 0) < 7 and state.get("retry_count", 0) < 2:
        return "retry"
    return "done"
