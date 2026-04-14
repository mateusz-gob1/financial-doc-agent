import re
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from schemas.financial import FinancialMetrics
from tools.llm_client import get_llm, get_langfuse_handler


COMPARISON_MODEL = "google/gemini-2.5-flash"

QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def period_sort_key(period: str) -> tuple[int, int]:
    """
    Parse a period string into a sortable (year, quarter) tuple.
    Examples: "FY2023" → (2023, 0), "Q4 2024" → (2024, 4), "2022" → (2022, 0)
    """
    period = period.strip()
    # Match quarter: "Q4 2024" or "2024 Q4"
    q_match = re.search(r'Q([1-4])', period, re.IGNORECASE)
    y_match = re.search(r'(20\d{2})', period)
    year = int(y_match.group(1)) if y_match else 0
    quarter = int(q_match.group(1)) if q_match else 0
    return (year, quarter)


def sort_results_by_period(results: list[dict]) -> list[dict]:
    """Sort a list of pipeline result dicts chronologically by metrics.period."""
    return sorted(results, key=lambda r: period_sort_key(r["metrics"].period))


@dataclass
class MetricDelta:
    label: str
    value_a: float | None
    value_b: float | None
    delta: float | None
    direction: str  # "up", "down", "flat", "n/a"


def compute_deltas(a: FinancialMetrics, b: FinancialMetrics) -> list[MetricDelta]:
    """Compute numeric deltas between two periods. a = older, b = newer."""

    def delta(va, vb, label) -> MetricDelta:
        if va is None or vb is None:
            return MetricDelta(label, va, vb, None, "n/a")
        diff = vb - va
        direction = "up" if diff > 0 else ("down" if diff < 0 else "flat")
        return MetricDelta(label, va, vb, round(diff, 2), direction)

    return [
        delta(a.revenue_usd_m, b.revenue_usd_m, "Revenue (USD M)"),
        delta(a.gross_margin_pct, b.gross_margin_pct, "Gross Margin (%)"),
        delta(a.ebitda_usd_m, b.ebitda_usd_m, "EBITDA (USD M)"),
        delta(a.net_income_usd_m, b.net_income_usd_m, "Net Income (USD M)"),
        delta(a.cash_usd_m, b.cash_usd_m, "Cash (USD M)"),
        delta(a.debt_usd_m, b.debt_usd_m, "Debt (USD M)"),
        delta(a.yoy_revenue_growth_pct, b.yoy_revenue_growth_pct, "YoY Revenue Growth (%)"),
    ]


def _format_deltas(deltas: list[MetricDelta]) -> str:
    lines = []
    for d in deltas:
        if d.direction == "n/a":
            lines.append(f"  {d.label}: {d.value_a} → {d.value_b} (no delta)")
        else:
            arrow = "▲" if d.direction == "up" else ("▼" if d.direction == "down" else "→")
            lines.append(f"  {d.label}: {d.value_a} → {d.value_b} ({arrow} {d.delta:+.1f})")
    return "\n".join(lines)


def generate_comparison_report(
    a: FinancialMetrics,
    b: FinancialMetrics,
    deltas: list[MetricDelta],
) -> str:
    """LLM generates a comparison narrative. a = older period, b = newer period."""
    llm = get_llm(COMPARISON_MODEL)
    langfuse = get_langfuse_handler()

    prompt = f"""You are a financial analyst writing a period-over-period comparison brief.

Company: {b.company_name}
Comparing: {a.period} (baseline) → {b.period} (current)

Metric changes:
{_format_deltas(deltas)}

Risk score: {a.risk_score} → {b.risk_score}
Management tone: {a.management_tone} → {b.management_tone}

Risks in {a.period}: {"; ".join(a.key_risks[:3])}
Risks in {b.period}: {"; ".join(b.key_risks[:3])}

Write a concise 3-paragraph comparison brief:
1. What improved — specific metrics, by how much, why it matters
2. What deteriorated or stayed concerning — specific metrics, risks that persist
3. Overall trajectory — is the company moving in the right direction? What to watch next

Use exact numbers. Be direct. No filler."""

    result = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [langfuse]},
    )
    return result.content


def compare_documents(a: FinancialMetrics, b: FinancialMetrics) -> dict:
    """
    Full comparison between two processed documents.
    Returns a dict ready for JSON serialization.
    """
    deltas = compute_deltas(a, b)
    report = generate_comparison_report(a, b, deltas)

    return {
        "company": b.company_name,
        "period_a": a.period,
        "period_b": b.period,
        "deltas": [
            {
                "label": d.label,
                "value_a": d.value_a,
                "value_b": d.value_b,
                "delta": d.delta,
                "direction": d.direction,
            }
            for d in deltas
        ],
        "risk_score_a": a.risk_score,
        "risk_score_b": b.risk_score,
        "management_tone_a": a.management_tone,
        "management_tone_b": b.management_tone,
        "comparison_report": report,
    }
