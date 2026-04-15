"""
Inject a demo peer comparison session (Grab vs Sea Limited, FY2024)
into SQLite history and demo_data.json — no API calls needed.

Usage:
    cd financial-doc-agent
    python scripts/inject_demo_peer.py
"""

import json
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.history_store import init_db, save_session

DB_PATH        = Path("data/history.db")
DEMO_DATA_PATH = Path("data/demo_data.json")
FRONTEND_DEMO  = Path("frontend/demo_data.json")


def fetch_doc(company_name: str, period: str) -> dict:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM analyses WHERE company_name=? AND period=? LIMIT 1",
        (company_name, period),
    ).fetchone()
    con.close()
    if not row:
        raise ValueError(f"No analysis found: {company_name} {period}")
    d = dict(row)
    d["metrics"]           = json.loads(d["metrics_json"])
    d["key_risks"]         = json.loads(d["key_risks_json"])
    d["red_flags_matched"] = json.loads(d["red_flags_json"])
    return d


def compute_deltas(m_a: dict, m_b: dict) -> list[dict]:
    fields = [
        ("Revenue (USD M)",        "revenue_usd_m"),
        ("Gross Margin (%)",        "gross_margin_pct"),
        ("EBITDA (USD M)",          "ebitda_usd_m"),
        ("Net Income (USD M)",      "net_income_usd_m"),
        ("Cash (USD M)",            "cash_usd_m"),
        ("Debt (USD M)",            "debt_usd_m"),
        ("YoY Revenue Growth (%)",  "yoy_revenue_growth_pct"),
    ]
    deltas = []
    for label, key in fields:
        va = m_a.get(key)
        vb = m_b.get(key)
        if va is None or vb is None:
            deltas.append({"label": label, "value_a": va, "value_b": vb, "delta": None, "direction": "n/a"})
        else:
            diff = round(vb - va, 2)
            direction = "up" if diff > 0 else ("down" if diff < 0 else "flat")
            deltas.append({"label": label, "value_a": va, "value_b": vb, "delta": diff, "direction": direction})
    return deltas


PEER_REPORT = """**Grab Holdings edges ahead on scale; Sea Limited leads on profitability.**

Grab reported FY2024 revenue of $2,815M USD — 19% above Sea's $2,363M USD — driven by its broader Southeast Asian footprint spanning ride-hailing, food delivery, and financial services across eight markets. However, Sea Limited demonstrated superior bottom-line discipline: net income of $306M USD versus Grab's -$95M USD loss, and positive EBITDA of $498M USD versus Grab's -$81M USD. Sea's gross margin advantage also reflects a more mature unit economics profile, particularly in its Garena gaming and Shopee e-commerce segments.

Grab carries significantly higher debt ($1,584M USD vs Sea's $454M USD) and remains in cash-burn mode, though its negative EBITDA narrowed materially year-over-year. Management tone at Grab is cautious with explicit guidance on the path to adjusted EBITDA breakeven; Sea's tone is positive, reflecting the turnaround achieved after its 2022–2023 cost-cutting cycle. Both companies operate in overlapping markets (Indonesia, Thailand, Vietnam) but with different competitive moats: Grab's is regulatory relationships and driver supply density, Sea's is digital payments penetration through SeaMoney.

Overall, Sea Limited is in a stronger financial position for FY2024 — profitable, lower-leveraged, and generating positive cash flow. Grab remains the larger operator by GMV and user count but must demonstrate that scale converts to earnings. Watch Grab's Q1-Q2 2025 EBITDA trend and Sea's Shopee take-rate trajectory as the key leading indicators for which company compounds value faster over the next 12 months."""


def main():
    init_db()

    grab = fetch_doc("Grab", "FY2024")
    sea  = fetch_doc("Sea Limited", "FY2024")

    deltas = compute_deltas(grab["metrics"], sea["metrics"])

    comparison = {
        "comparison_type":  "peer",
        "company_a":        grab["company_name"],
        "company_b":        sea["company_name"],
        "period_a":         grab["period"],
        "period_b":         sea["period"],
        "deltas":           deltas,
        "risk_score_a":     grab["risk_score"],
        "risk_score_b":     sea["risk_score"],
        "management_tone_a": grab["management_tone"],
        "management_tone_b": sea["management_tone"],
        "comparison_report": PEER_REPORT,
    }

    # Build doc dicts in the shape save_session expects
    def to_doc(d: dict) -> dict:
        return {
            "filename":          d.get("filename", ""),
            "company_name":      d["company_name"],
            "period":            d["period"],
            "document_type":     d["document_type"],
            "metrics":           d["metrics"],
            "guidance_summary":  d.get("guidance_summary"),
            "key_risks":         d["key_risks"],
            "management_tone":   d["management_tone"],
            "risk_score":        d["risk_score"],
            "risk_justification": d.get("risk_justification", ""),
            "red_flags_matched": d["red_flags_matched"],
            "report":            d["report"],
            "critique_score":    d.get("critique_score", 0),
            "page_count":        d.get("page_count", 0),
            "cost_usd":          d.get("cost_usd", 0.0),
        }

    documents  = [to_doc(grab), to_doc(sea)]
    total_cost = grab.get("cost_usd", 0.0) + sea.get("cost_usd", 0.0)

    session_id = save_session(documents, [comparison], total_cost, session_type="peer")
    print(f"Inserted peer session: {session_id}")

    # Build the full session object for demo_data.json
    new_session = {
        "session_id":      session_id,
        "session_type":    "peer",
        "documents":       documents,
        "comparisons":     [comparison],
        "total_cost_usd":  round(total_cost, 6),
    }

    # Update both demo_data.json files
    for path in [DEMO_DATA_PATH, FRONTEND_DEMO]:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"sessions": []}
        # Remove any existing peer session to avoid duplicates
        data["sessions"] = [s for s in data["sessions"] if s.get("session_type") != "peer"]
        data["sessions"].insert(0, new_session)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Updated: {path}")

    print("Done.")


if __name__ == "__main__":
    main()
