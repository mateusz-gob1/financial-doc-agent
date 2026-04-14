"""
Generate demo_data.json for HF Spaces deployment and populate SQLite history.

Sessions created:
  - Grab Holdings FY2022/FY2023/FY2024  (multi-doc, period comparison)
  - Sea Limited FY2024                  (single-doc)
  - Uber FY2024                         (single-doc)
  - Airbnb FY2024                       (single-doc)

All 4 sessions go to data/demo_data.json AND SQLite history.

Usage:
    cd financial-doc-agent
    python scripts/generate_demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agents.graph import build_graph
from tools.comparator import compare_documents, sort_results_by_period
from tools.history_store import init_db, save_session
from tools.vector_store import build_red_flags_index

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "data" / "sample_docs"
OUTPUT_PATH     = Path(__file__).parent.parent / "data" / "demo_data.json"

GRAB_PDFS   = sorted(SAMPLE_DOCS_DIR.glob("Grab*.pdf"))
SINGLE_PDFS = [
    SAMPLE_DOCS_DIR / "Sea Limited Form 20-F-2025.pdf",
    SAMPLE_DOCS_DIR / "Uber Form 10-K-2025.pdf",
    SAMPLE_DOCS_DIR / "Airbnb Form 10-K-2025.pdf",
]


def initial_state(pdf_path: str) -> dict:
    return {
        "pdf_path": pdf_path, "document_text": "", "page_count": 0,
        "metrics": None, "red_flags_context": [], "report": "",
        "critique_score": 0, "critique_feedback": "", "retry_count": 0,
        "cost_usd": 0.0, "error": "",
    }


def result_to_dict(result: dict, filename: str = "") -> dict:
    m = result["metrics"]
    return {
        "filename": filename,
        "company_name": m.company_name,
        "period": m.period,
        "document_type": m.document_type,
        "metrics": {
            "revenue_usd_m": m.revenue_usd_m,
            "gross_margin_pct": m.gross_margin_pct,
            "ebitda_usd_m": m.ebitda_usd_m,
            "net_income_usd_m": m.net_income_usd_m,
            "cash_usd_m": m.cash_usd_m,
            "debt_usd_m": m.debt_usd_m,
            "yoy_revenue_growth_pct": m.yoy_revenue_growth_pct,
        },
        "guidance_summary": m.guidance_summary,
        "key_risks": m.key_risks,
        "management_tone": m.management_tone,
        "risk_score": m.risk_score,
        "risk_justification": m.risk_justification,
        "red_flags_matched": result["red_flags_context"],
        "report": result["report"],
        "critique_score": result["critique_score"],
        "page_count": result["page_count"],
        "cost_usd": result.get("cost_usd", 0.0),
    }


def run_pdf(pdf: Path, graph) -> dict | None:
    print(f"  Processing: {pdf.name} ...")
    if not pdf.exists():
        print(f"    ✗ File not found, skipping.")
        return None
    result = graph.invoke(initial_state(str(pdf)))
    m = result.get("metrics")
    if not m:
        print(f"    ✗ Extraction failed.")
        return None
    print(f"    ✓ {m.company_name} | {m.period} | cost: ${result.get('cost_usd', 0):.4f}")
    return result


def build_session(raw: list[tuple[dict, str]]) -> dict:
    """Turn a list of (result, filename) into a session dict."""
    period_to_fn   = {r["metrics"].period: fn for r, fn in raw if r.get("metrics")}
    sorted_results = sort_results_by_period([r for r, _ in raw])
    documents      = [
        result_to_dict(r, filename=period_to_fn.get(r["metrics"].period, ""))
        for r in sorted_results
    ]
    comparisons = []
    for i in range(len(sorted_results) - 1):
        comp = compare_documents(sorted_results[i]["metrics"], sorted_results[i + 1]["metrics"])
        comparisons.append(comp)

    total_cost = sum(d["cost_usd"] for d in documents)
    session_id = save_session(documents, comparisons, total_cost)
    return {
        "session_id": session_id,
        "documents": documents,
        "comparisons": comparisons,
        "total_cost_usd": round(total_cost, 6),
    }


def main():
    init_db()
    build_red_flags_index()
    graph = build_graph()

    all_sessions = []
    total_cost_all = 0.0

    # ── Session 1: Grab multi-doc ────────────────────────────────────────────
    print(f"\n{'='*55}\nSESSION 1 — Grab Holdings ({len(GRAB_PDFS)} docs)\n{'='*55}")
    if GRAB_PDFS:
        raw = [(run_pdf(p, graph), p.name) for p in GRAB_PDFS]
        raw = [(r, fn) for r, fn in raw if r]
        if raw:
            session = build_session(raw)
            all_sessions.append(session)
            total_cost_all += session["total_cost_usd"]
            print(f"  → cost: ${session['total_cost_usd']:.4f}")

    # ── Sessions 2-4: single-doc ─────────────────────────────────────────────
    for i, pdf in enumerate(SINGLE_PDFS, start=2):
        print(f"\n{'='*55}\nSESSION {i} — {pdf.stem}\n{'='*55}")
        result = run_pdf(pdf, graph)
        if result:
            session = build_session([(result, pdf.name)])
            all_sessions.append(session)
            total_cost_all += session["total_cost_usd"]
            print(f"  → cost: ${session['total_cost_usd']:.4f}")

    # ── Write demo_data.json ─────────────────────────────────────────────────
    OUTPUT_PATH.write_text(json.dumps({"sessions": all_sessions}, indent=2, default=str))

    print(f"\n{'='*55}")
    print(f"Done. {len(all_sessions)} sessions generated.")
    print(f"Total cost: ${total_cost_all:.4f}")
    print(f"\nNext step:")
    print(f"  copy data\\demo_data.json frontend\\demo_data.json")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
