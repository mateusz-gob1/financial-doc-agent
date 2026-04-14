import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from agents.graph import build_graph
from tools.comparator import compare_documents, sort_results_by_period
from tools.history_store import get_session, init_db, list_sessions, save_session
from tools.vector_store import build_red_flags_index

app = FastAPI(title="Financial Document Intelligence Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
build_red_flags_index()
graph = build_graph()


def _initial_state(pdf_path: str) -> dict:
    return {
        "pdf_path": pdf_path,
        "document_text": "",
        "page_count": 0,
        "metrics": None,
        "red_flags_context": [],
        "report": "",
        "critique_score": 0,
        "critique_feedback": "",
        "retry_count": 0,
        "cost_usd": 0.0,
        "error": "",
    }


def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    return tmp.name


def _result_to_dict(result: dict, filename: str = "") -> dict:
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


@app.post("/api/analyze-multi")
async def analyze_multi(files: List[UploadFile] = File(...)):
    """
    Analyze 1–4 financial documents.
    Documents are auto-sorted chronologically by extracted period.
    Returns per-document analysis + pairwise comparisons between consecutive periods.
    """
    if not (1 <= len(files) <= 4):
        raise HTTPException(status_code=400, detail="Upload between 1 and 4 PDF files.")

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF.")

    filenames = [f.filename for f in files]
    paths = [_save_upload(f) for f in files]

    try:
        raw_results = [graph.invoke(_initial_state(p)) for p in paths]

        # Build period → filename map before sorting (upload order may differ from period order)
        period_to_filename = {
            r["metrics"].period: fn
            for r, fn in zip(raw_results, filenames)
            if r.get("metrics")
        }

        # Sort chronologically by extracted period (no manual ordering needed)
        sorted_results = sort_results_by_period(raw_results)

        documents = [
            _result_to_dict(r, filename=period_to_filename.get(r["metrics"].period, ""))
            for r in sorted_results
        ]

        # Pairwise comparisons between consecutive periods
        comparisons = []
        for i in range(len(sorted_results) - 1):
            a = sorted_results[i]["metrics"]
            b = sorted_results[i + 1]["metrics"]
            comparisons.append(compare_documents(a, b))

        total_cost = sum(d["cost_usd"] for d in documents)
        session_id = save_session(documents, comparisons, total_cost)

        return JSONResponse({
            "session_id": session_id,
            "documents": documents,
            "comparisons": comparisons,
            "total_cost_usd": round(total_cost, 6),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in paths:
            os.unlink(p)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/history")
def history():
    return JSONResponse(list_sessions(limit=20))


@app.get("/api/history/{session_id}")
def history_session(session_id: str):
    data = get_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    return JSONResponse(data)


frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
