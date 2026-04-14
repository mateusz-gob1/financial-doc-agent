import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = "data/history.db"


@contextmanager
def _conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                doc_count INTEGER NOT NULL,
                company_name TEXT,
                periods TEXT,        -- JSON list
                total_cost_usd REAL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                filename TEXT,
                company_name TEXT NOT NULL,
                period TEXT NOT NULL,
                document_type TEXT,
                metrics_json TEXT NOT NULL,
                key_risks_json TEXT NOT NULL,
                red_flags_json TEXT NOT NULL,
                report TEXT NOT NULL,
                critique_score INTEGER,
                management_tone TEXT,
                risk_score TEXT,
                risk_justification TEXT,
                guidance_summary TEXT,
                page_count INTEGER,
                cost_usd REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                period_a TEXT NOT NULL,
                period_b TEXT NOT NULL,
                deltas_json TEXT NOT NULL,
                comparison_report TEXT NOT NULL,
                risk_score_a TEXT,
                risk_score_b TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)


def save_session(documents: list[dict], comparisons: list[dict], total_cost: float) -> str:
    """Persist a full multi-document session. Returns the session ID."""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    company = documents[0]["company_name"] if documents else ""
    periods = [d["period"] for d in documents]

    with _conn() as con:
        con.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?)",
            (session_id, now, len(documents), company, json.dumps(periods), total_cost),
        )
        for doc in documents:
            con.execute(
                """INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()), session_id,
                    doc.get("filename", ""),
                    doc["company_name"], doc["period"], doc["document_type"],
                    json.dumps(doc["metrics"]),
                    json.dumps(doc["key_risks"]),
                    json.dumps(doc["red_flags_matched"]),
                    doc["report"], doc["critique_score"],
                    doc["management_tone"], doc["risk_score"],
                    doc.get("risk_justification", ""),
                    doc.get("guidance_summary", ""),
                    doc.get("page_count", 0),
                    doc.get("cost_usd", 0.0),
                    now,
                ),
            )
        for comp in comparisons:
            con.execute(
                "INSERT INTO comparisons VALUES (?,?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()), session_id,
                    comp["period_a"], comp["period_b"],
                    json.dumps(comp["deltas"]),
                    comp["comparison_report"],
                    comp.get("risk_score_a", ""), comp.get("risk_score_b", ""),
                ),
            )
    return session_id


def list_sessions(limit: int = 20) -> list[dict]:
    """Return recent sessions for the History tab."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    result = [dict(r) for r in rows]
    for r in result:
        r["session_id"] = r["id"]
    return result


def get_session(session_id: str) -> dict | None:
    """Return full session data including analyses and comparisons."""
    with _conn() as con:
        session = con.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if not session:
            return None

        analyses = con.execute(
            "SELECT * FROM analyses WHERE session_id=? ORDER BY created_at", (session_id,)
        ).fetchall()
        comparisons = con.execute(
            "SELECT * FROM comparisons WHERE session_id=?", (session_id,)
        ).fetchall()

    def parse_analysis(row):
        d = dict(row)
        d["metrics"] = json.loads(d["metrics_json"])
        d["key_risks"] = json.loads(d["key_risks_json"])
        d["red_flags_matched"] = json.loads(d["red_flags_json"])
        return d

    def parse_comparison(row):
        d = dict(row)
        d["deltas"] = json.loads(d["deltas_json"])
        return d

    return {
        **dict(session),
        "session_id": session["id"],
        "periods": json.loads(session["periods"]),
        "documents": [parse_analysis(a) for a in analyses],
        "comparisons": [parse_comparison(c) for c in comparisons],
    }
