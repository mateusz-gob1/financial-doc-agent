import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "history.db"
DEMO_DATA_PATH = Path(__file__).parent.parent / "data" / "demo_data.json"


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
                comparison_type TEXT DEFAULT 'temporal',
                company_a_name TEXT DEFAULT '',
                company_b_name TEXT DEFAULT '',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Migrations for databases created before these columns existed
        for migration in [
            "ALTER TABLE sessions ADD COLUMN session_type TEXT DEFAULT 'temporal'",
            "ALTER TABLE comparisons ADD COLUMN comparison_type TEXT DEFAULT 'temporal'",
            "ALTER TABLE comparisons ADD COLUMN company_a_name TEXT DEFAULT ''",
            "ALTER TABLE comparisons ADD COLUMN company_b_name TEXT DEFAULT ''",
        ]:
            try:
                con.execute(migration)
            except Exception:
                pass


def save_session(
    documents: list[dict],
    comparisons: list[dict],
    total_cost: float,
    session_type: str = "temporal",
) -> str:
    """Persist a full multi-document session. Returns the session ID."""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    if session_type == "peer" and len(documents) == 2:
        company = f"{documents[0]['company_name']} vs {documents[1]['company_name']}"
    else:
        company = documents[0]["company_name"] if documents else ""

    periods = [d["period"] for d in documents]

    with _conn() as con:
        con.execute(
            """INSERT INTO sessions
               (id, created_at, doc_count, company_name, periods, total_cost_usd, session_type)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, now, len(documents), company, json.dumps(periods), total_cost, session_type),
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
                """INSERT INTO comparisons
                   (id, session_id, period_a, period_b, deltas_json, comparison_report,
                    risk_score_a, risk_score_b, comparison_type, company_a_name, company_b_name)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()), session_id,
                    comp["period_a"], comp["period_b"],
                    json.dumps(comp["deltas"]),
                    comp["comparison_report"],
                    comp.get("risk_score_a", ""), comp.get("risk_score_b", ""),
                    comp.get("comparison_type", "temporal"),
                    comp.get("company_a", ""), comp.get("company_b", ""),
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


def seed_demo_data() -> None:
    """Populate the DB with pre-computed demo sessions if it is empty.

    Safe to call on every startup — does nothing when sessions already exist.
    Requires data/demo_data.json to be present (included in Docker image).
    """
    if not DEMO_DATA_PATH.exists():
        return

    with _conn() as con:
        count = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    if count > 0:
        return

    with open(DEMO_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    for session in data.get("sessions", []):
        session_type = session.get("session_type", "temporal")
        save_session(
            documents=session["documents"],
            comparisons=session["comparisons"],
            total_cost=session.get("total_cost_usd", 0.0),
            session_type=session_type,
        )

    print(f"Seeded {len(data.get('sessions', []))} demo sessions into DB.")
