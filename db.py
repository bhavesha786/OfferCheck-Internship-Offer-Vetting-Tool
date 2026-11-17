"""SQLite tracker for analysed offers."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.environ.get("OFFERCHECK_DB", Path(__file__).resolve().parent / "offercheck.db"))
STATUSES = ["Considering", "Negotiating", "Accepted", "Declined", "Avoided (risky)"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS offers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    company TEXT NOT NULL,
    role TEXT NOT NULL,
    final_risk_score INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Considering',
    notes TEXT NOT NULL DEFAULT '',
    report_json TEXT NOT NULL,
    offer_text TEXT NOT NULL
)
"""


@contextmanager
def _connect(path=None):
    conn = sqlite3.connect(str(path or DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path=None) -> None:
    with _connect(path) as conn:
        conn.execute(_SCHEMA)


def save_offer(report: dict, offer_text: str, path=None) -> int:
    with _connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO offers (created_at, company, role, final_risk_score, verdict, report_json, offer_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                report.get("company") or "Unknown",
                report.get("role") or "Unknown",
                int(report["final_risk_score"]),
                report["verdict"],
                json.dumps(report),
                offer_text,
            ),
        )
        return int(cur.lastrowid)


def list_offers(path=None) -> list:
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT id, created_at, company, role, final_risk_score, verdict, status, notes "
            "FROM offers ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_offer(offer_id: int, path=None) -> Optional[dict]:
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM offers WHERE id = ?", (offer_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["report"] = json.loads(data.pop("report_json"))
    return data


def update_offer(offer_id: int, status: str, notes: str, path=None) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    with _connect(path) as conn:
        conn.execute("UPDATE offers SET status = ?, notes = ? WHERE id = ?", (status, notes, offer_id))


def delete_offer(offer_id: int, path=None) -> None:
    with _connect(path) as conn:
        conn.execute("DELETE FROM offers WHERE id = ?", (offer_id,))
