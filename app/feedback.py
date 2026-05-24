import json
import sqlite3
from datetime import datetime
from typing import Dict, List

from app import config


def init_feedback_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS answers (
            answer_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            chunk_ids TEXT,
            local_sources TEXT,
            web_sources TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            answer_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            thumb TEXT NOT NULL,
            reason_tags TEXT,
            note TEXT,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()


def save_answer_record(
    conn: sqlite3.Connection,
    answer_id: str,
    session_id: str,
    query: str,
    answer: str,
    chunk_ids: List[str],
    local_sources: List[str],
    web_sources: List[str],
):
    conn.execute(
        """
        INSERT OR REPLACE INTO answers
        (answer_id, session_id, timestamp, query, answer, chunk_ids, local_sources, web_sources)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            answer_id,
            session_id,
            datetime.now().isoformat(),
            query,
            answer,
            json.dumps(chunk_ids),
            json.dumps(local_sources),
            json.dumps(web_sources),
        ),
    )
    conn.commit()


def save_feedback(
    conn: sqlite3.Connection,
    answer_id: str,
    session_id: str,
    thumb: str,
    reason_tags: List[str],
    note: str,
):
    conn.execute(
        """
        INSERT INTO feedback (answer_id, session_id, thumb, reason_tags, note, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            answer_id,
            session_id,
            thumb,
            json.dumps(reason_tags or []),
            note,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()


def compute_feedback_priors(conn: sqlite3.Connection) -> Dict[str, float]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.chunk_ids, f.thumb, COUNT(*)
        FROM feedback f
        JOIN answers a ON a.answer_id = f.answer_id
        GROUP BY a.chunk_ids, f.thumb
        """
    )

    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for chunk_ids_json, thumb, count in cur.fetchall():
        if not chunk_ids_json:
            continue
        chunk_ids = json.loads(chunk_ids_json)
        delta = 1.0 if thumb == "up" else -1.0
        for chunk_id in chunk_ids:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (delta * count)
            counts[chunk_id] = counts.get(chunk_id, 0) + count

    priors: Dict[str, float] = {}
    for chunk_id, raw_score in scores.items():
        total = counts.get(chunk_id, 0)
        if total < config.FEEDBACK_MIN_VOTES:
            continue
        if abs(raw_score) < 1.0:
            continue

        normalized = raw_score / float(total)
        decayed = normalized * config.FEEDBACK_DECAY
        bounded = max(config.FEEDBACK_MAX_PENALTY, min(config.FEEDBACK_MAX_BOOST, decayed))
        priors[chunk_id] = bounded

    return priors


def feedback_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM feedback WHERE thumb='up'")
    up = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM feedback WHERE thumb='down'")
    down = cur.fetchone()[0]
    return {
        "total_feedback": total,
        "up_votes": up,
        "down_votes": down,
        "satisfaction": round((up / total) * 100, 2) if total else 0.0,
    }
