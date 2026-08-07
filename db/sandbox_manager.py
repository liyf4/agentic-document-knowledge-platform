import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config import DB_NAME


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_NAME)
    connection.row_factory = sqlite3.Row
    return connection


def init_sandbox_db() -> None:
    connection = _connect()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_sandboxes (
                session_id TEXT PRIMARY KEY,
                sandbox_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                expires_at TEXT,
                error TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def get_session_sandbox(session_id: str) -> Optional[Dict[str, Any]]:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT * FROM session_sandboxes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def upsert_session_sandbox(
    session_id: str,
    sandbox_id: str,
    provider: str,
    status: str,
    *,
    expires_at: Optional[str] = None,
    error: str = "",
) -> Dict[str, Any]:
    now = datetime.now().isoformat()
    connection = _connect()
    try:
        connection.execute(
            """
            INSERT INTO session_sandboxes
                (session_id, sandbox_id, provider, status, created_at, last_used_at, expires_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                sandbox_id = excluded.sandbox_id,
                provider = excluded.provider,
                status = excluded.status,
                last_used_at = excluded.last_used_at,
                expires_at = excluded.expires_at,
                error = excluded.error
            """,
            (session_id, sandbox_id, provider, status, now, now, expires_at, error),
        )
        connection.commit()
    finally:
        connection.close()
    return get_session_sandbox(session_id) or {}


def update_session_sandbox(session_id: str, status: str, error: str = "") -> None:
    connection = _connect()
    try:
        connection.execute(
            """
            UPDATE session_sandboxes
            SET status = ?, error = ?, last_used_at = ?
            WHERE session_id = ?
            """,
            (status, error, datetime.now().isoformat(), session_id),
        )
        connection.commit()
    finally:
        connection.close()


def touch_session_sandbox(session_id: str) -> None:
    connection = _connect()
    try:
        connection.execute(
            "UPDATE session_sandboxes SET last_used_at = ? WHERE session_id = ?",
            (datetime.now().isoformat(), session_id),
        )
        connection.commit()
    finally:
        connection.close()


def touch_and_extend_session_sandbox(session_id: str, timeout_seconds: int) -> None:
    now = datetime.now()
    connection = _connect()
    try:
        connection.execute(
            "UPDATE session_sandboxes SET last_used_at = ?, expires_at = ?, status = 'running', error = '' WHERE session_id = ?",
            (now.isoformat(), (now + timedelta(seconds=timeout_seconds)).isoformat(), session_id),
        )
        connection.commit()
    finally:
        connection.close()


def list_idle_session_sandboxes(idle_seconds: int) -> List[Dict[str, Any]]:
    cutoff = (datetime.now() - timedelta(seconds=idle_seconds)).isoformat()
    connection = _connect()
    try:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM session_sandboxes WHERE last_used_at <= ?", (cutoff,)
        ).fetchall()]
    finally:
        connection.close()


def delete_session_sandbox(session_id: str) -> None:
    connection = _connect()
    try:
        connection.execute("DELETE FROM session_sandboxes WHERE session_id = ?", (session_id,))
        connection.commit()
    finally:
        connection.close()
