import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import DB_NAME


def _connect():
    return sqlite3.connect(DB_NAME)


def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_meta_db():
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                doc_summary TEXT,
                created_at TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_documents (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                file_path TEXT,
                chunk_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                uploaded_at TIMESTAMP NOT NULL
            )
            """
        )
        _ensure_column(conn, "session_documents", "file_path", "file_path TEXT")
        _ensure_column(conn, "sessions", "conversation_summary", "conversation_summary TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "sessions", "summarized_message_count", "summarized_message_count INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "sessions", "context_usage_json", "context_usage_json TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(conn, "sessions", "last_compacted_at", "last_compacted_at TEXT")
        conn.commit()
    finally:
        conn.close()


def create_new_session(title="新对话"):
    session_id = str(uuid.uuid4())
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO sessions (id, title, doc_summary, created_at) VALUES (?, ?, ?, ?)",
            (session_id, title, "", datetime.now()),
        )
        conn.commit()
        return session_id
    finally:
        conn.close()


def update_session_title(session_id, new_title):
    conn = _connect()
    try:
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        conn.commit()
    finally:
        conn.close()


def update_session_summary(session_id, summary):
    conn = _connect()
    try:
        conn.execute("UPDATE sessions SET doc_summary = ? WHERE id = ?", (summary, session_id))
        conn.commit()
    finally:
        conn.close()


def get_all_sessions():
    conn = _connect()
    try:
        return conn.execute("SELECT id, title FROM sessions ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()


def get_session_summary(session_id):
    conn = _connect()
    try:
        res = conn.execute("SELECT doc_summary FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return res[0] if res else ""
    finally:
        conn.close()


def get_conversation_context_state(session_id: str) -> Dict[str, object]:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT COALESCE(conversation_summary, ''), COALESCE(summarized_message_count, 0),
                   COALESCE(context_usage_json, '{}'), last_compacted_at
            FROM sessions WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if not row:
            return {"conversation_summary": "", "summarized_message_count": 0,
                    "context_usage": {}, "last_compacted_at": None}
        try:
            usage = json.loads(row[2] or "{}")
        except (TypeError, json.JSONDecodeError):
            usage = {}
        return {
            "conversation_summary": str(row[0] or ""),
            "summarized_message_count": int(row[1] or 0),
            "context_usage": usage if isinstance(usage, dict) else {},
            "last_compacted_at": row[3],
        }
    finally:
        conn.close()


def update_conversation_summary(
    session_id: str, summary: str, summarized_message_count: int, compacted_at: str
) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE sessions
            SET conversation_summary = ?, summarized_message_count = ?, last_compacted_at = ?
            WHERE id = ?
            """,
            (summary, int(summarized_message_count), compacted_at, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_context_usage(session_id: str, usage: Dict[str, object]) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE sessions SET context_usage_json = ? WHERE id = ?",
            (json.dumps(usage or {}, ensure_ascii=False), session_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_session_data(session_id):
    # Chat history lives in a separate LangChain database and must be cleared too.
    try:
        from core.history import get_session_history
        get_session_history(session_id).clear()
    except Exception:
        pass
    conn = _connect()
    try:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM session_documents WHERE session_id = ?", (session_id,))
        sandbox_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session_sandboxes'"
        ).fetchone()
        if sandbox_table:
            conn.execute("DELETE FROM session_sandboxes WHERE session_id = ?", (session_id,))
        for table, column in (
            ("document_facts", "session_id"),
            ("document_blocks", "session_id"),
            ("document_metadata", "session_id"),
            ("workspace_manifest_entries", "session_id"),
            ("artifacts", "source_session_id"),
            ("uploaded_files", "session_id"),
        ):
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            ).fetchone()
            if exists:
                conn.execute(f"DELETE FROM {table} WHERE {column} = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


def upsert_session_document(
    session_id: str,
    file_id: str,
    file_name: str,
    file_hash: str,
    chunk_count: int,
    status: str = "completed",
    file_path: str = "",
) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO session_documents
                (id, session_id, file_name, file_hash, file_path, chunk_count, status, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                session_id = excluded.session_id,
                file_name = excluded.file_name,
                file_hash = excluded.file_hash,
                file_path = excluded.file_path,
                chunk_count = excluded.chunk_count,
                status = excluded.status,
                uploaded_at = excluded.uploaded_at
            """,
            (file_id, session_id, file_name, file_hash, file_path, chunk_count, status, datetime.now()),
        )
        conn.commit()
    finally:
        conn.close()


def get_session_documents(session_id: str) -> List[Tuple[str, str, int, str, str, str]]:
    # New callers receive the durable-table projection; legacy schema remains for migration compatibility.
    try:
        from db.file_manager import list_uploaded_files
        rows = list_uploaded_files(session_id)
        if rows:
            return [
                (row["file_id"], row["original_name"], row["chunk_count"], row["index_status"],
                 row["created_at"], row["storage_path"])
                for row in rows
            ]
    except Exception:
        pass
    conn = _connect()
    try:
        return conn.execute(
            """
            SELECT id, file_name, chunk_count, status, uploaded_at, COALESCE(file_path, '')
            FROM session_documents
            WHERE session_id = ?
            ORDER BY uploaded_at DESC
            """,
            (session_id,),
        ).fetchall()
    finally:
        conn.close()


def get_session_document(session_id: str, file_id: str) -> Optional[Dict[str, str]]:
    try:
        from db.file_manager import get_uploaded_file
        record = get_uploaded_file(session_id, file_id)
        if record:
            return {
                "id": record["file_id"], "session_id": record["session_id"],
                "file_name": record["original_name"], "file_hash": record["sha256"],
                "file_path": record["storage_path"], "chunk_count": record["chunk_count"],
                "status": record["index_status"], "uploaded_at": record["created_at"],
            }
    except Exception:
        pass
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT id, session_id, file_name, file_hash, COALESCE(file_path, ''), chunk_count, status, uploaded_at
            FROM session_documents
            WHERE session_id = ? AND id = ?
            """,
            (session_id, file_id),
        ).fetchone()
        if not row:
            return None
        keys = ["id", "session_id", "file_name", "file_hash", "file_path", "chunk_count", "status", "uploaded_at"]
        return dict(zip(keys, row))
    finally:
        conn.close()


def update_session_document_status(session_id: str, file_id: str, status: str) -> None:
    try:
        from db.file_manager import update_uploaded_file
        update_uploaded_file(file_id, parse_status=status, index_status=status)
    except Exception:
        pass
    conn = _connect()
    try:
        conn.execute(
            "UPDATE session_documents SET status = ? WHERE session_id = ? AND id = ?",
            (status, session_id, file_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_session_document(session_id: str, file_id: str) -> None:
    try:
        from db.file_manager import delete_uploaded_file
        delete_uploaded_file(session_id, file_id)
    except Exception:
        pass
    conn = _connect()
    try:
        conn.execute("DELETE FROM session_documents WHERE session_id = ? AND id = ?", (session_id, file_id))
        conn.commit()
    finally:
        conn.close()
