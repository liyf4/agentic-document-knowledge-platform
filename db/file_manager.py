"""Metadata for durable uploads, generated artifacts, and workspace recovery.

The tables in this module are intentionally separate from the legacy
``session_documents`` table.  The latter remains populated as a compatibility
projection while these tables are the source of truth for new flows.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import DB_NAME


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _extension_category(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in {".txt", ".md", ".markdown", ".pdf", ".doc", ".docx"}:
        return "document"
    if ext in {".csv", ".xlsx", ".parquet"}:
        return "table"
    if ext in {".py", ".ipynb"}:
        return "code"
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    if ext in {".mp3", ".wav", ".m4a", ".ogg"}:
        return "audio"
    return "other"


def init_file_db() -> None:
    """Create durable-file tables and backfill old document metadata once."""
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS uploaded_files (
                file_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                owner_id TEXT,
                original_name TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                category TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                parse_status TEXT NOT NULL DEFAULT 'queued',
                ocr_status TEXT NOT NULL DEFAULT 'not_applicable',
                index_status TEXT NOT NULL DEFAULT 'queued',
                scan_status TEXT NOT NULL DEFAULT 'skipped',
                table_profile_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_uploaded_files_session ON uploaded_files(session_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS document_metadata (
                file_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                extraction_status TEXT NOT NULL DEFAULT 'pending',
                extraction_error TEXT NOT NULL DEFAULT '',
                extractor_version TEXT NOT NULL DEFAULT '',
                structure_version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_document_metadata_session
                ON document_metadata(session_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS document_facts (
                id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                raw_value TEXT NOT NULL,
                normalized_value TEXT NOT NULL DEFAULT '',
                unit TEXT NOT NULL DEFAULT '',
                currency TEXT NOT NULL DEFAULT '',
                qualifier TEXT NOT NULL DEFAULT '',
                physical_page INTEGER,
                printed_page TEXT NOT NULL DEFAULT '',
                section_path TEXT NOT NULL DEFAULT '',
                evidence_text TEXT NOT NULL DEFAULT '',
                chunk_id TEXT NOT NULL DEFAULT '',
                extractor TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0,
                extractor_version TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_document_facts_file_type
                ON document_facts(file_id, fact_type);
            CREATE INDEX IF NOT EXISTS idx_document_facts_normalized
                ON document_facts(session_id, fact_type, normalized_value);
            CREATE INDEX IF NOT EXISTS idx_document_facts_page
                ON document_facts(file_id, physical_page);

            CREATE TABLE IF NOT EXISTS document_blocks (
                id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                block_index INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                page_role TEXT NOT NULL DEFAULT 'body',
                retrieval_tier TEXT NOT NULL DEFAULT 'primary',
                physical_page INTEGER,
                printed_page TEXT NOT NULL DEFAULT '',
                section_path TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                bbox_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_document_blocks_file_order
                ON document_blocks(file_id, block_index);
            CREATE INDEX IF NOT EXISTS idx_document_blocks_role
                ON document_blocks(session_id, page_role, retrieval_tier);

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                source_session_id TEXT NOT NULL,
                source_sandbox_id TEXT NOT NULL DEFAULT '',
                original_file_id TEXT,
                name TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                generated_by_command TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                retention_policy TEXT NOT NULL DEFAULT 'session'
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(source_session_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS workspace_manifest_entries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                entry_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                sandbox_path TEXT NOT NULL,
                restore_order INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, entry_kind, source_id, sandbox_path)
            );
            CREATE INDEX IF NOT EXISTS idx_workspace_manifest_session
                ON workspace_manifest_entries(session_id, restore_order, created_at);
            """
        )
        legacy_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_documents'"
        ).fetchone()
        if legacy_exists:
            rows = conn.execute(
                """
                SELECT id, session_id, file_name, file_hash,
                       COALESCE(file_path, '') AS file_path,
                       chunk_count, status, uploaded_at
                FROM session_documents
                """
            ).fetchall()
            for row in rows:
                path = Path(row["file_path"]) if row["file_path"] else None
                size = path.stat().st_size if path and path.is_file() else 0
                status = row["status"] or "queued"
                created = str(row["uploaded_at"] or _now())
                conn.execute(
                    """
                    INSERT OR IGNORE INTO uploaded_files
                    (file_id, session_id, original_name, storage_path, sha256, mime_type,
                     category, size_bytes, chunk_count, parse_status, ocr_status,
                     index_status, scan_status, table_profile_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'not_applicable', ?,
                            'legacy_unknown', '{}', ?, ?)
                    """,
                    (
                        row["id"], row["session_id"], row["file_name"], row["file_path"],
                        row["file_hash"], "application/octet-stream", _extension_category(row["file_name"]),
                        size, row["chunk_count"] or 0, status, status, created, created,
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    value = dict(row)
    value["table_profile"] = json.loads(value.pop("table_profile_json", "{}") or "{}")
    return value


def create_uploaded_file(record: Dict[str, Any]) -> None:
    now = _now()
    values = {
        "file_id": record["file_id"], "session_id": record["session_id"],
        "owner_id": record.get("owner_id"), "original_name": record["original_name"],
        "storage_path": record["storage_path"], "sha256": record["sha256"],
        "mime_type": record["mime_type"], "category": record["category"],
        "size_bytes": int(record["size_bytes"]), "chunk_count": int(record.get("chunk_count", 0)),
        "parse_status": record.get("parse_status", "queued"),
        "ocr_status": record.get("ocr_status", "not_applicable"),
        "index_status": record.get("index_status", "queued"),
        "scan_status": record.get("scan_status", "skipped"),
        "table_profile_json": json.dumps(record.get("table_profile", {}), ensure_ascii=False, default=str),
        "created_at": record.get("created_at", now), "updated_at": now,
    }
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO uploaded_files
            (file_id, session_id, owner_id, original_name, storage_path, sha256, mime_type,
             category, size_bytes, chunk_count, parse_status, ocr_status, index_status,
             scan_status, table_profile_json, created_at, updated_at)
            VALUES (:file_id, :session_id, :owner_id, :original_name, :storage_path, :sha256,
                    :mime_type, :category, :size_bytes, :chunk_count, :parse_status,
                    :ocr_status, :index_status, :scan_status, :table_profile_json,
                    :created_at, :updated_at)
            """, values,
        )
        conn.commit()
    finally:
        conn.close()


def update_uploaded_file(file_id: str, **fields: Any) -> None:
    allowed = {"original_name", "storage_path", "mime_type", "category", "size_bytes", "chunk_count",
               "parse_status", "ocr_status", "index_status", "scan_status", "table_profile"}
    updates: Dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    if "table_profile" in updates:
        updates["table_profile_json"] = json.dumps(updates.pop("table_profile"), ensure_ascii=False, default=str)
    if not updates:
        return
    updates["updated_at"] = _now()
    assignments = ", ".join(f"{column} = :{column}" for column in updates)
    updates["file_id"] = file_id
    conn = _connect()
    try:
        conn.execute(f"UPDATE uploaded_files SET {assignments} WHERE file_id = :file_id", updates)
        conn.commit()
    finally:
        conn.close()


def replace_document_analysis(
    session_id: str,
    file_id: str,
    metadata: Dict[str, Any],
    facts: Iterable[Dict[str, Any]],
    blocks: Iterable[Dict[str, Any]],
    *,
    extraction_status: str,
    extraction_error: str = "",
    extractor_version: str = "",
    structure_version: int = 0,
) -> None:
    """Atomically replace derived structure and facts for one durable file."""
    now = _now()
    conn = _connect()
    try:
        conn.execute("DELETE FROM document_facts WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM document_blocks WHERE file_id = ?", (file_id,))
        conn.execute(
            """
            INSERT INTO document_metadata
            (file_id, session_id, metadata_json, extraction_status, extraction_error,
             extractor_version, structure_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_id) DO UPDATE SET
                session_id=excluded.session_id, metadata_json=excluded.metadata_json,
                extraction_status=excluded.extraction_status, extraction_error=excluded.extraction_error,
                extractor_version=excluded.extractor_version, structure_version=excluded.structure_version,
                updated_at=excluded.updated_at
            """,
            (file_id, session_id, json.dumps(metadata, ensure_ascii=False, default=str), extraction_status,
             extraction_error, extractor_version, int(structure_version), now, now),
        )
        for fact in facts:
            conn.execute(
                """INSERT INTO document_facts
                (id, file_id, session_id, fact_type, raw_value, normalized_value, unit, currency,
                 qualifier, physical_page, printed_page, section_path, evidence_text, chunk_id,
                 extractor, confidence, extractor_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fact["id"], file_id, session_id, fact["fact_type"], fact["raw_value"],
                 fact.get("normalized_value", ""), fact.get("unit", ""), fact.get("currency", ""),
                 fact.get("qualifier", ""), fact.get("physical_page"), fact.get("printed_page", ""),
                 fact.get("section_path", ""), fact.get("evidence_text", ""), fact.get("chunk_id", ""),
                 fact.get("extractor", ""), float(fact.get("confidence", 0)),
                 fact.get("extractor_version", extractor_version), now),
            )
        for block in blocks:
            conn.execute(
                """INSERT INTO document_blocks
                (id, file_id, session_id, block_index, block_type, page_role, retrieval_tier,
                 physical_page, printed_page, section_path, text, bbox_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (block["id"], file_id, session_id, int(block["block_index"]), block["block_type"],
                 block.get("page_role", "body"), block.get("retrieval_tier", "primary"),
                 block.get("physical_page"), block.get("printed_page", ""), block.get("section_path", ""),
                 block.get("text", ""), json.dumps(block.get("bbox", []), ensure_ascii=False),
                 json.dumps(block.get("metadata", {}), ensure_ascii=False, default=str), now),
            )
        conn.commit()
    finally:
        conn.close()


def delete_document_analysis(file_id: str) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM document_facts WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM document_blocks WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM document_metadata WHERE file_id = ?", (file_id,))
        conn.commit()
    finally:
        conn.close()


def get_document_metadata(session_id: str, file_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM document_metadata WHERE session_id = ? AND file_id = ?", (session_id, file_id)
        ).fetchone()
        if not row:
            return None
        value = dict(row)
        value["metadata"] = json.loads(value.pop("metadata_json", "{}") or "{}")
        return value
    finally:
        conn.close()


def update_document_metadata_status(file_id: str, extraction_status: str, extraction_error: str = "") -> None:
    """Update optional-enrichment state without replacing deterministic facts or blocks."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE document_metadata SET extraction_status = ?, extraction_error = ?, updated_at = ? WHERE file_id = ?",
            (extraction_status, extraction_error, _now(), file_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_document_metadata(session_id: str) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        values = []
        for row in conn.execute(
            "SELECT * FROM document_metadata WHERE session_id = ? ORDER BY updated_at DESC", (session_id,)
        ).fetchall():
            value = dict(row)
            value["metadata"] = json.loads(value.pop("metadata_json", "{}") or "{}")
            values.append(value)
        return values
    finally:
        conn.close()


def query_document_facts(
    session_id: str,
    *,
    file_id: Optional[str] = None,
    fact_types: Optional[Iterable[str]] = None,
    text_query: str = "",
    limit: int = 30,
) -> List[Dict[str, Any]]:
    clauses = ["session_id = ?"]
    params: List[Any] = [session_id]
    if file_id:
        clauses.append("file_id = ?")
        params.append(file_id)
    types = [str(item) for item in (fact_types or []) if str(item)]
    if types:
        clauses.append(f"fact_type IN ({','.join('?' for _ in types)})")
        params.extend(types)
    if text_query:
        clauses.append("(raw_value LIKE ? OR normalized_value LIKE ? OR qualifier LIKE ? OR evidence_text LIKE ?)")
        needle = f"%{text_query}%"
        params.extend([needle, needle, needle, needle])
    params.append(max(1, min(int(limit), 100)))
    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(
            f"SELECT * FROM document_facts WHERE {' AND '.join(clauses)} ORDER BY confidence DESC, physical_page LIMIT ?",
            params,
        ).fetchall()]
    finally:
        conn.close()


def get_uploaded_file(session_id: str, file_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        return _row(conn.execute(
            "SELECT * FROM uploaded_files WHERE session_id = ? AND file_id = ?", (session_id, file_id)
        ).fetchone())
    finally:
        conn.close()


def find_uploaded_by_hash(session_id: str, sha256: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        return _row(conn.execute(
            "SELECT * FROM uploaded_files WHERE session_id = ? AND sha256 = ?", (session_id, sha256)
        ).fetchone())
    finally:
        conn.close()


def list_uploaded_files(session_id: str) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        return [_row(row) for row in conn.execute(
            "SELECT * FROM uploaded_files WHERE session_id = ? ORDER BY created_at DESC", (session_id,)
        ).fetchall()]
    finally:
        conn.close()


def delete_uploaded_file(session_id: str, file_id: str) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM uploaded_files WHERE session_id = ? AND file_id = ?", (session_id, file_id))
        conn.commit()
    finally:
        conn.close()


def session_storage_bytes(session_id: str) -> int:
    conn = _connect()
    try:
        uploads = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM uploaded_files WHERE session_id = ?", (session_id,)).fetchone()[0]
        artifacts = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM artifacts WHERE source_session_id = ?", (session_id,)).fetchone()[0]
        return int(uploads or 0) + int(artifacts or 0)
    finally:
        conn.close()


def create_artifact(record: Dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO artifacts
            (artifact_id, source_session_id, source_sandbox_id, original_file_id, name,
             storage_path, sha256, size_bytes, mime_type, generated_by_command,
             created_at, retention_policy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (record["artifact_id"], record["source_session_id"], record.get("source_sandbox_id", ""),
             record.get("original_file_id"), record["name"], record["storage_path"], record["sha256"],
             int(record["size_bytes"]), record["mime_type"], record.get("generated_by_command", ""),
             record.get("created_at", _now()), record.get("retention_policy", "session")),
        )
        conn.commit()
    finally:
        conn.close()


def get_artifact(session_id: str, artifact_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM artifacts WHERE source_session_id = ? AND artifact_id = ?", (session_id, artifact_id)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def find_artifact(session_id: str, name: str, sha256: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM artifacts WHERE source_session_id = ? AND name = ? AND sha256 = ?", (session_id, name, sha256)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_artifacts(session_id: str) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM artifacts WHERE source_session_id = ? ORDER BY created_at DESC", (session_id,)
        ).fetchall()]
    finally:
        conn.close()


def add_manifest_entry(session_id: str, entry_kind: str, source_id: str, sandbox_path: str,
                       restore_order: int) -> None:
    import uuid
    now = _now()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO workspace_manifest_entries
            (id, session_id, entry_kind, source_id, sandbox_path, restore_order, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, entry_kind, source_id, sandbox_path) DO UPDATE SET
                restore_order = excluded.restore_order, updated_at = excluded.updated_at
            """, (str(uuid.uuid4()), session_id, entry_kind, source_id, sandbox_path, restore_order, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def next_manifest_order(session_id: str) -> int:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(restore_order), 0) + 1 FROM workspace_manifest_entries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def list_manifest_entries(session_id: str) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM workspace_manifest_entries WHERE session_id = ? ORDER BY restore_order, created_at",
            (session_id,),
        ).fetchall()]
    finally:
        conn.close()


def delete_manifest_entries(session_id: str) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM workspace_manifest_entries WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


def delete_session_file_records(session_id: str) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM workspace_manifest_entries WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM artifacts WHERE source_session_id = ?", (session_id,))
        conn.execute("DELETE FROM uploaded_files WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
