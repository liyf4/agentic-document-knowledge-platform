"""Controlled persistence boundary for uploads and generated workspace files."""

import hashlib
import mimetypes
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from config import (
    CLAMAV_COMMAND,
    DATA_DIR,
    MAX_ARTIFACT_MB,
    MAX_SESSION_STORAGE_MB,
    MAX_UPLOAD_MB,
    UPLOAD_SCAN_MODE,
)
from db.file_manager import (
    create_artifact,
    create_uploaded_file,
    find_artifact,
    find_uploaded_by_hash,
    get_uploaded_file,
    session_storage_bytes,
)


class StorageValidationError(ValueError):
    pass


def safe_file_name(name: str) -> str:
    candidate = Path(str(name or "")).name.replace("\x00", "").strip()
    if not candidate or candidate in {".", ".."}:
        raise StorageValidationError("A safe file name is required.")
    # Keep Unicode names but reject control characters and path separators.
    if any(ord(char) < 32 for char in candidate) or "/" in candidate or "\\" in candidate:
        raise StorageValidationError("The file name contains unsafe characters.")
    return candidate[:240]


def category_for_name(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in {".txt", ".md", ".markdown", ".pdf", ".doc", ".docx"}:
        return "document"
    if ext in {".csv", ".xlsx", ".parquet"}:
        return "table"
    if ext in {".zip"}:
        return "archive"
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    if ext in {".mp3", ".wav", ".m4a", ".ogg"}:
        return "audio"
    return "other"


def _looks_like_expected_type(name: str, content: bytes) -> bool:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return content.startswith(b"%PDF-")
    if ext in {".docx", ".xlsx"}:
        return content.startswith(b"PK\x03\x04")
    if ext == ".parquet":
        return len(content) >= 8 and content[:4] == b"PAR1" and content[-4:] == b"PAR1"
    if ext == ".csv":
        return b"\x00" not in content[:4096]
    if ext in {".txt", ".md", ".markdown"}:
        return b"\x00" not in content[:4096]
    # Legacy .doc has no reliable magic header; parsing is its final validation.
    return True


def _scan_content(path: Path) -> str:
    if UPLOAD_SCAN_MODE == "off":
        return "skipped"
    if UPLOAD_SCAN_MODE != "clamav_required":
        raise StorageValidationError("upload_scan_mode must be off or clamav_required.")
    try:
        completed = subprocess.run(
            [CLAMAV_COMMAND, "--no-summary", str(path)], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StorageValidationError(f"Malware scanner is unavailable: {exc}") from exc
    if completed.returncode == 0:
        return "clean"
    if completed.returncode == 1:
        raise StorageValidationError("Upload was rejected by the malware scanner.")
    raise StorageValidationError("Malware scanner did not complete successfully.")


class PersistentFileStore:
    def __init__(self, data_dir: str = DATA_DIR) -> None:
        self.root = Path(data_dir)

    def upload_path(self, session_id: str, file_id: str, original_name: str) -> Path:
        return self.root / "uploads" / session_id / file_id / safe_file_name(original_name)

    def artifact_path(self, session_id: str, artifact_id: str, name: str) -> Path:
        return self.root / "artifacts" / session_id / artifact_id / safe_file_name(name)

    def persist_upload(self, session_id: str, name: str, content: bytes, *, owner_id: Optional[str] = None) -> Dict[str, Any]:
        original_name = safe_file_name(name)
        supported = {".txt", ".md", ".markdown", ".pdf", ".doc", ".docx", ".csv", ".xlsx", ".parquet"}
        ext = Path(original_name).suffix.lower()
        if ext not in supported:
            raise StorageValidationError(f"Unsupported file type: {ext or '<none>'}.")
        max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
        if not content:
            raise StorageValidationError("Empty files cannot be uploaded.")
        if len(content) > max_bytes:
            raise StorageValidationError(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
        if not _looks_like_expected_type(original_name, content):
            raise StorageValidationError("File contents do not match its extension.")
        sha256 = hashlib.sha256(content).hexdigest()
        existing = find_uploaded_by_hash(session_id, sha256)
        if existing:
            return existing
        if session_storage_bytes(session_id) + len(content) > int(MAX_SESSION_STORAGE_MB * 1024 * 1024):
            raise StorageValidationError(f"Session storage exceeds the {MAX_SESSION_STORAGE_MB} MB limit.")
        file_id = f"{session_id[:8]}_{sha256[:16]}"
        # Scan a short-lived staging copy before committing the original to durable storage.
        with tempfile.NamedTemporaryFile(prefix="chatbot-upload-", suffix=ext, delete=False) as staged:
            staged.write(content)
            staged_path = Path(staged.name)
        try:
            scan_status = _scan_content(staged_path)
        finally:
            staged_path.unlink(missing_ok=True)
        path = self.upload_path(session_id, file_id, original_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        mime_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
        record = {
            "file_id": file_id, "session_id": session_id, "owner_id": owner_id,
            "original_name": original_name, "storage_path": str(path), "sha256": sha256,
            "mime_type": mime_type, "category": category_for_name(original_name), "size_bytes": len(content),
            "parse_status": "queued", "index_status": "queued", "ocr_status": "not_applicable",
            "scan_status": scan_status,
        }
        try:
            create_uploaded_file(record)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return get_uploaded_file(session_id, file_id) or record

    def persist_artifact(self, session_id: str, sandbox_id: str, name: str, content: bytes,
                         generated_by_command: str = "", original_file_id: Optional[str] = None) -> Dict[str, Any]:
        name = safe_file_name(name)
        max_bytes = int(MAX_ARTIFACT_MB * 1024 * 1024)
        if not content:
            raise StorageValidationError("Empty workspace files are not saved as artifacts.")
        if len(content) > max_bytes:
            raise StorageValidationError(f"Artifact exceeds the {MAX_ARTIFACT_MB} MB limit.")
        if session_storage_bytes(session_id) + len(content) > int(MAX_SESSION_STORAGE_MB * 1024 * 1024):
            raise StorageValidationError(f"Session storage exceeds the {MAX_SESSION_STORAGE_MB} MB limit.")
        sha256 = hashlib.sha256(content).hexdigest()
        existing = find_artifact(session_id, name, sha256)
        if existing:
            return existing
        artifact_id = str(uuid.uuid4())
        path = self.artifact_path(session_id, artifact_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        record = {
            "artifact_id": artifact_id, "source_session_id": session_id, "source_sandbox_id": sandbox_id,
            "original_file_id": original_file_id, "name": name, "storage_path": str(path), "sha256": sha256,
            "size_bytes": len(content), "mime_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
            "generated_by_command": generated_by_command, "retention_policy": "session",
        }
        try:
            create_artifact(record)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return record

    def remove_upload(self, record: Dict[str, Any]) -> None:
        path = Path(record.get("storage_path", ""))
        if path.is_file():
            shutil.rmtree(path.parent, ignore_errors=True)

    def remove_session(self, session_id: str) -> None:
        shutil.rmtree(self.root / "uploads" / session_id, ignore_errors=True)
        shutil.rmtree(self.root / "artifacts" / session_id, ignore_errors=True)
        # Existing Chroma and historical raw-file storage are session scoped too.
        shutil.rmtree(self.root / session_id, ignore_errors=True)


_store: Optional[PersistentFileStore] = None


def get_persistent_file_store() -> PersistentFileStore:
    global _store
    if _store is None:
        _store = PersistentFileStore()
    return _store
