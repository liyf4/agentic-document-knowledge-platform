import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from core.document_processor import DocumentProcessor
from core.persistent_storage import get_persistent_file_store
from db.file_manager import update_uploaded_file


class BufferedUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _set_job(task_id: str, **updates: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.setdefault(task_id, {})
        job.update(updates)
        job["updated_at"] = _now()


def _run_process_task(task_id: str, session_id: str, file_id: str) -> None:
    _set_job(task_id, status="processing", message="Indexing document.")
    update_uploaded_file(file_id, parse_status="processing", index_status="processing")

    try:
        ok, message = DocumentProcessor.process_stored_file(session_id, file_id)
        if ok:
            _set_job(task_id, status="completed", message=message, error="")
        else:
            update_uploaded_file(file_id, parse_status="failed", index_status="failed")
            _set_job(task_id, status="failed", message=message, error=message)
    except Exception as exc:
        logger.exception(f"Async indexing task {task_id} failed")
        update_uploaded_file(file_id, parse_status="failed", index_status="failed")
        _set_job(task_id, status="failed", message=f"Process failed: {exc}", error=str(exc))


def submit_process_file(uploaded_file, session_id: str) -> str:
    file_bytes = uploaded_file.getvalue()
    record = get_persistent_file_store().persist_upload(session_id, uploaded_file.name, file_bytes)

    task_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _JOBS[task_id] = {
            "task_id": task_id,
            "session_id": session_id,
            "file_id": record["file_id"],
            "file_name": record["original_name"],
            "status": "queued",
            "message": "Queued for indexing.",
            "error": "",
            "created_at": _now(),
            "updated_at": _now(),
        }

    worker = threading.Thread(
        target=_run_process_task,
        args=(task_id, session_id, record["file_id"]),
        daemon=True,
    )
    worker.start()
    return task_id


def get_index_task(task_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(task_id)
        return dict(job) if job else None


def list_index_tasks(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with _JOBS_LOCK:
        jobs = [dict(job) for job in _JOBS.values()]
    if session_id:
        jobs = [job for job in jobs if job.get("session_id") == session_id]
    return sorted(jobs, key=lambda job: job.get("created_at", ""), reverse=True)
