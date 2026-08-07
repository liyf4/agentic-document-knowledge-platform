"""Copy controlled files into/out of a sandbox without host-directory mounts."""

import posixpath
from pathlib import Path
from typing import Any, Dict, List

from config import MAX_WORKSPACE_IMPORT_MB, MAX_WORKSPACE_OUTPUT_FILES, MAX_WORKSPACE_OUTPUT_MB
from core.persistent_storage import get_persistent_file_store, safe_file_name
from db.file_manager import (
    add_manifest_entry,
    get_artifact,
    get_uploaded_file,
    list_manifest_entries,
    next_manifest_order,
)
from sandbox_runtime.base import SandboxValidationError


class WorkspacePersistenceService:
    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self.store = get_persistent_file_store()

    @staticmethod
    def _input_path(record: Dict[str, Any]) -> str:
        return f"/workspace/inputs/{record['file_id']}/{safe_file_name(record['original_name'])}"

    def list_session_files(self, session_id: str) -> List[Dict[str, Any]]:
        from db.file_manager import list_uploaded_files
        return list_uploaded_files(session_id)

    def import_uploaded_file(self, session_id: str, file_id: str) -> Dict[str, Any]:
        record = get_uploaded_file(session_id, file_id)
        if not record:
            raise SandboxValidationError("Uploaded file was not found in this session.")
        path = Path(record["storage_path"])
        if not path.is_file():
            raise SandboxValidationError("The durable upload is unavailable.")
        content = path.read_bytes()
        self._ensure_import_quota(len(content))
        sandbox_path = self._input_path(record)
        result = self.manager.stage_bytes(session_id, sandbox_path, content)
        add_manifest_entry(session_id, "uploaded_file", file_id, sandbox_path, next_manifest_order(session_id))
        return {"file_id": file_id, "sandbox_path": sandbox_path, **result}

    def import_artifact(self, session_id: str, artifact_id: str) -> Dict[str, Any]:
        record = get_artifact(session_id, artifact_id)
        if not record:
            raise SandboxValidationError("Artifact was not found in this session.")
        path = Path(record["storage_path"])
        if not path.is_file():
            raise SandboxValidationError("The durable artifact is unavailable.")
        content = path.read_bytes()
        self._ensure_import_quota(len(content))
        sandbox_path = f"/workspace/output/restored/{artifact_id}/{safe_file_name(record['name'])}"
        result = self.manager.stage_bytes(session_id, sandbox_path, content)
        add_manifest_entry(session_id, "artifact", artifact_id, sandbox_path, next_manifest_order(session_id))
        return {"artifact_id": artifact_id, "sandbox_path": sandbox_path, **result}

    def restore(self, session_id: str) -> Dict[str, int]:
        restored, skipped, total = 0, 0, 0
        for entry in list_manifest_entries(session_id):
            record = get_uploaded_file(session_id, entry["source_id"]) if entry["entry_kind"] == "uploaded_file" else get_artifact(session_id, entry["source_id"])
            if not record:
                skipped += 1
                continue
            path = Path(record.get("storage_path", ""))
            if not path.is_file():
                skipped += 1
                continue
            content = path.read_bytes()
            total += len(content)
            if total > int(MAX_WORKSPACE_IMPORT_MB * 1024 * 1024):
                raise SandboxValidationError(f"Restored workspace input exceeds the {MAX_WORKSPACE_IMPORT_MB} MB limit.")
            self.manager.stage_bytes(session_id, entry["sandbox_path"], content)
            restored += 1
        return {"restored": restored, "skipped": skipped}

    def collect_output(self, session_id: str, generated_by_command: str = "") -> List[Dict[str, Any]]:
        """Harvest only output files; reads/lists never call this method."""
        entries = self.manager.list_files(session_id, "/workspace/output", max_depth=8)
        files = [entry for entry in entries if not entry.is_dir and entry.path.startswith("/workspace/output/")]
        if len(files) > int(MAX_WORKSPACE_OUTPUT_FILES):
            raise SandboxValidationError(f"Workspace output has more than {MAX_WORKSPACE_OUTPUT_FILES} files.")
        total = sum(int(entry.size or 0) for entry in files)
        if total > int(MAX_WORKSPACE_OUTPUT_MB * 1024 * 1024):
            raise SandboxValidationError(f"Workspace output exceeds the {MAX_WORKSPACE_OUTPUT_MB} MB limit.")
        handle = self.manager.ensure(session_id)
        saved: List[Dict[str, Any]] = []
        for entry in files:
            content = self.manager.read_bytes(session_id, entry.path)
            artifact = self.store.persist_artifact(
                session_id, handle.sandbox_id, safe_file_name(posixpath.basename(entry.path)), content,
                generated_by_command=generated_by_command,
            )
            add_manifest_entry(session_id, "artifact", artifact["artifact_id"], entry.path, next_manifest_order(session_id))
            saved.append({"artifact_id": artifact["artifact_id"], "name": artifact["name"], "size_bytes": artifact["size_bytes"]})
        return saved

    def save_workspace_file(self, session_id: str, workspace_path: str) -> Dict[str, Any]:
        safe_path = self.manager.workspace_path(workspace_path)
        content = self.manager.read_bytes(session_id, safe_path)
        handle = self.manager.ensure(session_id)
        artifact = self.store.persist_artifact(
            session_id, handle.sandbox_id, safe_file_name(posixpath.basename(safe_path)), content,
            generated_by_command="explicit_workspace_save",
        )
        add_manifest_entry(session_id, "artifact", artifact["artifact_id"], safe_path, next_manifest_order(session_id))
        return artifact

    @staticmethod
    def _ensure_import_quota(size: int) -> None:
        if size > int(MAX_WORKSPACE_IMPORT_MB * 1024 * 1024):
            raise SandboxValidationError(f"Import exceeds the {MAX_WORKSPACE_IMPORT_MB} MB workspace limit.")


def get_workspace_persistence(manager: Any) -> WorkspacePersistenceService:
    service = getattr(manager, "_persistence_service", None)
    if service is None:
        service = WorkspacePersistenceService(manager)
        manager.set_persistence_service(service)
    return service
