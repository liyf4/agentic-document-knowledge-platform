import json
from typing import Any, Callable, Optional

from langchain_core.tools import BaseTool, tool

from sandbox_runtime.base import SandboxError
from sandbox_runtime.manager import WorkspaceSandboxManager, get_sandbox_manager
from sandbox_runtime.persistence import get_workspace_persistence


def _json_result(callback: Callable[[], Any]) -> str:
    try:
        value = callback()
        return json.dumps({"ok": True, "result": value}, ensure_ascii=False, default=str)
    except SandboxError as exc:
        return json.dumps(
            {"ok": False, "error": str(exc), "error_type": type(exc).__name__},
            ensure_ascii=False,
        )


def create_workspace_tools(
    session_id: str,
    manager: Optional[WorkspaceSandboxManager] = None,
) -> list[BaseTool]:
    runtime = manager or get_sandbox_manager()
    persistence = get_workspace_persistence(runtime)

    @tool("workspace_list_session_files")
    def workspace_list_session_files() -> str:
        """List durable files uploaded to this session. Use before importing a file into the sandbox."""
        return _json_result(lambda: persistence.list_session_files(session_id))

    @tool("workspace_import_uploaded_file")
    def workspace_import_uploaded_file(file_id: str) -> str:
        """Copy one durable uploaded file into /workspace/inputs/<file_id>/ and remember it for recovery."""
        return _json_result(lambda: persistence.import_uploaded_file(session_id, file_id))

    @tool("workspace_list_artifacts")
    def workspace_list_artifacts() -> str:
        """List durable result files previously saved from this session workspace."""
        from db.file_manager import list_artifacts
        return _json_result(lambda: list_artifacts(session_id))

    @tool("workspace_import_artifact")
    def workspace_import_artifact(artifact_id: str) -> str:
        """Restore one saved artifact into /workspace/output/restored/ and remember it for recovery."""
        return _json_result(lambda: persistence.import_artifact(session_id, artifact_id))

    @tool("workspace_save_file")
    def workspace_save_file(path: str) -> str:
        """Explicitly export any file below /workspace as a durable session artifact."""
        return _json_result(lambda: persistence.save_workspace_file(session_id, path))

    @tool("workspace_terminal")
    def workspace_terminal(
        command: str,
        cwd: str = "/workspace",
        timeout_seconds: Optional[int] = None,
        background: bool = False,
    ) -> str:
        """Run a shell command inside the isolated session workspace. Never runs on the application host."""
        return _json_result(
            lambda: runtime.execute(
                session_id,
                command,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                background=background,
            ).to_dict()
        )

    @tool("workspace_list_files")
    def workspace_list_files(path: str = "/workspace", max_depth: int = 3) -> str:
        """List files below /workspace in the isolated session sandbox."""
        return _json_result(lambda: [item.to_dict() for item in runtime.list_files(session_id, path, max_depth)])

    @tool("workspace_read_file")
    def workspace_read_file(path: str) -> str:
        """Read a UTF-8 text file below /workspace in the isolated session sandbox."""
        return _json_result(lambda: {"path": runtime.workspace_path(path), "content": runtime.read_file(session_id, path)})

    @tool("workspace_write_file")
    def workspace_write_file(path: str, content: str) -> str:
        """Write a UTF-8 text file below /workspace in the isolated session sandbox."""
        return _json_result(lambda: runtime.write_file(session_id, path, content))

    @tool("workspace_process")
    def workspace_process(process_id: str, action: str = "status") -> str:
        """Inspect logs/status or stop a background sandbox process; action is status, logs, or stop."""
        return _json_result(lambda: runtime.process(session_id, process_id, action).to_dict())

    return [
        workspace_list_session_files,
        workspace_import_uploaded_file,
        workspace_list_artifacts,
        workspace_import_artifact,
        workspace_save_file,
        workspace_terminal,
        workspace_list_files,
        workspace_read_file,
        workspace_write_file,
        workspace_process,
    ]
