import posixpath
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config import (
    SANDBOX_ALLOWED_EGRESS,
    SANDBOX_API_KEY_ENV,
    SANDBOX_COMMAND_TIMEOUT_SECONDS,
    CONTEXT_TOOL_OBSERVATION_MAX_TOKENS,
    SANDBOX_CPU,
    SANDBOX_DOMAIN,
    SANDBOX_ENABLED,
    SANDBOX_IMAGE,
    SANDBOX_IDLE_TIMEOUT_SECONDS,
    SANDBOX_MAX_COMMAND_CHARS,
    SANDBOX_MAX_OUTPUT_CHARS,
    SANDBOX_MEMORY,
    SANDBOX_NETWORK_ENABLED,
    SANDBOX_PROVIDER,
    SANDBOX_PROTOCOL,
    SANDBOX_REQUEST_TIMEOUT_SECONDS,
    SANDBOX_TIMEOUT_SECONDS,
    SANDBOX_USE_SERVER_PROXY,
)
from db.sandbox_manager import (
    delete_session_sandbox,
    get_session_sandbox,
    list_idle_session_sandboxes,
    touch_and_extend_session_sandbox,
    update_session_sandbox,
    upsert_session_sandbox,
)
from sandbox_runtime.base import SandboxError, SandboxProvider, SandboxUnavailableError, SandboxValidationError
from sandbox_runtime.models import CommandResult, FileEntry, ProcessResult, SandboxHandle
from sandbox_runtime.opensandbox_provider import OpenSandboxProvider


WORKSPACE_ROOT = "/workspace"


class WorkspaceSandboxManager:
    def __init__(
        self,
        provider: SandboxProvider,
        *,
        enabled: bool = True,
        max_command_chars: int = SANDBOX_MAX_COMMAND_CHARS,
        max_output_chars: int = SANDBOX_MAX_OUTPUT_CHARS,
        command_timeout_seconds: int = SANDBOX_COMMAND_TIMEOUT_SECONDS,
        persistence: bool = True,
    ) -> None:
        self.provider = provider
        self.enabled = enabled
        self.max_command_chars = max_command_chars
        self.max_output_chars = max_output_chars
        self.command_timeout_seconds = command_timeout_seconds
        self.persistence = persistence
        self._handles: Dict[str, SandboxHandle] = {}
        self._lock = threading.RLock()
        self._persistence_service: Any = None

    def set_persistence_service(self, service: Any) -> None:
        """Attach the durable workspace bridge after both objects are constructed."""
        self._persistence_service = service

    def capabilities(self) -> Dict[str, Any]:
        sdk_available = self.provider.is_available()
        return {
            "enabled": self.enabled,
            "provider": self.provider.name,
            "sdk_available": sdk_available,
            "available": bool(self.enabled and sdk_available),
            "network_enabled": SANDBOX_NETWORK_ENABLED,
            "workspace_root": WORKSPACE_ROOT,
        }

    def status(self, session_id: str) -> Dict[str, Any]:
        record = get_session_sandbox(session_id) if self.persistence else None
        return {"mode": "workspace", "runtime": record, **self.capabilities()}

    def execute(
        self,
        session_id: str,
        command: str,
        *,
        cwd: str = WORKSPACE_ROOT,
        timeout_seconds: Optional[int] = None,
        background: bool = False,
    ) -> CommandResult:
        command = str(command or "")
        if not command.strip():
            raise SandboxValidationError("Command cannot be empty.")
        if len(command) > self.max_command_chars:
            raise SandboxValidationError(f"Command exceeds the {self.max_command_chars}-character limit.")
        safe_cwd = self.workspace_path(cwd)
        timeout = int(timeout_seconds or self.command_timeout_seconds)
        if timeout < 1 or timeout > self.command_timeout_seconds:
            raise SandboxValidationError(
                f"timeout_seconds must be between 1 and {self.command_timeout_seconds}."
            )
        result = self._with_recovery(
            session_id, lambda handle: self.provider.execute(handle.sandbox_id, command, safe_cwd, timeout, background)
        )
        result.stdout = self._truncate(result.stdout)
        result.stderr = self._truncate(result.stderr)
        self._touch(session_id)
        self._collect_outputs(session_id, command)
        return result

    def list_files(self, session_id: str, path: str = WORKSPACE_ROOT, max_depth: int = 3) -> List[FileEntry]:
        if max_depth < 0 or max_depth > 8:
            raise SandboxValidationError("max_depth must be between 0 and 8.")
        safe_path = self.workspace_path(path)
        entries = self._with_recovery(session_id, lambda handle: self.provider.list_files(handle.sandbox_id, safe_path, max_depth))
        self._touch(session_id)
        return entries[:1000]

    def read_file(self, session_id: str, path: str) -> str:
        safe_path = self.workspace_path(path)
        content = self._with_recovery(session_id, lambda handle: self.provider.read_file(handle.sandbox_id, safe_path))
        self._touch(session_id)
        return self._truncate(content)

    def write_file(self, session_id: str, path: str, content: str) -> Dict[str, Any]:
        safe_path = self.workspace_path(path)
        if safe_path == WORKSPACE_ROOT:
            raise SandboxValidationError("A file path below /workspace is required.")
        encoded_size = len(str(content).encode("utf-8"))
        if encoded_size > self.max_output_chars:
            raise SandboxValidationError(f"File content exceeds the {self.max_output_chars}-byte tool limit.")
        self._with_recovery(session_id, lambda handle: self.provider.write_file(handle.sandbox_id, safe_path, str(content)))
        self._touch(session_id)
        if safe_path.startswith(WORKSPACE_ROOT + "/output/"):
            self._collect_outputs(session_id, "workspace_write_file")
        return {"path": safe_path, "bytes_written": encoded_size}

    def process(self, session_id: str, process_id: str, action: str = "status") -> ProcessResult:
        normalized_action = str(action).casefold()
        if normalized_action not in {"status", "logs", "stop"}:
            raise SandboxValidationError("action must be one of: status, logs, stop.")
        if not process_id or len(process_id) > 200:
            raise SandboxValidationError("A valid process_id is required.")
        result = self._with_recovery(session_id, lambda handle: self.provider.process(handle.sandbox_id, process_id, normalized_action))
        result.stdout = self._truncate(result.stdout)
        result.stderr = self._truncate(result.stderr)
        self._touch(session_id)
        return result

    def ensure(self, session_id: str) -> SandboxHandle:
        self._require_available()
        if not session_id or len(session_id) > 128:
            raise SandboxValidationError("A valid session id is required.")
        with self._lock:
            cached = self._handles.get(session_id)
            if cached is not None:
                return cached
            record = get_session_sandbox(session_id) if self.persistence else None
            if record:
                if record.get("provider") != self.provider.name:
                    raise SandboxUnavailableError("The session sandbox belongs to a different provider.")
                if self._record_expired(record):
                    delete_session_sandbox(session_id)
                    record = None
            if record:
                try:
                    handle = self.provider.connect(str(record["sandbox_id"]))
                    self._handles[session_id] = handle
                    update_session_sandbox(session_id, "running")
                    return handle
                except Exception as exc:
                    if self._is_expired_error(exc):
                        delete_session_sandbox(session_id)
                    else:
                        update_session_sandbox(session_id, "error", str(exc))
                        raise
            handle = self.provider.create(session_id)
            self._handles[session_id] = handle
            if self.persistence:
                try:
                    expires = (datetime.now() + timedelta(seconds=SANDBOX_TIMEOUT_SECONDS)).isoformat()
                    upsert_session_sandbox(session_id, handle.sandbox_id, self.provider.name, handle.status, expires_at=expires)
                    self.provider.renew(handle.sandbox_id, SANDBOX_TIMEOUT_SECONDS)
                    self._restore_workspace(session_id, handle)
                except Exception:
                    self._handles.pop(session_id, None)
                    self.provider.destroy(handle.sandbox_id)
                    delete_session_sandbox(session_id)
                    raise
            return handle

    def destroy(self, session_id: str) -> bool:
        with self._lock:
            handle = self._handles.get(session_id)
            record = get_session_sandbox(session_id) if self.persistence else None
            sandbox_id = handle.sandbox_id if handle else str((record or {}).get("sandbox_id", ""))
            if not sandbox_id:
                return False
            if not self.provider.is_available():
                raise SandboxUnavailableError("The sandbox provider SDK is unavailable, so cleanup cannot be confirmed.")
            self.provider.destroy(sandbox_id)
            self._handles.pop(session_id, None)
            if self.persistence:
                delete_session_sandbox(session_id)
            return True

    def stage_bytes(self, session_id: str, path: str, content: bytes) -> Dict[str, Any]:
        safe_path = self.workspace_path(path)
        if safe_path == WORKSPACE_ROOT:
            raise SandboxValidationError("A file path below /workspace is required.")
        parent = posixpath.dirname(safe_path)
        def _stage(handle: SandboxHandle) -> None:
            self.provider.execute(handle.sandbox_id, f"mkdir -p {self._quote_path(parent)}", WORKSPACE_ROOT, self.command_timeout_seconds)
            self.provider.write_file(handle.sandbox_id, safe_path, content)
        self._with_recovery(session_id, _stage)
        self._touch(session_id)
        return {"path": safe_path, "bytes_written": len(content)}

    def read_bytes(self, session_id: str, path: str) -> bytes:
        safe_path = self.workspace_path(path)
        content = self._with_recovery(session_id, lambda handle: self.provider.read_file_bytes(handle.sandbox_id, safe_path))
        self._touch(session_id)
        return content

    def reap_idle(self) -> int:
        """Destroy only confirmed idle mappings; transient provider failures keep their mapping."""
        if not self.persistence or not self.provider.is_available():
            return 0
        removed = 0
        for record in list_idle_session_sandboxes(SANDBOX_IDLE_TIMEOUT_SECONDS):
            session_id, sandbox_id = record["session_id"], record["sandbox_id"]
            try:
                self.provider.destroy(sandbox_id)
            except Exception as exc:
                if self._is_expired_error(exc):
                    pass
                else:
                    update_session_sandbox(session_id, "error", str(exc))
                    continue
            with self._lock:
                self._handles.pop(session_id, None)
            delete_session_sandbox(session_id)
            removed += 1
        return removed

    @staticmethod
    def workspace_path(path: str) -> str:
        value = str(path or WORKSPACE_ROOT).strip()
        if "\x00" in value or "\\" in value:
            raise SandboxValidationError("Workspace paths must be POSIX paths without null bytes.")
        parts = value.split("/")
        if ".." in parts:
            raise SandboxValidationError("Workspace paths cannot contain '..'.")
        if value.startswith("/") and value != WORKSPACE_ROOT and not value.startswith(WORKSPACE_ROOT + "/"):
            raise SandboxValidationError("Absolute paths must stay below /workspace.")
        if not value.startswith("/"):
            value = f"{WORKSPACE_ROOT}/{value}"
        normalized = posixpath.normpath(value)
        if normalized != WORKSPACE_ROOT and not normalized.startswith(WORKSPACE_ROOT + "/"):
            raise SandboxValidationError("Path escapes the workspace root.")
        return normalized

    def _require_available(self) -> None:
        if not self.enabled:
            raise SandboxUnavailableError(
                "Workspace mode is disabled. Set sandbox.enabled=true after configuring OpenSandbox."
            )
        if not self.provider.is_available():
            raise SandboxUnavailableError(
                "Workspace mode is enabled but the optional OpenSandbox SDK is not installed."
            )

    def _truncate(self, value: str) -> str:
        text = str(value or "")
        context_char_limit = max(1000, CONTEXT_TOOL_OBSERVATION_MAX_TOKENS * 3)
        effective_limit = min(self.max_output_chars, context_char_limit)
        if len(text) <= effective_limit:
            return text
        half = max(1, (effective_limit - 64) // 2)
        return text[:half] + "\n... <sandbox output truncated for context budget> ...\n" + text[-half:]

    def _touch(self, session_id: str) -> None:
        if self.persistence:
            handle = self._handles.get(session_id)
            if handle:
                try:
                    self.provider.renew(handle.sandbox_id, SANDBOX_TIMEOUT_SECONDS)
                except Exception:
                    # A completed operation remains valid; a later use will reconnect/recover if it expired.
                    pass
            touch_and_extend_session_sandbox(session_id, SANDBOX_TIMEOUT_SECONDS)

    def _with_recovery(self, session_id: str, operation: Any) -> Any:
        handle = self.ensure(session_id)
        try:
            return operation(handle)
        except Exception as exc:
            if not self._is_expired_error(exc):
                raise
            with self._lock:
                self._handles.pop(session_id, None)
            if self.persistence:
                delete_session_sandbox(session_id)
            handle = self.ensure(session_id)
            return operation(handle)

    def _restore_workspace(self, session_id: str, handle: SandboxHandle) -> None:
        self.provider.execute(handle.sandbox_id, "mkdir -p /workspace/inputs /workspace/output", WORKSPACE_ROOT, self.command_timeout_seconds)
        if self._persistence_service is not None:
            self._persistence_service.restore(session_id)

    def _collect_outputs(self, session_id: str, generated_by_command: str) -> None:
        if self._persistence_service is not None:
            self._persistence_service.collect_output(session_id, generated_by_command)

    @staticmethod
    def _quote_path(path: str) -> str:
        return "'" + path.replace("'", "'\\''") + "'"

    @staticmethod
    def _record_expired(record: Dict[str, Any]) -> bool:
        raw = record.get("expires_at")
        if not raw:
            return False
        try:
            return datetime.fromisoformat(str(raw)) <= datetime.now()
        except ValueError:
            return False

    @staticmethod
    def _is_expired_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(token in message for token in ("not found", "404", "expired", "terminated", "does not exist", "unknown sandbox"))


_manager: Optional[WorkspaceSandboxManager] = None


def get_sandbox_manager() -> WorkspaceSandboxManager:
    global _manager
    if _manager is None:
        if SANDBOX_PROVIDER != "opensandbox":
            raise SandboxUnavailableError(f"Unsupported sandbox provider: {SANDBOX_PROVIDER}")
        provider = OpenSandboxProvider(
            domain=SANDBOX_DOMAIN,
            protocol=SANDBOX_PROTOCOL,
            api_key_env=SANDBOX_API_KEY_ENV,
            image=SANDBOX_IMAGE,
            sandbox_timeout_seconds=SANDBOX_TIMEOUT_SECONDS,
            request_timeout_seconds=SANDBOX_REQUEST_TIMEOUT_SECONDS,
            cpu=SANDBOX_CPU,
            memory=SANDBOX_MEMORY,
            network_enabled=SANDBOX_NETWORK_ENABLED,
            allowed_egress=SANDBOX_ALLOWED_EGRESS,
            use_server_proxy=SANDBOX_USE_SERVER_PROXY,
        )
        _manager = WorkspaceSandboxManager(provider, enabled=SANDBOX_ENABLED)
        from sandbox_runtime.persistence import get_workspace_persistence
        get_workspace_persistence(_manager)
    return _manager
