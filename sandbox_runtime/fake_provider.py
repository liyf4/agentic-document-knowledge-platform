import itertools
import posixpath
from typing import Dict, List

from sandbox_runtime.base import SandboxError
from sandbox_runtime.models import CommandResult, FileEntry, ProcessResult, SandboxHandle


class FakeSandboxProvider:
    """In-memory provider for deterministic unit tests. It never runs host commands."""

    name = "fake"

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self.sandboxes: Dict[str, Dict[str, object]] = {}

    def is_available(self) -> bool:
        return True

    def create(self, session_id: str) -> SandboxHandle:
        sandbox_id = f"fake-{next(self._ids)}"
        self.sandboxes[sandbox_id] = {"session_id": session_id, "files": {}, "processes": {}}
        return SandboxHandle(sandbox_id=sandbox_id, provider=self.name)

    def connect(self, sandbox_id: str) -> SandboxHandle:
        self._require(sandbox_id)
        return SandboxHandle(sandbox_id=sandbox_id, provider=self.name)

    def execute(
        self,
        sandbox_id: str,
        command: str,
        cwd: str,
        timeout_seconds: int,
        background: bool = False,
    ) -> CommandResult:
        sandbox = self._require(sandbox_id)
        if background:
            process_id = f"process-{len(sandbox['processes']) + 1}"
            sandbox["processes"][process_id] = ProcessResult(process_id=process_id, status="running")
            return CommandResult(command=command, exit_code=None, process_id=process_id, background=True)
        return CommandResult(command=command, exit_code=0, stdout=f"fake:{cwd}$ {command}\n")

    def list_files(self, sandbox_id: str, path: str, max_depth: int) -> List[FileEntry]:
        files: Dict[str, bytes] = self._require(sandbox_id)["files"]
        prefix = path.rstrip("/") + "/"
        return [
            FileEntry(path=file_path, size=len(content))
            for file_path, content in sorted(files.items())
            if file_path == path or file_path.startswith(prefix)
        ]

    def read_file(self, sandbox_id: str, path: str) -> str:
        files: Dict[str, bytes] = self._require(sandbox_id)["files"]
        if path not in files:
            raise SandboxError(f"Workspace file not found: {path}")
        return files[path].decode("utf-8")

    def read_file_bytes(self, sandbox_id: str, path: str) -> bytes:
        files: Dict[str, bytes] = self._require(sandbox_id)["files"]
        if path not in files:
            raise SandboxError(f"Workspace file not found: {path}")
        return files[path]

    def write_file(self, sandbox_id: str, path: str, content: bytes | str) -> None:
        files: Dict[str, bytes] = self._require(sandbox_id)["files"]
        files[posixpath.normpath(path)] = content.encode("utf-8") if isinstance(content, str) else content

    def renew(self, sandbox_id: str, timeout_seconds: int) -> None:
        self._require(sandbox_id)

    def process(self, sandbox_id: str, process_id: str, action: str) -> ProcessResult:
        processes: Dict[str, ProcessResult] = self._require(sandbox_id)["processes"]
        result = processes.get(process_id)
        if result is None:
            raise SandboxError(f"Unknown sandbox process: {process_id}")
        if action == "stop":
            result.status = "stopped"
        return result

    def destroy(self, sandbox_id: str) -> None:
        self.sandboxes.pop(sandbox_id, None)

    def _require(self, sandbox_id: str) -> Dict[str, object]:
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None:
            raise SandboxError(f"Unknown sandbox: {sandbox_id}")
        return sandbox
