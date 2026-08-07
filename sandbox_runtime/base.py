from typing import List, Protocol

from sandbox_runtime.models import CommandResult, FileEntry, ProcessResult, SandboxHandle


class SandboxError(RuntimeError):
    """Base error returned by the isolated workspace runtime."""


class SandboxUnavailableError(SandboxError):
    """Raised when workspace mode is disabled or its provider is unavailable."""


class SandboxValidationError(SandboxError):
    """Raised before an unsafe or malformed request reaches the provider."""


class SandboxProvider(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def create(self, session_id: str) -> SandboxHandle: ...

    def connect(self, sandbox_id: str) -> SandboxHandle: ...

    def execute(
        self,
        sandbox_id: str,
        command: str,
        cwd: str,
        timeout_seconds: int,
        background: bool = False,
    ) -> CommandResult: ...

    def list_files(self, sandbox_id: str, path: str, max_depth: int) -> List[FileEntry]: ...

    def read_file(self, sandbox_id: str, path: str) -> str: ...

    def read_file_bytes(self, sandbox_id: str, path: str) -> bytes: ...

    def write_file(self, sandbox_id: str, path: str, content: bytes | str) -> None: ...

    def renew(self, sandbox_id: str, timeout_seconds: int) -> None: ...

    def process(self, sandbox_id: str, process_id: str, action: str) -> ProcessResult: ...

    def destroy(self, sandbox_id: str) -> None: ...
