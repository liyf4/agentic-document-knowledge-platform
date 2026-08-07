from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SandboxHandle:
    sandbox_id: str
    provider: str
    status: str = "running"


@dataclass
class CommandResult:
    command: str
    exit_code: Optional[int]
    stdout: str = ""
    stderr: str = ""
    process_id: str = ""
    background: bool = False
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileEntry:
    path: str
    is_dir: bool = False
    size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessResult:
    process_id: str
    status: str
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
