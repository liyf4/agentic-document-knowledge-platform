import importlib.util
import os
from datetime import timedelta
from typing import Any, Dict, Iterable, List

from sandbox_runtime.base import SandboxError, SandboxUnavailableError
from sandbox_runtime.models import CommandResult, FileEntry, ProcessResult, SandboxHandle


def _text_lines(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, Iterable):
        return str(value)
    lines = []
    for item in value:
        lines.append(str(getattr(item, "text", item)))
    return "".join(lines)


class OpenSandboxProvider:
    """Thin, delayed-import adapter around the official OpenSandbox Python SDK."""

    name = "opensandbox"

    def __init__(
        self,
        *,
        domain: str,
        protocol: str,
        api_key_env: str,
        image: str,
        sandbox_timeout_seconds: int,
        request_timeout_seconds: int,
        cpu: str,
        memory: str,
        network_enabled: bool,
        allowed_egress: List[str],
        use_server_proxy: bool,
    ) -> None:
        if domain.startswith("https://"):
            protocol = "https"
        elif domain.startswith("http://"):
            protocol = "http"
        self.domain = domain.removeprefix("http://").removeprefix("https://").rstrip("/")
        self.protocol = protocol
        self.api_key_env = api_key_env
        self.image = image
        self.sandbox_timeout_seconds = sandbox_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.cpu = cpu
        self.memory = memory
        self.network_enabled = network_enabled
        self.allowed_egress = list(allowed_egress)
        self.use_server_proxy = use_server_proxy
        self._clients: Dict[str, Any] = {}

    def is_available(self) -> bool:
        return importlib.util.find_spec("opensandbox") is not None

    def create(self, session_id: str) -> SandboxHandle:
        sdk = self._sdk()
        config = self._connection_config(sdk)
        create_kwargs: Dict[str, Any] = {
            "connection_config": config,
            "timeout": timedelta(seconds=self.sandbox_timeout_seconds),
            "resource": {"cpu": self.cpu, "memory": self.memory},
            "metadata": {"project": "chatbot", "session_id": session_id},
        }
        create_kwargs["network_policy"] = self._network_policy(sdk)
        sandbox: Any = None
        sandbox_id = ""
        try:
            sandbox = sdk["SandboxSync"].create(self.image, **create_kwargs)
            sandbox_id = str(getattr(sandbox, "id", "") or getattr(sandbox, "sandbox_id", ""))
            if not sandbox_id:
                raise SandboxError("OpenSandbox created an instance without returning its id.")
            self._clients[sandbox_id] = sandbox
            sandbox.commands.run("mkdir -p /workspace/inputs /workspace/output")
            return SandboxHandle(sandbox_id=sandbox_id, provider=self.name)
        except Exception as exc:
            if sandbox is not None:
                try:
                    sandbox.destroy()
                except Exception:
                    pass
            if sandbox_id:
                self._clients.pop(sandbox_id, None)
            if isinstance(exc, SandboxError):
                raise
            raise SandboxUnavailableError(f"OpenSandbox create failed: {exc}") from exc

    def connect(self, sandbox_id: str) -> SandboxHandle:
        if sandbox_id in self._clients:
            return SandboxHandle(sandbox_id=sandbox_id, provider=self.name)
        sdk = self._sdk()
        config = self._connection_config(sdk)
        connector = getattr(sdk["SandboxSync"], "connect", None)
        if connector is None:
            raise SandboxUnavailableError(
                "The installed OpenSandbox SDK cannot reconnect to an existing sandbox; upgrade opensandbox."
            )
        try:
            try:
                sandbox = connector(sandbox_id=sandbox_id, connection_config=config)
            except TypeError:
                sandbox = connector(sandbox_id, connection_config=config)
            self._clients[sandbox_id] = sandbox
            return SandboxHandle(sandbox_id=sandbox_id, provider=self.name)
        except Exception as exc:
            raise SandboxUnavailableError(f"OpenSandbox reconnect failed: {exc}") from exc

    def execute(
        self,
        sandbox_id: str,
        command: str,
        cwd: str,
        timeout_seconds: int,
        background: bool = False,
    ) -> CommandResult:
        sandbox = self._client(sandbox_id)
        try:
            sdk_opts = self._sdk()["RunCommandOpts"](
                background=background,
                working_directory=cwd,
                timeout=timedelta(seconds=timeout_seconds),
            )
            execution = sandbox.commands.run(command, opts=sdk_opts)
            logs = getattr(execution, "logs", None)
            stdout = _text_lines(getattr(logs, "stdout", None))
            stderr = _text_lines(getattr(logs, "stderr", None))
            process_id = str(
                getattr(execution, "id", "")
                or getattr(execution, "process_id", "")
                or getattr(execution, "command_id", "")
            )
            exit_code = getattr(execution, "exit_code", None)
            if exit_code is None:
                exit_code = getattr(execution, "code", None)
            return CommandResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                process_id=process_id,
                background=background,
            )
        except Exception as exc:
            raise SandboxError(f"Sandbox command failed: {exc}") from exc

    def list_files(self, sandbox_id: str, path: str, max_depth: int) -> List[FileEntry]:
        sandbox = self._client(sandbox_id)
        try:
            list_entry = self._sdk()["DirectoryListEntry"](path=path, depth=max_depth)
            raw_entries = sandbox.files.list_directory(list_entry)
            entries = []
            for item in raw_entries or []:
                item_path = str(getattr(item, "path", "") or getattr(item, "name", ""))
                entries.append(
                    FileEntry(
                        path=item_path,
                        is_dir=bool(
                            getattr(item, "is_dir", False)
                            or getattr(item, "entry_type", "") in {"directory", "dir"}
                        ),
                        size=getattr(item, "size", None),
                    )
                )
            return entries
        except Exception as exc:
            raise SandboxError(f"Sandbox file listing failed: {exc}") from exc

    def read_file(self, sandbox_id: str, path: str) -> str:
        try:
            value = self._client(sandbox_id).files.read_file(path)
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
        except Exception as exc:
            raise SandboxError(f"Sandbox file read failed: {exc}") from exc

    def read_file_bytes(self, sandbox_id: str, path: str) -> bytes:
        try:
            files = self._client(sandbox_id).files
            reader = getattr(files, "read_bytes", None)
            value = reader(path) if reader else files.read_file(path)
            return value if isinstance(value, bytes) else str(value).encode("utf-8")
        except Exception as exc:
            raise SandboxError(f"Sandbox binary file read failed: {exc}") from exc

    def write_file(self, sandbox_id: str, path: str, content: bytes | str) -> None:
        try:
            self._client(sandbox_id).files.write_file(path, content, mode=0o644)
        except Exception as exc:
            raise SandboxError(f"Sandbox file write failed: {exc}") from exc

    def renew(self, sandbox_id: str, timeout_seconds: int) -> None:
        try:
            sandbox = self._client(sandbox_id)
            renew = getattr(sandbox, "renew", None)
            if renew is not None:
                renew(timedelta(seconds=timeout_seconds))
        except Exception as exc:
            raise SandboxError(f"OpenSandbox renew failed: {exc}") from exc

    def process(self, sandbox_id: str, process_id: str, action: str) -> ProcessResult:
        commands = self._client(sandbox_id).commands
        try:
            if action == "status":
                value = commands.get_command_status(process_id)
                running = getattr(value, "running", None)
                exit_code = getattr(value, "exit_code", None)
                if running is True:
                    status = "running"
                elif running is False:
                    status = "completed" if exit_code == 0 else "failed"
                else:
                    status = "unknown"
                return ProcessResult(process_id=process_id, status=status, exit_code=exit_code)
            if action == "logs":
                value = commands.get_background_command_logs(process_id)
                return ProcessResult(
                    process_id=process_id,
                    status="unknown",
                    stdout=str(getattr(value, "content", "")),
                    metadata={"cursor": getattr(value, "cursor", None)},
                )
            commands.interrupt(process_id)
            return ProcessResult(process_id=process_id, status="stopped")
        except Exception as exc:
            raise SandboxError(f"Sandbox process {action} failed: {exc}") from exc

    def destroy(self, sandbox_id: str) -> None:
        sandbox = self._client(sandbox_id)
        try:
            destroy = getattr(sandbox, "destroy", None) or getattr(sandbox, "kill", None)
            if destroy is None:
                raise SandboxError("The installed OpenSandbox SDK has no destroy method.")
            destroy()
            self._clients.pop(sandbox_id, None)
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f"OpenSandbox destroy failed: {exc}") from exc

    def _client(self, sandbox_id: str) -> Any:
        if sandbox_id not in self._clients:
            self.connect(sandbox_id)
        return self._clients[sandbox_id]

    def _sdk(self) -> Dict[str, Any]:
        if not self.is_available():
            raise SandboxUnavailableError(
                "Workspace mode requires the optional 'opensandbox' package and an OpenSandbox server."
            )
        try:
            from opensandbox import SandboxSync
            from opensandbox.config import ConnectionConfigSync
            from opensandbox.models.execd import RunCommandOpts
            from opensandbox.models.filesystem import DirectoryListEntry
            from opensandbox.models.sandboxes import NetworkPolicy, NetworkRule

            return {
                "SandboxSync": SandboxSync,
                "ConnectionConfigSync": ConnectionConfigSync,
                "NetworkPolicy": NetworkPolicy,
                "NetworkRule": NetworkRule,
                "RunCommandOpts": RunCommandOpts,
                "DirectoryListEntry": DirectoryListEntry,
            }
        except Exception as exc:
            raise SandboxUnavailableError(f"Could not load the OpenSandbox SDK: {exc}") from exc

    def _connection_config(self, sdk: Dict[str, Any]) -> Any:
        kwargs: Dict[str, Any] = {
            "domain": self.domain,
            "protocol": self.protocol,
            "request_timeout": timedelta(seconds=self.request_timeout_seconds),
            "use_server_proxy": self.use_server_proxy,
        }
        api_key = os.getenv(self.api_key_env)
        if api_key:
            kwargs["api_key"] = api_key
        return sdk["ConnectionConfigSync"](**kwargs)

    def _network_policy(self, sdk: Dict[str, Any]) -> Any:
        if not self.network_enabled:
            return self._construct_model(
                sdk["NetworkPolicy"],
                [{"defaultAction": "deny", "egress": []}, {"default_action": "deny", "egress": []}],
            )
        if self.allowed_egress:
            rules = [sdk["NetworkRule"](action="allow", target=target) for target in self.allowed_egress]
            return self._construct_model(
                sdk["NetworkPolicy"],
                [{"defaultAction": "deny", "egress": rules}, {"default_action": "deny", "egress": rules}],
            )
        return self._construct_model(
            sdk["NetworkPolicy"],
            [{"defaultAction": "allow", "egress": []}, {"default_action": "allow", "egress": []}],
        )

    @staticmethod
    def _construct_model(model: Any, variants: List[Dict[str, Any]]) -> Any:
        last_error: Exception | None = None
        for kwargs in variants:
            try:
                return model(**kwargs)
            except Exception as exc:
                last_error = exc
        raise SandboxUnavailableError(f"OpenSandbox SDK model is incompatible: {last_error}")
