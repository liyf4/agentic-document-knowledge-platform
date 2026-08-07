"""Run the local chatbot development services in one terminal."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
IS_WINDOWS = os.name == "nt"


@dataclass(frozen=True)
class Service:
    name: str
    command: Sequence[str] | str
    cwd: Path
    shell: bool = False


def _npm_command() -> str:
    executable = shutil.which("npm.cmd" if IS_WINDOWS else "npm")
    if not executable:
        raise RuntimeError("找不到 npm，请先安装 Node.js。")
    return executable


def _services(args: argparse.Namespace) -> list[Service]:
    services = [
        Service(
            "backend",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                args.host,
                "--port",
                str(args.backend_port),
                *(["--reload"] if args.reload else []),
            ],
            ROOT,
        ),
        Service(
            "frontend",
            [_npm_command(), "run", "dev", "--", "--host", args.host, "--port", str(args.frontend_port)],
            FRONTEND,
        ),
    ]

    sandbox_command = args.sandbox_command or os.getenv("CHATBOT_SANDBOX_COMMAND", "").strip()
    if sandbox_command:
        services.insert(0, Service("sandbox", sandbox_command, ROOT, shell=True))
    return services


def _stream_output(name: str, pipe: object) -> None:
    for line in iter(pipe.readline, ""):
        output = f"[{name:<8}] {line}"
        console_encoding = sys.stdout.encoding or "utf-8"
        safe_output = output.encode(console_encoding, errors="replace").decode(console_encoding)
        print(safe_output, end="", flush=True)
    pipe.close()


def _start(service: Service) -> subprocess.Popen[str]:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
    process = subprocess.Popen(
        service.command,
        cwd=service.cwd,
        shell=service.shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
        start_new_session=not IS_WINDOWS,
    )
    assert process.stdout is not None
    threading.Thread(target=_stream_output, args=(service.name, process.stdout), daemon=True).start()
    return process


def _stop_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在一个终端中启动 Chatbot 前后端服务。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--no-reload", action="store_false", dest="reload", help="关闭后端热重载")
    parser.add_argument(
        "--sandbox-command",
        help="可选：同时托管 OpenSandbox 的启动命令；也可设置 CHATBOT_SANDBOX_COMMAND。",
    )
    parser.set_defaults(reload=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not (FRONTEND / "package.json").is_file():
        print("找不到 frontend/package.json。", file=sys.stderr)
        return 2
    if not (FRONTEND / "node_modules").is_dir():
        print("前端依赖尚未安装，请先运行：cd frontend; npm install", file=sys.stderr)
        return 2

    try:
        services = _services(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    running: list[tuple[Service, subprocess.Popen[str]]] = []
    try:
        for service in services:
            process = _start(service)
            running.append((service, process))
            print(f"[{service.name:<8}] 已启动 (PID {process.pid})", flush=True)

        print(f"\n前端：http://{args.host}:{args.frontend_port}")
        print(f"后端：http://{args.host}:{args.backend_port}/api/health")
        print("按 Ctrl+C 停止全部服务。\n", flush=True)

        while True:
            for service, process in running:
                code = process.poll()
                if code is not None:
                    print(f"\n[{service.name}] 已退出，退出码 {code}；正在停止其他服务。")
                    return code or 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n正在停止全部服务……")
        return 0
    finally:
        for _, process in reversed(running):
            _stop_tree(process)


if __name__ == "__main__":
    raise SystemExit(main())
