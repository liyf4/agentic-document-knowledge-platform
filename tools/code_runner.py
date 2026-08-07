import ast
import base64
import os
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_TIMEOUT_SECONDS = 5
DEFAULT_OUTPUT_LIMIT = 8000
DEFAULT_MAX_CODE_CHARS = 12000

ALLOWED_IMPORTS = {
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
}

BANNED_IMPORTS = {
    "builtins",
    "ctypes",
    "glob",
    "importlib",
    "inspect",
    "io",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "posix",
    "shutil",
    "site",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
}

BANNED_NAMES = {
    "__builtins__",
    "__debug__",
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "vars",
}

BANNED_ATTRS = {
    "__base__",
    "__bases__",
    "__class__",
    "__dict__",
    "__globals__",
    "__mro__",
    "__subclasses__",
    "__getattribute__",
}


class CodeValidationError(ValueError):
    pass


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


def _iter_imported_modules(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            yield ""
        else:
            yield node.module or ""


def validate_python_code(code: str, max_code_chars: int = DEFAULT_MAX_CODE_CHARS) -> None:
    if len(code) > max_code_chars:
        raise CodeValidationError(f"Code is too long: {len(code)} characters, limit is {max_code_chars}.")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CodeValidationError(f"Syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for module in _iter_imported_modules(node):
                root = _module_root(module)
                if not root or root in BANNED_IMPORTS or root not in ALLOWED_IMPORTS:
                    raise CodeValidationError(f"Import is not allowed: {module or '<relative import>'}.")

        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise CodeValidationError(f"Name is not allowed: {node.id}.")

        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            raise CodeValidationError(f"Attribute is not allowed: {node.attr}.")


def _print_last_value(code: str) -> str:
    tree = ast.parse(code, mode="exec")
    if not tree.body:
        return code

    last = tree.body[-1]
    value_to_print = None
    if isinstance(last, ast.Expr):
        if isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name) and last.value.func.id == "print":
            return code
        value_to_print = last.value
        tree.body.pop()
    elif isinstance(last, ast.Assign):
        for target in last.targets:
            if isinstance(target, ast.Name):
                value_to_print = ast.Name(id=target.id, ctx=ast.Load())
                break
    elif isinstance(last, ast.AnnAssign) and isinstance(last.target, ast.Name):
        value_to_print = ast.Name(id=last.target.id, ctx=ast.Load())
    elif isinstance(last, ast.AugAssign) and isinstance(last.target, ast.Name):
        value_to_print = ast.Name(id=last.target.id, ctx=ast.Load())

    if value_to_print is None:
        return code

    tree.body.append(
        ast.copy_location(
            ast.Expr(value=ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[value_to_print], keywords=[])),
            last,
        )
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _truncate_output(text: str, output_limit: int) -> str:
    if len(text) <= output_limit:
        return text
    omitted = len(text) - output_limit
    return f"{text[:output_limit]}\n... <truncated {omitted} characters>"


def _runner_source(encoded_code: str) -> str:
    return textwrap.dedent(
        f"""
        import base64
        import builtins
        import sys

        _ALLOWED_IMPORTS = {sorted(ALLOWED_IMPORTS)!r}
        _SAFE_BUILTINS = {{
            "abs": builtins.abs,
            "all": builtins.all,
            "any": builtins.any,
            "bool": builtins.bool,
            "chr": builtins.chr,
            "dict": builtins.dict,
            "divmod": builtins.divmod,
            "enumerate": builtins.enumerate,
            "filter": builtins.filter,
            "float": builtins.float,
            "format": builtins.format,
            "frozenset": builtins.frozenset,
            "hash": builtins.hash,
            "hex": builtins.hex,
            "int": builtins.int,
            "isinstance": builtins.isinstance,
            "issubclass": builtins.issubclass,
            "len": builtins.len,
            "list": builtins.list,
            "map": builtins.map,
            "max": builtins.max,
            "min": builtins.min,
            "next": builtins.next,
            "oct": builtins.oct,
            "ord": builtins.ord,
            "pow": builtins.pow,
            "print": builtins.print,
            "range": builtins.range,
            "repr": builtins.repr,
            "reversed": builtins.reversed,
            "round": builtins.round,
            "set": builtins.set,
            "slice": builtins.slice,
            "sorted": builtins.sorted,
            "str": builtins.str,
            "sum": builtins.sum,
            "tuple": builtins.tuple,
            "zip": builtins.zip,
            "ArithmeticError": builtins.ArithmeticError,
            "AssertionError": builtins.AssertionError,
            "Exception": builtins.Exception,
            "IndexError": builtins.IndexError,
            "KeyError": builtins.KeyError,
            "LookupError": builtins.LookupError,
            "RuntimeError": builtins.RuntimeError,
            "TypeError": builtins.TypeError,
            "ValueError": builtins.ValueError,
        }}

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level:
                raise ImportError("Relative imports are not allowed.")
            root = name.split(".", 1)[0]
            if root not in _ALLOWED_IMPORTS:
                raise ImportError(f"Import is not allowed: {{name}}")
            return builtins.__import__(name, globals, locals, fromlist, level)

        _SAFE_BUILTINS["__import__"] = _safe_import
        _code = base64.b64decode({encoded_code!r}).decode("utf-8")
        _globals = {{"__builtins__": _SAFE_BUILTINS}}
        exec(_code, _globals, _globals)
        """
    ).strip()


def execute_python_code(
    code: str,
    *,
    session_id: str = "default",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
    max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
    work_root: Optional[Path] = None,
) -> str:
    """Run a small Python snippet in a constrained child interpreter."""
    try:
        validate_python_code(code, max_code_chars=max_code_chars)
    except CodeValidationError as exc:
        return f"Code rejected: {exc}"

    code = _print_last_value(code)
    safe_session = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)[:80] or "default"
    root = Path(work_root) if work_root else Path(tempfile.gettempdir()) / "chatbot_agent_code"
    run_dir = root / safe_session / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)

    encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
    runner_path = run_dir / "runner.py"
    runner_path.write_text(_runner_source(encoded_code), encoding="utf-8")

    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    system_root = os.environ.get("SystemRoot")
    if system_root:
        env["SystemRoot"] = system_root
    path = os.environ.get("PATH")
    if path:
        env["PATH"] = path

    try:
        completed = subprocess.run(
            [sys.executable, str(runner_path)],
            cwd=str(run_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return f"Code execution timed out after {timeout_seconds} seconds."

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    pieces = []
    if stdout:
        pieces.append(f"stdout:\n{stdout}")
    if stderr:
        pieces.append(f"stderr:\n{stderr}")
    if completed.returncode != 0:
        pieces.append(f"exit_code: {completed.returncode}")
    if not pieces:
        pieces.append("Code executed successfully with no output.")

    return _truncate_output("\n\n".join(pieces), output_limit)


def create_python_code_tool(
    *,
    session_id: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
    max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
):
    def run_python_code(code: str) -> str:
        """Run a short, self-contained Python snippet for calculations or data analysis."""
        return execute_python_code(
            code,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
            output_limit=output_limit,
            max_code_chars=max_code_chars,
        )

    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=run_python_code,
        name="run_python_code",
        description=(
            "Run a short, self-contained Python snippet for calculations, data processing, "
            "or algorithm checks. Do not use it for file access, network access, project edits, "
            "system commands, or long-running jobs. Print the result you need."
        ),
    )
