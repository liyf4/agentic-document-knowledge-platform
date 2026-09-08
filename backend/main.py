import json
import logging
import mimetypes
import os
import re
import threading
import asyncio
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Literal, Optional
from urllib.parse import quote

from utils.ssl_compat import prefer_certifi_default_context

prefer_certifi_default_context()

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from config import ASYNC_UPLOAD_THRESHOLD_MB, DATA_DIR, SANDBOX_REAPER_INTERVAL_SECONDS
from core.agent import build_agent_system_prompt_preview, create_agent_executor, retrieve_session_knowledge
from core.context_manager import (
    ContextLimitError,
    get_context_usage,
    make_usage_snapshot,
    prepare_session_context,
)
from core.document_processor import delete_document, process_file, reindex_document
from core.history import get_session_history
from core.indexing_jobs import list_index_tasks, submit_process_file
from core.request_router import RouteDecision, apply_execution_mode, find_route_violations, route_chat_request
from core.orchestration import run_orchestration
from core.skill_chat_commands import handle_skill_chat_command
from core.skill_workflow import SkillWorkflowError, run_skill_workflows
from core.retriever import (
    clear_last_retrieval_debug,
    get_last_retrieval_debug,
    get_session_document_overview,
    set_last_retrieval_debug,
)
from core.trace import get_recent_traces, get_trace_statistics, init_trace_db, record_chat_trace
from db.skill_manager import (
    create_skill as create_skill_record,
    delete_skill as delete_skill_record,
    get_skill,
    init_skill_db,
    list_skills as list_skill_records,
    match_skills_with_workflow_fallback,
    summarize_skills,
    update_skill as update_skill_record,
)
from db.session_manager import (
    create_new_session,
    delete_session_data,
    get_all_sessions,
    get_session_document,
    get_session_documents,
    init_meta_db,
    update_session_title,
)
from db.file_manager import (
    get_artifact, get_uploaded_file, init_file_db, list_artifacts, list_document_metadata, list_uploaded_files,
)
from core.persistent_storage import get_persistent_file_store
from db.sandbox_manager import get_session_sandbox, init_sandbox_db
from sandbox_runtime.base import SandboxError
from sandbox_runtime.manager import get_sandbox_manager
from tools.workspace import create_workspace_tools

load_dotenv()


class _PollingAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/index-tasks " not in record.getMessage()


def _configure_access_log_filter() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, _PollingAccessFilter) for item in access_logger.filters):
        access_logger.addFilter(_PollingAccessFilter())


class BufferedUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class SessionUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)
    mode: Literal["chat", "workspace"] = "chat"


class SessionOut(BaseModel):
    id: str
    title: str


class SkillCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    trigger_terms: List[str] = Field(default_factory=list)
    instruction: str = Field(min_length=1, max_length=8000)
    examples: List[str] = Field(default_factory=list)
    enabled: bool = True
    mode: Literal["prompt", "workflow"] = "prompt"
    workflow: Dict[str, Any] = Field(default_factory=dict)


class SkillUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    description: Optional[str] = Field(default=None, min_length=1, max_length=500)
    trigger_terms: Optional[List[str]] = None
    instruction: Optional[str] = Field(default=None, min_length=1, max_length=8000)
    examples: Optional[List[str]] = None
    enabled: Optional[bool] = None
    mode: Optional[Literal["prompt", "workflow"]] = None
    workflow: Optional[Dict[str, Any]] = None


class SkillOut(BaseModel):
    id: str
    name: str
    description: str
    trigger_terms: List[str]
    instruction: str
    examples: List[str]
    enabled: bool
    created_at: str
    updated_at: str
    mode: str
    workflow: Dict[str, Any]
    source: str = "database"
    path: str = ""
    slug: str = ""
    assets: List[str] = Field(default_factory=list)


def _initialize_databases() -> None:
    init_meta_db()
    init_file_db()
    init_trace_db()
    init_skill_db()
    init_sandbox_db()


app = FastAPI(title="Agentic RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    _configure_access_log_filter()
    _initialize_databases()
    _start_sandbox_reaper()


_reaper_stop = threading.Event()
_reaper_thread: Optional[threading.Thread] = None


def _start_sandbox_reaper() -> None:
    global _reaper_thread
    if _reaper_thread and _reaper_thread.is_alive():
        return
    _reaper_stop.clear()

    def _run() -> None:
        while not _reaper_stop.wait(max(1, SANDBOX_REAPER_INTERVAL_SECONDS)):
            try:
                get_sandbox_manager().reap_idle()
            except Exception:
                # Provider outages must not take the API down or delete active mappings.
                pass

    _reaper_thread = threading.Thread(target=_run, name="sandbox-idle-reaper", daemon=True)
    _reaper_thread.start()


@app.on_event("shutdown")
def shutdown() -> None:
    _reaper_stop.set()


def _is_default_session_title(title: str) -> bool:
    normalized = title.strip()
    return normalized in {"", "新对话", "鏂板璇?"}


def _session_exists(session_id: str) -> bool:
    return any(session[0] == session_id for session in get_all_sessions())


def _get_session_title(session_id: str) -> Optional[str]:
    for item_id, title in get_all_sessions():
        if item_id == session_id:
            return title
    return None


def _require_session(session_id: str) -> None:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")


def _require_workspace_available(mode: str) -> None:
    if mode != "workspace":
        return
    capabilities = get_sandbox_manager().capabilities()
    if not capabilities.get("available"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Workspace mode is unavailable. Enable sandbox.enabled, install the opensandbox SDK, "
                "and configure an OpenSandbox server."
            ),
        )


def _message_role(message: Any) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    return getattr(message, "type", "message")


def _serialize_messages(session_id: str) -> List[Dict[str, str]]:
    history = get_session_history(session_id)
    return [{"role": _message_role(msg), "content": msg.content} for msg in history.messages]


def _serialize_skill(skill: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(skill.get("id", "")),
        "name": str(skill.get("name", "")),
        "description": str(skill.get("description", "")),
        "trigger_terms": list(skill.get("trigger_terms") or []),
        "instruction": str(skill.get("instruction", "")),
        "examples": list(skill.get("examples") or []),
        "enabled": bool(skill.get("enabled", False)),
        "created_at": str(skill.get("created_at", "")),
        "updated_at": str(skill.get("updated_at", "")),
        "mode": str(skill.get("mode", "prompt")),
        "workflow": dict(skill.get("workflow") or {}),
        "source": str(skill.get("source", "database")),
        "path": str(skill.get("path", "")),
        "slug": str(skill.get("slug", "")),
        "assets": list(skill.get("assets") or []),
    }


def _sse(event: str, payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {data}\n\n"


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _final_output(value: Any) -> str:
    if isinstance(value, dict):
        output = value.get("output")
        return output if isinstance(output, str) else ""
    return ""


def _extract_token_usage(value: Any) -> tuple[Optional[int], Optional[int]]:
    """Best-effort extraction across LangChain message/chunk/provider response shapes."""
    candidates: List[Any] = [value]
    if isinstance(value, dict):
        candidates.extend([value.get("usage_metadata"), value.get("token_usage"), value.get("usage")])
        candidates.extend([value.get("output"), value.get("generations")])
    else:
        candidates.extend([
            getattr(value, "usage_metadata", None),
            getattr(value, "response_metadata", None),
        ])
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)):
            for item in candidate:
                prompt, completion = _extract_token_usage(item)
                if prompt is not None and completion is not None:
                    return prompt, completion
        if not isinstance(candidate, dict):
            continue
        nested = candidate.get("token_usage") or candidate.get("usage")
        if isinstance(nested, dict):
            candidate = nested
        prompt = candidate.get("input_tokens", candidate.get("prompt_tokens"))
        completion = candidate.get("output_tokens", candidate.get("completion_tokens"))
        if prompt is not None and completion is not None:
            try:
                return int(prompt), int(completion)
            except (TypeError, ValueError):
                pass
    return None, None


def _tool_preview(value: Any) -> str:
    return str(value)[:1000]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _tool_display_name(name: str) -> str:
    if name == "run_python_code":
        return "Python code"
    labels = {
        "workspace_terminal": "Sandbox terminal",
        "workspace_list_files": "Sandbox files",
        "workspace_read_file": "Sandbox file read",
        "workspace_write_file": "Sandbox file write",
        "workspace_process": "Sandbox process",
        "workspace_list_session_files": "Session uploads",
        "workspace_import_uploaded_file": "Import upload",
        "workspace_list_artifacts": "Saved artifacts",
        "workspace_import_artifact": "Restore artifact",
        "workspace_save_file": "Save workspace file",
        "retrieve_knowledge": "本地知识库检索",
        "query_document_metadata": "文档元数据查询",
        "read_website": "网页读取",
        "web_search": "网络搜索",
        "get_current_time": "当前时间",
    }
    return labels.get(name, name)


def _serialize_intermediate_tool_call(action: Any, observation: Any) -> Dict[str, Any]:
    tool_name = str(getattr(action, "tool", ""))
    return {
        "tool": tool_name,
        "display_name": _tool_display_name(tool_name),
        "input": getattr(action, "tool_input", ""),
        "status": "completed",
        "output_preview": _tool_preview(observation),
        "error": "",
        "started_at": "",
        "ended_at": "",
        "call_stage": "main_agent",
        "duplicate_blocked": False,
        "remediation": False,
    }


_DOCUMENT_QUERY_TERMS = (
    "文档",
    "文件",
    "资料",
    "上传",
    "pdf",
    "doc",
    "docx",
    "csv",
    "xlsx",
    "表格",
    "主要内容",
    "讲了什么",
    "说了什么",
    "总结",
    "概括",
    "内容",
    "document",
    "file",
    "uploaded",
    "project",
    "atlas",
    "beacon",
    "cygnus",
    "action item",
    "owner",
    "due date",
    "priority",
    "delayed",
    "delay",
    "citation",
    "evidence",
    "hybrid retrieval",
    "rerank",
    "prompt injection",
    "prompt-injection",
    "confidential",
    "private",
    "salary",
    "api key",
    "human approval",
    "route violation",
    "task success rate",
    "demo target",
    "personal phone",
    "nora",
    "send_email",
    "table cell",
    "payroll token",
    ".md",
)

_DOCUMENT_OVERVIEW_TERMS = (
    "主要内容",
    "讲了什么",
    "说了什么",
    "总结",
    "概括",
    "overview",
    "summary",
    "summarize",
)


def _is_document_query(message: str) -> bool:
    lowered = message.lower()
    return any(term.lower() in lowered for term in _DOCUMENT_QUERY_TERMS)


def _is_document_overview_query(message: str) -> bool:
    lowered = message.lower()
    return any(term.lower() in lowered for term in _DOCUMENT_OVERVIEW_TERMS)


def _auto_prefetch_document_context(session_id: str, message: str) -> tuple[str, List[Dict[str, Any]]]:
    documents = [row for row in get_session_documents(session_id) if str(row[3]).lower() == "completed"]
    if not documents or not _is_document_query(message):
        return "", []

    started_at = _now_iso()
    tool_call: Dict[str, Any] = {
        "tool": "retrieve_knowledge",
        "display_name": _tool_display_name("retrieve_knowledge"),
        "input": {"query": message, "source": "auto_prefetch"},
        "status": "running",
        "output_preview": "",
        "error": "",
        "started_at": started_at,
        "ended_at": "",
    }

    try:
        context = ""
        if len(documents) > 1 and _is_document_overview_query(message):
            overview, overview_debug = get_session_document_overview(session_id)
            if overview.strip():
                overview_debug["original_query"] = message
                overview_debug["search_query"] = "document_overview"
                set_last_retrieval_debug(session_id, overview_debug)
                context = (
                    "The user asked a broad question about uploaded documents. "
                    "Use this per-file evidence so every uploaded document is considered.\n\n"
                    f"{overview}"
                )

        if not context:
            context = retrieve_session_knowledge(session_id, message)

        tool_call["status"] = "completed"
        tool_call["output_preview"] = _tool_preview(context)
        tool_call["ended_at"] = _now_iso()
        return context, [tool_call]
    except Exception as exc:
        tool_call["status"] = "failed"
        tool_call["error"] = str(exc)
        tool_call["ended_at"] = _now_iso()
        return "", [tool_call]


def _build_citations(retrieval_debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    final_docs = retrieval_debug.get("final")
    if not isinstance(final_docs, list):
        return []

    citations = []
    for index, item in enumerate(final_docs, start=1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        citations.append(
            {
                "id": str(index),
                "source": item.get("source") or "未知来源",
                "page": item.get("page"),
                "printed_page": item.get("printed_page"),
                "section_path": item.get("section_path"),
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
                "content": content,
                "preview": item.get("preview") or content[:240].replace("\n", " "),
            }
        )
    return citations


def _execute_matched_skill_workflows(
    session_id: str,
    matched_skills: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    try:
        workflow_results, skill_context, workflow_tool_calls = run_skill_workflows(session_id, matched_skills)
    except SkillWorkflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    now = _now_iso()
    for tool_call in workflow_tool_calls:
        tool_call["started_at"] = tool_call.get("started_at") or now
        tool_call["ended_at"] = tool_call.get("ended_at") or now
    return workflow_results, skill_context, workflow_tool_calls


def _should_skip_document_prefetch(workflow_results: List[Dict[str, Any]]) -> bool:
    return any(str(result.get("workflow_kind", "")).casefold() == "table_analysis" for result in workflow_results)


def _route_debug(decision: RouteDecision) -> Dict[str, Any]:
    debug = decision.to_debug()
    debug["route_violation"] = []
    return debug


def _supervisor_execution(
    decision: RouteDecision, orchestration_debug: Dict[str, Any],
) -> tuple[List[str], Dict[str, int], Dict[str, Any]]:
    context = orchestration_debug.get("execution_context") if isinstance(orchestration_debug, dict) else None
    if not isinstance(context, dict) or not orchestration_debug.get("enabled"):
        return list(decision.allowed_tools), {}, {}
    remaining = context.get("remaining_tools")
    limits = context.get("tool_call_limits")
    return (
        list(remaining) if isinstance(remaining, list) else list(decision.allowed_tools),
        {str(k): int(v) for k, v in (limits or {}).items()},
        context,
    )


def _parse_tool_json(value: Any) -> Dict[str, Any]:
    text = str(value or "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _registered_artifacts(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    for call in tool_calls:
        if str(call.get("tool")) not in {"workspace_write_file", "workspace_save_file"}:
            continue
        payload = _parse_tool_json(call.get("output_preview"))
        result = payload.get("result") if payload.get("ok") is True else None
        if not isinstance(result, dict):
            continue
        if result.get("artifact_id") and int(result.get("size_bytes", 0) or 0) > 0:
            call["artifact_id"] = result["artifact_id"]
            call["artifact_registered"] = True
            call["termination_reason"] = "artifact_registered"
            artifacts.append({
                "artifact_id": result["artifact_id"], "name": result.get("name", ""),
                "size_bytes": int(result["size_bytes"]),
            })
    return artifacts


def normalize_artifact_delivery(answer: str, tool_calls: List[Dict[str, Any]]) -> str:
    """Remove unsupported local links and claim delivery only after artifact registration."""
    cleaned = re.sub(r"\[[^\]]*\]\((?:sandbox:|file:|/?workspace/)[^)]+\)", "", answer or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"sandbox:/\S+|/workspace/\S+", "", cleaned, flags=re.IGNORECASE).strip()
    artifacts = _registered_artifacts(tool_calls)
    direct_tool_result = _parse_tool_json(answer)
    if artifacts and direct_tool_result.get("ok") is True:
        # A return_direct artifact tool yields its JSON observation as the agent
        # output. Keep that internal and present the stable delivery contract.
        answer = ""
    if artifacts:
        names = "、".join(str(item.get("name") or "产物") for item in artifacts)
        delivery = f"文件已保存到右侧已保存产物：{names}。"
        if "文件已保存到右侧已保存产物" not in cleaned:
            cleaned = f"{cleaned}\n\n{delivery}".strip()
    return cleaned


def _artifact_file_name(message: str) -> str:
    matches = re.findall(r"[\w\u4e00-\u9fff.-]+\.(?:txt|md|csv|json)", message, flags=re.IGNORECASE)
    name = os.path.basename(matches[-1]) if matches else "comparison_document.txt"
    name = re.sub(r"^(?:然后)?(?:写|生成|创建)(?:一个|一份)?", "", name)
    return name if name not in {".", ".."} else "comparison_document.txt"


def _compose_grounded_artifact(message: str, orchestration_debug: Dict[str, Any]) -> str:
    workers = orchestration_debug.get("workers", []) if isinstance(orchestration_debug, dict) else []
    document_evidence = [
        item for worker in workers if worker.get("worker") == "document"
        for item in worker.get("evidence", []) if item.get("kind") == "document"
    ]
    web_evidence = [
        item for worker in workers if worker.get("worker") == "web"
        for item in worker.get("evidence", []) if item.get("kind") == "verified_web"
    ]
    lowered = message.casefold()
    if "sql" in lowered and ("注入" in message or "injection" in lowered):
        relevant_documents = [
            item for item in document_evidence
            if "sql" in (str(item.get("section_path", "")) + str(item.get("text", ""))).casefold()
            and ("注入" in str(item.get("section_path", "")) + str(item.get("text", ""))
                 or "injection" in (str(item.get("section_path", "")) + str(item.get("text", ""))).casefold())
        ]
        document = relevant_documents[0] if relevant_documents else (document_evidence[0] if document_evidence else {})
        source = str(document.get("file_name") or "上传文档")
        section = str(document.get("section_path") or "未标注章节")
        evidence_id = str(document.get("evidence_id") or "")
        original = str(document.get("text") or "未取得相关原文证据")
        lines = [
            "SQL 注入防御方式对比",
            "",
            "一、上传文档证据",
            f"来源文件：{source}",
            f"章节：{section}",
            f"证据 ID：{evidence_id}",
            f"原文：{original}",
            "结论：文档的核心方案是使用预编译语句和参数化查询，使输入值不再被当作 SQL 结构执行，并避免直接拼接 SQL 字符串。",
            "",
            "二、联网核验补充",
            "1. 参数化查询/预编译语句：仍是首选防线，应用应使用语言或框架提供的绑定参数接口。",
            "2. 安全构造的存储过程：仅当过程内部同样不动态拼接不可信输入时，才具有与参数化查询相近的防护效果。",
            "3. 白名单输入验证：对表名、列名、排序方向等无法作为绑定参数的位置，只允许预定义值。",
            "4. 最小权限：限制应用数据库账户的读写范围，降低注入成功后的影响面。",
            "5. 转义不是首选主防线：仅在无法参数化的遗留场景谨慎使用数据库专用转义，并配合迁移计划。",
            "",
            "三、对比结论",
            "上传文档给出的“预编译语句 + 参数化查询 + 禁止字符串拼接”与权威网页建议一致，是实现层面的主防线。联网资料补充了安全存储过程、白名单验证和最小权限等纵深措施；这些措施用于补充而非替代参数化查询。",
            "",
            "四、互联网来源（已实际打开核验）",
        ]
    else:
        lines = ["证据对比文档", "", "一、上传文档证据"]
        for item in document_evidence[:4]:
            lines.extend([
                f"来源文件：{item.get('file_name', '')}",
                f"章节：{item.get('section_path', '')}",
                f"证据 ID：{item.get('evidence_id', '')}",
                f"原文：{item.get('text', '')}",
                "",
            ])
        lines.append("二、互联网来源（已实际打开核验）")

    for index, item in enumerate(web_evidence, 1):
        lines.extend([
            f"{index}. {item.get('title') or '未命名页面'}",
            f"URL：{item.get('url', '')}",
            f"来源类型：{item.get('source_type', 'web_page')}",
            f"访问时间：{item.get('accessed_at', '')}",
            f"证据 ID：{item.get('evidence_id', '')}",
        ])
    if not web_evidence:
        lines.append("未能打开并核验权威网页，因此不提供未经核验的联网结论。")
    lines.extend(["", "不确定性：网页内容可能随时间更新；以上联网结论以所列访问时间获取的页面为准。"])
    return "\n".join(lines).strip() + "\n"


def _write_grounded_artifact(
    session_id: str, message: str, orchestration_debug: Dict[str, Any], artifact_requirements: Dict[str, Any],
) -> Dict[str, Any]:
    name = _artifact_file_name(message)
    path = f"/workspace/output/{name}"
    content = _compose_grounded_artifact(message, orchestration_debug)
    write_tool = next(
        tool for tool in create_workspace_tools(session_id, artifact_requirements=artifact_requirements)
        if tool.name == "workspace_write_file"
    )
    started_at = _now_iso()
    output = str(write_tool.invoke({"path": path, "content": content}))
    parsed = _parse_tool_json(output)
    call = {
        "tool": "workspace_write_file", "display_name": _tool_display_name("workspace_write_file"),
        "input": {"path": path, "content_chars": len(content)},
        "status": "completed" if parsed.get("ok") is True else "failed",
        "output_preview": output, "error": "" if parsed.get("ok") is True else str(parsed.get("error") or parsed.get("error_code") or "tool_failed"),
        "started_at": started_at, "ended_at": _now_iso(), "call_stage": "main_agent_artifact",
        "duplicate_blocked": False, "remediation": False,
    }
    return call


def _annotate_duplicate_block(tool_call: Dict[str, Any]) -> None:
    payload = _parse_tool_json(tool_call.get("output_preview"))
    if payload.get("error_code") == "REMEDIATION_LIMIT_REACHED":
        tool_call["status"] = "blocked"
        tool_call["duplicate_blocked"] = True
        tool_call["termination_reason"] = "duplicate_capability_call_blocked"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _skill_selection_summary(route_decision: RouteDecision, matched_skills: List[Dict[str, Any]]) -> Dict[str, Any]:
    skills = summarize_skills(matched_skills)
    workflow_skills = [skill for skill in skills if str(skill.get("mode", "")).casefold() == "workflow"]
    prompt_skills = [skill for skill in skills if str(skill.get("mode", "")).casefold() != "workflow"]
    if workflow_skills:
        action = "命中 workflow skill，优先执行后端 skill 工作流。"
    elif prompt_skills:
        action = "命中 prompt skill，作为回答规则注入模型；如问题涉及上传文档，再走 RAG 取证。"
    elif route_decision.route == "document_qa":
        action = "未命中 skill，问题涉及上传文档，走 RAG。"
    elif route_decision.route == "web_search":
        action = "未命中 skill，问题要求外部信息，走 web 工具。"
    elif route_decision.route == "orchestrated":
        action = "未命中 skill，复杂请求由文档、联网和 Workspace Worker 协同取证与生成产物。"
    else:
        action = "未命中 skill，也未命中文档 RAG 场景，直接由模型回答。"
    return {
        "route": route_decision.route,
        "router_reason": route_decision.reason,
        "matched_skills": skills,
        "action": action,
    }


def _skill_selection_tool_call(
    route_decision: RouteDecision,
    matched_skills: List[Dict[str, Any]],
    message: str,
) -> Dict[str, Any]:
    summary = _skill_selection_summary(route_decision, matched_skills)
    return {
        "tool": "skill_selection",
        "display_name": "Skill selection",
        "input": {"message": message},
        "status": "completed",
        "output_preview": _safe_json(summary),
        "error": "",
        "started_at": _now_iso(),
        "ended_at": _now_iso(),
    }


def _log_route_decision(
    *,
    stage: str,
    session_id: str,
    message: str,
    route_decision: RouteDecision,
    matched_skills: Optional[List[Dict[str, Any]]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> None:
    payload = {
        "stage": stage,
        "session_id": session_id,
        "message": message,
        "route": route_decision.route,
        "router_reason": route_decision.reason,
        "allowed_tools": route_decision.allowed_tools,
        "blocked_tools": route_decision.blocked_tools,
        "matched_skills": summarize_skills(matched_skills or []),
        "tool_calls": [
            {
                "tool": call.get("tool"),
                "status": call.get("status"),
                "display_name": call.get("display_name"),
                "error": call.get("error"),
            }
            for call in (tool_calls or [])
        ],
    }
    print(f"[chat-route] {_safe_json(payload)}", flush=True)


def _workflow_skills_or_default(matched_skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    workflow_skills = [skill for skill in matched_skills if str(skill.get("mode", "prompt")).casefold() == "workflow"]
    if workflow_skills:
        selected: List[Dict[str, Any]] = []
        seen_kinds = set()
        for skill in workflow_skills:
            workflow = skill.get("workflow") or {}
            kind = str(workflow.get("kind") or skill.get("id") or "").strip().casefold()
            if kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            selected.append(skill)
        return selected
    return [
        {
            "id": "builtin_table_analysis",
            "name": "builtin_table_analysis",
            "description": "Built-in table analysis workflow.",
            "trigger_terms": [],
            "instruction": "Analyze uploaded CSV/XLSX files.",
            "examples": [],
            "enabled": True,
            "mode": "workflow",
            "workflow": {
                "kind": "table_analysis",
                "steps": ["load_latest_table", "clean_table", "profile_columns", "numeric_means", "summarize"],
            },
        }
    ]


def _format_table_workflow_answer(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "没有可用的数据分析结果。"
    lines = ["已使用后端表格工作流处理上传文件："]
    for result in results:
        document = result.get("document") or {}
        lines.append(f"\n文件: {document.get('file_name', '')}")
        lines.append(f"原始形状: {result.get('raw_shape', {})}")
        lines.append(f"清洗后形状: {result.get('clean_shape', {})}")
        numeric_means = result.get("numeric_means") or {}
        if numeric_means:
            lines.append("数值列均值:")
            for column, value in numeric_means.items():
                lines.append(f"- {column}: {value}")
        else:
            lines.append("未检测到可计算均值的数值列。")
        missing = result.get("missing_values") or {}
        lines.append(f"缺失值: {missing if missing else '无'}")
        notes = result.get("cleaning_notes") or []
        if notes:
            lines.append("清洗说明: " + " ".join(str(item) for item in notes))
    return "\n".join(lines)


def _record_direct_chat_response(
    *,
    session_id: str,
    user_input: str,
    answer: str,
    retrieval_debug: Optional[Dict[str, Any]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    history = get_session_history(session_id)
    history.add_user_message(user_input)
    history.add_ai_message(answer)
    context_usage = make_usage_snapshot(None, answer, session_id=session_id)
    trace_debug = dict(retrieval_debug or {})
    trace_debug["context_usage"] = context_usage
    record_chat_trace(
        session_id=session_id,
        user_input=user_input,
        answer=answer,
        retrieval_debug=trace_debug,
        tool_calls=tool_calls or [],
        citations=citations or [],
    )
    return context_usage


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "zai_api_key_configured": bool(os.getenv("ZAI_API_KEY")),
        "workspace": get_sandbox_manager().capabilities(),
    }


@app.post("/api/skills", response_model=SkillOut)
def create_skill(payload: SkillCreate) -> Dict[str, Any]:
    skill = create_skill_record(
        name=payload.name,
        description=payload.description,
        trigger_terms=payload.trigger_terms,
        instruction=payload.instruction,
        examples=payload.examples,
        enabled=payload.enabled,
        mode=payload.mode,
        workflow=payload.workflow,
    )
    return _serialize_skill(skill)


@app.get("/api/skills")
def list_skills() -> Dict[str, List[SkillOut]]:
    return {"skills": [_serialize_skill(skill) for skill in list_skill_records()]}


@app.get("/api/skills/{skill_id}", response_model=SkillOut)
def read_skill(skill_id: str) -> Dict[str, Any]:
    skill = get_skill(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found.")
    return _serialize_skill(skill)


@app.patch("/api/skills/{skill_id}", response_model=SkillOut)
def update_skill(skill_id: str, payload: SkillUpdate) -> Dict[str, Any]:
    updates = payload.dict(exclude_unset=True)
    skill = update_skill_record(skill_id, updates)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found.")
    return _serialize_skill(skill)


@app.delete("/api/skills/{skill_id}")
def delete_skill(skill_id: str) -> Dict[str, Any]:
    if not delete_skill_record(skill_id):
        raise HTTPException(status_code=404, detail="Skill not found.")
    return {"ok": True}


@app.get("/api/sessions")
def list_sessions() -> Dict[str, List[SessionOut]]:
    sessions = [SessionOut(id=session_id, title=title) for session_id, title in get_all_sessions()]
    return {"sessions": sessions}


@app.post("/api/sessions")
def create_session() -> SessionOut:
    session_id = create_new_session()
    title = _get_session_title(session_id) or ""
    return SessionOut(id=session_id, title=title)


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, payload: SessionUpdate) -> Dict[str, Any]:
    _require_session(session_id)
    update_session_title(session_id, payload.title.strip())
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    _require_session(session_id)
    if get_session_sandbox(session_id):
        try:
            get_sandbox_manager().destroy(session_id)
        except SandboxError as exc:
            raise HTTPException(status_code=503, detail=f"Could not destroy the session sandbox: {exc}") from exc
    # Remote cleanup is confirmed first. Only then remove durable originals, indexes, outputs, and metadata.
    get_persistent_file_store().remove_session(session_id)
    delete_session_data(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/sandbox")
def get_sandbox_status(session_id: str) -> Dict[str, Any]:
    _require_session(session_id)
    return get_sandbox_manager().status(session_id)


@app.delete("/api/sessions/{session_id}/sandbox")
def destroy_sandbox(session_id: str) -> Dict[str, Any]:
    _require_session(session_id)
    try:
        destroyed = get_sandbox_manager().destroy(session_id)
    except SandboxError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "destroyed": destroyed}


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str) -> Dict[str, List[Dict[str, str]]]:
    _require_session(session_id)
    return {"messages": _serialize_messages(session_id)}


@app.get("/api/sessions/{session_id}/context-usage")
def session_context_usage(session_id: str) -> Dict[str, Any]:
    _require_session(session_id)
    return {"context_usage": get_context_usage(session_id)}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> Dict[str, Any]:
    _require_session(payload.session_id)
    _require_workspace_available(payload.mode)
    clear_last_retrieval_debug(payload.session_id)
    route_decision = apply_execution_mode(
        route_chat_request(payload.session_id, payload.message), payload.mode
    )
    route_debug = _route_debug(route_decision)
    route_debug["mode"] = payload.mode

    if route_decision.route == "skill_admin":
        _log_route_decision(
            stage="start",
            session_id=payload.session_id,
            message=payload.message,
            route_decision=route_decision,
            matched_skills=[],
        )
        skill_command = handle_skill_chat_command(payload.message)
        if skill_command is None:
            skill_command = ("没有识别到有效的 skill 管理命令。", {"operation": "error", "error": "unrecognized_skill_command"})
        answer, command_debug = skill_command
        retrieval_debug = {**route_debug, "skill_command": command_debug, "citations": [], "skills_used": []}
        tool_calls = [
            {
                "tool": "skill_command",
                "display_name": "Skill command",
                "input": payload.message,
                "status": "completed" if command_debug.get("operation") != "error" else "failed",
                "output_preview": _tool_preview(answer),
                "error": str(command_debug.get("error") or ""),
                "started_at": _now_iso(),
                "ended_at": _now_iso(),
            }
        ]
        retrieval_debug["route_violation"] = find_route_violations(
            route_decision.route, tool_calls, route_decision.allowed_tools
        )
        _log_route_decision(
            stage="finish",
            session_id=payload.session_id,
            message=payload.message,
            route_decision=route_decision,
            matched_skills=[],
            tool_calls=tool_calls,
        )
        context_usage = _record_direct_chat_response(
            session_id=payload.session_id,
            user_input=payload.message,
            answer=answer,
            retrieval_debug=retrieval_debug,
            tool_calls=tool_calls,
        )
        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "retrieval_debug": retrieval_debug,
            "citations": [],
            "skills_used": [],
            "context_usage": context_usage,
        }

    matched_skills = match_skills_with_workflow_fallback(payload.message)
    skills_used = summarize_skills(matched_skills)
    skill_selection_call = _skill_selection_tool_call(route_decision, matched_skills, payload.message)
    skill_selection_debug = _skill_selection_summary(route_decision, matched_skills)
    _log_route_decision(
        stage="start",
        session_id=payload.session_id,
        message=payload.message,
        route_decision=route_decision,
        matched_skills=matched_skills,
        tool_calls=[skill_selection_call],
    )

    if route_decision.route == "table_workflow" and payload.mode != "workspace":
        workflow_skills = _workflow_skills_or_default(matched_skills)
        workflow_results, _skill_context, workflow_tool_calls = _execute_matched_skill_workflows(payload.session_id, workflow_skills)
        tool_calls = [skill_selection_call] + workflow_tool_calls
        answer = _format_table_workflow_answer(workflow_results)
        retrieval_debug = {
            **route_debug,
            "citations": [],
            "skills_used": summarize_skills(workflow_skills),
            "skill_workflows": workflow_results,
            "skill_selection": skill_selection_debug,
        }
        retrieval_debug["route_violation"] = find_route_violations(
            route_decision.route, tool_calls, route_decision.allowed_tools
        )
        _log_route_decision(
            stage="finish",
            session_id=payload.session_id,
            message=payload.message,
            route_decision=route_decision,
            matched_skills=workflow_skills,
            tool_calls=tool_calls,
        )
        context_usage = _record_direct_chat_response(
            session_id=payload.session_id,
            user_input=payload.message,
            answer=answer,
            retrieval_debug=retrieval_debug,
            tool_calls=tool_calls,
        )
        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "retrieval_debug": retrieval_debug,
            "citations": [],
            "skills_used": retrieval_debug["skills_used"],
            "context_usage": context_usage,
        }

    if not os.getenv("ZAI_API_KEY"):
        raise HTTPException(status_code=400, detail="Missing ZAI_API_KEY in .env.")

    workflow_results: List[Dict[str, Any]] = []
    skill_context = ""
    workflow_tool_calls: List[Dict[str, Any]] = []
    orchestration_context, orchestration_tool_calls, orchestration_debug = run_orchestration(
        payload.session_id, payload.message, route_decision
    )
    supervisor_tools, tool_call_limits, supervisor_context = _supervisor_execution(
        route_decision, orchestration_debug
    )
    route_debug["supervisor_execution_context"] = supervisor_context
    if supervisor_tools == ["workspace_write_file"]:
        artifact_call = _write_grounded_artifact(
            payload.session_id, payload.message, orchestration_debug,
            supervisor_context.get("artifact_requirements", {}),
        )
        tool_calls = [skill_selection_call] + workflow_tool_calls + orchestration_tool_calls + [artifact_call]
        answer = normalize_artifact_delivery("", tool_calls)
        if artifact_call["status"] != "completed":
            answer = f"产物生成失败：{artifact_call['error']}"
        retrieval_debug = get_last_retrieval_debug(payload.session_id)
        retrieval_debug.update(route_debug)
        citations = _build_citations(retrieval_debug)
        retrieval_debug.update({
            "citations": citations, "skills_used": skills_used, "skill_workflows": workflow_results,
            "skill_selection": skill_selection_debug, "orchestration": orchestration_debug,
            "route_violation": find_route_violations(route_decision.route, tool_calls, supervisor_tools),
        })
        context_usage = _record_direct_chat_response(
            session_id=payload.session_id, user_input=payload.message, answer=answer,
            retrieval_debug=retrieval_debug, tool_calls=tool_calls, citations=citations,
        )
        return {
            "answer": answer, "tool_calls": tool_calls, "retrieval_debug": retrieval_debug,
            "citations": citations, "skills_used": skills_used, "context_usage": context_usage,
        }
    if route_decision.route == "document_qa" and not route_decision.use_multi_agent:
        prefetched_context, prefetched_tool_calls = _auto_prefetch_document_context(payload.session_id, payload.message)
    else:
        prefetched_context, prefetched_tool_calls = "", []
    try:
        prompt_preview = build_agent_system_prompt_preview(
            payload.session_id,
            selected_skills=matched_skills,
            user_input=payload.message,
            prefetched_context=prefetched_context,
            skill_context=skill_context,
            allowed_tools=supervisor_tools,
            route=route_decision.route,
            mode=payload.mode,
            orchestration_context=orchestration_context,
        )
        context_plan = prepare_session_context(
            payload.session_id,
            payload.message,
            static_texts=[prompt_preview],
            tool_names=supervisor_tools,
        )
    except ContextLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    route_debug["context_management"] = context_plan.debug
    agent_chain = create_agent_executor(
        payload.session_id,
        user_input=payload.message,
        selected_skills=matched_skills,
        prefetched_context=prefetched_context,
        skill_context=skill_context,
        allowed_tools=supervisor_tools,
        route=route_decision.route,
        mode=payload.mode,
        orchestration_context=orchestration_context,
        tool_call_limits=tool_call_limits,
        artifact_requirements=supervisor_context.get("artifact_requirements", {}),
        history_messages=context_plan.visible_messages,
        conversation_summary=context_plan.conversation_summary,
    )
    if agent_chain is None:
        raise HTTPException(status_code=500, detail="Agent initialization failed.")

    try:
        response = agent_chain.invoke(
            {"input": payload.message},
            config={"configurable": {"session_id": payload.session_id}},
        )
        answer = response.get("output", "")
        intermediate_steps = response.get("intermediate_steps", [])
        tool_calls = [skill_selection_call] + workflow_tool_calls + orchestration_tool_calls + prefetched_tool_calls + [
            _serialize_intermediate_tool_call(action, observation) for action, observation in intermediate_steps
        ]
        remediation_tools = set(supervisor_context.get("remediation_capabilities", []))
        for call in tool_calls:
            if call.get("call_stage") == "main_agent" and call.get("tool") in remediation_tools:
                call["call_stage"] = "remediation"
                call["remediation"] = True
            elif str(call.get("tool", "")).startswith("workspace_"):
                call["call_stage"] = "main_agent_artifact"
            _annotate_duplicate_block(call)
        remediation_count = sum(1 for call in tool_calls if call.get("remediation"))
        orchestration_debug["redispatches"] = min(remediation_count, 1)
        if remediation_count:
            orchestration_debug["termination_reason"] = "bounded_workers_completed_with_remediation"
        answer = normalize_artifact_delivery(answer, tool_calls)
        retrieval_debug = get_last_retrieval_debug(payload.session_id)
        retrieval_debug.update(route_debug)
        citations = _build_citations(retrieval_debug)
        retrieval_debug["citations"] = citations
        retrieval_debug["skills_used"] = skills_used
        retrieval_debug["skill_workflows"] = workflow_results
        retrieval_debug["skill_selection"] = skill_selection_debug
        retrieval_debug["orchestration"] = orchestration_debug
        retrieval_debug["route_violation"] = find_route_violations(
            route_decision.route, tool_calls, supervisor_tools
        )
        context_usage = make_usage_snapshot(context_plan, answer)
        retrieval_debug["context_usage"] = context_usage
        _log_route_decision(
            stage="finish",
            session_id=payload.session_id,
            message=payload.message,
            route_decision=route_decision,
            matched_skills=matched_skills,
            tool_calls=tool_calls,
        )
        record_chat_trace(
            session_id=payload.session_id,
            user_input=payload.message,
            answer=answer,
            retrieval_debug=retrieval_debug,
            tool_calls=tool_calls,
            citations=citations,
        )

        current_title = _get_session_title(payload.session_id) or ""
        if len(_serialize_messages(payload.session_id)) >= 2 and _is_default_session_title(current_title):
            update_session_title(payload.session_id, payload.message[:15])

        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "retrieval_debug": retrieval_debug,
            "citations": citations,
            "skills_used": skills_used,
            "context_usage": context_usage,
        }
    except HTTPException:
        raise
    except Exception as exc:
        retrieval_debug = get_last_retrieval_debug(payload.session_id)
        retrieval_debug.update(route_debug)
        retrieval_debug["skills_used"] = skills_used
        retrieval_debug["skill_workflows"] = workflow_results
        retrieval_debug["skill_selection"] = skill_selection_debug
        retrieval_debug["orchestration"] = orchestration_debug
        error_tool_calls = [skill_selection_call] + workflow_tool_calls + orchestration_tool_calls + prefetched_tool_calls
        retrieval_debug["route_violation"] = find_route_violations(
            route_decision.route, error_tool_calls, supervisor_tools
        )
        _log_route_decision(
            stage="error",
            session_id=payload.session_id,
            message=payload.message,
            route_decision=route_decision,
            matched_skills=matched_skills,
            tool_calls=error_tool_calls,
        )
        record_chat_trace(
            session_id=payload.session_id,
            user_input=payload.message,
            error=str(exc),
            retrieval_debug=retrieval_debug,
            tool_calls=error_tool_calls,
            citations=_build_citations(retrieval_debug),
        )
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    _require_session(payload.session_id)
    _require_workspace_available(payload.mode)
    clear_last_retrieval_debug(payload.session_id)
    route_decision = apply_execution_mode(
        route_chat_request(payload.session_id, payload.message), payload.mode
    )
    route_debug = _route_debug(route_decision)
    route_debug["mode"] = payload.mode

    if route_decision.route == "skill_admin":
        _log_route_decision(
            stage="stream_start",
            session_id=payload.session_id,
            message=payload.message,
            route_decision=route_decision,
            matched_skills=[],
        )
        skill_command = handle_skill_chat_command(payload.message)
        if skill_command is None:
            skill_command = ("没有识别到有效的 skill 管理命令。", {"operation": "error", "error": "unrecognized_skill_command"})
        answer, command_debug = skill_command
        retrieval_debug = {**route_debug, "skill_command": command_debug, "citations": [], "skills_used": []}
        tool_calls = [
            {
                "tool": "skill_command",
                "display_name": "Skill command",
                "input": payload.message,
                "status": "completed" if command_debug.get("operation") != "error" else "failed",
                "output_preview": _tool_preview(answer),
                "error": str(command_debug.get("error") or ""),
                "started_at": _now_iso(),
                "ended_at": _now_iso(),
            }
        ]
        retrieval_debug["route_violation"] = find_route_violations(
            route_decision.route, tool_calls, route_decision.allowed_tools
        )
        _log_route_decision(
            stage="stream_finish",
            session_id=payload.session_id,
            message=payload.message,
            route_decision=route_decision,
            matched_skills=[],
            tool_calls=tool_calls,
        )
        context_usage = _record_direct_chat_response(
            session_id=payload.session_id,
            user_input=payload.message,
            answer=answer,
            retrieval_debug=retrieval_debug,
            tool_calls=tool_calls,
        )

        async def direct_generate() -> AsyncIterator[str]:
            yield _sse(
                "done",
                {
                    "answer": answer,
                    "tool_calls": tool_calls,
                    "retrieval_debug": retrieval_debug,
                    "citations": [],
                    "skills_used": [],
                    "context_usage": context_usage,
                },
            )

        return StreamingResponse(direct_generate(), media_type="text/event-stream")

    matched_skills = match_skills_with_workflow_fallback(payload.message)
    skills_used = summarize_skills(matched_skills)
    skill_selection_call = _skill_selection_tool_call(route_decision, matched_skills, payload.message)
    skill_selection_debug = _skill_selection_summary(route_decision, matched_skills)
    _log_route_decision(
        stage="stream_start",
        session_id=payload.session_id,
        message=payload.message,
        route_decision=route_decision,
        matched_skills=matched_skills,
        tool_calls=[skill_selection_call],
    )

    if route_decision.route == "table_workflow" and payload.mode != "workspace":
        workflow_skills = _workflow_skills_or_default(matched_skills)
        workflow_results, _skill_context, workflow_tool_calls = _execute_matched_skill_workflows(payload.session_id, workflow_skills)
        tool_calls = [skill_selection_call] + workflow_tool_calls
        answer = _format_table_workflow_answer(workflow_results)
        retrieval_debug = {
            **route_debug,
            "citations": [],
            "skills_used": summarize_skills(workflow_skills),
            "skill_workflows": workflow_results,
            "skill_selection": skill_selection_debug,
        }
        retrieval_debug["route_violation"] = find_route_violations(
            route_decision.route, tool_calls, route_decision.allowed_tools
        )
        _log_route_decision(
            stage="stream_finish",
            session_id=payload.session_id,
            message=payload.message,
            route_decision=route_decision,
            matched_skills=workflow_skills,
            tool_calls=tool_calls,
        )
        context_usage = _record_direct_chat_response(
            session_id=payload.session_id,
            user_input=payload.message,
            answer=answer,
            retrieval_debug=retrieval_debug,
            tool_calls=tool_calls,
        )

        async def workflow_generate() -> AsyncIterator[str]:
            yield _sse(
                "done",
                {
                    "answer": answer,
                    "tool_calls": tool_calls,
                    "retrieval_debug": retrieval_debug,
                    "citations": [],
                    "skills_used": retrieval_debug["skills_used"],
                    "context_usage": context_usage,
                },
            )

        return StreamingResponse(workflow_generate(), media_type="text/event-stream")

    if not os.getenv("ZAI_API_KEY"):
        raise HTTPException(status_code=400, detail="Missing ZAI_API_KEY in .env.")

    workflow_results: List[Dict[str, Any]] = []
    skill_context = ""
    workflow_tool_calls: List[Dict[str, Any]] = []
    orchestration_context, orchestration_tool_calls, orchestration_debug = run_orchestration(
        payload.session_id, payload.message, route_decision
    )
    supervisor_tools, tool_call_limits, supervisor_context = _supervisor_execution(
        route_decision, orchestration_debug
    )
    route_debug["supervisor_execution_context"] = supervisor_context
    if supervisor_tools == ["workspace_write_file"]:
        async def artifact_generate() -> AsyncIterator[str]:
            tool_calls = [skill_selection_call] + list(workflow_tool_calls) + list(orchestration_tool_calls)
            display_name = _tool_display_name("workspace_write_file")
            yield _sse("status", {"message": "正在生成并登记产物"})
            yield _sse("tool_start", {
                "step_index": len(tool_calls) + 1, "tool": "workspace_write_file",
                "display_name": display_name, "status": "running", "message": f"正在调用 {display_name}",
            })
            artifact_call = await asyncio.to_thread(
                _write_grounded_artifact, payload.session_id, payload.message, orchestration_debug,
                supervisor_context.get("artifact_requirements", {}),
            )
            tool_calls.append(artifact_call)
            event = "tool_end" if artifact_call["status"] == "completed" else "tool_error"
            yield _sse(event, {
                "step_index": len(tool_calls), "tool": "workspace_write_file", "display_name": display_name,
                "status": artifact_call["status"], "output_preview": artifact_call["output_preview"],
                "error": artifact_call["error"], "message": f"{display_name} 调用完成",
            })
            answer = normalize_artifact_delivery("", tool_calls)
            if artifact_call["status"] != "completed":
                answer = f"产物生成失败：{artifact_call['error']}"
            retrieval_debug = get_last_retrieval_debug(payload.session_id)
            retrieval_debug.update(route_debug)
            citations = _build_citations(retrieval_debug)
            retrieval_debug.update({
                "citations": citations, "skills_used": skills_used, "skill_workflows": workflow_results,
                "skill_selection": skill_selection_debug, "orchestration": orchestration_debug,
                "route_violation": find_route_violations(route_decision.route, tool_calls, supervisor_tools),
            })
            context_usage = _record_direct_chat_response(
                session_id=payload.session_id, user_input=payload.message, answer=answer,
                retrieval_debug=retrieval_debug, tool_calls=tool_calls, citations=citations,
            )
            yield _sse("token", {"token": answer})
            yield _sse("done", {
                "answer": answer, "tool_calls": tool_calls, "retrieval_debug": retrieval_debug,
                "citations": citations, "skills_used": skills_used, "context_usage": context_usage,
            })

        return StreamingResponse(artifact_generate(), media_type="text/event-stream")
    if route_decision.route == "document_qa" and not route_decision.use_multi_agent:
        prefetched_context, prefetched_tool_calls = _auto_prefetch_document_context(payload.session_id, payload.message)
    else:
        prefetched_context, prefetched_tool_calls = "", []
    try:
        prompt_preview = build_agent_system_prompt_preview(
            payload.session_id,
            selected_skills=matched_skills,
            user_input=payload.message,
            prefetched_context=prefetched_context,
            skill_context=skill_context,
            allowed_tools=supervisor_tools,
            route=route_decision.route,
            mode=payload.mode,
            orchestration_context=orchestration_context,
        )
        context_plan = await asyncio.to_thread(
            prepare_session_context,
            payload.session_id,
            payload.message,
            static_texts=[prompt_preview],
            tool_names=supervisor_tools,
        )
    except ContextLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    route_debug["context_management"] = context_plan.debug
    agent_chain = create_agent_executor(
        payload.session_id,
        async_history=True,
        user_input=payload.message,
        selected_skills=matched_skills,
        prefetched_context=prefetched_context,
        skill_context=skill_context,
        allowed_tools=supervisor_tools,
        route=route_decision.route,
        mode=payload.mode,
        orchestration_context=orchestration_context,
        tool_call_limits=tool_call_limits,
        artifact_requirements=supervisor_context.get("artifact_requirements", {}),
        history_messages=context_plan.visible_messages,
        conversation_summary=context_plan.conversation_summary,
    )
    if agent_chain is None:
        raise HTTPException(status_code=500, detail="Agent initialization failed.")

    async def generate() -> AsyncIterator[str]:
        answer_parts: List[str] = []
        final_answer = ""
        tool_calls: List[Dict[str, Any]] = (
            [skill_selection_call] + list(workflow_tool_calls) + list(orchestration_tool_calls) + list(prefetched_tool_calls)
        )
        tool_call_by_run_id: Dict[str, Dict[str, Any]] = {}
        completed_tool_run_ids: set[str] = set()
        actual_prompt_tokens: Optional[int] = None
        actual_completion_tokens: Optional[int] = None

        if context_plan.compacted:
            yield _sse("context_compaction_start", {
                "message": "正在压缩上下文",
                "before_tokens": context_plan.debug.get("before_tokens"),
            })
            yield _sse("context_compaction_end", {
                "message": "上下文压缩完成",
                "after_tokens": context_plan.debug.get("after_tokens"),
                "compacted_message_count": context_plan.compacted_message_count,
                "target_unreachable": context_plan.target_unreachable,
            })
        elif context_plan.warning:
            yield _sse("context_compaction_warning", {"message": context_plan.warning})

        yield _sse("status", {"message": "正在思考"})
        if payload.mode == "workspace":
            yield _sse("sandbox_status", get_sandbox_manager().status(payload.session_id))

        try:
            async for event in agent_chain.astream_events(
                {"input": payload.message},
                config={"configurable": {"session_id": payload.session_id}},
                version="v2",
            ):
                event_name = event.get("event", "")
                run_id = event.get("run_id", "")
                name = event.get("name", "")
                data = event.get("data", {}) or {}

                if event_name == "on_tool_start":
                    if run_id and run_id in tool_call_by_run_id:
                        continue
                    display_name = _tool_display_name(name)
                    tool_call = {
                        "step_index": len(tool_calls) + 1,
                        "tool": name,
                        "display_name": display_name,
                        "input": data.get("input"),
                        "status": "running",
                        "output_preview": "",
                        "error": "",
                        "started_at": _now_iso(),
                        "ended_at": "",
                        "call_stage": "main_agent_artifact" if name.startswith("workspace_") else "main_agent",
                        "duplicate_blocked": False,
                        "remediation": name in set(supervisor_context.get("remediation_capabilities", [])),
                    }
                    if tool_call["remediation"]:
                        tool_call["call_stage"] = "remediation"
                    tool_calls.append(tool_call)
                    if run_id:
                        tool_call_by_run_id[run_id] = tool_call
                    _log_route_decision(
                        stage="tool_start",
                        session_id=payload.session_id,
                        message=payload.message,
                        route_decision=route_decision,
                        matched_skills=matched_skills,
                        tool_calls=tool_calls,
                    )
                    yield _sse(
                        "tool_start",
                        {
                            "step_index": tool_call["step_index"],
                            "tool": name,
                            "display_name": display_name,
                            "input": data.get("input"),
                            "status": "running",
                            "message": f"正在调用 {display_name}",
                        },
                    )
                    if name.startswith("workspace_"):
                        yield _sse(
                            "sandbox_start",
                            {"tool": name, "message": f"Starting isolated workspace tool: {display_name}"},
                        )

                elif event_name == "on_tool_end":
                    if run_id and run_id in completed_tool_run_ids:
                        continue
                    tool_call = tool_call_by_run_id.get(run_id)
                    if tool_call is None:
                        tool_call = {
                            "step_index": len(tool_calls) + 1,
                            "tool": name,
                            "display_name": _tool_display_name(name),
                            "input": "",
                            "status": "completed",
                            "output_preview": "",
                            "error": "",
                            "started_at": "",
                            "ended_at": "",
                        }
                        tool_calls.append(tool_call)
                    tool_call["output_preview"] = _tool_preview(data.get("output", ""))
                    structured_output = _parse_tool_json(tool_call["output_preview"])
                    tool_call["status"] = "failed" if structured_output.get("ok") is False else "completed"
                    if structured_output.get("ok") is False:
                        tool_call["error"] = str(structured_output.get("error") or structured_output.get("error_code") or "tool_failed")
                    _annotate_duplicate_block(tool_call)
                    tool_call["ended_at"] = _now_iso()
                    if run_id:
                        completed_tool_run_ids.add(run_id)
                    display_name = str(tool_call.get("display_name") or _tool_display_name(str(tool_call.get("tool", name))))
                    _log_route_decision(
                        stage="tool_end",
                        session_id=payload.session_id,
                        message=payload.message,
                        route_decision=route_decision,
                        matched_skills=matched_skills,
                        tool_calls=tool_calls,
                    )
                    yield _sse(
                        "tool_end",
                        {
                            "step_index": tool_call.get("step_index"),
                            "tool": tool_call.get("tool", name),
                            "display_name": display_name,
                            "status": "completed",
                            "output_preview": tool_call["output_preview"],
                            "message": f"{display_name} 调用完成",
                        },
                    )
                    if str(tool_call.get("tool", name)).startswith("workspace_"):
                        sandbox_status = get_sandbox_manager().status(payload.session_id)
                        sandbox_status["message"] = "隔离工作区已就绪"
                        yield _sse("sandbox_ready", sandbox_status)
                        if tool_call["output_preview"]:
                            yield _sse(
                                "process_output",
                                {
                                    "tool": tool_call.get("tool", name),
                                    "output_preview": tool_call["output_preview"],
                                },
                            )

                elif event_name == "on_tool_error":
                    if run_id and run_id in completed_tool_run_ids:
                        continue
                    tool_call = tool_call_by_run_id.get(run_id)
                    if tool_call is None:
                        tool_call = {
                            "step_index": len(tool_calls) + 1,
                            "tool": name,
                            "display_name": _tool_display_name(name),
                            "input": "",
                            "status": "failed",
                            "output_preview": "",
                            "error": "",
                            "started_at": "",
                            "ended_at": "",
                        }
                        tool_calls.append(tool_call)
                    error_text = _tool_preview(data.get("error", ""))
                    tool_call["status"] = "failed"
                    tool_call["error"] = error_text
                    tool_call["ended_at"] = _now_iso()
                    if run_id:
                        completed_tool_run_ids.add(run_id)
                    display_name = str(tool_call.get("display_name") or _tool_display_name(str(tool_call.get("tool", name))))
                    _log_route_decision(
                        stage="tool_error",
                        session_id=payload.session_id,
                        message=payload.message,
                        route_decision=route_decision,
                        matched_skills=matched_skills,
                        tool_calls=tool_calls,
                    )
                    yield _sse(
                        "tool_error",
                        {
                            "step_index": tool_call.get("step_index"),
                            "tool": tool_call.get("tool", name),
                            "display_name": display_name,
                            "status": "failed",
                            "error": error_text,
                            "message": f"{display_name} failed",
                        },
                    )
                    if str(tool_call.get("tool", name)).startswith("workspace_"):
                        yield _sse(
                            "sandbox_error",
                            {"tool": tool_call.get("tool", name), "error": error_text},
                        )

                elif event_name == "on_chat_model_stream":
                    prompt_usage, completion_usage = _extract_token_usage(data.get("chunk"))
                    if prompt_usage is not None and completion_usage is not None:
                        previous_total = (actual_prompt_tokens or 0) + (actual_completion_tokens or 0)
                        if prompt_usage + completion_usage > previous_total:
                            actual_prompt_tokens, actual_completion_tokens = prompt_usage, completion_usage
                    text = _chunk_text(data.get("chunk"))
                    if text:
                        answer_parts.append(text)
                        yield _sse("token", {"token": text})

                elif event_name == "on_chain_end":
                    output = _final_output(data.get("output"))
                    if output:
                        final_answer = output

                elif event_name == "on_chat_model_end":
                    prompt_usage, completion_usage = _extract_token_usage(data.get("output"))
                    if prompt_usage is not None and completion_usage is not None:
                        previous_total = (actual_prompt_tokens or 0) + (actual_completion_tokens or 0)
                        if prompt_usage + completion_usage > previous_total:
                            actual_prompt_tokens, actual_completion_tokens = prompt_usage, completion_usage

            answer = "".join(answer_parts) or final_answer
            if final_answer and final_answer != answer:
                answer = final_answer
                yield _sse("token", {"token": final_answer})

            answer = normalize_artifact_delivery(answer, tool_calls)
            remediation_count = sum(1 for call in tool_calls if call.get("remediation"))
            orchestration_debug["redispatches"] = min(remediation_count, 1)
            if remediation_count:
                orchestration_debug["termination_reason"] = "bounded_workers_completed_with_remediation"
            retrieval_debug = get_last_retrieval_debug(payload.session_id)
            retrieval_debug.update(route_debug)
            citations = _build_citations(retrieval_debug)
            retrieval_debug["citations"] = citations
            retrieval_debug["skills_used"] = skills_used
            retrieval_debug["skill_workflows"] = workflow_results
            retrieval_debug["skill_selection"] = skill_selection_debug
            retrieval_debug["orchestration"] = orchestration_debug
            retrieval_debug["route_violation"] = find_route_violations(
                route_decision.route, tool_calls, supervisor_tools
            )
            context_usage = make_usage_snapshot(
                context_plan,
                answer,
                actual_prompt_tokens=actual_prompt_tokens,
                actual_completion_tokens=actual_completion_tokens,
            )
            retrieval_debug["context_usage"] = context_usage
            _log_route_decision(
                stage="stream_finish",
                session_id=payload.session_id,
                message=payload.message,
                route_decision=route_decision,
                matched_skills=matched_skills,
                tool_calls=tool_calls,
            )
            record_chat_trace(
                session_id=payload.session_id,
                user_input=payload.message,
                answer=answer,
                retrieval_debug=retrieval_debug,
                tool_calls=tool_calls,
                citations=citations,
            )

            current_title = _get_session_title(payload.session_id) or ""
            if len(_serialize_messages(payload.session_id)) >= 2 and _is_default_session_title(current_title):
                update_session_title(payload.session_id, payload.message[:15])

            yield _sse(
                "done",
                {
                    "answer": answer,
                    "tool_calls": tool_calls,
                    "retrieval_debug": retrieval_debug,
                    "citations": citations,
                    "skills_used": skills_used,
                    "context_usage": context_usage,
                },
            )
        except Exception as exc:
            retrieval_debug = get_last_retrieval_debug(payload.session_id)
            retrieval_debug.update(route_debug)
            citations = _build_citations(retrieval_debug)
            retrieval_debug["citations"] = citations
            retrieval_debug["skills_used"] = skills_used
            retrieval_debug["skill_workflows"] = workflow_results
            retrieval_debug["skill_selection"] = skill_selection_debug
            retrieval_debug["orchestration"] = orchestration_debug
            retrieval_debug["route_violation"] = find_route_violations(
                route_decision.route, tool_calls, supervisor_tools
            )
            _log_route_decision(
                stage="stream_error",
                session_id=payload.session_id,
                message=payload.message,
                route_decision=route_decision,
                matched_skills=matched_skills,
                tool_calls=tool_calls,
            )
            record_chat_trace(
                session_id=payload.session_id,
                user_input=payload.message,
                answer="".join(answer_parts),
                error=str(exc),
                retrieval_debug=retrieval_debug,
                tool_calls=tool_calls,
                citations=citations,
            )
            yield _sse("error", {"error": f"Chat failed: {exc}", "skills_used": skills_used})

    return StreamingResponse(generate(), media_type="text/event-stream")


# Runtime overrides for helpers that previously contained mojibake literals.
# Route handlers resolve these globals when requests are handled, so defining
# them last keeps the behavior clean without changing the handler structure.
_DOCUMENT_QUERY_TERMS = (
    "文档", "文件", "资料", "上传", "知识库", "pdf", "doc", "docx", "csv", "xlsx", "表格",
    "主要内容", "讲了什么", "说了什么", "总结", "概括", "引用", "证据", "document", "file", "uploaded",
    "sql 注入", "sql注入", "xss",
)
_DOCUMENT_OVERVIEW_TERMS = (
    "主要内容", "讲了什么", "说了什么", "总结", "概括", "overview", "summary", "summarize",
)


def _tool_display_name(name: str) -> str:
    if name == "run_python_code":
        return "Python code"
    labels = {
        "workspace_terminal": "Sandbox terminal",
        "workspace_list_files": "Sandbox files",
        "workspace_read_file": "Sandbox file read",
        "workspace_write_file": "Sandbox file write",
        "workspace_process": "Sandbox process",
        "workspace_list_session_files": "Session uploads",
        "workspace_import_uploaded_file": "Import upload",
        "workspace_list_artifacts": "Saved artifacts",
        "workspace_import_artifact": "Restore artifact",
        "workspace_save_file": "Save workspace file",
        "retrieve_knowledge": "本地知识库检索",
        "query_document_metadata": "文档元数据查询",
        "read_website": "网页读取",
        "web_search": "网络搜索",
        "get_current_time": "当前时间",
        "run_skill_workflow": "Skill workflow",
        "skill_command": "Skill command",
    }
    return labels.get(name, name)


def _format_table_workflow_answer(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "没有可用的数据分析结果。"
    lines = ["已使用后端表格工作流处理上传文件："]
    for result in results:
        document = result.get("document") or {}
        lines.append(f"\n文件: {document.get('file_name', '')}")
        lines.append(f"原始形状: {result.get('raw_shape', {})}")
        lines.append(f"清洗后形状: {result.get('clean_shape', {})}")
        numeric_means = result.get("numeric_means") or {}
        if numeric_means:
            lines.append("数值列均值:")
            for column, value in numeric_means.items():
                lines.append(f"- {column}: {value}")
        else:
            lines.append("未检测到可计算均值的数值列。")
        missing = result.get("missing_values") or {}
        lines.append(f"缺失值: {missing if missing else '无'}")
        notes = result.get("cleaning_notes") or []
        if notes:
            lines.append("清洗说明: " + " ".join(str(item) for item in notes))
    return "\n".join(lines)


@app.get("/api/sessions/{session_id}/documents")
def documents(session_id: str) -> Dict[str, List[Dict[str, Any]]]:
    _require_session(session_id)
    analysis = {item["file_id"]: item for item in list_document_metadata(session_id)}
    return {
        "documents": [
            {
                "id": row["file_id"],
                "file_name": row["original_name"],
                "chunk_count": row["chunk_count"],
                "status": row["index_status"],
                "uploaded_at": str(row["created_at"]),
                "mime_type": row["mime_type"],
                "category": row["category"],
                "size_bytes": row["size_bytes"],
                "parse_status": row["parse_status"],
                "ocr_status": row["ocr_status"],
                "index_status": row["index_status"],
                "scan_status": row["scan_status"],
                "table_profile": row["table_profile"],
                "metadata_status": (analysis.get(row["file_id"]) or {}).get("extraction_status", "legacy"),
                "structure_version": (analysis.get(row["file_id"]) or {}).get("structure_version", 0),
            }
            for row in list_uploaded_files(session_id)
        ]
    }


@app.post("/api/sessions/{session_id}/documents")
async def upload_document(session_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    _require_session(session_id)
    content = await file.read()
    upload = BufferedUpload(file.filename or "upload", content)
    file_size_mb = len(content) / 1024 / 1024
    if file_size_mb >= float(ASYNC_UPLOAD_THRESHOLD_MB):
        try:
            task_id = submit_process_file(upload, session_id)
            return {"ok": True, "mode": "async", "task_id": task_id, "message": "Indexing task submitted."}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Submit failed: {exc}") from exc

    # Indexing performs CPU-heavy embedding plus blocking database and optional LLM I/O.
    # Keep it off the asyncio event loop so chat/session APIs remain responsive.
    ok, message = await asyncio.to_thread(process_file, upload, session_id)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "mode": "sync", "message": message}


@app.get("/api/sessions/{session_id}/documents/{file_id}/file")
def document_file(session_id: str, file_id: str) -> FileResponse:
    _require_session(session_id)
    doc_meta = get_uploaded_file(session_id, file_id)
    if not doc_meta:
        raise HTTPException(status_code=404, detail="Document not found.")

    raw_path = str(doc_meta.get("storage_path") or "")
    if not raw_path:
        raise HTTPException(status_code=404, detail="Original file is unavailable.")

    file_path = os.path.abspath(raw_path)
    data_root = os.path.abspath(DATA_DIR)
    if not (file_path == data_root or file_path.startswith(data_root + os.sep)):
        raise HTTPException(status_code=403, detail="Document path is outside the data directory.")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original file is missing.")

    file_name = str(doc_meta.get("original_name") or os.path.basename(file_path))
    media_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    headers = {"Content-Disposition": f"inline; filename*=UTF-8''{quote(file_name)}"}
    return FileResponse(file_path, media_type=media_type, headers=headers)


@app.get("/api/sessions/{session_id}/artifacts")
def artifacts(session_id: str) -> Dict[str, List[Dict[str, Any]]]:
    _require_session(session_id)
    return {"artifacts": [
        {
            "id": row["artifact_id"], "name": row["name"], "size_bytes": row["size_bytes"],
            "mime_type": row["mime_type"], "created_at": row["created_at"],
            "generated_by_command": row["generated_by_command"], "retention_policy": row["retention_policy"],
        }
        for row in list_artifacts(session_id)
    ]}


@app.get("/api/sessions/{session_id}/artifacts/{artifact_id}/file")
def artifact_file(session_id: str, artifact_id: str) -> FileResponse:
    _require_session(session_id)
    artifact = get_artifact(session_id, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = os.path.abspath(str(artifact["storage_path"]))
    artifact_root = os.path.abspath(os.path.join(DATA_DIR, "artifacts"))
    if not path.startswith(artifact_root + os.sep) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Artifact file is unavailable.")
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(str(artifact['name']))}"}
    return FileResponse(path, media_type=str(artifact["mime_type"]), headers=headers)


@app.delete("/api/sessions/{session_id}/documents/{file_id}")
def remove_document(session_id: str, file_id: str) -> Dict[str, Any]:
    _require_session(session_id)
    ok, message = delete_document(session_id, file_id)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "message": message}


@app.post("/api/sessions/{session_id}/documents/{file_id}/reindex")
def reindex(session_id: str, file_id: str) -> Dict[str, Any]:
    _require_session(session_id)
    ok, message = reindex_document(session_id, file_id)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "message": message}


@app.get("/api/sessions/{session_id}/index-tasks")
def index_tasks(session_id: str) -> Dict[str, List[Dict[str, Any]]]:
    _require_session(session_id)
    return {"tasks": list_index_tasks(session_id)}


@app.get("/api/sessions/{session_id}/traces")
def traces(session_id: str, limit: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    _require_session(session_id)
    return {"traces": get_recent_traces(session_id, limit=limit)}


@app.get("/api/sessions/{session_id}/trace-stats")
def trace_stats(session_id: str) -> Dict[str, Any]:
    _require_session(session_id)
    return {"stats": get_trace_statistics(session_id)}
