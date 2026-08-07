import json
import mimetypes
import os
import threading
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
from core.agent import create_agent_executor, retrieve_session_knowledge
from core.document_processor import delete_document, process_file, reindex_document
from core.history import get_session_history
from core.indexing_jobs import list_index_tasks, submit_process_file
from core.request_router import RouteDecision, apply_execution_mode, find_route_violations, route_chat_request
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
from db.file_manager import get_artifact, get_uploaded_file, init_file_db, list_artifacts, list_uploaded_files
from core.persistent_storage import get_persistent_file_store
from db.sandbox_manager import get_session_sandbox, init_sandbox_db
from sandbox_runtime.base import SandboxError
from sandbox_runtime.manager import get_sandbox_manager

load_dotenv()


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
) -> None:
    history = get_session_history(session_id)
    history.add_user_message(user_input)
    history.add_ai_message(answer)
    record_chat_trace(
        session_id=session_id,
        user_input=user_input,
        answer=answer,
        retrieval_debug=retrieval_debug or {},
        tool_calls=tool_calls or [],
        citations=citations or [],
    )


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
        _record_direct_chat_response(
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
        _record_direct_chat_response(
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
        }

    if not os.getenv("ZAI_API_KEY"):
        raise HTTPException(status_code=400, detail="Missing ZAI_API_KEY in .env.")

    workflow_results: List[Dict[str, Any]] = []
    skill_context = ""
    workflow_tool_calls: List[Dict[str, Any]] = []
    if route_decision.route == "document_qa":
        prefetched_context, prefetched_tool_calls = _auto_prefetch_document_context(payload.session_id, payload.message)
    else:
        prefetched_context, prefetched_tool_calls = "", []
    agent_chain = create_agent_executor(
        payload.session_id,
        user_input=payload.message,
        selected_skills=matched_skills,
        prefetched_context=prefetched_context,
        skill_context=skill_context,
        allowed_tools=route_decision.allowed_tools,
        route=route_decision.route,
        mode=payload.mode,
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
        tool_calls = [skill_selection_call] + workflow_tool_calls + prefetched_tool_calls + [
            _serialize_intermediate_tool_call(action, observation) for action, observation in intermediate_steps
        ]
        retrieval_debug = get_last_retrieval_debug(payload.session_id)
        retrieval_debug.update(route_debug)
        citations = _build_citations(retrieval_debug)
        retrieval_debug["citations"] = citations
        retrieval_debug["skills_used"] = skills_used
        retrieval_debug["skill_workflows"] = workflow_results
        retrieval_debug["skill_selection"] = skill_selection_debug
        retrieval_debug["route_violation"] = find_route_violations(
            route_decision.route, tool_calls, route_decision.allowed_tools
        )
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
        }
    except HTTPException:
        raise
    except Exception as exc:
        retrieval_debug = get_last_retrieval_debug(payload.session_id)
        retrieval_debug.update(route_debug)
        retrieval_debug["skills_used"] = skills_used
        retrieval_debug["skill_workflows"] = workflow_results
        retrieval_debug["skill_selection"] = skill_selection_debug
        error_tool_calls = [skill_selection_call] + workflow_tool_calls + prefetched_tool_calls
        retrieval_debug["route_violation"] = find_route_violations(
            route_decision.route, error_tool_calls, route_decision.allowed_tools
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
        _record_direct_chat_response(
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
        _record_direct_chat_response(
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
                },
            )

        return StreamingResponse(workflow_generate(), media_type="text/event-stream")

    if not os.getenv("ZAI_API_KEY"):
        raise HTTPException(status_code=400, detail="Missing ZAI_API_KEY in .env.")

    workflow_results: List[Dict[str, Any]] = []
    skill_context = ""
    workflow_tool_calls: List[Dict[str, Any]] = []
    if route_decision.route == "document_qa":
        prefetched_context, prefetched_tool_calls = _auto_prefetch_document_context(payload.session_id, payload.message)
    else:
        prefetched_context, prefetched_tool_calls = "", []
    agent_chain = create_agent_executor(
        payload.session_id,
        async_history=True,
        user_input=payload.message,
        selected_skills=matched_skills,
        prefetched_context=prefetched_context,
        skill_context=skill_context,
        allowed_tools=route_decision.allowed_tools,
        route=route_decision.route,
        mode=payload.mode,
    )
    if agent_chain is None:
        raise HTTPException(status_code=500, detail="Agent initialization failed.")

    async def generate() -> AsyncIterator[str]:
        answer_parts: List[str] = []
        final_answer = ""
        tool_calls: List[Dict[str, Any]] = [skill_selection_call] + list(workflow_tool_calls) + list(prefetched_tool_calls)
        tool_call_by_run_id: Dict[str, Dict[str, Any]] = {}

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
                    }
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
                    tool_call["status"] = "completed"
                    tool_call["output_preview"] = _tool_preview(data.get("output", ""))
                    tool_call["ended_at"] = _now_iso()
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
                    text = _chunk_text(data.get("chunk"))
                    if text:
                        answer_parts.append(text)
                        yield _sse("token", {"token": text})

                elif event_name == "on_chain_end":
                    output = _final_output(data.get("output"))
                    if output:
                        final_answer = output

            answer = "".join(answer_parts) or final_answer
            if final_answer and final_answer != answer:
                answer = final_answer
                yield _sse("token", {"token": final_answer})

            retrieval_debug = get_last_retrieval_debug(payload.session_id)
            retrieval_debug.update(route_debug)
            citations = _build_citations(retrieval_debug)
            retrieval_debug["citations"] = citations
            retrieval_debug["skills_used"] = skills_used
            retrieval_debug["skill_workflows"] = workflow_results
            retrieval_debug["skill_selection"] = skill_selection_debug
            retrieval_debug["route_violation"] = find_route_violations(
                route_decision.route, tool_calls, route_decision.allowed_tools
            )
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
            retrieval_debug["route_violation"] = find_route_violations(
                route_decision.route, tool_calls, route_decision.allowed_tools
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

    ok, message = process_file(upload, session_id)
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
