import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from db.session_manager import get_session_documents
from db.skill_manager import match_skills_with_workflow_fallback


ALL_AGENT_TOOLS = [
    "retrieve_knowledge",
    "read_website",
    "web_search",
    "get_current_time",
    "run_python_code",
    "workspace_terminal",
    "workspace_list_files",
    "workspace_read_file",
    "workspace_write_file",
    "workspace_process",
    "workspace_list_session_files",
    "workspace_import_uploaded_file",
    "workspace_list_artifacts",
    "workspace_import_artifact",
    "workspace_save_file",
]

WORKSPACE_TOOLS = [
    "workspace_terminal",
    "workspace_list_files",
    "workspace_read_file",
    "workspace_write_file",
    "workspace_process",
    "workspace_list_session_files",
    "workspace_import_uploaded_file",
    "workspace_list_artifacts",
    "workspace_import_artifact",
    "workspace_save_file",
]

ROUTE_ALLOWED_TOOLS: Dict[str, List[str]] = {
    "skill_admin": [],
    "table_workflow": [],
    "document_qa": ["retrieve_knowledge"],
    "web_search": ["web_search", "read_website"],
    "general_chat": ["get_current_time"],
}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    allowed_tools: List[str]
    blocked_tools: List[str]
    reason: str
    matched_skill_ids: List[str]

    def to_debug(self) -> Dict[str, object]:
        return {
            "route": self.route,
            "allowed_tools": self.allowed_tools,
            "blocked_tools": self.blocked_tools,
            "router_reason": self.reason,
            "matched_skill_ids": self.matched_skill_ids,
        }


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def _has_url(text: str) -> bool:
    return bool(re.search(r"https?://|www\.", text, flags=re.IGNORECASE))


def _completed_table_count(session_id: str) -> int:
    count = 0
    for _file_id, file_name, _chunk_count, status, _uploaded_at, _file_path in get_session_documents(session_id):
        if str(status).casefold() != "completed":
            continue
        if Path(str(file_name)).suffix.lower() in {".csv", ".xlsx", ".parquet"}:
            count += 1
    return count


def _completed_document_count(session_id: str) -> int:
    return sum(1 for row in get_session_documents(session_id) if str(row[3]).casefold() == "completed")


def _decision(route: str, reason: str, matched_skill_ids: List[str] | None = None) -> RouteDecision:
    allowed = list(ROUTE_ALLOWED_TOOLS.get(route, []))
    blocked = [tool for tool in ALL_AGENT_TOOLS if tool not in allowed]
    return RouteDecision(
        route=route,
        allowed_tools=allowed,
        blocked_tools=blocked,
        reason=reason,
        matched_skill_ids=matched_skill_ids or [],
    )


def route_chat_request(session_id: str, message: str) -> RouteDecision:
    text = message.strip()
    lowered = text.casefold()

    skill_execution_terms = ["data_analysis", "run_skill_workflow", "使用", "用", "处理", "分析", "workflow"]
    skill_admin_terms = [
        "查看skill",
        "查看 skill",
        "列出skill",
        "列出 skill",
        "有哪些skill",
        "有哪些 skill",
        "创建skill",
        "创建 skill",
        "新增skill",
        "新增 skill",
        "修改skill",
        "修改 skill",
        "更新skill",
        "更新 skill",
        "启用skill",
        "启用 skill",
        "禁用skill",
        "禁用 skill",
        "删除skill",
        "删除 skill",
    ]
    if _contains_any(lowered, skill_admin_terms):
        return _decision("skill_admin", "Matched explicit skill administration command.")
    if _contains_any(
        lowered,
        ["list skill", "show skill", "create skill", "add skill", "update skill", "enable skill", "disable skill", "delete skill"],
    ):
        return _decision("skill_admin", "Matched explicit English skill administration command.")
    if "skill" in lowered and len(lowered) <= 40 and not _contains_any(lowered, skill_execution_terms):
        return _decision("skill_admin", "Matched short skill administration/list request.")

    matched_skills = match_skills_with_workflow_fallback(message)
    workflow_skills = [skill for skill in matched_skills if str(skill.get("mode", "prompt")).casefold() == "workflow"]
    if workflow_skills:
        return _decision(
            "table_workflow",
            "Matched enabled workflow skill for a table workflow request.",
            [str(skill.get("id", "")) for skill in workflow_skills],
        )

    if _has_url(text) or _contains_any(
        lowered,
        ["联网", "网络搜索", "网上查", "网上找", "搜索", "最新", "today", "latest", "search web", "web search"],
    ):
        return _decision("web_search", "Matched explicit external/web information request.")

    table_count = _completed_table_count(session_id)
    if table_count and _contains_any(
        lowered,
        ["csv", "xlsx", "excel", "表格", "数据处理", "数据分析", "表格分析", "清洗", "均值", "平均", "缺失值", "分组", "统计"],
    ):
        return _decision("table_workflow", "Matched table-analysis intent with uploaded table file.")

    if _completed_document_count(session_id) and _contains_any(
        lowered,
        [
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
        ],
    ):
        return _decision("document_qa", "Matched uploaded-document business question.")

    if _completed_document_count(session_id) and _contains_any(
        lowered,
        ["上传", "文档", "文件", "资料", "pdf", "docx", "总结", "概括", "讲了什么", "主要内容", "引用", "document", "file", "uploaded", "summarize"],
    ):
        return _decision("document_qa", "Matched uploaded-document question.")

    return _decision("general_chat", "No specialized route matched.")


def apply_execution_mode(decision: RouteDecision, mode: str = "chat") -> RouteDecision:
    """Add workspace capabilities without changing the route selected for existing features."""
    if mode != "workspace":
        return decision
    allowed = list(dict.fromkeys([*decision.allowed_tools, *WORKSPACE_TOOLS]))
    blocked = [tool for tool in ALL_AGENT_TOOLS if tool not in allowed]
    return RouteDecision(
        route=decision.route,
        allowed_tools=allowed,
        blocked_tools=blocked,
        reason=f"{decision.reason} Workspace capabilities were explicitly enabled for this request.",
        matched_skill_ids=list(decision.matched_skill_ids),
    )


def find_route_violations(
    route: str,
    tool_calls: List[Dict[str, object]],
    allowed_tools: Sequence[str] | None = None,
) -> List[str]:
    allowed = set(allowed_tools if allowed_tools is not None else ROUTE_ALLOWED_TOOLS.get(route, []))
    direct_route_tools = {"skill_admin": {"skill_command"}, "table_workflow": {"run_skill_workflow"}}
    allowed.update(direct_route_tools.get(route, set()))
    allowed.add("skill_selection")
    violations = []
    for tool_call in tool_calls:
        tool_name = str(tool_call.get("tool") or "")
        if tool_name and tool_name not in allowed:
            violations.append(tool_name)
    return violations
