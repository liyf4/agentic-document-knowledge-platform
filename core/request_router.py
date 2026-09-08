"""Deterministic UTF-8 request routing and execution-mode tool policy."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from config import (
    LLM_MODEL_NAME, LLM_TYPE, METADATA_LLM_MAX_RETRIES, METADATA_LLM_TIMEOUT_SECONDS,
    ORCHESTRATION_ENABLED, ORCHESTRATION_MIN_INDEPENDENT_SUBTASKS, ZAI_API_BASE,
)
from core.factory import ComponentFactory
from db.session_manager import get_session_documents
from db.skill_manager import match_skills_with_workflow_fallback

ALL_AGENT_TOOLS = [
    "retrieve_knowledge", "query_document_metadata", "read_website", "web_search", "get_current_time",
    "run_python_code", "workspace_terminal", "workspace_list_files", "workspace_read_file",
    "workspace_write_file", "workspace_process", "workspace_list_session_files",
    "workspace_import_uploaded_file", "workspace_list_artifacts", "workspace_import_artifact",
    "workspace_save_file",
]

WORKSPACE_TOOLS = [
    "workspace_terminal", "workspace_list_files", "workspace_read_file", "workspace_write_file",
    "workspace_process", "workspace_list_session_files", "workspace_import_uploaded_file",
    "workspace_list_artifacts", "workspace_import_artifact", "workspace_save_file",
]

ROUTE_ALLOWED_TOOLS: Dict[str, List[str]] = {
    "skill_admin": [], "table_workflow": [], "document_qa": ["retrieve_knowledge"],
    "metadata_query": ["query_document_metadata", "retrieve_knowledge"],
    "web_search": ["web_search", "read_website"],
    "orchestrated": ["query_document_metadata", "retrieve_knowledge", "web_search", "read_website"],
    "general_chat": ["get_current_time"],
}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    allowed_tools: List[str]
    blocked_tools: List[str]
    reason: str
    matched_skill_ids: List[str]
    needs_documents: bool = False
    needs_web: bool = False
    needs_workspace: bool = False
    needs_structured_facts: bool = False
    use_multi_agent: bool = False
    confidence: float = 1.0

    def to_debug(self) -> Dict[str, object]:
        return {
            "route": self.route, "allowed_tools": self.allowed_tools, "blocked_tools": self.blocked_tools,
            "router_reason": self.reason, "matched_skill_ids": self.matched_skill_ids,
            "needs_documents": self.needs_documents, "needs_web": self.needs_web,
            "needs_workspace": self.needs_workspace, "needs_structured_facts": self.needs_structured_facts,
            "use_multi_agent": self.use_multi_agent, "route_confidence": self.confidence,
        }


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def _has_url(text: str) -> bool:
    return bool(re.search(r"https?://|www\.", text, flags=re.IGNORECASE))


def _completed_table_count(session_id: str) -> int:
    return sum(
        1 for row in get_session_documents(session_id)
        if str(row[3]).casefold() == "completed" and Path(str(row[1])).suffix.lower() in {".csv", ".xlsx", ".parquet"}
    )


def _completed_document_count(session_id: str) -> int:
    return sum(1 for row in get_session_documents(session_id) if str(row[3]).casefold() == "completed")


def _decision(
    route: str, reason: str, matched_skill_ids: List[str] | None = None, *,
    needs_documents: bool = False, needs_web: bool = False, needs_workspace: bool = False,
    needs_structured_facts: bool = False, use_multi_agent: bool = False, confidence: float = 1.0,
) -> RouteDecision:
    allowed = list(ROUTE_ALLOWED_TOOLS.get(route, []))
    if route == "orchestrated":
        allowed = []
        if needs_documents:
            if needs_structured_facts:
                allowed.append("query_document_metadata")
            allowed.append("retrieve_knowledge")
        if needs_web:
            allowed.extend(["web_search", "read_website"])
    return RouteDecision(
        route=route, allowed_tools=allowed, blocked_tools=[tool for tool in ALL_AGENT_TOOLS if tool not in allowed],
        reason=reason, matched_skill_ids=matched_skill_ids or [], needs_documents=needs_documents,
        needs_web=needs_web, needs_workspace=needs_workspace, needs_structured_facts=needs_structured_facts,
        use_multi_agent=use_multi_agent, confidence=confidence,
    )


def _subtask_count(text: str) -> int:
    separators = len(re.findall(r"(?:^|\s)(?:\d+[.、]|[-*])\s*|[；;]\s*|\n+", text))
    conjunctions = len(re.findall(r"并且|同时|另外|然后|以及|and then|also", text, re.IGNORECASE))
    return max(1, separators + conjunctions + 1)


def _llm_route_fallback(message: str, has_documents: bool) -> Dict[str, object] | None:
    """Resolve only ambiguous search wording; explicit deterministic rules remain authoritative."""
    if not has_documents or not os.getenv("ZAI_API_KEY"):
        return None
    if not _contains_any(message, ("搜索", "查找", "查询", "找一个", "search", "lookup")):
        return None
    try:
        llm = ComponentFactory.get_component(
            "llm", LLM_TYPE, api_key=os.getenv("ZAI_API_KEY"), base_url=ZAI_API_BASE,
            model_name=LLM_MODEL_NAME, temperature=0, timeout=METADATA_LLM_TIMEOUT_SECONDS,
            max_retries=METADATA_LLM_MAX_RETRIES,
        )
        prompt = (
            "Return strict JSON with route=document_qa|metadata_query|web_search and confidence 0..1. "
            "Use web_search only when external/current internet information is explicitly intended. Request: " + message
        )
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(llm.invoke(prompt).content).strip(), flags=re.IGNORECASE)
        value = json.loads(content)
        return value if value.get("route") in {"document_qa", "metadata_query", "web_search"} else None
    except Exception:
        return None


def route_chat_request(session_id: str, message: str) -> RouteDecision:
    text, lowered = message.strip(), message.strip().casefold()
    if _contains_any(lowered, (
        "查看skill", "查看 skill", "列出skill", "列出 skill", "创建skill", "创建 skill",
        "修改skill", "更新skill", "启用skill", "禁用skill", "删除skill",
        "list skill", "show skill", "create skill", "update skill", "delete skill",
    )):
        return _decision("skill_admin", "Matched explicit skill administration command.")

    matched_skills = match_skills_with_workflow_fallback(message)
    workflow_skills = [skill for skill in matched_skills if str(skill.get("mode", "prompt")).casefold() == "workflow"]
    if workflow_skills:
        return _decision(
            "table_workflow", "Matched enabled workflow skill for a table workflow request.",
            [str(skill.get("id", "")) for skill in workflow_skills],
        )

    document_count = _completed_document_count(session_id)
    explicit_document = bool(document_count) and _contains_any(
        lowered, ("上传", "文档", "文件", "资料", "知识库", "根据材料", "pdf", "docx", "uploaded", "document", "local file"),
    )
    explicit_web = _has_url(text) or _contains_any(
        lowered, ("联网", "互联网", "网上", "最新", "权威来源", "外部来源", "today", "latest", "search web", "web search"),
    )
    metadata_terms = (
        "多少页", "页数", "金额", "币种", "合同号", "合同编号", "发票号", "项目号", "作者", "版本号",
        "签署日期", "签字人", "甲方", "乙方", "电话", "邮箱", "page count", "amount", "currency",
    )
    needs_metadata = bool(document_count) and _contains_any(lowered, metadata_terms)

    if ORCHESTRATION_ENABLED and explicit_document and explicit_web:
        return _decision(
            "orchestrated", "Request requires both uploaded-document and verified internet evidence.",
            needs_documents=True, needs_web=True, needs_structured_facts=needs_metadata, use_multi_agent=True,
        )
    if needs_metadata:
        return _decision("metadata_query", "Matched exact uploaded-document fact query.", needs_documents=True, needs_structured_facts=True)
    if _completed_table_count(session_id) and _contains_any(
        lowered, ("csv", "xlsx", "excel", "表格", "数据处理", "数据分析", "清洗", "均值", "平均", "缺失值", "分组", "统计"),
    ):
        return _decision("table_workflow", "Matched table-analysis intent with an uploaded table.")

    ambiguous = _llm_route_fallback(text, bool(document_count))
    if ambiguous and float(ambiguous.get("confidence", 0)) >= 0.65:
        route = str(ambiguous["route"])
        return _decision(
            route, "Ambiguous search intent resolved by model fallback.",
            needs_documents=route in {"document_qa", "metadata_query"}, needs_web=route == "web_search",
            needs_structured_facts=route == "metadata_query", confidence=float(ambiguous.get("confidence", 0.65)),
        )

    if explicit_web:
        return _decision("web_search", "Matched explicit external/web information request.", needs_web=True)
    if explicit_document or (document_count and _contains_any(
        lowered, ("总结", "概括", "主要内容", "引用", "证据", "sql 注入", "sql注入", "xss", "项目", "summarize"),
    )):
        return _decision("document_qa", "Matched uploaded-document question.", needs_documents=True)
    if ORCHESTRATION_ENABLED and _subtask_count(text) >= ORCHESTRATION_MIN_INDEPENDENT_SUBTASKS:
        return _decision(
            "orchestrated", "Request contains multiple independently executable subtasks.",
            needs_documents=bool(document_count), use_multi_agent=True, confidence=0.8,
        )
    return _decision("general_chat", "No specialized route matched.")


def apply_execution_mode(decision: RouteDecision, mode: str = "chat") -> RouteDecision:
    if mode != "workspace":
        return decision
    use_multi_agent = decision.use_multi_agent or (ORCHESTRATION_ENABLED and decision.needs_documents)
    route = "orchestrated" if use_multi_agent else decision.route
    allowed = list(dict.fromkeys([*decision.allowed_tools, *WORKSPACE_TOOLS]))
    return RouteDecision(
        route=route, allowed_tools=allowed, blocked_tools=[tool for tool in ALL_AGENT_TOOLS if tool not in allowed],
        reason=f"{decision.reason} Workspace capabilities were explicitly enabled for this request.",
        matched_skill_ids=list(decision.matched_skill_ids), needs_documents=decision.needs_documents,
        needs_web=decision.needs_web, needs_workspace=True, needs_structured_facts=decision.needs_structured_facts,
        use_multi_agent=use_multi_agent, confidence=decision.confidence,
    )


def find_route_violations(
    route: str, tool_calls: List[Dict[str, object]], allowed_tools: Sequence[str] | None = None,
) -> List[str]:
    allowed = set(allowed_tools if allowed_tools is not None else ROUTE_ALLOWED_TOOLS.get(route, []))
    allowed.update({"skill_admin": {"skill_command"}, "table_workflow": {"run_skill_workflow"}}.get(route, set()))
    allowed.add("skill_selection")
    violations: List[str] = []
    for tool_call in tool_calls:
        if str(tool_call.get("call_stage") or "") == "worker":
            continue
        tool_name = str(tool_call.get("tool") or "")
        if tool_name and tool_name not in allowed:
            violations.append(tool_name)
    return violations
