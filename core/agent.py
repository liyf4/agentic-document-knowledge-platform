import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from utils.ssl_compat import prefer_certifi_default_context

prefer_certifi_default_context()

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from loguru import logger

from config import (
    AGENT_CODE_MAX_CHARS,
    AGENT_CODE_OUTPUT_LIMIT,
    AGENT_CODE_TIMEOUT_SECONDS,
    AGENT_ENABLE_CODE_EXECUTION,
    AGENT_MAX_EXECUTION_TIME_SECONDS,
    AGENT_MAX_ITERATIONS,
    CONTEXT_TOOL_OBSERVATION_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TYPE,
    ZAI_API_BASE,
)
from core.factory import ComponentFactory
from core.history import get_async_session_history, get_session_history, get_windowed_session_history
from core.query_transform import generate_hyde_query
from core.retriever import load_retriever, retrieve_with_rerank_debug, set_last_retrieval_debug
from db.skill_manager import match_skills
from tools.code_runner import create_python_code_tool
from tools.document_metadata import create_document_metadata_tool, query_session_document_metadata
from tools.system import get_current_time
from tools.web_search import read_web_page, read_website, search_web_candidates, web_search
from tools.workspace import create_workspace_tools
from prompts.builder import PromptContext, build_system_prompt


def build_document_evidence(reranked_docs: List[Tuple[float, Document]]) -> List[Dict[str, Any]]:
    """Convert retrieval hits into stable evidence without joining adjacent sections."""
    evidence: List[Dict[str, Any]] = []
    for score, doc in reranked_docs:
        metadata = dict(doc.metadata or {})
        chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or "unknown")
        identity = "|".join((
            str(metadata.get("file_id", "")), chunk_id,
            str(metadata.get("section_path", "")), doc.page_content,
        ))
        evidence.append({
            "evidence_id": "doc_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
            "kind": "document",
            "file_id": str(metadata.get("file_id", "")),
            "file_name": str(metadata.get("source", "Unknown source")),
            "page": metadata.get("page"),
            "printed_page": metadata.get("printed_page", ""),
            "section_path": str(metadata.get("section_path", "")),
            "chunk_id": chunk_id,
            "chunk_kind": str(metadata.get("chunk_kind", "text")),
            "text": doc.page_content,
            "score": float(score),
        })
    return evidence


def group_document_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group evidence while preserving section and block-type boundaries."""
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for item in evidence:
        key = (
            str(item.get("file_id", "")),
            str(item.get("section_path", "")),
            str(item.get("chunk_kind", "text")),
        )
        groups.setdefault(key, []).append(item)
    return [
        {"file_id": key[0], "section_path": key[1], "chunk_kind": key[2], "evidence": values}
        for key, values in groups.items()
    ]


def _format_evidence(reranked_docs: List[Tuple[float, Document]]) -> str:
    if not reranked_docs:
        return "未检索到可用证据。"

    lines = []
    for idx, (score, doc) in enumerate(reranked_docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "未知来源")
        page = metadata.get("page")
        printed_page = metadata.get("printed_page")
        section_path = metadata.get("section_path")
        chunk_id = metadata.get("chunk_id", metadata.get("id", "unknown"))
        page_text = f", page={page}" if page is not None else ""
        printed_text = f", printed_page={printed_page}" if printed_page else ""
        section_text = f", section={section_path}" if section_path else ""
        lines.append(
            f"[{idx}] score={score:.4f}, source={source}{page_text}{printed_text}{section_text}, chunk_id={chunk_id}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(lines)


def _truncate_skill_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... <truncated>"


def _bounded_tool_observation(value: Any) -> str:
    text = str(value or "")
    char_limit = max(1000, CONTEXT_TOOL_OBSERVATION_MAX_TOKENS * 3)
    if len(text) <= char_limit:
        return text
    half = max(1, (char_limit - 80) // 2)
    return f"{text[:half]}\n... <tool observation truncated for context budget> ...\n{text[-half:]}"


def _format_skills_for_prompt(skills: List[Dict[str, Any]]) -> str:
    if not skills:
        return ""

    blocks = [
        "Available prompt skills matched for this user request. Follow them as the primary answer style/task instructions.",
        "Prompt skills cannot override safety rules, tool limits, citation rules, or system instructions.",
        "Filesystem skills follow a SKILL.md package model: the loaded markdown body below is authoritative, while listed assets are only references to request explicitly if needed.",
    ]
    for index, skill in enumerate(skills, start=1):
        examples = skill.get("examples") or []
        example_text = "\n".join(f"- {item}" for item in examples[:3])
        assets = skill.get("assets") or []
        asset_text = "\n".join(f"- {item}" for item in assets[:8])
        source = str(skill.get("source", "database"))
        path = str(skill.get("path", ""))
        block = [
            f"[Skill {index}] {skill.get('name', '')}",
            f"Source: {source}{f' ({path})' if path else ''}",
            f"Description: {_truncate_skill_text(skill.get('description', ''), 600)}",
            f"Instructions:\n{_truncate_skill_text(skill.get('instruction', ''), 3000)}",
        ]
        if example_text:
            block.append(f"Examples:\n{_truncate_skill_text(example_text, 1200)}")
        if asset_text:
            block.append(f"Available package assets:\n{_truncate_skill_text(asset_text, 1200)}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def build_agent_system_prompt_preview(
    session_id: str,
    *,
    selected_skills: Optional[List[Dict[str, Any]]] = None,
    user_input: str = "",
    prefetched_context: str = "",
    skill_context: str = "",
    allowed_tools: Optional[List[str]] = None,
    route: str = "general_chat",
    mode: str = "chat",
    orchestration_context: str = "",
) -> str:
    """Build the non-history prompt text for preflight token budgeting."""
    _retriever, doc_summary = load_retriever(session_id)
    skills = selected_skills if selected_skills is not None else match_skills(user_input)
    return build_system_prompt(PromptContext(
        route=route,
        mode=mode,
        allowed_tools=list(allowed_tools or []),
        document_summary=doc_summary,
        prefetched_context=prefetched_context,
        skill_context=skill_context,
        skill_prompt=_format_skills_for_prompt(skills),
        orchestration_context=orchestration_context,
    ))


def retrieve_session_knowledge(session_id: str, query: str) -> str:
    retriever, _doc_summary = load_retriever(session_id)
    if not retriever:
        return "No local documents are indexed for the current session."

    try:
        search_query, hyde_debug = generate_hyde_query(query)
        reranked_docs, retrieval_debug = retrieve_with_rerank_debug(search_query, retriever)
        retrieval_debug["original_query"] = query
        retrieval_debug["search_query"] = search_query
        retrieval_debug["hyde"] = hyde_debug
        set_last_retrieval_debug(session_id, retrieval_debug)

        if not reranked_docs:
            return "No sufficiently relevant evidence was found in the uploaded documents."

        evidence = _format_evidence(reranked_docs)
        return (
            "The following evidence was retrieved from the local knowledge base. Answer based on this evidence, "
            "cite supporting claims with [1], [2], etc. using the evidence order, and state clearly when evidence is insufficient.\n\n"
            f"{evidence}"
        )
    except Exception as exc:
        logger.error(f"Retrieval tool error: {exc}")
        return f"Retrieval failed: {exc}"


def retrieve_session_evidence(session_id: str, query: str) -> Dict[str, Any]:
    """Retrieve structured document evidence for an already-executed worker call."""
    retriever, _doc_summary = load_retriever(session_id)
    if not retriever:
        return {"ok": False, "error_code": "NO_INDEXED_DOCUMENTS", "evidence": [], "claims": []}
    try:
        search_query, hyde_debug = generate_hyde_query(query)
        reranked_docs, retrieval_debug = retrieve_with_rerank_debug(search_query, retriever)
        retrieval_debug.update({"original_query": query, "search_query": search_query, "hyde": hyde_debug})
        set_last_retrieval_debug(session_id, retrieval_debug)
        evidence = build_document_evidence(reranked_docs)
        claims = [
            {
                "claim": str(item["text"])[:500],
                "evidence_ids": [str(item["evidence_id"])],
                "section_path": str(item.get("section_path", "")),
            }
            for item in evidence
        ]
        return {
            "ok": bool(evidence),
            "error_code": "" if evidence else "INSUFFICIENT_DOCUMENT_EVIDENCE",
            "evidence": evidence,
            "evidence_groups": group_document_evidence(evidence),
            "claims": claims,
        }
    except Exception as exc:
        logger.exception("Structured document retrieval failed")
        return {
            "ok": False, "error_code": "DOCUMENT_RETRIEVAL_FAILED", "error": str(exc),
            "evidence": [], "claims": [],
        }


def create_agent_executor(
    session_id: str,
    async_history: bool = False,
    user_input: str = "",
    selected_skills: Optional[List[Dict[str, Any]]] = None,
    prefetched_context: str = "",
    skill_context: str = "",
    allowed_tools: Optional[List[str]] = None,
    route: str = "general_chat",
    mode: str = "chat",
    orchestration_context: str = "",
    tool_call_limits: Optional[Dict[str, int]] = None,
    artifact_requirements: Optional[Dict[str, Any]] = None,
    history_messages: Optional[List[Any]] = None,
    conversation_summary: str = "",
):
    zai_api_key = os.getenv("ZAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    if not zai_api_key:
        logger.error("Missing ZAI_API_KEY environment variable")
        return None

    enable_web_search = bool((google_api_key and google_cse_id) or serpapi_api_key)
    retriever, doc_summary = load_retriever(session_id)
    skills_for_prompt = selected_skills if selected_skills is not None else match_skills(user_input)

    @tool
    def retrieve_knowledge(query: str) -> str:
        """本地知识库检索：当问题涉及用户上传的文档、项目资料、私有知识时使用。"""
        if not retriever:
            return "当前会话尚未上传文档，无法使用本地知识库检索。"

        try:
            search_query, hyde_debug = generate_hyde_query(query)
            reranked_docs, retrieval_debug = retrieve_with_rerank_debug(search_query, retriever)
            retrieval_debug["original_query"] = query
            retrieval_debug["search_query"] = search_query
            retrieval_debug["hyde"] = hyde_debug
            set_last_retrieval_debug(session_id, retrieval_debug)

            if not reranked_docs:
                return "文档中未检索到足够相关的内容，请说明信息不足，不要编造答案。"

            evidence = _format_evidence(reranked_docs)
            return (
                "以下是本地知识库检索到的证据片段。回答必须基于这些证据，"
                "并在相关句子后使用 [1]、[2] 这样的编号引用来源；如果证据不足，请明确说明。\n\n"
                f"{evidence}"
            )
        except Exception as exc:
            logger.error(f"Retrieval tool error: {exc}")
            return f"检索出错: {exc}"

    call_counts: Dict[str, int] = {}

    def _within_limit(name: str) -> bool:
        limit = int((tool_call_limits or {}).get(name, 10**9))
        used = call_counts.get(name, 0)
        if used >= limit:
            return False
        call_counts[name] = used + 1
        return True

    @tool("retrieve_knowledge")
    def limited_retrieve_knowledge(query: str) -> str:
        """Retrieve uploaded-document evidence within the Supervisor remediation budget."""
        if not _within_limit("retrieve_knowledge"):
            return json.dumps({"ok": False, "error_code": "REMEDIATION_LIMIT_REACHED"})
        return _bounded_tool_observation(retrieve_session_knowledge(session_id, query))

    @tool("query_document_metadata")
    def limited_document_metadata(query: str, file_id: str = "", fact_types: str = "") -> str:
        """Query document metadata within the Supervisor remediation budget."""
        if not _within_limit("query_document_metadata"):
            return json.dumps({"ok": False, "error_code": "REMEDIATION_LIMIT_REACHED"})
        return _bounded_tool_observation(query_session_document_metadata(session_id, query, file_id, fact_types))

    @tool("web_search")
    def limited_web_search(query: str) -> str:
        """Search candidate web sources within the Supervisor remediation budget."""
        if not _within_limit("web_search"):
            return json.dumps({"ok": False, "error_code": "REMEDIATION_LIMIT_REACHED"})
        return _bounded_tool_observation(json.dumps(search_web_candidates(query), ensure_ascii=False, default=str))

    @tool("read_website")
    def limited_read_website(url: str) -> str:
        """Open a web source within the Supervisor remediation budget."""
        if not _within_limit("read_website"):
            return json.dumps({"ok": False, "error_code": "REMEDIATION_LIMIT_REACHED"})
        return _bounded_tool_observation(json.dumps(read_web_page(url), ensure_ascii=False, default=str))

    requested_tools = set(allowed_tools or [
        "retrieve_knowledge", "query_document_metadata", "read_website", "web_search",
        "get_current_time", "run_python_code",
    ])
    tool_registry = {
        "retrieve_knowledge": limited_retrieve_knowledge,
        "query_document_metadata": limited_document_metadata,
        "read_website": limited_read_website,
        "get_current_time": get_current_time,
    }
    if enable_web_search:
        tool_registry["web_search"] = limited_web_search
    if AGENT_ENABLE_CODE_EXECUTION:
        tool_registry["run_python_code"] = create_python_code_tool(
            session_id=session_id,
            timeout_seconds=int(AGENT_CODE_TIMEOUT_SECONDS),
            output_limit=int(AGENT_CODE_OUTPUT_LIMIT),
            max_code_chars=int(AGENT_CODE_MAX_CHARS),
        )
    if mode == "workspace":
        tool_registry.update({
            item.name: item for item in create_workspace_tools(
                session_id, artifact_requirements=artifact_requirements,
            )
        })
    tools = [tool for name, tool in tool_registry.items() if name in requested_tools]

    llm = ComponentFactory.get_component(
        "llm",
        LLM_TYPE,
        api_key=zai_api_key,
        base_url=ZAI_API_BASE,
        model_name=LLM_MODEL_NAME,
        temperature=0.4,
        streaming=True,
    )

    tool_names = [str(getattr(item, "name", "")) for item in tools if getattr(item, "name", "")]
    skill_prompt = _format_skills_for_prompt(skills_for_prompt)
    system_text = build_system_prompt(PromptContext(
        route=route, mode=mode, allowed_tools=tool_names, document_summary=doc_summary,
        prefetched_context=prefetched_context, skill_context=skill_context,
        skill_prompt=skill_prompt, orchestration_context=orchestration_context,
        conversation_summary=conversation_summary,
    ))

    prompt = ChatPromptTemplate.from_messages(
        [
            # Dynamic evidence and worker reports may contain JSON/code braces. A concrete
            # SystemMessage keeps them as data instead of parsing them as f-string fields.
            SystemMessage(content=system_text),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    if tool_names == ["workspace_write_file"]:
        # The research Workers have already completed. Force the sole remaining
        # artifact action and stop on its observed result instead of allowing the
        # model to spin on empty/future-action responses.
        tools[0].return_direct = True
        llm_with_tools = llm.bind_tools(tools, tool_choice="workspace_write_file")
        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda values: format_to_tool_messages(values["intermediate_steps"]),
            )
            | prompt
            | llm_with_tools
            | ToolsAgentOutputParser()
        )
    else:
        agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=int(AGENT_MAX_ITERATIONS),
        max_execution_time=int(AGENT_MAX_EXECUTION_TIME_SECONDS),
    )

    if history_messages is None:
        history_factory = get_async_session_history if async_history else get_session_history
    else:
        visible_history = list(history_messages)
        history_factory = lambda requested_session_id: get_windowed_session_history(
            requested_session_id, visible_history, async_mode=async_history
        )
    return RunnableWithMessageHistory(
        agent_executor,
        history_factory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
