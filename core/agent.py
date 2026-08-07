import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from loguru import logger

from config import (
    AGENT_CODE_MAX_CHARS,
    AGENT_CODE_OUTPUT_LIMIT,
    AGENT_CODE_TIMEOUT_SECONDS,
    AGENT_ENABLE_CODE_EXECUTION,
    AGENT_MAX_EXECUTION_TIME_SECONDS,
    AGENT_MAX_ITERATIONS,
    LLM_MODEL_NAME,
    LLM_TYPE,
    ZAI_API_BASE,
)
from core.factory import ComponentFactory
from core.history import get_async_session_history, get_session_history
from core.query_transform import generate_hyde_query
from core.retriever import load_retriever, retrieve_with_rerank_debug, set_last_retrieval_debug
from db.skill_manager import match_skills
from tools.code_runner import create_python_code_tool
from tools.system import get_current_time
from tools.web_search import read_website, web_search
from tools.workspace import create_workspace_tools


def _format_evidence(reranked_docs: List[Tuple[float, Document]]) -> str:
    if not reranked_docs:
        return "未检索到可用证据。"

    lines = []
    for idx, (score, doc) in enumerate(reranked_docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "未知来源")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id", metadata.get("id", "unknown"))
        page_text = f", page={page}" if page is not None else ""
        lines.append(
            f"[{idx}] score={score:.4f}, source={source}{page_text}, chunk_id={chunk_id}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(lines)


def _truncate_skill_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... <truncated>"


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

    requested_tools = set(allowed_tools or ["retrieve_knowledge", "read_website", "web_search", "get_current_time", "run_python_code"])
    tool_registry = {
        "retrieve_knowledge": retrieve_knowledge,
        "read_website": read_website,
        "get_current_time": get_current_time,
    }
    if enable_web_search:
        tool_registry["web_search"] = web_search
    if AGENT_ENABLE_CODE_EXECUTION:
        tool_registry["run_python_code"] = create_python_code_tool(
            session_id=session_id,
            timeout_seconds=int(AGENT_CODE_TIMEOUT_SECONDS),
            output_limit=int(AGENT_CODE_OUTPUT_LIMIT),
            max_code_chars=int(AGENT_CODE_MAX_CHARS),
        )
    if mode == "workspace":
        tool_registry.update({item.name: item for item in create_workspace_tools(session_id)})
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

    system_text = """你是一个严谨的 AI 助手，擅长使用工具解决复杂问题。

工作规则：
1. 如果问题涉及上传文档、私有资料、当前会话知识库，必须调用 retrieve_knowledge。
2. 如果问题包含 URL，优先调用 read_website。
3. 如果问题需要实时或外部信息，并且 web_search 可用，可以调用 web_search。
4. 使用本地知识库证据回答时，必须在关键结论后标注引用编号，例如 [1]、[2]；编号必须来自 retrieve_knowledge 返回的证据顺序，不允许虚构不存在的编号。
5. 如果检索证据不足以回答，直接说明“根据当前文档无法确认”，不要编造。
6. 回答尽量结构化、简洁，并明确区分文档内容、互联网内容和你的推理。
"""

    system_text += """

Advanced agent rules:
1. For multi-step tasks, think through the next useful step, call the right tool, inspect the observation, then continue.
2. Use run_python_code for arithmetic, statistics, data transformations, or small algorithm checks when exact computation is useful.
3. Do not claim that a tool was used unless a tool observation is available.
4. Do not invent code execution, retrieval, search, or website results.
5. Keep code snippets short and self-contained; never use code execution for files, network access, project edits, shell commands, or long-running jobs.
"""

    tool_names = [str(getattr(item, "name", "")) for item in tools if getattr(item, "name", "")]
    system_text = (
        "You are a careful assistant. The backend has already routed this request.\n"
        f"Route: {route}\n"
        f"Allowed tools: {', '.join(tool_names) if tool_names else 'none'}\n"
        "Decision policy before answering: first use any matched skill instructions supplied below; "
        "if no skill fits and this is an uploaded-document request, use RAG evidence; "
        "if neither skill nor RAG applies, answer directly from general knowledge. "
        "Use only available tools. Do not invent tool results. "
        "Do not claim a tool was used unless an observation is available.\n"
    )
    if mode == "workspace":
        system_text += (
            "Workspace mode is active. Use workspace_terminal and workspace file/process tools for tasks that require "
            "commands, code execution, file edits, builds, or tests. All workspace paths must stay below /workspace. "
            "For a session upload, first call workspace_list_session_files then workspace_import_uploaded_file; "
            "write generated files below /workspace/output so they are automatically saved as artifacts. "
            "Treat tool errors as real failures and never imply that a command succeeded without its result.\n"
        )
    if route == "document_qa":
        system_text += (
            "This is an uploaded-document question. Base document-specific claims on retrieved evidence "
            "and cite evidence numbers when present.\n"
        )
    elif route == "web_search":
        system_text += "This is an external information request. Use web tools only for external facts.\n"
    elif route == "general_chat" and mode != "workspace":
        system_text += "This is general chat. Do not use document retrieval, web search, or code execution.\n"

    if doc_summary:
        system_text += f"\n当前会话已加载文档摘要：\n{doc_summary}\n"
    else:
        system_text += "\n当前会话尚未加载本地文档。\n"

    if prefetched_context.strip():
        system_text += (
            "\nAuto-retrieved document evidence for this request:\n"
            f"{prefetched_context.strip()}\n"
            "Use this evidence when it is relevant. If you cite it, use the bracketed evidence numbers already shown.\n"
        )

    if skill_context.strip():
        system_text += (
            "\nComputed skill workflow context for this request:\n"
            f"{skill_context.strip()}\n"
            "Base any workflow-specific answer on these computed values.\n"
        )

    skill_prompt = _format_skills_for_prompt(skills_for_prompt)
    if skill_prompt:
        system_text += (
            f"\n\n{skill_prompt}\n"
            "A matched prompt skill is considered active for this request. Make the final answer visibly follow that skill.\n"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

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

    history_factory = get_async_session_history if async_history else get_session_history
    return RunnableWithMessageHistory(
        agent_executor,
        history_factory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
