import logging
import os

from utils.ssl_compat import prefer_certifi_default_context

prefer_certifi_default_context()

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from config import ASYNC_UPLOAD_THRESHOLD_MB
from core.agent import create_agent_executor
from core.document_processor import delete_document, process_file, reindex_document
from core.history import get_session_history
from core.indexing_jobs import list_index_tasks, submit_process_file
from core.retriever import clear_last_retrieval_debug, get_last_retrieval_debug
from core.trace import get_recent_traces, get_trace_statistics, init_trace_db, record_chat_trace
from db.session_manager import (
    create_new_session,
    delete_session_data,
    get_all_sessions,
    get_session_documents,
    init_meta_db,
    update_session_title,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    filename="chatbot.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")

with open("static/style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _ensure_session() -> str:
    if "current_session_id" not in st.session_state:
        sessions = get_all_sessions()
        st.session_state.current_session_id = sessions[0][0] if sessions else create_new_session()
    return st.session_state.current_session_id


def _render_retrieval_debug(session_id: str) -> None:
    debug = get_last_retrieval_debug(session_id)
    if not debug:
        return

    with st.expander("检索调试链路", expanded=False):
        hyde = debug.get("hyde", {})
        if hyde:
            st.markdown("**HyDE 查询扩展**")
            st.write(
                {
                    "状态": hyde.get("status"),
                    "原始问题": hyde.get("original_query"),
                    "假想文档": hyde.get("hyde_query"),
                }
            )

        timing = debug.get("timing_ms", {})
        if timing:
            st.markdown("**耗时统计(ms)**")
            st.dataframe([timing], use_container_width=True)

        for key, title in [
            ("dense", "Dense / MMR 结果"),
            ("bm25", "BM25 结果"),
            ("rrf", "RRF 融合结果"),
            ("rerank", "CrossEncoder Rerank 结果"),
            ("final", "最终注入上下文"),
        ]:
            rows = debug.get(key, [])
            if rows:
                st.markdown(f"**{title}**")
                st.dataframe(
                    [
                        {
                            "rank": row.get("rank"),
                            "score": row.get("score"),
                            "source": row.get("source"),
                            "page": row.get("page"),
                            "chunk_id": row.get("chunk_id"),
                            "preview": row.get("preview"),
                        }
                        for row in rows
                    ],
                    use_container_width=True,
                )


def _render_recent_traces(session_id: str) -> None:
    stats = get_trace_statistics(session_id)
    traces = get_recent_traces(session_id, limit=8)
    if not traces and not stats.get("total"):
        return

    with st.expander("最近执行 Trace", expanded=False):
        st.dataframe([stats], use_container_width=True)
        for trace in traces:
            status = "ERROR" if trace.get("error") else "OK"
            label = f"#{trace['id']} · {status} · {trace.get('created_at', '')}"
            with st.expander(label, expanded=False):
                st.caption("User input")
                st.write(trace.get("user_input", ""))

                retrieval_debug = trace.get("retrieval_debug", {})
                timing = retrieval_debug.get("timing_ms", {})
                if timing:
                    st.caption("Timing(ms)")
                    st.dataframe([timing], use_container_width=True)

                final_docs = retrieval_debug.get("final", [])
                if final_docs:
                    st.caption("Final retrieval context")
                    st.dataframe(
                        [
                            {
                                "rank": doc.get("rank"),
                                "score": doc.get("score"),
                                "source": doc.get("source"),
                                "page": doc.get("page"),
                                "chunk_id": doc.get("chunk_id"),
                                "preview": doc.get("preview"),
                            }
                            for doc in final_docs
                        ],
                        use_container_width=True,
                    )

                tool_calls = trace.get("tool_calls", [])
                if tool_calls:
                    st.caption("Tool calls")
                    st.dataframe(tool_calls, use_container_width=True)

                if trace.get("error"):
                    st.error(trace["error"])


def _render_sidebar(curr_id: str) -> None:
    with st.sidebar:
        st.header("会话管理")
        if st.button("新建对话", use_container_width=True):
            st.session_state.current_session_id = create_new_session()
            st.rerun()

        st.markdown("---")
        st.caption("历史会话")
        sessions = get_all_sessions()
        for session_id, title in sessions:
            col1, col2 = st.columns([0.78, 0.22])
            with col1:
                if st.button(
                    f"💬 {title}",
                    key=f"btn_{session_id}",
                    disabled=session_id == curr_id,
                    use_container_width=True,
                ):
                    st.session_state.current_session_id = session_id
                    st.rerun()
            with col2:
                if st.button("删", key=f"del_{session_id}", use_container_width=True):
                    delete_session_data(session_id)
                    remaining = get_all_sessions()
                    st.session_state.current_session_id = remaining[0][0] if remaining else create_new_session()
                    st.rerun()

        st.markdown("---")
        st.subheader("当前会话文档")
        uploaded_file = st.file_uploader(
            "上传文档",
            type=["txt", "pdf", "md", "doc", "docx", "csv", "xlsx"],
            key=f"uploader_{curr_id}",
        )
        if uploaded_file and st.button("解析并写入知识库", use_container_width=True):
            with st.spinner("正在解析、切分并写入向量库..."):
                file_size_mb = len(uploaded_file.getvalue()) / 1024 / 1024
                if file_size_mb >= float(ASYNC_UPLOAD_THRESHOLD_MB):
                    try:
                        task_id = submit_process_file(uploaded_file, curr_id)
                        success, message = True, f"Indexing task submitted: {task_id}"
                    except Exception as exc:
                        success, message = False, f"Submit failed: {exc}"
                else:
                    success, message = process_file(uploaded_file, curr_id)
            if success:
                st.success(message)
            else:
                st.error(message)

        jobs = list_index_tasks(curr_id)
        if jobs:
            st.markdown("---")
            st.caption("Indexing tasks")
            for job in jobs[:5]:
                st.caption(f"{job.get('file_name')} 路 {job.get('status')} 路 {job.get('message')}")

        documents = get_session_documents(curr_id)
        if documents:
            st.markdown("---")
            st.caption("已入库文档")
            for file_id, file_name, chunk_count, status, uploaded_at, file_path in documents:
                col1, col2, col3 = st.columns([0.58, 0.21, 0.21])
                with col1:
                    st.caption(f"{file_name} · {chunk_count} chunks · {status}")
                with col2:
                    if st.button("Reindex", key=f"doc_reindex_{file_id}", use_container_width=True):
                        success, message = reindex_document(curr_id, file_id)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                with col3:
                    if st.button("删", key=f"doc_del_{file_id}", use_container_width=True):
                        success, message = delete_document(curr_id, file_id)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

        st.markdown("---")
        _render_recent_traces(curr_id)


def main():
    init_meta_db()
    init_trace_db()
    curr_id = _ensure_session()
    _render_sidebar(curr_id)

    sessions = get_all_sessions()
    st.title("Agentic RAG Chatbot")

    history = get_session_history(curr_id)
    messages = history.messages

    if len(messages) >= 2:
        current_title = next((session[1] for session in sessions if session[0] == curr_id), "新对话")
        if current_title == "新对话":
            first_user_msg = messages[0].content
            update_session_title(curr_id, first_user_msg[:15])
            st.rerun()

    for msg in messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

    prompt = st.chat_input("请输入问题...")
    if not prompt:
        return

    if not os.getenv("ZAI_API_KEY"):
        st.error("请先在 .env 中配置 ZAI_API_KEY")
        st.stop()

    agent_chain = create_agent_executor(curr_id)
    if agent_chain is None:
        st.error("Agent 初始化失败，请检查模型配置和 API Key。")
        st.stop()

    st.chat_message("user").write(prompt)
    clear_last_retrieval_debug(curr_id)

    with st.chat_message("assistant"):
        container = st.empty()
        with st.spinner("思考中..."):
            try:
                response = agent_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": curr_id}},
                )

                final_answer = response.get("output", "")
                intermediate_steps = response.get("intermediate_steps", [])
                tool_calls = [
                    {
                        "tool": action.tool,
                        "input": action.tool_input,
                        "output_preview": str(observation)[:1000],
                    }
                    for action, observation in intermediate_steps
                ]

                container.write(final_answer)

                if intermediate_steps:
                    with st.expander("工具调用", expanded=False):
                        for action, observation in intermediate_steps:
                            st.markdown(f"**工具:** `{action.tool}`")
                            st.markdown(f"**输入:** `{action.tool_input}`")
                            st.markdown(f"**输出预览:** {str(observation)[:800]}")

                _render_retrieval_debug(curr_id)
                record_chat_trace(
                    session_id=curr_id,
                    user_input=prompt,
                    answer=final_answer,
                    retrieval_debug=get_last_retrieval_debug(curr_id),
                    tool_calls=tool_calls,
                )

            except Exception as exc:
                record_chat_trace(
                    session_id=curr_id,
                    user_input=prompt,
                    error=str(exc),
                    retrieval_debug=get_last_retrieval_debug(curr_id),
                )
                st.error(f"发生错误: {exc}")

    st.rerun()


if __name__ == "__main__":
    main()
