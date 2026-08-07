from langchain_core.tools import BaseTool
from typing import Any, Type
from pydantic import Field, BaseModel
from core.retriever import load_retriever, retrieve_with_rerank

class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="The query to search for in the knowledge base.")

# 该类继承自 langchain 的 BaseTool，用于把“知识库检索”封装成一个可被智能体调用的工具。
class KnowledgeBaseTool(BaseTool):
    name = "knowledge_base_search"
    description = "【本地知识库检索】当问题涉及已上传的文档内容时使用。"
    args_schema: Type[BaseModel] = KnowledgeSearchInput

    def _run(self, query: str, **kwargs: Any) -> Any:
        session_id = kwargs.get("session_id")
        retriever, _ = load_retriever(session_id)
        if not retriever:
            return "⚠️ 错误：用户尚未上传文档。"
        try:
            reranked_docs = retrieve_with_rerank(query, retriever)
            if not reranked_docs:
                return "文档中未找到相关内容。"
            res = ""
            for i, (score, doc) in enumerate(reranked_docs[:4]):
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page")
                page_info = f", page={page}" if page is not None else ""
                res += (
                    f"\n[片段 {i+1}, score={score:.2f}, source={source}{page_info}]: "
                    f"{doc.page_content}\n"
                )
            return res
        except Exception as e:
            return f"检索出错: {e}"
