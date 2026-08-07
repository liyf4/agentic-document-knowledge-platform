from typing import List, Tuple, Optional, Any
from loguru import logger
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import (
    LLM_MODEL_NAME, 
    LLM_TYPE,
    ZAI_API_BASE,
    RERANK_TOP_K
)
from core.factory import ComponentFactory
from core.retriever import load_retriever, retrieve_with_rerank

class RagPipeline:
    """
    Unified RAG Pipeline combining retrieval, reranking, and generation.
    Inspired by BishengRagPipeline.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.retriever, self.summary = load_retriever(session_id)
        self.api_key = os.getenv("ZAI_API_KEY")
        
    def run_retrieval(self, query: str) -> List[Tuple[float, Document]]:
        """Executes the hybrid retrieval and reranking flow."""
        if not self.retriever:
            logger.warning(f"No retriever loaded for session {self.session_id}")
            return []
            
        try:
            return retrieve_with_rerank(query, self.retriever)
        except Exception as e:
            logger.error(f"Retrieval error in pipeline: {e}")
            return []

    def format_context(self, reranked_docs: List[Tuple[float, Document]]) -> str:
        """Formats retrieved documents into a context string."""
        res = ""
        for i, (score, doc) in enumerate(reranked_docs[:RERANK_TOP_K]):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            page_info = f", page={page}" if page is not None else ""
            res += (
                f"\n[片段 {i+1}, score={score:.2f}, source={source}{page_info}]: "
                f"{doc.page_content}\n"
            )
        return res

    def generate_answer(self, query: str, context: str) -> str:
        """Generates an answer using the LLM based on context."""
        try:
            llm = ComponentFactory.get_component(
                "llm",
                LLM_TYPE,
                api_key=self.api_key,
                base_url=ZAI_API_BASE,
                model_name=LLM_MODEL_NAME,
                temperature=0.4
            )
            
            prompt = f"""根据以下上下文回答问题。如果上下文不足以回答，请明确告知。
            
            上下文：
            {context}
            
            问题：
            {query}
            
            回答："""
            
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"生成回答时出错: {str(e)}"

import os
