import os
from typing import Dict, Tuple

from loguru import logger

from config import (
    ENABLE_HYDE,
    HYDE_MAX_OUTPUT_CHARS,
    HYDE_MAX_QUERY_LENGTH,
    LLM_MODEL_NAME,
    LLM_TYPE,
    ZAI_API_BASE,
)
from core.factory import ComponentFactory


HYDE_PROMPT = """你是一个用于增强 RAG 检索的查询改写器。

请根据用户问题生成一段“假想文档摘要”，用于帮助向量检索命中更相关的资料。
要求：
1. 保留问题中的核心实体、产品名、技术名和专有名词。
2. 补充可能出现在答案文档中的相关术语。
3. 不要回答问题，不要列步骤。
4. 输出不超过 100 个中文字符。
"""


def generate_hyde_query(query: str) -> Tuple[str, Dict[str, str]]:
    """Generate a HyDE query and return a safe fallback when unavailable."""
    debug = {
        "enabled": str(bool(ENABLE_HYDE)),
        "original_query": query,
        "hyde_query": "",
        "search_query": query,
        "status": "disabled",
        "error": "",
    }

    clean_query = (query or "").strip()
    if not ENABLE_HYDE:
        return clean_query, debug

    if not clean_query:
        debug["status"] = "empty_query"
        return clean_query, debug

    if len(clean_query) > HYDE_MAX_QUERY_LENGTH:
        debug["status"] = "skipped_long_query"
        return clean_query, debug

    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        debug["status"] = "missing_api_key"
        return clean_query, debug

    try:
        llm = ComponentFactory.get_component(
            "llm",
            LLM_TYPE,
            api_key=api_key,
            base_url=ZAI_API_BASE,
            model_name=LLM_MODEL_NAME,
            temperature=0.2,
        )
        response = llm.invoke(
            [
                ("system", HYDE_PROMPT),
                ("human", clean_query),
            ]
        )
        hyde_query = (response.content or "").strip()
        if not hyde_query:
            debug["status"] = "empty_hyde"
            return clean_query, debug

        hyde_query = hyde_query[:HYDE_MAX_OUTPUT_CHARS]
        search_query = f"{clean_query}\n{hyde_query}"
        debug.update(
            {
                "hyde_query": hyde_query,
                "search_query": search_query,
                "status": "generated",
            }
        )
        return search_query, debug
    except Exception as exc:
        logger.warning(f"HyDE generation failed, fallback to original query: {exc}")
        debug["status"] = "failed"
        debug["error"] = str(exc)
        return clean_query, debug
