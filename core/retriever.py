import os
import re
import time
import streamlit as st
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from loguru import logger

from config import (
    BM25_K,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_TYPE,
    RERANK_CANDIDATES,
    RERANK_MIN_SCORE,
    RERANK_MODEL_NAME,
    RERANK_SKIP_ENABLED,
    RERANK_SKIP_RRF_ABS_MARGIN,
    RERANK_SKIP_RRF_RATIO,
    RERANK_SKIP_SHORT_QUERY_TOKENS,
    RERANK_TOP_K,
    RRF_K,
    VECTOR_FETCH_K,
    VECTOR_MMR_LAMBDA,
    VECTOR_SEARCH_K,
)
from db.session_manager import get_session_documents, get_session_summary
from core.factory import ComponentFactory

_LAST_RETRIEVAL_DEBUG: Dict[str, Dict[str, Any]] = {}
_RETRIEVER_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_STATS: Dict[str, int] = {"hits": 0, "misses": 0, "invalidations": 0}

@st.cache_resource
def get_embedding_model():
    """
    Load embedding model using ComponentFactory.
    """
    return ComponentFactory.get_component(
        "embeddings", 
        EMBEDDING_TYPE, 
        model_name=EMBEDDING_MODEL_NAME
    )

@st.cache_resource
def get_rerank_model():
    """
    Load rerank model (CrossEncoder).
    """
    logger.info(f"Loading rerank model: {RERANK_MODEL_NAME}")
    return CrossEncoder(RERANK_MODEL_NAME, max_length=512)


def _doc_identity(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    chunk_id = str(doc.metadata.get("chunk_id", doc.metadata.get("id", "")))
    return f"{source}|{page}|{chunk_id}|{doc.page_content[:160]}"


def _doc_debug(doc: Document, rank: int, score: Optional[float] = None) -> Dict[str, Any]:
    metadata = doc.metadata or {}
    return {
        "rank": rank,
        "score": score,
        "source": metadata.get("source", "未知来源"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id", metadata.get("id")),
        "content": doc.page_content,
        "preview": doc.page_content[:240].replace("\n", " "),
    }


def set_last_retrieval_debug(session_id: str, debug: Dict[str, Any]) -> None:
    _LAST_RETRIEVAL_DEBUG[session_id] = deepcopy(debug)


def get_last_retrieval_debug(session_id: str) -> Dict[str, Any]:
    return deepcopy(_LAST_RETRIEVAL_DEBUG.get(session_id, {}))


def clear_last_retrieval_debug(session_id: str) -> None:
    _LAST_RETRIEVAL_DEBUG.pop(session_id, None)


def invalidate_retriever_cache(session_id: Optional[str] = None) -> None:
    if session_id is None:
        _RETRIEVER_CACHE.clear()
    else:
        _RETRIEVER_CACHE.pop(session_id, None)
    _CACHE_STATS["invalidations"] += 1


def get_retriever_cache_stats() -> Dict[str, Any]:
    stats = dict(_CACHE_STATS)
    stats["sessions"] = sorted(_RETRIEVER_CACHE.keys())
    return stats


def get_session_document_overview(
    session_id: str,
    max_chunks_per_file: int = 3,
    max_chars_per_file: int = 1800,
) -> Tuple[str, Dict[str, Any]]:
    """Build a compact, per-file context block for broad document overview questions."""
    persist_dir = f"{DATA_DIR}/{session_id}/chroma"
    debug: Dict[str, Any] = {
        "query": "document_overview",
        "mode": "document_overview",
        "final": [],
        "timing_ms": {},
        "error": "",
    }
    if not os.path.exists(persist_dir):
        debug["error"] = "vector store does not exist"
        return "", debug

    start_time = time.perf_counter()
    rows = get_session_documents(session_id)
    if not rows:
        debug["error"] = "no session documents"
        return "", debug

    try:
        embeddings = get_embedding_model()
        vectorstore = ComponentFactory.get_component(
            "vectorstores",
            "Chroma",
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"collection_{session_id}",
        )
        stored = vectorstore.get(include=["documents", "metadatas"])
        all_texts = stored.get("documents", [])
        all_metadatas = stored.get("metadatas", []) or [{} for _ in all_texts]

        grouped: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {}
        for text, metadata in zip(all_texts, all_metadatas):
            safe_metadata = dict(metadata or {})
            file_id = str(safe_metadata.get("file_id", ""))
            if not file_id:
                continue
            try:
                chunk_index = int(safe_metadata.get("chunk_index", 0))
            except (TypeError, ValueError):
                chunk_index = 0
            grouped.setdefault(file_id, []).append((chunk_index, str(text or ""), safe_metadata))

        blocks = []
        final_docs = []
        rank = 1
        for file_id, file_name, chunk_count, status, _uploaded_at, _file_path in rows:
            chunks = sorted(grouped.get(file_id, []), key=lambda item: item[0])
            if not chunks:
                continue
            selected_text = "\n".join(text for _idx, text, _metadata in chunks[:max_chunks_per_file]).strip()
            if len(selected_text) > max_chars_per_file:
                selected_text = f"{selected_text[:max_chars_per_file]}\n... <truncated>"
            metadata = chunks[0][2] if chunks else {}
            blocks.append(
                f"[{rank}] source={file_name}, file_id={file_id}, status={status}, chunks={chunk_count}\n"
                f"{selected_text}"
            )
            final_docs.append(
                {
                    "rank": rank,
                    "score": None,
                    "source": file_name,
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "content": selected_text,
                    "preview": selected_text[:240].replace("\n", " "),
                }
            )
            rank += 1

        debug["final"] = final_docs
        debug["timing_ms"]["total"] = round((time.perf_counter() - start_time) * 1000, 2)
        return "\n\n".join(blocks), debug
    except Exception as exc:
        logger.error(f"Failed to build document overview for session {session_id}: {exc}")
        debug["error"] = str(exc)
        debug["timing_ms"]["total"] = round((time.perf_counter() - start_time) * 1000, 2)
        return "", debug


def _session_document_fingerprint(session_id: str) -> Tuple[Tuple[Any, ...], ...]:
    rows = get_session_documents(session_id)
    return tuple(
        sorted(
            (
                file_id,
                file_name,
                int(chunk_count),
                status,
                str(uploaded_at),
                file_path,
            )
            for file_id, file_name, chunk_count, status, uploaded_at, file_path in rows
        )
    )


def _rrf_fuse(dense_docs: List[Document], sparse_docs: List[Document], rrf_k: int = RRF_K) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) implementation for hybrid search.
    """
    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_docs in (dense_docs, sparse_docs):
        for rank, doc in enumerate(ranked_docs, start=1):
            key = _doc_identity(doc)
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    ranked_keys = sorted(score_map.keys(), key=lambda k: score_map[k], reverse=True)
    return [doc_map[k] for k in ranked_keys]


def _rrf_fuse_with_scores(
    dense_docs: List[Document], sparse_docs: List[Document], rrf_k: int = RRF_K
) -> Tuple[List[Tuple[float, Document]], Dict[str, float]]:
    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_docs in (dense_docs, sparse_docs):
        for rank, doc in enumerate(ranked_docs, start=1):
            key = _doc_identity(doc)
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    ranked_keys = sorted(score_map.keys(), key=lambda k: score_map[k], reverse=True)
    return [(score_map[k], doc_map[k]) for k in ranked_keys], score_map


def _query_token_count(query: str) -> int:
    return len(re.findall(r"[\w.-]+", query, flags=re.UNICODE))


def _looks_like_exact_lookup(query: str) -> bool:
    stripped = query.strip()
    if not stripped:
        return False
    token_count = _query_token_count(stripped)
    lowered = stripped.lower()
    has_identifier_shape = bool(re.search(r"[_#./\\-]|\b[a-f0-9]{8,}\b|\d", stripped))
    has_lookup_word = any(word in lowered for word in ("marker", "id", "file", "source", "chunk"))
    has_file_ext = bool(re.search(r"\.(md|txt|pdf|docx?)\b", lowered))
    return token_count <= RERANK_SKIP_SHORT_QUERY_TOKENS and (has_identifier_shape or has_lookup_word or has_file_ext)


def _should_skip_rerank(query: str, candidates_with_scores: List[Tuple[float, Document]]) -> Tuple[bool, str]:
    if not RERANK_SKIP_ENABLED:
        return False, ""
    if not candidates_with_scores:
        return True, "no_candidates"
    if len(candidates_with_scores) <= RERANK_TOP_K:
        return True, "candidate_count_at_or_below_top_k"
    if _looks_like_exact_lookup(query):
        return True, "short_exact_lookup_query"

    if len(candidates_with_scores) >= 2:
        top_score = float(candidates_with_scores[0][0])
        second_score = float(candidates_with_scores[1][0])
        if second_score > 0:
            abs_margin = top_score - second_score
            ratio = top_score / second_score
            if abs_margin >= RERANK_SKIP_RRF_ABS_MARGIN and ratio >= RERANK_SKIP_RRF_RATIO:
                return True, "rrf_top_score_dominates"
    return False, ""


def rerank(query: str, docs: List[Document]) -> List[Tuple[float, Document]]:
    """
    Rerank documents using a CrossEncoder model.
    """
    if not docs:
        return []

    model = get_rerank_model()
    pairs = [[query, doc.page_content[:1200]] for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
    return [(float(score), doc) for score, doc in ranked]


class HybridRRFRetriever:
    """
    Hybrid retriever combining dense (vector) and sparse (BM25) search with RRF.
    """
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def invoke(self, query: str) -> List[Document]:
        docs, _ = self.search(query)
        return docs

    def search(self, query: str) -> Tuple[List[Document], Dict[str, Any]]:
        start_time = time.perf_counter()
        debug: Dict[str, Any] = {
            "query": query,
            "dense": [],
            "bm25": [],
            "rrf": [],
            "candidate_count": 0,
            "timing_ms": {},
            "error": "",
        }
        try:
            dense_start = time.perf_counter()
            dense_docs = self.vector_retriever.invoke(query)
            debug["timing_ms"]["dense"] = round((time.perf_counter() - dense_start) * 1000, 2)

            sparse_start = time.perf_counter()
            sparse_docs = self.bm25_retriever.invoke(query)
            debug["timing_ms"]["bm25"] = round((time.perf_counter() - sparse_start) * 1000, 2)

            rrf_start = time.perf_counter()
            fused_with_scores, _ = _rrf_fuse_with_scores(dense_docs, sparse_docs)
            debug["timing_ms"]["rrf"] = round((time.perf_counter() - rrf_start) * 1000, 2)
            fused_docs = [doc for _, doc in fused_with_scores]

            debug["dense"] = [_doc_debug(doc, i + 1) for i, doc in enumerate(dense_docs)]
            debug["bm25"] = [_doc_debug(doc, i + 1) for i, doc in enumerate(sparse_docs)]
            debug["rrf"] = [
                _doc_debug(doc, i + 1, float(score)) for i, (score, doc) in enumerate(fused_with_scores)
            ]
            debug["_rrf_candidates_with_scores"] = fused_with_scores[:RERANK_CANDIDATES]
            debug["candidate_count"] = len(debug["_rrf_candidates_with_scores"])
            debug["timing_ms"]["total_retrieval"] = round((time.perf_counter() - start_time) * 1000, 2)
            return fused_docs[:RERANK_CANDIDATES], debug
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            debug["error"] = str(e)
            return [], debug


def retrieve_with_rerank(query: str, retriever: "HybridRRFRetriever") -> List[Tuple[float, Document]]:
    """
    Perform hybrid retrieval followed by reranking and filtering.
    """
    reranked, _ = retrieve_with_rerank_debug(query, retriever)
    return reranked


def retrieve_with_rerank_debug(
    query: str, retriever: "HybridRRFRetriever"
) -> Tuple[List[Tuple[float, Document]], Dict[str, Any]]:
    """
    Perform hybrid retrieval followed by reranking and return a debug trace.
    """
    start_time = time.perf_counter()
    candidates, debug = retriever.search(query)
    candidates_with_scores = debug.pop("_rrf_candidates_with_scores", [])
    skip_rerank, skip_reason = _should_skip_rerank(query, candidates_with_scores)

    if skip_rerank:
        reranked = [(float(score), doc) for score, doc in candidates_with_scores]
        debug.setdefault("timing_ms", {})["rerank"] = 0.0
        debug["rerank_skipped"] = True
        debug["rerank_skip_reason"] = skip_reason
        debug["rerank_count"] = 0
        debug["rerank"] = []
        debug["timing_ms"]["total"] = round((time.perf_counter() - start_time) * 1000, 2)
    else:
        rerank_start = time.perf_counter()
        reranked = rerank(query, candidates)
        debug.setdefault("timing_ms", {})["rerank"] = round((time.perf_counter() - rerank_start) * 1000, 2)
        debug["rerank_skipped"] = False
        debug["rerank_skip_reason"] = ""
        debug["rerank_count"] = len(reranked)
        debug["rerank"] = [
            _doc_debug(doc, i + 1, float(score)) for i, (score, doc) in enumerate(reranked)
        ]
        debug["timing_ms"]["total"] = round((time.perf_counter() - start_time) * 1000, 2)

    if not reranked:
        debug["final"] = []
        return [], debug

    filtered = [x for x in reranked if x[0] >= RERANK_MIN_SCORE]
    if not filtered:
        filtered = reranked
    final_docs = filtered[:RERANK_TOP_K]
    debug["final"] = [
        _doc_debug(doc, i + 1, float(score)) for i, (score, doc) in enumerate(final_docs)
    ]
    return final_docs, debug


def load_retriever(session_id: str) -> Tuple[Optional[HybridRRFRetriever], str]:
    """
    Load vector and BM25 retrievers for a specific session.
    """
    persist_dir = f"{DATA_DIR}/{session_id}/chroma"
    if not os.path.exists(persist_dir):
        return None, ""

    try:
        fingerprint = _session_document_fingerprint(session_id)
        cached = _RETRIEVER_CACHE.get(session_id)
        if cached and cached.get("fingerprint") == fingerprint:
            _CACHE_STATS["hits"] += 1
            return cached["retriever"], cached["summary"]

        _CACHE_STATS["misses"] += 1
        embeddings = get_embedding_model()
        vectorstore = ComponentFactory.get_component(
            "vectorstores",
            "Chroma",
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"collection_{session_id}"
        )

        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": VECTOR_SEARCH_K,
                "fetch_k": VECTOR_FETCH_K,
                "lambda_mult": VECTOR_MMR_LAMBDA
            }
        )

        stored = vectorstore.get(include=["documents", "metadatas"])
        all_texts = stored.get("documents", [])
        all_metadatas = stored.get("metadatas", []) or [{} for _ in all_texts]
        all_ids = stored.get("ids", []) or [None for _ in all_texts]
        if not all_texts:
            return None, ""

        bm25_docs = []
        for text, metadata, doc_id in zip(all_texts, all_metadatas, all_ids):
            safe_metadata = dict(metadata or {})
            if doc_id and "chunk_id" not in safe_metadata:
                safe_metadata["chunk_id"] = doc_id
            bm25_docs.append(Document(page_content=text, metadata=safe_metadata))

        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = BM25_K

        hybrid_retriever = HybridRRFRetriever(vector_retriever, bm25_retriever)
        summary = get_session_summary(session_id)
        _RETRIEVER_CACHE[session_id] = {
            "fingerprint": fingerprint,
            "retriever": hybrid_retriever,
            "summary": summary,
            "doc_count": len(bm25_docs),
            "created_at": time.time(),
        }
        return hybrid_retriever, summary
    except Exception as e:
        logger.error(f"Failed to load retriever for session {session_id}: {e}")
        return None, ""
