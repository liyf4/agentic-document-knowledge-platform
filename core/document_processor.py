import json
import os
import threading
from typing import Any, List, Tuple

import pandas as pd

from utils.ssl_compat import prefer_certifi_default_context

prefer_certifi_default_context()

from langchain_core.documents import Document
from loguru import logger

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DOCUMENT_STRUCTURE_VERSION,
    LLM_MODEL_NAME,
    LLM_TYPE,
    MAX_CHUNKS_PER_FILE,
    MAX_UPLOAD_MB,
    METADATA_ENABLED,
    METADATA_EXTRACTOR_VERSION,
    METADATA_LLM_EXTRACTION_ENABLED,
    METADATA_LLM_MAX_RETRIES,
    METADATA_LLM_TIMEOUT_SECONDS,
    ZAI_API_BASE,
)
from core.factory import ComponentFactory
from core.document_loaders import load_documents, supported_extensions
from core.document_analysis import build_document_analysis
from core.persistent_storage import StorageValidationError, get_persistent_file_store
from core.retriever import get_embedding_model, invalidate_retriever_cache
from db.file_manager import (
    delete_document_analysis,
    delete_uploaded_file,
    get_uploaded_file,
    replace_document_analysis,
    update_uploaded_file,
)
from db.session_manager import (
    delete_session_document,
    update_session_summary,
    upsert_session_document,
)


class DocumentProcessor:
    """Load, split, index, summarize, and reindex uploaded documents."""

    SUPPORTED_EXTENSIONS = set(supported_extensions())

    @staticmethod
    def _metadata_failure_status(exc: Exception) -> str:
        text = str(exc).casefold()
        type_name = type(exc).__name__.casefold()
        return "llm_timeout" if (
            isinstance(exc, TimeoutError) or "timeout" in type_name
            or "timeout" in text or "timed out" in text
        ) else "partial"

    @staticmethod
    def _max_upload_bytes() -> int:
        return int(float(MAX_UPLOAD_MB) * 1024 * 1024)

    @staticmethod
    def _validate_file_size(file_bytes: bytes, file_name: str) -> None:
        max_bytes = DocumentProcessor._max_upload_bytes()
        if len(file_bytes) > max_bytes:
            actual_mb = len(file_bytes) / 1024 / 1024
            raise ValueError(f"{file_name} is {actual_mb:.2f} MB; max upload size is {MAX_UPLOAD_MB} MB")

    @staticmethod
    def _validate_file_type(file_name: str) -> None:
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext not in DocumentProcessor.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(DocumentProcessor.SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported file type: {file_ext or '<none>'}. Supported types: {supported}")

    @staticmethod
    def _validate_chunk_count(splits: List[Document], file_name: str) -> None:
        if len(splits) > int(MAX_CHUNKS_PER_FILE):
            raise ValueError(
                f"{file_name} produced {len(splits)} chunks; max chunks per file is {MAX_CHUNKS_PER_FILE}"
            )

    @staticmethod
    def _is_pre_chunked(docs: List[Document]) -> bool:
        return bool(docs) and all(bool((doc.metadata or {}).get("pre_chunked")) for doc in docs)

    @staticmethod
    def _sanitize_metadata_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        return {
            key: sanitized
            for key, value in metadata.items()
            if (sanitized := DocumentProcessor._sanitize_metadata_value(value)) is not None
        }

    @staticmethod
    def get_loader(file_path: str):
        raise NotImplementedError("Use core.document_loaders.load_documents instead.")

    @staticmethod
    def process_file(uploaded_file, session_id: str) -> Tuple[bool, str]:
        try:
            record = get_persistent_file_store().persist_upload(session_id, uploaded_file.name, uploaded_file.getvalue())
            return DocumentProcessor.process_stored_file(session_id, record["file_id"])
        except Exception as exc:
            logger.exception(f"Failed to process file {uploaded_file.name}")
            return False, f"Process failed: {exc}"

    @staticmethod
    def process_stored_file(session_id: str, file_id: str) -> Tuple[bool, str]:
        """Index a durable original without copying it into a temporary workspace."""
        record = get_uploaded_file(session_id, file_id)
        if not record:
            return False, "Uploaded file metadata was not found."
        file_path = record["storage_path"]
        if not os.path.isfile(file_path):
            update_uploaded_file(file_id, parse_status="failed", index_status="failed")
            return False, "Original file is unavailable."
        update_uploaded_file(file_id, parse_status="processing", index_status="processing")
        try:
            splits = DocumentProcessor._index_file(
                file_path=file_path,
                file_name=record["original_name"],
                session_id=session_id,
                file_id=file_id,
                file_hash=record["sha256"],
                replace_existing=True,
            )
            updates = {"chunk_count": len(splits), "parse_status": "completed", "index_status": "completed"}
            if record["category"] == "table":
                updates["table_profile"] = DocumentProcessor._table_profile(file_path, record["original_name"])
            update_uploaded_file(file_id, **updates)
            # Compatibility projection for installations that still read session_documents.
            upsert_session_document(
                session_id=session_id, file_id=file_id, file_name=record["original_name"],
                file_hash=record["sha256"], chunk_count=len(splits), status="completed", file_path=file_path,
            )
            invalidate_retriever_cache(session_id)
            DocumentProcessor._start_optional_enrichment(
                splits=splits, session_id=session_id, file_id=file_id, file_path=file_path,
                file_name=record["original_name"], api_key=os.getenv("ZAI_API_KEY"),
            )
            return True, f"Document indexed with {len(splits)} chunks."
        except Exception as exc:
            updates = {"parse_status": "failed", "index_status": "failed"}
            if "OCR is required" in str(exc) or "scanned" in str(exc).casefold():
                updates["ocr_status"] = "required"
            update_uploaded_file(file_id, **updates)
            if METADATA_ENABLED:
                try:
                    replace_document_analysis(
                        session_id, file_id,
                        {"file_name": record["original_name"], "ocr_required": updates.get("ocr_status") == "required"},
                        [], [], extraction_status="failed", extraction_error=str(exc),
                        extractor_version=METADATA_EXTRACTOR_VERSION,
                        structure_version=DOCUMENT_STRUCTURE_VERSION,
                    )
                except Exception as metadata_exc:
                    logger.warning(f"Could not persist failed metadata state for {file_id}: {metadata_exc}")
            logger.exception(f"Failed to index stored file {file_id}")
            return False, f"Process failed: {exc}"

    @staticmethod
    def _table_profile(file_path: str, file_name: str) -> dict:
        """Small metadata preview kept outside the sandbox; heavy work stays in it."""
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".csv":
            frame = pd.read_csv(file_path)
        elif ext == ".xlsx":
            frame = pd.read_excel(file_path, engine="openpyxl")
        elif ext == ".parquet":
            frame = pd.read_parquet(file_path)
        else:
            return {}
        return {
            "columns": [str(column) for column in frame.columns[:100]],
            "row_count": int(len(frame)), "column_count": int(len(frame.columns)),
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "missing": {str(column): int(value) for column, value in frame.isna().sum().items()},
            "sample_rows": frame.head(5).where(frame.notna(), None).to_dict(orient="records"),
        }

    @staticmethod
    def _index_file(
        file_path: str,
        file_name: str,
        session_id: str,
        file_id: str,
        file_hash: str,
        replace_existing: bool = False,
    ) -> List[Document]:
        logger.info(f"Loading file: {file_name}")
        docs = load_documents(file_path, file_name)
        if not docs:
            raise ValueError("empty file")

        if DocumentProcessor._is_pre_chunked(docs):
            logger.info(f"Using {len(docs)} pre-chunked documents for session: {session_id}")
            splits = docs
        else:
            logger.info(f"Splitting documents for session: {session_id}")
            splitter = ComponentFactory.get_component(
                "splitters",
                "RecursiveCharacterTextSplitter",
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            splits = splitter.split_documents(docs)
        DocumentProcessor._validate_chunk_count(splits, file_name)
        for idx, split in enumerate(splits):
            split.metadata = dict(split.metadata or {})
            split.metadata["source"] = file_name
            split.metadata["session_id"] = session_id
            split.metadata["file_id"] = file_id
            split.metadata["file_hash"] = file_hash
            split.metadata["chunk_index"] = idx
            split.metadata["chunk_id"] = f"{file_id}_{idx}"
        current_heading_id = ""
        for idx, split in enumerate(splits):
            split.metadata["previous_chunk_id"] = f"{file_id}_{idx - 1}" if idx > 0 else ""
            split.metadata["next_chunk_id"] = f"{file_id}_{idx + 1}" if idx + 1 < len(splits) else ""
            if split.metadata.get("chunk_kind") == "heading":
                current_heading_id = split.metadata["chunk_id"]
                split.metadata["parent_chunk_id"] = ""
            else:
                split.metadata["parent_chunk_id"] = current_heading_id
            split.metadata = DocumentProcessor._sanitize_metadata(split.metadata)

        persist_dir = f"{DATA_DIR}/{session_id}/chroma"
        embeddings = get_embedding_model()
        vectorstore = ComponentFactory.get_component(
            "vectorstores",
            "Chroma",
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"collection_{session_id}",
        )

        if replace_existing:
            stored = vectorstore.get(where={"file_id": file_id})
            ids = stored.get("ids", [])
            if ids:
                vectorstore.delete(ids=ids)

        logger.info(f"Indexing {len(splits)} chunks into Chroma at {persist_dir}")
        vectorstore.add_documents(
            documents=splits,
            ids=[split.metadata["chunk_id"] for split in splits],
        )
        if METADATA_ENABLED:
            try:
                metadata, facts, blocks = build_document_analysis(
                    splits, file_path=file_path, file_name=file_name, file_id=file_id, session_id=session_id,
                    include_llm=False,
                )
                pending_llm = bool(METADATA_LLM_EXTRACTION_ENABLED and os.getenv("ZAI_API_KEY"))
                replace_document_analysis(
                    session_id, file_id, metadata, facts, blocks,
                    extraction_status="partial" if pending_llm else "completed",
                    extractor_version=METADATA_EXTRACTOR_VERSION,
                    structure_version=DOCUMENT_STRUCTURE_VERSION,
                )
            except Exception as analysis_exc:
                logger.warning(f"Document metadata extraction failed for {file_id}: {analysis_exc}")
                replace_document_analysis(
                    session_id, file_id, {}, [], [], extraction_status="failed",
                    extraction_error=str(analysis_exc), extractor_version=METADATA_EXTRACTOR_VERSION,
                    structure_version=DOCUMENT_STRUCTURE_VERSION,
                )
        return splits

    @staticmethod
    def _start_optional_enrichment(
        *, splits: List[Document], session_id: str, file_id: str, file_path: str,
        file_name: str, api_key: str | None,
    ) -> None:
        """Run optional metadata LLM and summary after the document is queryable."""
        if not api_key:
            return

        def run() -> None:
            if METADATA_ENABLED and METADATA_LLM_EXTRACTION_ENABLED:
                try:
                    metadata, facts, blocks = build_document_analysis(
                        splits, file_path=file_path, file_name=file_name, file_id=file_id,
                        session_id=session_id, include_llm=True, raise_on_llm_error=True,
                    )
                    replace_document_analysis(
                        session_id, file_id, metadata, facts, blocks, extraction_status="completed",
                        extractor_version=METADATA_EXTRACTOR_VERSION,
                        structure_version=DOCUMENT_STRUCTURE_VERSION,
                    )
                except Exception as exc:
                    status = DocumentProcessor._metadata_failure_status(exc)
                    logger.warning(f"Optional metadata enrichment ended with status={status} for {file_id}: {exc}")
                    # Preserve deterministic facts and blocks; update only the existing status/error fields.
                    from db.file_manager import update_document_metadata_status
                    update_document_metadata_status(file_id, status, str(exc))
            try:
                DocumentProcessor._generate_summary(splits, session_id, api_key)
            except Exception as summary_exc:
                logger.warning(f"Summary generation skipped after indexing {file_name}: {summary_exc}")

        threading.Thread(target=run, name=f"metadata-enrichment-{file_id[:8]}", daemon=True).start()

    @staticmethod
    def _generate_summary(splits: List[Document], session_id: str, api_key: str):
        """Generate a short summary for the uploaded documents."""
        if not api_key:
            return

        try:
            preview_text = "\n".join([doc.page_content for doc in splits[:4]])[:2000]
            summary_llm = ComponentFactory.get_component(
                "llm",
                LLM_TYPE,
                api_key=api_key,
                base_url=ZAI_API_BASE,
                model_name=LLM_MODEL_NAME,
                timeout=METADATA_LLM_TIMEOUT_SECONDS,
                max_retries=METADATA_LLM_MAX_RETRIES,
            )
            summary = summary_llm.invoke(f"Summarize this document in 300 Chinese characters or fewer:\n{preview_text}").content
            update_session_summary(session_id, summary)
            logger.info(f"Summary generated for session {session_id}")
        except Exception as exc:
            logger.warning(f"Failed to generate summary: {exc}")


def process_file(uploaded_file, session_id):
    """Legacy compatibility wrapper."""
    return DocumentProcessor.process_file(uploaded_file, session_id)


def delete_document(session_id: str, file_id: str) -> Tuple[bool, str]:
    persist_dir = f"{DATA_DIR}/{session_id}/chroma"
    doc_meta = get_uploaded_file(session_id, file_id)
    if not os.path.exists(persist_dir):
        delete_document_analysis(file_id)
        if doc_meta:
            get_persistent_file_store().remove_upload(doc_meta)
        delete_session_document(session_id, file_id)
        invalidate_retriever_cache(session_id)
        return True, "Document metadata deleted; vector store does not exist."

    try:
        embeddings = get_embedding_model()
        vectorstore = ComponentFactory.get_component(
            "vectorstores",
            "Chroma",
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"collection_{session_id}",
        )
        stored = vectorstore.get(where={"file_id": file_id})
        ids = stored.get("ids", [])
        if ids:
            vectorstore.delete(ids=ids)

        delete_document_analysis(file_id)
        if doc_meta:
            get_persistent_file_store().remove_upload(doc_meta)
        delete_session_document(session_id, file_id)
        invalidate_retriever_cache(session_id)
        return True, f"Deleted document and {len(ids)} vector chunks."
    except Exception as exc:
        logger.error(f"Failed to delete document {file_id}: {exc}")
        return False, f"Delete failed: {exc}"


def reindex_document(session_id: str, file_id: str) -> Tuple[bool, str]:
    doc_meta = get_uploaded_file(session_id, file_id)
    if not doc_meta:
        return False, "Document metadata not found."

    file_path = doc_meta.get("storage_path", "")
    if not file_path or not os.path.exists(file_path):
        return False, "Original file is unavailable, so this document cannot be reindexed."

    try:
        splits = DocumentProcessor._index_file(
            file_path=file_path,
            file_name=doc_meta["original_name"],
            session_id=session_id,
            file_id=file_id,
            file_hash=doc_meta["sha256"],
            replace_existing=True,
        )
        update_uploaded_file(file_id, chunk_count=len(splits), parse_status="completed", index_status="completed")
        upsert_session_document(session_id, file_id, doc_meta["original_name"], doc_meta["sha256"], len(splits), "completed", file_path)
        invalidate_retriever_cache(session_id)
        return True, f"Reindexed {len(splits)} chunks."
    except Exception as exc:
        logger.error(f"Failed to reindex document {file_id}: {exc}")
        return False, f"Reindex failed: {exc}"
