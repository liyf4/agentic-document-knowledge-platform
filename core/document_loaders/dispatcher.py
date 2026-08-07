from pathlib import Path
from typing import List

from langchain_core.documents import Document

from config import LOADER_TYPE
from core.document_loaders.pdf_loader import load_pdf_documents
from core.document_loaders.table_loader import load_table_documents
from core.document_loaders.text_loader import load_text_documents
from core.document_loaders.word_loader import load_docx_documents
from core.factory import ComponentFactory


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".doc", ".docx", ".csv", ".xlsx", ".parquet"}


def supported_extensions() -> List[str]:
    return sorted(SUPPORTED_EXTENSIONS)


def load_documents(file_path: str, file_name: str) -> List[Document]:
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext or '<none>'}")
    if ext in {".txt", ".md", ".markdown"}:
        return load_text_documents(file_path, file_name)
    if ext == ".pdf":
        return load_pdf_documents(file_path, file_name)
    if ext == ".docx":
        return load_docx_documents(file_path, file_name)
    if ext in {".csv", ".xlsx", ".parquet"}:
        return load_table_documents(file_path, file_name)

    loader = ComponentFactory.get_component("loaders", LOADER_TYPE, file_path=file_path)
    return loader.load()
