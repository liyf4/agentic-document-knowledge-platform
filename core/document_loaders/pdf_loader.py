from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf_documents(file_path: str, file_name: str) -> List[Document]:
    reader = PdfReader(file_path)
    docs: List[Document] = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "file_type": "pdf",
                    "chunk_kind": "page_text",
                    "page": page_index + 1,
                    "page_count": len(reader.pages),
                },
            )
        )

    if not docs and reader.pages:
        raise ValueError(
            f"{Path(file_name).name} contains no extractable text. "
            "It may be a scanned/image-only PDF; OCR is not enabled."
        )
    return docs
