from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_text_documents(file_path: str, file_name: str) -> List[Document]:
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": file_name,
                "file_type": Path(file_name).suffix.lower().lstrip(".") or "text",
                "chunk_kind": "text",
            },
        )
    ]
