from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOP_LEVEL_BULLET_RE = re.compile(r"^-\s+\*\*(.+?)\*\*\s*(?::|：)?")


def _markdown_documents(text: str, file_name: str) -> List[Document]:
    """Split Markdown on headings and semantic top-level list subjects.

    A bold top-level bullet is treated as a subject boundary. This keeps
    relations such as SQL-injection defenses separate from adjacent XSS
    defenses even when both appear below the same Markdown heading.
    """
    heading_stack: Dict[int, str] = {}
    documents: List[Document] = []
    buffer: List[str] = []
    buffer_path = ""
    buffer_kind = "text"

    def current_heading_path() -> str:
        return " > ".join(heading_stack[level] for level in sorted(heading_stack))

    def flush() -> None:
        nonlocal buffer, buffer_path, buffer_kind
        content = "\n".join(buffer).strip()
        if content:
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": file_name,
                    "file_type": "md",
                    "chunk_kind": buffer_kind,
                    "section_path": buffer_path,
                    "retrieval_tier": "primary",
                    "pre_chunked": True,
                },
            ))
        buffer = []
        buffer_path = current_heading_path()
        buffer_kind = "text"

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            for existing in [value for value in heading_stack if value >= level]:
                heading_stack.pop(existing, None)
            heading_stack[level] = title
            path = current_heading_path()
            documents.append(Document(
                page_content=line,
                metadata={
                    "source": file_name, "file_type": "md", "chunk_kind": "heading",
                    "section_path": path, "retrieval_tier": "primary", "pre_chunked": True,
                },
            ))
            buffer_path = path
            continue

        subject = _TOP_LEVEL_BULLET_RE.match(line)
        if subject:
            flush()
            subject_name = subject.group(1).strip()
            base = current_heading_path()
            buffer_path = f"{base} > {subject_name}" if base else subject_name
            buffer_kind = "list_item"
            buffer = [line]
            continue

        if line.strip() == "---":
            flush()
            continue
        if not buffer and line.strip():
            buffer_path = current_heading_path()
        buffer.append(line)

    flush()
    return documents


def load_text_documents(file_path: str, file_name: str) -> List[Document]:
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    suffix = Path(file_name).suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _markdown_documents(text, file_name)
    return [
        Document(
            page_content=text,
            metadata={
                "source": file_name,
                "file_type": suffix.lstrip(".") or "text",
                "chunk_kind": "text",
            },
        )
    ]
