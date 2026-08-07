from pathlib import Path
from typing import List

from langchain_core.documents import Document


def _table_to_markdown(table) -> str:
    rows = []
    for row in table.rows:
        rows.append([cell.text.strip().replace("\n", " ") for cell in row.cells])
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""

    max_width = max(len(row) for row in rows)
    normalized = [row + [""] * (max_width - len(row)) for row in rows]
    header = normalized[0]
    separator = ["---"] * max_width
    body = normalized[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def load_docx_documents(file_path: str, file_name: str) -> List[Document]:
    from docx import Document as DocxDocument

    docx = DocxDocument(file_path)
    docs: List[Document] = []

    paragraph_parts = []
    for index, paragraph in enumerate(docx.paragraphs):
        text = paragraph.text.strip()
        if not text:
            continue
        style = paragraph.style.name if paragraph.style else ""
        paragraph_parts.append(text)
        if style.lower().startswith("heading"):
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "file_type": "docx",
                        "chunk_kind": "heading",
                        "paragraph_index": index,
                        "style": style,
                    },
                )
            )

    if paragraph_parts:
        docs.append(
            Document(
                page_content="\n\n".join(paragraph_parts),
                metadata={
                    "source": file_name,
                    "file_type": "docx",
                    "chunk_kind": "paragraphs",
                },
            )
        )

    for table_index, table in enumerate(docx.tables):
        table_text = _table_to_markdown(table)
        if not table_text:
            continue
        docs.append(
            Document(
                page_content=f"Table {table_index + 1}\n{table_text}",
                metadata={
                    "source": file_name,
                    "file_type": "docx",
                    "chunk_kind": "table",
                    "table_index": table_index + 1,
                },
            )
        )

    if not docs:
        raise ValueError(f"{Path(file_name).name} contains no readable text.")
    return docs
