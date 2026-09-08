from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

from langchain_core.documents import Document

from config import COVER_MAX_PAGES, HEADER_FOOTER_REPEAT_RATIO, TOC_MAX_PAGES


def _normalized_repeat_text(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip().casefold()


def _looks_like_toc(lines: List[str]) -> bool:
    joined = "\n".join(lines[:80])
    if re.search(r"(?:^|\n)\s*(目录|contents?)\s*(?:\n|$)", joined, re.IGNORECASE):
        return True
    entries = sum(bool(re.search(r"\.{2,}\s*\d+\s*$|\s{3,}\d+\s*$", line)) for line in lines)
    return entries >= 3


def _page_role(page_number: int, lines: List[str]) -> str:
    joined = "\n".join(lines)
    if page_number <= TOC_MAX_PAGES and _looks_like_toc(lines):
        return "toc"
    if re.search(r"(?:^|\n)\s*(附录|appendix)\b", joined, re.IGNORECASE):
        return "appendix"
    if page_number <= COVER_MAX_PAGES and len(joined) < 1200 and len(lines) <= 25:
        return "cover"
    return "body"


def _document_properties(pdf: Any) -> Dict[str, str]:
    raw = pdf.metadata or {}
    mapping = {
        "title": "title", "author": "author", "subject": "subject", "creator": "creator",
        "producer": "producer", "creationDate": "creation_date", "modDate": "modification_date",
    }
    return {f"document_{target}": str(raw.get(source) or "").strip() for source, target in mapping.items()
            if str(raw.get(source) or "").strip()}


def load_pdf_documents(file_path: str, file_name: str) -> List[Document]:
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for structure-aware PDF parsing.") from exc

    pdf = fitz.open(file_path)
    pages: List[Dict[str, Any]] = []
    repeat_counts: Counter[str] = Counter()
    for page_index, page in enumerate(pdf):
        height = float(page.rect.height or 1)
        raw_blocks = page.get_text("dict").get("blocks", [])
        blocks = []
        for raw in raw_blocks:
            if raw.get("type") != 0:
                continue
            spans = [span for line in raw.get("lines", []) for span in line.get("spans", []) if span.get("text", "").strip()]
            text = "\n".join(
                "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                for line in raw.get("lines", [])
            ).strip()
            if not text:
                continue
            bbox = [round(float(value), 2) for value in raw.get("bbox", (0, 0, 0, 0))]
            font_size = max((float(span.get("size", 0)) for span in spans), default=0)
            block = {"text": text, "bbox": bbox, "font_size": font_size, "height": height}
            blocks.append(block)
            if bbox[1] <= height * 0.12 or bbox[3] >= height * 0.88:
                repeat_counts[_normalized_repeat_text(text)] += 1
        pages.append({"blocks": blocks, "image_count": len(page.get_images(full=True))})

    threshold = max(2, int(len(pages) * HEADER_FOOTER_REPEAT_RATIO + 0.999))
    repeated = {text for text, count in repeat_counts.items() if text and count >= threshold}
    properties = _document_properties(pdf)
    docs: List[Document] = []
    section_path = ""
    for page_index, page_data in enumerate(pages, start=1):
        blocks = page_data["blocks"]
        lines = [line.strip() for block in blocks for line in block["text"].splitlines() if line.strip()]
        role = _page_role(page_index, lines)
        body_sizes = [block["font_size"] for block in blocks if block["font_size"] > 0]
        typical_size = median(body_sizes) if body_sizes else 0
        printed_page = ""
        for block_index, block in enumerate(blocks):
            text = block["text"]
            bbox = block["bbox"]
            normalized = _normalized_repeat_text(text)
            at_top = bbox[1] <= block["height"] * 0.12
            at_bottom = bbox[3] >= block["height"] * 0.88
            if at_bottom and re.fullmatch(r"\s*(?:第\s*)?[ivxlcdm\d]+(?:\s*页)?\s*", text, re.IGNORECASE):
                printed_page = text.strip()
            if normalized in repeated and at_top:
                kind = "header"
            elif normalized in repeated and at_bottom:
                kind = "footer"
            elif typical_size and block["font_size"] >= typical_size * 1.25 and len(text) <= 160:
                kind = "heading"
                section_path = text.replace("\n", " ").strip()
            elif at_bottom and len(text) <= 240:
                kind = "footnote"
            else:
                kind = "page_text"
            tier = "secondary" if role in {"cover", "toc"} or kind in {"header", "footer"} else "primary"
            docs.append(Document(page_content=text, metadata={
                "source": file_name, "file_type": "pdf", "chunk_kind": kind,
                "page": page_index, "page_count": len(pages), "printed_page": printed_page,
                "page_role": role, "retrieval_tier": tier, "section_path": section_path,
                "bbox": bbox, "font_size": block["font_size"], "pre_chunked": True,
                "image_count": page_data["image_count"] if block_index == 0 else 0, **properties,
            }))
    pdf.close()
    if not docs and pages:
        raise ValueError(
            f"{Path(file_name).name} contains no extractable text. It may be scanned; OCR is required."
        )
    return docs
