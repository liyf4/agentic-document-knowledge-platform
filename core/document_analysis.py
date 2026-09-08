"""Structured document metadata, block, and fact extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from loguru import logger

from config import (
    DOCUMENT_STRUCTURE_VERSION,
    LLM_MODEL_NAME,
    LLM_TYPE,
    METADATA_EXTRACTOR_VERSION,
    METADATA_LLM_EXTRACTION_CHARS_PER_CHUNK,
    METADATA_LLM_EXTRACTION_ENABLED,
    METADATA_LLM_EXTRACTION_INPUT_CHARS,
    METADATA_LLM_EXTRACTION_MAX_CHUNKS,
    METADATA_LLM_EXTRACTION_MAX_RETRIES,
    METADATA_LLM_EXTRACTION_MAX_TOKENS,
    METADATA_LLM_EXTRACTION_TIMEOUT_SECONDS,
    ZAI_API_BASE,
)
from core.factory import ComponentFactory


FACT_SCHEMA_TYPES = {
    "amount", "percentage", "date", "duration", "person", "organization", "address",
    "phone", "email", "contract_number", "invoice_number", "project_number", "product_model",
    "quantity", "payment_term", "signature", "metric",
}

_AMOUNT_RE = re.compile(
    r"(?P<prefix>人民币|RMB|CNY|USD|EUR|美元|欧元|￥|¥|\$|€)?\s*"
    r"(?P<value>\d{1,3}(?:[,，]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>亿元|万元|千元|元|million|billion|万|千)?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"(?:百分之\s*(?P<prefix_value>\d+(?:\.\d+)?)|"
    r"(?P<suffix_value>\d+(?:\.\d+)?)\s*"
    r"(?:%|％|percent(?:age)?|per[\s-]*cent|pct\.?)(?![A-Za-z]))",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?:\d{4}[年./-]\d{1,2}[月./-]\d{1,2}日?|\d{4}年\d{1,2}月|\d{4}-\d{1,2}-\d{1,2})"
)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?(?:1[3-9]\d{9}|0\d{2,3}[- ]?\d{7,8})(?!\d)")
_IDENTIFIER_PATTERNS = {
    "contract_number": re.compile(r"(?:合同|协议)(?:编号|号)\s*[:：]?\s*([A-Za-z0-9._/-]+)", re.IGNORECASE),
    "invoice_number": re.compile(r"(?:发票)(?:编号|号码|号)\s*[:：]?\s*([A-Za-z0-9._/-]+)", re.IGNORECASE),
    "project_number": re.compile(r"(?:项目)(?:编号|号码|号)\s*[:：]?\s*([A-Za-z0-9._/-]+)", re.IGNORECASE),
}


def _stable_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _context(text: str, start: int, end: int, radius: int = 80) -> str:
    return text[max(0, start - radius):min(len(text), end + radius)].replace("\n", " ").strip()


def _amount_qualifier(context: str) -> str:
    labels = (
        ("tax", ("税额", "税款")), ("tax_inclusive_total", ("含税总额", "价税合计")),
        ("paid", ("已支付", "已付款")), ("unit_price", ("单价",)),
        ("contract_total", ("合同总额", "合同金额", "总价", "总金额")),
    )
    for label, terms in labels:
        if any(term in context for term in terms):
            return label
    return "unspecified_amount"


def _normalize_amount(value: str, unit: str) -> str:
    number = float(value.replace(",", "").replace("，", ""))
    multiplier = {"千": 1_000, "千元": 1_000, "万": 10_000, "万元": 10_000,
                  "million": 1_000_000, "亿元": 100_000_000, "billion": 1_000_000_000}.get(unit.lower(), 1)
    normalized = number * multiplier
    return str(int(normalized)) if normalized.is_integer() else str(normalized)


def _currency(prefix: str, unit: str) -> str:
    value = (prefix or "").upper()
    if value in {"人民币", "RMB", "CNY", "￥", "¥"} or unit.endswith("元"):
        return "CNY"
    if value in {"USD", "美元", "$"}:
        return "USD"
    if value in {"EUR", "欧元", "€"}:
        return "EUR"
    return ""


def _fact(doc: Document, fact_type: str, raw: str, normalized: str, evidence: str, **extra: Any) -> Dict[str, Any]:
    metadata = doc.metadata or {}
    return {
        "id": _stable_id(metadata.get("file_id", ""), fact_type, normalized, metadata.get("page"), evidence),
        "fact_type": fact_type,
        "raw_value": raw,
        "normalized_value": normalized,
        "unit": extra.get("unit", ""),
        "currency": extra.get("currency", ""),
        "qualifier": extra.get("qualifier", ""),
        "physical_page": metadata.get("page"),
        "printed_page": metadata.get("printed_page", ""),
        "section_path": metadata.get("section_path", ""),
        "evidence_text": evidence[:500],
        "chunk_id": metadata.get("chunk_id", ""),
        "extractor": extra.get("extractor", "rules"),
        "confidence": float(extra.get("confidence", 0.9)),
        "extractor_version": METADATA_EXTRACTOR_VERSION,
    }


def extract_rule_facts(docs: Sequence[Document]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for doc in docs:
        text = doc.page_content
        for match in _AMOUNT_RE.finditer(text):
            prefix, value, unit = match.group("prefix") or "", match.group("value"), match.group("unit") or ""
            nearby = _context(text, match.start(), match.end())
            # Bare small integers are overwhelmingly page/list numbers, not monetary facts.
            if not prefix and not unit and not any(term in nearby for term in ("金额", "价格", "费用", "合计", "付款", "税")):
                continue
            facts.append(_fact(doc, "amount", match.group(0).strip(), _normalize_amount(value, unit), nearby,
                               unit=unit, currency=_currency(prefix, unit), qualifier=_amount_qualifier(nearby)))
        for match in _PERCENT_RE.finditer(text):
            value = match.group("prefix_value") or match.group("suffix_value")
            facts.append(_fact(doc, "percentage", match.group(0), value,
                               _context(text, match.start(), match.end()), unit="percent"))
        for match in _DATE_RE.finditer(text):
            facts.append(_fact(doc, "date", match.group(0), match.group(0), _context(text, match.start(), match.end())))
        for match in _EMAIL_RE.finditer(text):
            facts.append(_fact(doc, "email", match.group(0), match.group(0).lower(),
                               _context(text, match.start(), match.end())))
        for match in _PHONE_RE.finditer(text):
            normalized = re.sub(r"\D", "", match.group(0))
            facts.append(_fact(doc, "phone", match.group(0), normalized, _context(text, match.start(), match.end())))
        for fact_type, pattern in _IDENTIFIER_PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(1)
                facts.append(_fact(doc, fact_type, value, value.casefold(), _context(text, match.start(), match.end())))
    unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for fact in facts:
        key = (fact["fact_type"], fact["normalized_value"], fact.get("physical_page"), fact["evidence_text"])
        unique[key] = fact
    return list(unique.values())


def _parse_llm_json(content: Any) -> List[Dict[str, Any]]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    payload = json.loads(text)
    values = payload.get("facts", []) if isinstance(payload, dict) else []
    if not isinstance(values, list):
        raise ValueError("facts must be a JSON array")
    clean = []
    for value in values[:100]:
        if not isinstance(value, dict) or value.get("fact_type") not in FACT_SCHEMA_TYPES:
            continue
        raw = str(value.get("raw_value", "")).strip()
        if not raw:
            continue
        clean.append({
            "fact_type": value["fact_type"], "raw_value": raw,
            "normalized_value": str(value.get("normalized_value", raw)),
            "unit": str(value.get("unit", "")), "currency": str(value.get("currency", "")),
            "qualifier": str(value.get("qualifier", "")),
        })
    return clean


def extract_llm_facts(
    docs: Sequence[Document], api_key: str | None, *, raise_on_error: bool = False,
) -> List[Dict[str, Any]]:
    if not METADATA_LLM_EXTRACTION_ENABLED or not api_key:
        return []
    primary_docs = [
        doc for doc in docs
        if (doc.metadata or {}).get("retrieval_tier", "primary") == "primary" and doc.page_content.strip()
    ]
    max_chunks = max(1, METADATA_LLM_EXTRACTION_MAX_CHUNKS)
    if len(primary_docs) <= max_chunks:
        candidates = primary_docs
    elif max_chunks == 1:
        candidates = [primary_docs[0]]
    else:
        # Sample the whole document instead of sending only its opening chunks.
        last = len(primary_docs) - 1
        indexes = {
            round(index * last / (max_chunks - 1))
            for index in range(max_chunks)
        }
        candidates = [primary_docs[index] for index in sorted(indexes)]
    if not candidates:
        return []
    evidence = "\n\n".join(
        f"[chunk_id={doc.metadata.get('chunk_id','')}, page={doc.metadata.get('page','')}]\n"
        f"{doc.page_content[:METADATA_LLM_EXTRACTION_CHARS_PER_CHUNK]}"
        for doc in candidates
    )[:METADATA_LLM_EXTRACTION_INPUT_CHARS]
    prompt = (
        "Extract only explicit document facts as strict JSON: {\"facts\":[{\"fact_type\": one of "
        f"{sorted(FACT_SCHEMA_TYPES)}, \"raw_value\":str, \"normalized_value\":str, \"unit\":str, "
        "\"currency\":str, \"qualifier\":str, \"chunk_id\":str}]}. Do not infer missing facts.\n\n" + evidence
    )
    try:
        logger.info(
            f"Starting optional LLM metadata extraction for {len(candidates)} chunks "
            f"with timeout={METADATA_LLM_EXTRACTION_TIMEOUT_SECONDS}s, "
            f"retries={METADATA_LLM_EXTRACTION_MAX_RETRIES}, input_chars={len(evidence)}"
        )
        llm = ComponentFactory.get_component("llm", LLM_TYPE, api_key=api_key, base_url=ZAI_API_BASE,
                                             model_name=LLM_MODEL_NAME, temperature=0,
                                             timeout=METADATA_LLM_EXTRACTION_TIMEOUT_SECONDS,
                                             max_retries=METADATA_LLM_EXTRACTION_MAX_RETRIES,
                                             max_tokens=METADATA_LLM_EXTRACTION_MAX_TOKENS)
        values = _parse_llm_json(llm.invoke(prompt).content)
        docs_by_id = {str(doc.metadata.get("chunk_id", "")): doc for doc in candidates}
        results = []
        for value in values:
            raw = value["raw_value"]
            doc = next((item for item in candidates if raw in item.page_content), None)
            if doc is None:
                continue
            start = doc.page_content.find(raw)
            results.append(_fact(doc, value["fact_type"], raw, value["normalized_value"],
                                 _context(doc.page_content, start, start + len(raw)),
                                 unit=value["unit"], currency=value["currency"], qualifier=value["qualifier"],
                                 extractor="llm", confidence=0.75))
        logger.info(f"Optional LLM metadata extraction completed with {len(results)} verified facts")
        return results
    except Exception as exc:
        logger.warning(f"Optional LLM metadata extraction failed: {exc}")
        if raise_on_error:
            raise
        return []


def build_document_analysis(
    docs: Sequence[Document], *, file_path: str, file_name: str, file_id: str, session_id: str,
    include_llm: bool = True, raise_on_llm_error: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metadata: Dict[str, Any] = {
        "file_name": file_name, "extension": Path(file_name).suffix.lower(),
        "size_bytes": os.path.getsize(file_path), "physical_page_count": 0,
        "body_page_count": 0, "printed_page_range": [], "cover_pages": [], "toc_pages": [],
        "body_pages": [], "appendix_pages": [], "section_count": 0, "table_count": 0,
        "image_count": 0, "ocr_required": False,
    }
    blocks: List[Dict[str, Any]] = []
    page_roles: Dict[int, str] = {}
    printed_pages: List[str] = []
    headings = set()
    for index, doc in enumerate(docs):
        source = dict(doc.metadata or {})
        for key in ("title", "author", "subject", "creator", "producer", "creation_date", "modification_date"):
            if source.get(f"document_{key}") and not metadata.get(key):
                metadata[key] = source[f"document_{key}"]
        page = source.get("page")
        if source.get("page_count"):
            metadata["physical_page_count"] = max(
                int(metadata.get("physical_page_count", 0)), int(source.get("page_count", 0))
            )
        role = str(source.get("page_role", "body"))
        if isinstance(page, int):
            page_roles[page] = role
        if source.get("printed_page"):
            printed_pages.append(str(source["printed_page"]))
        if source.get("chunk_kind") == "heading":
            headings.add(str(source.get("section_path") or doc.page_content))
        if source.get("chunk_kind") == "table":
            metadata["table_count"] += 1
        metadata["image_count"] += int(source.get("image_count", 0) or 0)
        blocks.append({
            "id": _stable_id(file_id, "block", index, page, doc.page_content[:80]),
            "block_index": index, "block_type": source.get("chunk_kind", "text"),
            "page_role": role, "retrieval_tier": source.get("retrieval_tier", "primary"),
            "physical_page": page, "printed_page": source.get("printed_page", ""),
            "section_path": source.get("section_path", ""), "text": doc.page_content,
            "bbox": source.get("bbox", []),
            "metadata": {k: v for k, v in source.items() if k not in {"bbox"}},
        })
    if page_roles:
        metadata["physical_page_count"] = max(int(metadata.get("physical_page_count", 0)), max(page_roles))
        for page, role in sorted(page_roles.items()):
            metadata.setdefault(f"{role}_pages", []).append(page)
        metadata["body_page_count"] = len(metadata.get("body_pages", []))
    metadata["printed_page_range"] = printed_pages
    metadata["section_count"] = len(headings)
    metadata["ocr_required"] = not any(doc.page_content.strip() for doc in docs)
    preview = "\n".join(doc.page_content for doc in docs[:8])[:6000]
    if not metadata.get("title"):
        title_doc = next((doc for doc in docs if (doc.metadata or {}).get("chunk_kind") == "heading"), None)
        if title_doc is None:
            title_doc = next((doc for doc in docs if (doc.metadata or {}).get("page_role") == "cover"), None)
        metadata["title"] = (title_doc.page_content.splitlines()[0][:300] if title_doc else Path(file_name).stem)
    metadata["language"] = "zh" if len(re.findall(r"[\u4e00-\u9fff]", preview)) > len(preview) * 0.08 else "en"
    # Document type is a title-level property. Body-wide keyword matching made
    # security documents containing phrases such as "协议攻击" look like contracts.
    type_context = f"{metadata.get('title', '')} {file_name} {preview[:500]}".casefold()
    type_markers = (
        ("contract", ("合同书", "合同编号", "协议书", "甲方", "乙方")),
        ("invoice", ("发票", "invoice")), ("report", ("报告", "report")),
        ("paper", ("摘要", "abstract")), ("tender", ("招标书", "投标书")),
    )
    metadata["document_type"] = next(
        (label for label, terms in type_markers if any(term.casefold() in type_context for term in terms)), "document"
    )

    facts = extract_rule_facts(docs)
    if include_llm:
        facts.extend(extract_llm_facts(docs, os.getenv("ZAI_API_KEY"), raise_on_error=raise_on_llm_error))
    unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for fact in facts:
        key = (fact["fact_type"], fact["normalized_value"], fact.get("physical_page"), fact["evidence_text"])
        current = unique.get(key)
        if current is None or fact["confidence"] > current["confidence"]:
            unique[key] = fact
    metadata["fact_count"] = len(unique)
    metadata["structure_version"] = DOCUMENT_STRUCTURE_VERSION
    metadata["extractor_version"] = METADATA_EXTRACTOR_VERSION
    return metadata, list(unique.values()), blocks
