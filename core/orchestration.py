"""Bounded, opt-in evidence-worker orchestration for complex requests."""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import urlparse

from config import ORCHESTRATION_MAX_TOOL_CALLS_PER_WORKER, ORCHESTRATION_MAX_WORKERS
from core.agent import retrieve_session_evidence
from core.request_router import RouteDecision
from tools.document_metadata import query_session_document_metadata
from tools.web_search import read_web_page, search_web_candidates


@dataclass
class WorkerResult:
    worker: str
    status: str
    claims: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    completed_capabilities: List[str] = field(default_factory=list)
    termination_reason: str = ""
    remediation_used: bool = False
    elapsed_ms: float = 0
    tool_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SupervisorExecutionContext:
    completed_capabilities: List[str]
    remediation_capabilities: List[str]
    remaining_tools: List[str]
    tool_call_limits: Dict[str, int]
    artifact_requirements: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _run_worker(name: str, callback: Callable[[], WorkerResult]) -> WorkerResult:
    started = time.perf_counter()
    try:
        result = callback()
    except Exception as exc:
        result = WorkerResult(
            worker=name, status="failed", errors=[{"error_code": "WORKER_EXCEPTION", "message": str(exc)}],
            termination_reason="uncaught_exception",
        )
    result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return result


def _metadata_evidence(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    evidence: List[Dict[str, Any]] = []
    claims: List[Dict[str, Any]] = []
    for fact in payload.get("facts", []):
        identity = "|".join((str(fact.get("file_id", "")), str(fact.get("id", "")), str(fact.get("evidence_text", ""))))
        evidence_id = "meta_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        item = {
            "evidence_id": evidence_id, "kind": "structured_metadata",
            "file_id": str(fact.get("file_id", "")), "file_name": str(fact.get("source", "")),
            "page": fact.get("physical_page"), "printed_page": fact.get("printed_page", ""),
            "section_path": str(fact.get("section_path", "")), "chunk_id": str(fact.get("chunk_id", "")),
            "chunk_kind": "fact", "text": str(fact.get("evidence_text", "")),
            "fact_type": str(fact.get("fact_type", "")), "value": fact.get("raw_value"),
            "score": float(fact.get("confidence", 0)),
        }
        evidence.append(item)
        claims.append({"claim": f"{item['fact_type']}: {item['value']}", "evidence_ids": [evidence_id]})
    return evidence, claims


def _document_worker(session_id: str, message: str, include_metadata: bool) -> WorkerResult:
    evidence: List[Dict[str, Any]] = []
    claims: List[Dict[str, Any]] = []
    capabilities: List[str] = []
    errors: List[Dict[str, str]] = []
    calls = 0
    if include_metadata:
        try:
            payload = json.loads(query_session_document_metadata(session_id, message))
            metadata_evidence, metadata_claims = _metadata_evidence(payload)
            evidence.extend(metadata_evidence)
            claims.extend(metadata_claims)
            capabilities.append("query_document_metadata")
        except Exception as exc:
            errors.append({"error_code": "METADATA_QUERY_FAILED", "message": str(exc)})
        calls += 1

    retrieval = retrieve_session_evidence(session_id, message)
    calls += 1
    if retrieval.get("ok"):
        evidence.extend(retrieval.get("evidence", []))
        claims.extend(retrieval.get("claims", []))
        capabilities.append("retrieve_knowledge")
    else:
        errors.append({
            "error_code": str(retrieval.get("error_code") or "DOCUMENT_RETRIEVAL_FAILED"),
            "message": str(retrieval.get("error") or "No relevant document evidence was found."),
        })
    status = "completed" if "retrieve_knowledge" in capabilities else "failed"
    return WorkerResult(
        "document", status, claims=claims, evidence=evidence,
        uncertainties=[] if status == "completed" else ["Uploaded-document evidence is insufficient."],
        errors=errors, completed_capabilities=capabilities,
        termination_reason="evidence_collected" if status == "completed" else "insufficient_evidence",
        tool_calls=calls,
    )


def _authority_score(candidate: Dict[str, Any]) -> Tuple[int, int]:
    host = urlparse(str(candidate.get("url", ""))).netloc.casefold()
    score = 0
    if host.endswith(".gov") or ".gov." in host:
        score += 100
    if host.endswith(".edu") or ".edu." in host:
        score += 80
    if any(domain in host for domain in ("nist.gov", "owasp.org", "w3.org", "ietf.org", "cve.org")):
        score += 90
    if any(domain in host for domain in (
        "microsoft.com", "google.com", "mozilla.org", "cloudflare.com",
        "oracle.com", "postgresql.org", "huawei.com",
    )):
        score += 60
    return score, -len(str(candidate.get("url", "")))


def _web_research_query(message: str) -> str:
    lowered = message.casefold()
    if "sql" in lowered and ("注入" in message or "injection" in lowered):
        return "SQL injection prevention official guidance parameterized queries"
    if "xss" in lowered or "跨站脚本" in message:
        return "XSS prevention official guidance CSP output encoding"
    cleaned = re.sub(
        r"上传|文档|文件|联网|搜索|然后|生成|写一个|写一份|对比|\.txt|\.docx|\.pdf",
        " ", message, flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())[:180] or message[:180]


def _web_retry_query(message: str) -> str:
    lowered = message.casefold()
    if "sql" in lowered and ("注入" in message or "injection" in lowered):
        return "site:owasp.org SQL Injection Prevention Cheat Sheet"
    if "xss" in lowered or "跨站脚本" in message:
        return "site:owasp.org Cross Site Scripting Prevention Cheat Sheet"
    return _web_research_query(message) + " official documentation"


def _web_topic_relevant(item: Dict[str, Any], message: str) -> bool:
    haystack = " ".join(str(item.get(key, "")) for key in ("title", "url", "snippet", "text")).casefold()
    lowered = message.casefold()
    if "sql" in lowered and ("注入" in message or "injection" in lowered):
        return "sql" in haystack and ("injection" in haystack or "注入" in haystack)
    if "xss" in lowered or "跨站脚本" in message:
        return "xss" in haystack or "cross site scripting" in haystack or "跨站脚本" in haystack
    terms = [term for term in re.findall(r"[a-z0-9]{4,}|[\u4e00-\u9fff]{2,}", lowered) if term not in {"official", "documentation"}]
    return not terms or any(term in haystack for term in terms)


def _web_worker(message: str) -> WorkerResult:
    search = search_web_candidates(_web_research_query(message))
    remediation_used = False
    calls = 1
    errors: List[Dict[str, str]] = []
    relevant_candidates = [item for item in search.get("candidates", []) if _web_topic_relevant(item, message)]
    if not search.get("ok") or not relevant_candidates:
        errors.append({
            "error_code": str(search.get("error_code") or "NO_TOPIC_RELEVANT_RESULTS"),
            "message": json.dumps(search.get("errors", []) or "No topic-relevant candidate was returned.", ensure_ascii=False),
        })
        remediation_used = True
        retry_query = _web_retry_query(message)
        search = search_web_candidates(retry_query)
        calls += 1
        relevant_candidates = [item for item in search.get("candidates", []) if _web_topic_relevant(item, message)]
        if not search.get("ok") or not relevant_candidates:
            errors.append({
                "error_code": str(search.get("error_code") or "NO_TOPIC_RELEVANT_RESULTS"),
                "message": json.dumps(search.get("errors", []) or "No topic-relevant candidate was returned.", ensure_ascii=False),
            })
            return WorkerResult(
                "web", "failed", uncertainties=["No verified internet source could be collected."],
                errors=errors, termination_reason="search_failed_after_remediation",
                remediation_used=True, tool_calls=calls,
            )

    candidates = sorted(relevant_candidates, key=_authority_score, reverse=True)
    evidence: List[Dict[str, Any]] = []
    remaining_calls = max(0, int(ORCHESTRATION_MAX_TOOL_CALLS_PER_WORKER) - calls)
    max_opens = min(2, remaining_calls)
    for candidate in candidates[:max_opens]:
        page = read_web_page(str(candidate.get("url", "")))
        calls += 1
        if page.get("ok") and _web_topic_relevant(page, message):
            evidence.append(page)
        elif page.get("ok"):
            errors.append({
                "error_code": "WEB_PAGE_TOPIC_MISMATCH",
                "message": f"Opened page is not relevant to the research topic: {page.get('url', '')}",
            })
        else:
            errors.append({
                "error_code": str(page.get("error_code", "WEB_PAGE_OPEN_FAILED")),
                "message": str(page.get("error", "")),
            })
    claims = [
        {"claim": str(item.get("text", ""))[:500], "evidence_ids": [str(item.get("evidence_id", ""))]}
        for item in evidence
    ]
    status = "completed" if evidence else "failed"
    return WorkerResult(
        "web", status, claims=claims, evidence=evidence, errors=errors,
        uncertainties=[] if evidence else ["Candidate sources were found, but no authoritative page was successfully opened."],
        completed_capabilities=["web_search", "read_website"] if evidence else [],
        termination_reason=(
            "verified_sources_opened_after_remediation" if evidence and remediation_used
            else "verified_sources_opened" if evidence else "source_open_failed"
        ),
        remediation_used=remediation_used, tool_calls=calls,
    )


def _analysis_worker(message: str) -> WorkerResult:
    subtasks = [item.strip(" -\t") for item in re.split(r"(?:\n+|[；;]|\bthen\b|另外|同时|然后)", message) if item.strip()]
    claims = [{"claim": item, "evidence_ids": []} for item in subtasks[:8]]
    return WorkerResult("analysis", "completed", claims=claims, termination_reason="subtasks_identified")


def _workspace_worker() -> WorkerResult:
    return WorkerResult(
        "workspace", "completed",
        claims=[{"claim": "Generate the requested artifact from supplied evidence and verify registration.", "evidence_ids": []}],
        completed_capabilities=[], termination_reason="artifact_work_deferred_to_supervisor",
    )


def _supervisor_context(decision: RouteDecision, results: List[WorkerResult]) -> SupervisorExecutionContext:
    completed = sorted({cap for result in results for cap in result.completed_capabilities})
    requested = set(decision.allowed_tools)
    remaining = [tool for tool in decision.allowed_tools if tool not in completed]
    remediation: List[str] = []
    limits: Dict[str, int] = {}

    for capability in ("retrieve_knowledge", "query_document_metadata", "web_search", "read_website"):
        if capability in requested and capability not in completed:
            remediation.append(capability)
            limits[capability] = 1

    if decision.needs_workspace and "retrieve_knowledge" in completed:
        # Evidence is already in the worker report. Workspace only creates/registers the deliverable.
        blocked_source_tools = {
            "workspace_terminal", "workspace_process", "workspace_list_session_files",
            "workspace_import_uploaded_file", "workspace_read_file", "workspace_import_artifact",
            "workspace_save_file",
        }
        remaining = [tool for tool in remaining if tool not in blocked_source_tools]
        required_evidence_complete = (
            (not decision.needs_documents or "retrieve_knowledge" in completed)
            and (not decision.needs_web or "read_website" in completed)
        )
        if required_evidence_complete:
            # Research is complete. The Supervisor has one job left: create and register the deliverable.
            remaining = [tool for tool in remaining if tool == "workspace_write_file"]

    verified_web_urls = [
        str(item.get("url"))
        for result in results for item in result.evidence
        if item.get("kind") == "verified_web" and item.get("url")
    ]
    document_sources = sorted({
        str(item.get("file_name"))
        for result in results for item in result.evidence
        if item.get("kind") == "document" and item.get("file_name")
    })

    return SupervisorExecutionContext(
        completed_capabilities=completed,
        remediation_capabilities=sorted(set(remediation)),
        remaining_tools=remaining,
        tool_call_limits=limits,
        artifact_requirements={
            "must_register": bool(decision.needs_workspace), "must_be_non_empty": True,
            "forbid_local_links": True, "delivery_text": "文件已保存到右侧已保存产物",
            "require_verified_web": bool(decision.needs_web),
            "verified_web_urls": verified_web_urls,
            "require_document_citation": bool(decision.needs_documents),
            "document_sources": document_sources,
        },
    )


def _verify(results: List[WorkerResult]) -> WorkerResult:
    uncertainties: List[str] = []
    if not results:
        uncertainties.append("No worker was selected.")
    if any(result.status != "completed" for result in results):
        uncertainties.append("At least one worker did not complete successfully; one bounded remediation is allowed.")
    if sum(len(result.evidence) for result in results) == 0:
        uncertainties.append("No document or verified web evidence was produced.")
    return WorkerResult("verification", "completed", uncertainties=uncertainties, termination_reason="worker_results_checked")


def run_orchestration(
    session_id: str, message: str, decision: RouteDecision,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    if not decision.use_multi_agent:
        return "", [], {"enabled": False, "reason": "complexity gate not met", "workers": [], "execution_context": {}}

    jobs: Dict[str, Callable[[], WorkerResult]] = {}
    if decision.needs_documents:
        jobs["document"] = lambda: _document_worker(session_id, message, decision.needs_structured_facts)
    if decision.needs_web:
        jobs["web"] = lambda: _web_worker(message)
    if decision.needs_workspace:
        jobs["workspace"] = _workspace_worker
    if not jobs:
        jobs["analysis"] = lambda: _analysis_worker(message)

    results: List[WorkerResult] = []
    with ThreadPoolExecutor(max_workers=min(max(1, ORCHESTRATION_MAX_WORKERS), len(jobs))) as executor:
        futures = {executor.submit(_run_worker, name, callback): name for name, callback in jobs.items()}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item.worker)
    verification = _verify(results)
    execution = _supervisor_context(decision, results)

    tool_calls: List[Dict[str, Any]] = []
    for result in results:
        for capability in result.completed_capabilities:
            tool_calls.append({
                "tool": capability, "input": {"query": message, "worker": result.worker},
                "status": "completed", "output_preview": f"{result.worker} worker completed {capability}.",
                "call_stage": "worker_remediation" if result.remediation_used else "worker",
                "duplicate_blocked": False, "remediation": result.remediation_used,
                "evidence_ids": [item.get("evidence_id") for item in result.evidence],
                "termination_reason": result.termination_reason,
            })
        if result.status != "completed" and result.worker in {"document", "web"}:
            tool_calls.append({
                "tool": f"{result.worker}_worker", "input": {"query": message}, "status": result.status,
                "output_preview": json.dumps(result.errors, ensure_ascii=False), "call_stage": "worker",
                "duplicate_blocked": False, "remediation": False, "evidence_ids": [],
                "termination_reason": result.termination_reason,
            })

    reports = [result.to_dict() for result in results] + [verification.to_dict()]
    debug = {
        "enabled": True, "reason": decision.reason, "worker_count": len(results), "workers": reports,
        "execution_context": execution.to_dict(), "redispatches": 0,
        "termination_reason": "bounded_workers_completed",
    }
    if any(result.remediation_used for result in results):
        debug["redispatches"] = 1
        debug["termination_reason"] = "bounded_workers_completed_with_remediation"
    context = json.dumps({"worker_reports": reports, "execution_context": execution.to_dict()}, ensure_ascii=False, default=str)
    return context, tool_calls, debug
