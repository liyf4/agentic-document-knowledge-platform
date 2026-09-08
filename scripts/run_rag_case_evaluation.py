"""Run an auditable per-case RAG evaluation against the long-document fixture."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.main import app
from core.document_processor import process_file
from core.retriever import load_retriever, rerank
from db.file_manager import list_document_metadata, list_uploaded_files, query_document_facts
from db.session_manager import create_new_session
from tests.support.isolated_runtime import isolated_runtime


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "agent_rag_case_study"
EVIDENCE_PATTERN = re.compile(
    r"\[EVIDENCE:([A-Z]{2,3}-\d{3})\]|\|\s*(REG-\d{3})\s*\||^(REG-\d{3}),",
    flags=re.MULTILINE,
)


@dataclass
class BufferedUpload:
    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


def load_cases(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for row in rows:
        case_id = str(row.get("id") or "")
        if not case_id or case_id in seen:
            raise ValueError(f"Missing or duplicate case id: {case_id!r}")
        seen.add(case_id)
        if not row.get("evidence_ids") or not row.get("answer_checks"):
            raise ValueError(f"Case {case_id} must define evidence_ids and answer_checks")
    return rows


def extract_evidence_ids(text: str) -> List[str]:
    found = {
        match.group(1) or match.group(2) or match.group(3)
        for match in EVIDENCE_PATTERN.finditer(text or "")
    }
    return sorted(found)


def evidence_ids_for_docs(docs: Sequence[Dict[str, Any]]) -> List[str]:
    found: set[str] = set()
    for doc in docs:
        found.update(extract_evidence_ids(str(doc.get("content") or doc.get("preview") or "")))
    return sorted(found)


def score_answer(answer: str, checks: Sequence[Sequence[str]]) -> Dict[str, Any]:
    lowered = answer.casefold()
    matched: List[bool] = []
    for alternatives in checks:
        matched.append(any(str(value).casefold() in lowered for value in alternatives))
    score = sum(matched) / max(len(matched), 1)
    return {
        "accuracy": round(score, 4),
        "matched_checks": sum(matched),
        "total_checks": len(matched),
        "missing_checks": [list(checks[index]) for index, value in enumerate(matched) if not value],
    }


def doc_view(doc: Dict[str, Any], rank: int) -> Dict[str, Any]:
    content = str(doc.get("content") or doc.get("preview") or "")
    return {
        "rank": rank,
        "source": doc.get("source"),
        "evidence_ids": extract_evidence_ids(content),
        "score": doc.get("score"),
        "preview": content[:500].replace("\n", " "),
    }


def index_fixture_documents() -> tuple[str, List[Dict[str, Any]]]:
    session_id = create_new_session("long-document-rag-case-study")
    uploads: List[Dict[str, Any]] = []
    for path in sorted((FIXTURE_DIR / "documents").iterdir()):
        if not path.is_file():
            continue
        started = time.perf_counter()
        ok, message = process_file(BufferedUpload(path.name, path.read_bytes()), session_id)
        uploads.append({
            "file": path.name,
            "ok": ok,
            "message": message,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "bytes": path.stat().st_size,
        })
    if not uploads or not all(item["ok"] for item in uploads):
        raise RuntimeError(f"Fixture indexing failed: {uploads}")
    return session_id, uploads


def metadata_evaluation(session_id: str, retriever: Any, cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    stored = retriever.vector_retriever.vectorstore.get(include=["documents", "metadatas"])
    documents = stored.get("documents", []) or []
    metadatas = stored.get("metadatas", []) or [{} for _ in documents]
    ids = stored.get("ids", []) or ["" for _ in documents]
    required_ids = sorted({value for row in cases for value in row["evidence_ids"]})
    expected_sources: Dict[str, str] = {}
    for path in (FIXTURE_DIR / "documents").glob("*.md"):
        for evidence_id in extract_evidence_ids(path.read_text(encoding="utf-8")):
            expected_sources[evidence_id] = path.name

    evidence_results = []
    for evidence_id in required_ids:
        matches = [
            (text, dict(metadata or {}), str(doc_id or ""))
            for text, metadata, doc_id in zip(documents, metadatas, ids)
            if evidence_id in extract_evidence_ids(str(text))
        ]
        source_ok = any(str(metadata.get("source")) == expected_sources.get(evidence_id) for _, metadata, _ in matches)
        section_ok = any(bool(metadata.get("section_path")) for _, metadata, _ in matches)
        chunk_id_ok = any(bool(metadata.get("chunk_id") or doc_id) for _, metadata, doc_id in matches)
        evidence_results.append({
            "evidence_id": evidence_id,
            "expected_source": expected_sources.get(evidence_id, ""),
            "resolved": bool(matches),
            "source_correct": source_ok,
            "section_path_present": section_ok,
            "chunk_id_present": chunk_id_ok,
            "match_count": len(matches),
        })

    document_rows = list_document_metadata(session_id)
    uploaded = list_uploaded_files(session_id)
    uploaded_by_id = {item["file_id"]: item for item in uploaded}
    document_results = []
    for row in document_rows:
        metadata = row.get("metadata", {})
        upload = uploaded_by_id.get(row["file_id"], {})
        document_results.append({
            "file": upload.get("original_name", metadata.get("file_name", "")),
            "extension": metadata.get("extension"),
            "language": metadata.get("language"),
            "section_count": metadata.get("section_count", 0),
            "fact_count": metadata.get("fact_count", 0),
            "extraction_status": row.get("extraction_status"),
            "passed": bool(
                metadata.get("extension") == ".md"
                and metadata.get("language") == "en"
                and int(metadata.get("section_count", 0)) > 0
            ),
            "llm_enrichment_completed": row.get("extraction_status") == "completed",
        })

    expected_facts = [
        ("amount", "185000"), ("percentage", "82"), ("percentage", "70"),
        ("percentage", "3"), ("date", "2026-03-08"), ("date", "2026-04-17"),
        ("date", "2026-04-22"), ("date", "2026-03-03"), ("date", "2026-02-07"),
    ]
    facts = query_document_facts(session_id, limit=100)
    fact_keys = {(str(item["fact_type"]), str(item["normalized_value"])) for item in facts}
    fact_results = [
        {"fact_type": fact_type, "normalized_value": value, "found": (fact_type, value) in fact_keys}
        for fact_type, value in expected_facts
    ]
    linked_facts = [item for item in facts if item.get("chunk_id") and item.get("section_path")]
    count = max(len(evidence_results), 1)
    return {
        "evidence_resolve_rate": round(sum(item["resolved"] for item in evidence_results) / count, 4),
        "source_accuracy": round(sum(item["source_correct"] for item in evidence_results) / count, 4),
        "section_path_completeness": round(sum(item["section_path_present"] for item in evidence_results) / count, 4),
        "chunk_id_completeness": round(sum(item["chunk_id_present"] for item in evidence_results) / count, 4),
        "document_metadata_pass_rate": round(sum(item["passed"] for item in document_results) / max(len(document_results), 1), 4),
        "llm_metadata_enrichment_rate": round(sum(item["llm_enrichment_completed"] for item in document_results) / max(len(document_results), 1), 4),
        "rule_fact_recall": round(sum(item["found"] for item in fact_results) / len(fact_results), 4),
        "fact_evidence_link_rate": round(len(linked_facts) / max(len(facts), 1), 4),
        "evidence_results": evidence_results,
        "document_results": document_results,
        "fact_results": fact_results,
        "fact_count": len(facts),
    }


def percentile(values: Iterable[float], ratio: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio) - 1))
    return round(ordered[index], 2)


def evaluate(
    output_dir: Path,
    require_llm: bool,
    reused_answers: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    cases = load_cases(FIXTURE_DIR / "cases.jsonl")
    session_id, uploads = index_fixture_documents()
    retriever, summary = load_retriever(session_id)
    if retriever is None:
        raise RuntimeError("Retriever was not created")

    results: List[Dict[str, Any]] = []
    with TestClient(app) as client:
        for row in cases:
            request_text = "According to the uploaded documents, " + str(row["question"])
            retrieval_started = time.perf_counter()
            rrf_candidates, rrf_debug = retriever.search(request_text)
            rrf_docs = rrf_debug.get("rrf", [])
            rerank_started = time.perf_counter()
            reranked = rerank(request_text, rrf_candidates)
            rerank_ms = round((time.perf_counter() - rerank_started) * 1000, 2)
            final_docs = []
            for rank, (score, doc) in enumerate(reranked[:4], start=1):
                metadata = dict(doc.metadata or {})
                final_docs.append({
                    "rank": rank,
                    "score": float(score),
                    "source": metadata.get("source"),
                    "section_path": metadata.get("section_path"),
                    "chunk_id": metadata.get("chunk_id", metadata.get("id")),
                    "content": doc.page_content,
                    "preview": doc.page_content[:240].replace("\n", " "),
                })
            retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 2)

            required = set(str(value) for value in row["evidence_ids"])
            rrf_top4_ids = set(evidence_ids_for_docs(rrf_docs[:4]))
            crossencoder_top1_ids = set(evidence_ids_for_docs(final_docs[:1]))
            rrf_coverage = len(required & rrf_top4_ids) / max(len(required), 1)
            rrf_accuracy = float(rrf_coverage == 1.0)
            crossencoder_accuracy = float(bool(required & crossencoder_top1_ids))

            answer = ""
            citations: List[Dict[str, Any]] = []
            answer_ms = 0.0
            api_status = 0
            api_error = ""
            reused = (reused_answers or {}).get(row["id"])
            if reused is not None:
                answer = str(reused.get("model_answer") or "")
                citations = list(reused.get("citations") or [])
                api_status = int(reused.get("api_status") or 0)
                api_error = str(reused.get("api_error") or "")
                answer_ms = float((reused.get("timing_ms") or {}).get("answer_api", 0))
            else:
                answer_started = time.perf_counter()
                try:
                    response = client.post(
                        "/api/chat",
                        json={"session_id": session_id, "message": request_text, "mode": "chat"},
                    )
                    answer_ms = round((time.perf_counter() - answer_started) * 1000, 2)
                    api_status = response.status_code
                    if response.status_code == 200:
                        payload = response.json()
                        answer = str(payload.get("answer") or "")
                        citations = list(payload.get("citations") or [])
                    else:
                        api_error = response.text
                except Exception as exc:
                    answer_ms = round((time.perf_counter() - answer_started) * 1000, 2)
                    api_error = f"{type(exc).__name__}: {exc}"

            answer_score = score_answer(answer, row["answer_checks"])
            answer_passed = answer_score["accuracy"] == 1.0 and bool(citations) and api_status == 200
            citation_ids = set(evidence_ids_for_docs(citations))
            citation_coverage = len(required & citation_ids) / max(len(required), 1)
            results.append({
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "request_text": request_text,
                "standard_answer": row["standard_answer"],
                "required_evidence_ids": sorted(required),
                "crossencoder_accuracy_at_1": crossencoder_accuracy,
                "crossencoder_passed": crossencoder_accuracy == 1.0,
                "rrf_accuracy_at_4": rrf_accuracy,
                "rrf_evidence_coverage_at_4": round(rrf_coverage, 4),
                "rrf_passed": rrf_accuracy == 1.0,
                "answer_accuracy": answer_score["accuracy"],
                "answer_passed": answer_passed,
                "answer_checks": row["answer_checks"],
                "missing_answer_checks": answer_score["missing_checks"],
                "model_answer": answer,
                "citations": citations,
                "citation_evidence_coverage": round(citation_coverage, 4),
                "citation_evidence_ids": sorted(citation_ids),
                "api_status": api_status,
                "api_error": api_error,
                "rrf_top4": [doc_view(doc, index + 1) for index, doc in enumerate(rrf_docs[:4])],
                "crossencoder_top4": [doc_view(doc, index + 1) for index, doc in enumerate(final_docs)],
                "rerank_executed": True,
                "rerank_skipped": False,
                "rerank_skip_reason": "",
                "timing_ms": {
                    "retrieval_total": retrieval_ms,
                    "retrieval_stages": {**rrf_debug.get("timing_ms", {}), "crossencoder": rerank_ms},
                    "answer_api": answer_ms,
                },
            })

    count = len(results)
    metadata_metrics = metadata_evaluation(session_id, retriever, cases)
    summary_data = {
        "passed": all(
            item["crossencoder_passed"] and item["rrf_passed"] and item["answer_passed"]
            for item in results
        ),
        "case_count": count,
        "crossencoder_accuracy_at_1": round(statistics.mean(item["crossencoder_accuracy_at_1"] for item in results), 4),
        "crossencoder_rank1_cases": sum(item["crossencoder_passed"] for item in results),
        "rrf_accuracy_at_4": round(statistics.mean(item["rrf_accuracy_at_4"] for item in results), 4),
        "rrf_pass_cases": sum(item["rrf_passed"] for item in results),
        "model_answer_accuracy": round(statistics.mean(item["answer_accuracy"] for item in results), 4),
        "model_answer_pass_cases": sum(item["answer_passed"] for item in results),
        "api_error_cases": sum(bool(item["api_error"]) for item in results),
        "rerank_executed_cases": sum(item["rerank_executed"] for item in results),
        "retrieval_p95_ms": percentile((item["timing_ms"]["retrieval_total"] for item in results), 0.95),
        "answer_p95_ms": percentile((item["timing_ms"]["answer_api"] for item in results), 0.95),
        "citation_evidence_accuracy": round(statistics.mean(item["citation_evidence_coverage"] for item in results), 4),
        "citation_presence_rate": round(sum(bool(item["citations"]) for item in results) / count, 4),
        "require_llm": require_llm,
    }
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixture_dir": str(FIXTURE_DIR.resolve()),
        "retriever_summary": summary,
        "uploads": uploads,
        "summary": summary_data,
        "metadata_metrics": metadata_metrics,
        "results": results,
    }


def markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 长文档 Agent/RAG 逐题对照报告",
        "",
        f"- 总体通过：**{summary['passed']}**",
        f"- 用例数：`{summary['case_count']}`",
        f"- CrossEncoder Accuracy@1：`{summary['crossencoder_accuracy_at_1']}`（Rank 1 命中 {summary['crossencoder_rank1_cases']}/{summary['case_count']}）",
        f"- RRF Accuracy@4：`{summary['rrf_accuracy_at_4']}`（Top-4 完整覆盖 {summary['rrf_pass_cases']}/{summary['case_count']}）",
        f"- 模型回答准确度：`{summary['model_answer_accuracy']}`（事实与引用全通过 {summary['model_answer_pass_cases']}/{summary['case_count']}）",
        f"- 引用证据准确度：`{summary['citation_evidence_accuracy']}`；引用存在率：`{summary['citation_presence_rate']}`",
        f"- 检索 P95：`{summary['retrieval_p95_ms']} ms`",
        f"- 回答 API P95：`{summary['answer_p95_ms']} ms`",
        "",
        "## 文档索引",
        "",
        "| 文件 | 字节 | 索引结果 | 耗时 ms |",
        "| --- | ---: | --- | ---: |",
    ]
    for item in report["uploads"]:
        lines.append(f"| {item['file']} | {item['bytes']} | {item['message']} | {item['elapsed_ms']} |")
    metadata = report.get("metadata_metrics", {})
    lines.extend([
        "", "## 元数据与证据质量", "",
        f"- 标准证据可解析率：`{metadata.get('evidence_resolve_rate', 0)}`",
        f"- 来源元数据准确率：`{metadata.get('source_accuracy', 0)}`",
        f"- 章节路径完整率：`{metadata.get('section_path_completeness', 0)}`",
        f"- Chunk ID 完整率：`{metadata.get('chunk_id_completeness', 0)}`",
        f"- 文档元数据通过率：`{metadata.get('document_metadata_pass_rate', 0)}`",
        f"- 可选 LLM 元数据增强完成率：`{metadata.get('llm_metadata_enrichment_rate', 0)}`",
        f"- 规则事实 Recall：`{metadata.get('rule_fact_recall', 0)}`",
        f"- 事实到证据链接率：`{metadata.get('fact_evidence_link_rate', 0)}`",
    ])
    lines.extend([
        "", "## 汇总表", "",
        "| ID | 类型 | CrossEncoder@1 | RRF@4 | Answer | Citation evidence | 重排 |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ])
    for item in report["results"]:
        lines.append(
            f"| {item['id']} | {item['category']} | {item['crossencoder_accuracy_at_1']} | "
            f"{item['rrf_accuracy_at_4']} | {item['answer_accuracy']} | {item['citation_evidence_coverage']} | "
            f"{item['rerank_executed']} |"
        )
    lines.extend(["", "## 逐题对照", ""])
    for item in report["results"]:
        lines.extend([
            f"### {item['id']} · {item['category']}", "",
            f"**问题**：{item['question']}", "",
            f"**标准答案**：{item['standard_answer']}", "",
            f"**模型答案**：{item['model_answer'] or '[无答案]'}", "",
            f"- 标准证据：`{item['required_evidence_ids']}`",
            f"- CrossEncoder Accuracy@1：`{item['crossencoder_accuracy_at_1']}`；Rank-1 证据：`{item['crossencoder_top4'][0]['evidence_ids'] if item['crossencoder_top4'] else []}`",
            f"- RRF Accuracy@4：`{item['rrf_accuracy_at_4']}`；证据覆盖率：`{item['rrf_evidence_coverage_at_4']}`；Top-4 证据：`{[e for d in item['rrf_top4'] for e in d['evidence_ids']]}`",
            f"- Answer Accuracy：`{item['answer_accuracy']}`；缺失检查项：`{item['missing_answer_checks']}`",
            f"- 引用证据覆盖率：`{item['citation_evidence_coverage']}`；引用：`{item['citations']}`",
            f"- API：`{item['api_status']}`；错误：`{item['api_error']}`",
            "", "**CrossEncoder Top-4**", "",
        ])
        for doc in item["crossencoder_top4"]:
            lines.append(
                f"- Rank {doc['rank']} · `{doc['source']}` · evidence={doc['evidence_ids']} · {doc['preview']}"
            )
        lines.append("")
    return "\n".join(lines)


def rescore_existing_report(report: Dict[str, Any]) -> Dict[str, Any]:
    cases = {row["id"]: row for row in load_cases(FIXTURE_DIR / "cases.jsonl")}
    for item in report.get("results", []):
        row = cases[item["id"]]
        answer_score = score_answer(str(item.get("model_answer") or ""), row["answer_checks"])
        item["standard_answer"] = row["standard_answer"]
        item["answer_checks"] = row["answer_checks"]
        item["answer_accuracy"] = answer_score["accuracy"]
        item["missing_answer_checks"] = answer_score["missing_checks"]
        item["answer_passed"] = bool(
            answer_score["accuracy"] == 1.0
            and item.get("citations")
            and item.get("api_status") == 200
        )
    results = report.get("results", [])
    summary = report["summary"]
    summary["model_answer_accuracy"] = round(
        statistics.mean(item["answer_accuracy"] for item in results), 4
    )
    summary["model_answer_pass_cases"] = sum(item["answer_passed"] for item in results)
    summary["passed"] = all(
        item["crossencoder_passed"] and item["rrf_passed"] and item["answer_passed"]
        for item in results
    )
    metadata = report.get("metadata_metrics", {})
    document_results = metadata.get("document_results", [])
    for item in document_results:
        item["passed"] = bool(
            item.get("extension") == ".md"
            and item.get("language") == "en"
            and int(item.get("section_count", 0)) > 0
        )
        item["llm_enrichment_completed"] = item.get("extraction_status") == "completed"
    if document_results:
        metadata["document_metadata_pass_rate"] = round(
            sum(item["passed"] for item in document_results) / len(document_results), 4
        )
        metadata["llm_metadata_enrichment_rate"] = round(
            sum(item["llm_enrichment_completed"] for item in document_results) / len(document_results), 4
        )
    report["rescored_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run long-document per-case Agent/RAG evaluation")
    parser.add_argument("--output-dir", default="test-results/agent-rag/case-study")
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--rescore-existing", action="store_true")
    parser.add_argument("--reuse-existing-answers", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.rescore_existing:
        report_path = output_dir / "report.json"
        if not report_path.exists():
            raise SystemExit(f"Existing report not found: {report_path}")
        report = rescore_existing_report(json.loads(report_path.read_text(encoding="utf-8")))
    else:
        reused_answers = None
        if args.reuse_existing_answers:
            report_path = output_dir / "report.json"
            if not report_path.exists():
                raise SystemExit(f"Existing report not found: {report_path}")
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            reused_answers = {item["id"]: item for item in previous.get("results", [])}
        with isolated_runtime(hide_zai_key=False):
            if args.require_llm and not os.getenv("ZAI_API_KEY"):
                raise SystemExit("ZAI_API_KEY is required")
            report = evaluate(output_dir, args.require_llm, reused_answers)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "comparison.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Fixture directory: {FIXTURE_DIR.resolve()}")
    print(f"Comparison report: {(output_dir / 'comparison.md').resolve()}")
    if not report["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
