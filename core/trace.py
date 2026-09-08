import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config import DB_NAME


def init_trace_db() -> None:
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                answer TEXT,
                retrieval_debug TEXT,
                tool_calls TEXT,
                citations TEXT,
                spans TEXT,
                metrics TEXT,
                safety_events TEXT,
                error TEXT,
                created_at TIMESTAMP NOT NULL
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_traces)").fetchall()}
        if "citations" not in columns:
            conn.execute("ALTER TABLE chat_traces ADD COLUMN citations TEXT")
        if "spans" not in columns:
            conn.execute("ALTER TABLE chat_traces ADD COLUMN spans TEXT")
        if "metrics" not in columns:
            conn.execute("ALTER TABLE chat_traces ADD COLUMN metrics TEXT")
        if "safety_events" not in columns:
            conn.execute("ALTER TABLE chat_traces ADD COLUMN safety_events TEXT")
        conn.commit()
    finally:
        conn.close()


def record_chat_trace(
    session_id: str,
    user_input: str,
    answer: str = "",
    retrieval_debug: Optional[Dict[str, Any]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
    spans: Optional[List[Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    safety_events: Optional[List[Dict[str, Any]]] = None,
    error: str = "",
) -> None:
    if spans is None or metrics is None:
        built_spans, built_metrics = build_trace_payload(
            user_input=user_input,
            retrieval_debug=retrieval_debug or {},
            tool_calls=tool_calls or [],
            answer=answer,
            error=error,
        )
        if spans is None:
            spans = built_spans
        if metrics is None:
            metrics = built_metrics
    if safety_events is None:
        safety_events = detect_safety_events(
            user_input=user_input,
            retrieval_debug=retrieval_debug or {},
            tool_calls=tool_calls or [],
        )

    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute(
            """
            INSERT INTO chat_traces
                (session_id, user_input, answer, retrieval_debug, tool_calls, citations, spans, metrics, safety_events, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_input,
                answer,
                json.dumps(retrieval_debug or {}, ensure_ascii=False),
                json.dumps(tool_calls or [], ensure_ascii=False),
                json.dumps(citations or [], ensure_ascii=False),
                json.dumps(spans or [], ensure_ascii=False),
                json.dumps(metrics or {}, ensure_ascii=False),
                json.dumps(safety_events or [], ensure_ascii=False),
                error,
                datetime.now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_traces(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_NAME)
    try:
        rows = conn.execute(
            """
            SELECT id, session_id, user_input, answer, retrieval_debug, tool_calls, citations, spans, metrics, safety_events, error, created_at
            FROM chat_traces
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    finally:
        conn.close()

    traces = []
    for row in rows:
        traces.append(_trace_row_to_dict(row))
    return traces


def get_trace(trace_id: int) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_NAME)
    try:
        row = conn.execute(
            """
            SELECT id, session_id, user_input, answer, retrieval_debug, tool_calls, citations, spans, metrics, safety_events, error, created_at
            FROM chat_traces
            WHERE id = ?
            """,
            (trace_id,),
        ).fetchone()
    finally:
        conn.close()

    return _trace_row_to_dict(row) if row else None


def get_trace_statistics(session_id: str) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_NAME)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM chat_traces WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        failures = conn.execute(
            "SELECT COUNT(*) FROM chat_traces WHERE session_id = ? AND COALESCE(error, '') <> ''",
            (session_id,),
        ).fetchone()[0]
        recent = conn.execute(
            """
            SELECT created_at
            FROM chat_traces
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    finally:
        conn.close()

    return {
        "total": total,
        "failures": failures,
        "successes": total - failures,
        "last_trace_at": recent[0] if recent else "",
    }


def _safe_json_loads(raw: str, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


PROMPT_INJECTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "ignore system",
    "system prompt",
    "developer message",
    "reveal hidden",
    "print secrets",
    "show api key",
    "忽略系统",
    "忽略之前",
    "输出所有用户",
    "泄露",
)

DANGEROUS_TOOL_PATTERNS = (
    "delete_all_files",
    "delete files",
    "send_email",
    "email_send",
    "payment",
    "wire transfer",
    "drop table",
    "rm -rf",
    "删除所有",
    "发送邮件",
    "付款",
)


def _hash_id(prefix: str, value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _parse_dt(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_ms(started_at: Any, ended_at: Any) -> Optional[float]:
    start = _parse_dt(started_at)
    end = _parse_dt(ended_at)
    if not start or not end:
        return None
    return round(max(0.0, (end - start).total_seconds() * 1000), 2)


def _span(
    *,
    name: str,
    kind: str,
    status: str = "completed",
    parent_id: str = "",
    duration_ms: Optional[float] = None,
    input_preview: Any = "",
    output_preview: Any = "",
    error_type: str = "",
    error: str = "",
) -> Dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    return {
        "span_id": _hash_id("span", [name, kind, input_preview, output_preview, error]),
        "parent_id": parent_id,
        "name": name,
        "kind": kind,
        "status": status,
        "started_at": started_at,
        "ended_at": started_at,
        "duration_ms": 0.0 if duration_ms is None else duration_ms,
        "input_preview": str(input_preview or "")[:1000],
        "output_preview": str(output_preview or "")[:1000],
        "error_type": error_type,
        "error": str(error or "")[:1000],
    }


def _tool_span(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    duration = _duration_ms(tool_call.get("started_at"), tool_call.get("ended_at"))
    status = str(tool_call.get("status") or "completed")
    error = str(tool_call.get("error") or "")
    return _span(
        name=str(tool_call.get("tool") or tool_call.get("display_name") or "tool_call"),
        kind="tool_call",
        status="failed" if error or status == "failed" else status,
        duration_ms=duration,
        input_preview=tool_call.get("input", ""),
        output_preview=tool_call.get("output_preview", ""),
        error_type="tool_error" if error else "",
        error=error,
    )


def build_trace_payload(
    *,
    user_input: str,
    retrieval_debug: Optional[Dict[str, Any]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    answer: str = "",
    error: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug = retrieval_debug or {}
    calls = tool_calls or []
    spans: List[Dict[str, Any]] = []

    route = str(debug.get("route") or "unknown")
    spans.append(
        _span(
            name="route",
            kind="router",
            status="completed",
            input_preview=user_input,
            output_preview={"route": route, "reason": debug.get("router_reason", "")},
        )
    )

    if debug.get("skill_selection"):
        spans.append(
            _span(
                name="skill_selection",
                kind="agent_step",
                status="completed",
                input_preview=user_input,
                output_preview=debug.get("skill_selection"),
            )
        )

    timing = debug.get("timing_ms") if isinstance(debug.get("timing_ms"), dict) else {}
    for name in ("dense", "bm25", "rrf", "rerank", "total_retrieval", "total"):
        if name in timing:
            spans.append(
                _span(
                    name=name,
                    kind="retrieval",
                    status="completed",
                    duration_ms=float(timing.get(name) or 0.0),
                    input_preview=debug.get("query", user_input),
                    output_preview={
                        "candidate_count": debug.get("candidate_count", 0),
                        "final_count": len(debug.get("final", []) or []),
                    },
                )
            )

    for call in calls:
        if isinstance(call, dict):
            spans.append(_tool_span(call))

    latency_candidates = [
        float(timing.get(key) or 0.0)
        for key in ("total", "total_retrieval")
        if key in timing
    ]
    if not latency_candidates:
        latency_candidates = [
            float(span.get("duration_ms") or 0.0)
            for span in spans
            if span.get("kind") in {"tool_call", "retrieval"}
        ]
    latency_ms = round(max(latency_candidates) if latency_candidates else 0.0, 2)
    retry_count = sum(int((call.get("execution") or {}).get("retry_count", 0)) for call in calls if isinstance(call, dict))
    metrics = {
        "latency_ms": latency_ms,
        "tool_call_count": len([call for call in calls if isinstance(call, dict)]),
        "retry_count": retry_count,
        "first_token_latency_ms": None,
        "token_usage": dict(debug.get("context_usage") or {}),
        "estimated_cost": None,
        "answer_chars": len(answer or ""),
        "error": str(error or ""),
    }
    return spans, metrics


def _scan_text_for_events(text: str, source: str) -> List[Dict[str, Any]]:
    lowered = text.casefold()
    events = []
    if any(pattern in lowered for pattern in PROMPT_INJECTION_PATTERNS):
        events.append(
            {
                "event_id": _hash_id("safety", [source, "prompt_injection", text[:240]]),
                "type": "prompt_injection_signal",
                "source": source,
                "severity": "medium",
                "message": "Prompt-injection-like text was detected and treated as data.",
                "matched_preview": text[:240],
            }
        )
    if any(pattern in lowered for pattern in DANGEROUS_TOOL_PATTERNS):
        events.append(
            {
                "event_id": _hash_id("safety", [source, "tool_injection", text[:240]]),
                "type": "tool_injection_signal",
                "source": source,
                "severity": "high",
                "message": "Dangerous-tool instruction text was detected and not trusted as an instruction.",
                "matched_preview": text[:240],
            }
        )
    return events


def detect_safety_events(
    *,
    user_input: str,
    retrieval_debug: Optional[Dict[str, Any]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    events = _scan_text_for_events(user_input or "", "user_input")
    debug = retrieval_debug or {}
    for index, doc in enumerate(debug.get("final", []) or [], start=1):
        if isinstance(doc, dict):
            events.extend(_scan_text_for_events(str(doc.get("content") or doc.get("preview") or ""), f"retrieved_chunk_{index}"))

    violations = debug.get("route_violation") or []
    for violation in violations:
        events.append(
            {
                "event_id": _hash_id("safety", ["route_violation", violation]),
                "type": "route_violation",
                "source": "tool_authorization",
                "severity": "high",
                "message": f"Tool call was outside the route allow-list: {violation}",
                "matched_preview": str(violation),
            }
        )

    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        if str(call.get("status") or "").casefold() == "failed" and call.get("error"):
            events.append(
                {
                    "event_id": _hash_id("safety", ["tool_error", call.get("tool"), call.get("error")]),
                    "type": "tool_error",
                    "source": str(call.get("tool") or "tool_call"),
                    "severity": "low",
                    "message": "Tool execution failed and was recorded for audit.",
                    "matched_preview": str(call.get("error"))[:240],
                }
            )
    return events


def _trace_row_to_dict(row) -> Dict[str, Any]:
    return {
        "id": row[0],
        "session_id": row[1],
        "user_input": row[2],
        "answer": row[3] or "",
        "retrieval_debug": _safe_json_loads(row[4], {}),
        "tool_calls": _safe_json_loads(row[5], []),
        "citations": _safe_json_loads(row[6], []),
        "spans": _safe_json_loads(row[7], []),
        "metrics": _safe_json_loads(row[8], {}),
        "safety_events": _safe_json_loads(row[9], []),
        "error": row[10] or "",
        "created_at": row[11],
    }
