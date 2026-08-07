import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from config import DATA_DIR
from db.session_manager import get_session_documents


class SkillWorkflowError(ValueError):
    pass


TABLE_EXTENSIONS = {".csv", ".xlsx", ".parquet"}
SUPPORTED_WORKFLOW_KINDS = {"table_analysis"}


class SkillExecutionRecord(BaseModel):
    call_id: str
    status: Literal["completed", "failed"]
    retry_count: int = 0
    timing_ms: float = 0.0
    idempotency_key: str
    error_type: Optional[str] = None


class TableAnalysisWorkflowInput(BaseModel):
    kind: Literal["table_analysis"] = "table_analysis"
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    sheet_name: Optional[str | int] = 0
    max_retries: int = Field(default=0, ge=0, le=2)
    steps: List[str] = Field(default_factory=list)


class TableAnalysisWorkflowOutput(BaseModel):
    skill_id: str
    skill_name: str
    workflow_kind: Literal["table_analysis"]
    document: Dict[str, Any]
    raw_shape: Dict[str, int]
    clean_shape: Dict[str, int]
    column_names: List[str]
    column_types: Dict[str, str]
    missing_values: Dict[str, int]
    numeric_means: Dict[str, Optional[float]]
    cleaning_notes: List[str]
    errors: List[str] = Field(default_factory=list)
    execution: SkillExecutionRecord


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _idempotency_key(session_id: str, skill: Dict[str, Any], workflow: Dict[str, Any]) -> str:
    raw = f"{session_id}|{skill.get('id', '')}|{_canonical_json(workflow)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _execution_record(
    *,
    call_id: str,
    status: Literal["completed", "failed"],
    retry_count: int,
    timing_ms: float,
    idempotency_key: str,
    error_type: Optional[str] = None,
) -> Dict[str, Any]:
    return _model_dump(
        SkillExecutionRecord(
            call_id=call_id,
            status=status,
            retry_count=retry_count,
            timing_ms=round(timing_ms, 2),
            idempotency_key=idempotency_key,
            error_type=error_type,
        )
    )


def _safe_data_path(raw_path: str) -> Path:
    if not raw_path:
        raise SkillWorkflowError("Document file path is missing.")

    file_path = Path(raw_path).resolve()
    data_root = Path(DATA_DIR).resolve()
    if file_path != data_root and data_root not in file_path.parents:
        raise SkillWorkflowError("Document path is outside the data directory.")
    if not file_path.exists():
        raise SkillWorkflowError("Original uploaded file is missing.")
    return file_path


def _select_table_document(session_id: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
    target_file_id = str(workflow.get("file_id") or "").strip()
    target_file_name = str(workflow.get("file_name") or "").strip().casefold()
    candidates = []

    for file_id, file_name, chunk_count, status, uploaded_at, file_path in get_session_documents(session_id):
        ext = Path(str(file_name)).suffix.lower()
        if str(status).casefold() != "completed" or ext not in TABLE_EXTENSIONS:
            continue
        if target_file_id and str(file_id) != target_file_id:
            continue
        if target_file_name and target_file_name not in str(file_name).casefold():
            continue
        candidates.append(
            {
                "id": str(file_id),
                "file_name": str(file_name),
                "chunk_count": int(chunk_count or 0),
                "status": str(status),
                "uploaded_at": str(uploaded_at),
                "file_path": str(file_path),
                "extension": ext,
            }
        )

    if not candidates:
        raise SkillWorkflowError("当前会话没有可分析的 CSV/XLSX 文件。")
    return candidates[0]


def _dedupe_columns(columns: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    result = []
    for index, column in enumerate(columns, start=1):
        normalized = str(column).strip() or f"column_{index}"
        count = seen.get(normalized, 0)
        seen[normalized] = count + 1
        result.append(normalized if count == 0 else f"{normalized}_{count + 1}")
    return result


def _load_frame(file_path: Path, extension: str, workflow: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    if extension == ".csv":
        return pd.read_csv(file_path), "CSV"
    if extension == ".parquet":
        return pd.read_parquet(file_path), "Parquet"

    sheet_name = workflow.get("sheet_name", 0)
    loaded = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    if isinstance(loaded, dict):
        first_sheet_name = next(iter(loaded))
        return loaded[first_sheet_name], str(first_sheet_name)
    return loaded, str(sheet_name if sheet_name != 0 else "first sheet")


def _clean_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    notes = []
    original_rows, original_cols = frame.shape
    cleaned = frame.copy()
    cleaned.columns = _dedupe_columns([str(column) for column in cleaned.columns])
    notes.append("Normalized column names and deduplicated repeated names.")

    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    dropped_rows = original_rows - len(cleaned)
    dropped_cols = original_cols - len(cleaned.columns)
    if dropped_rows or dropped_cols:
        notes.append(f"Dropped {dropped_rows} empty rows and {dropped_cols} empty columns.")
    else:
        notes.append("No fully empty rows or columns were removed.")

    converted = []
    for column in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            continue
        as_text = cleaned[column].astype(str).str.replace(",", "", regex=False).str.strip()
        numeric = pd.to_numeric(as_text, errors="coerce")
        non_null_original = int(cleaned[column].notna().sum())
        non_null_numeric = int(numeric.notna().sum())
        if non_null_original and non_null_numeric / non_null_original >= 0.8:
            cleaned[column] = numeric
            converted.append(str(column))

    if converted:
        notes.append(f"Converted likely numeric columns: {', '.join(converted)}.")
    else:
        notes.append("No additional text columns met the numeric conversion threshold.")
    return cleaned, notes


def _table_profile(frame: pd.DataFrame) -> Dict[str, Any]:
    column_types = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    missing_values = {str(column): int(count) for column, count in frame.isna().sum().items() if int(count) > 0}
    numeric_means = {
        str(column): None if pd.isna(value) else float(value)
        for column, value in frame.select_dtypes(include="number").mean(numeric_only=True).items()
    }
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": [str(column) for column in frame.columns],
        "column_types": column_types,
        "missing_values": missing_values,
        "numeric_means": numeric_means,
    }


def run_table_analysis_workflow(session_id: str, skill: Dict[str, Any]) -> Dict[str, Any]:
    workflow = _model_dump(TableAnalysisWorkflowInput(**dict(skill.get("workflow") or {})))
    document = _select_table_document(session_id, workflow)
    file_path = _safe_data_path(document["file_path"])

    raw_frame, sheet_name = _load_frame(file_path, document["extension"], workflow)
    cleaned_frame, cleaning_notes = _clean_frame(raw_frame)
    profile = _table_profile(cleaned_frame)
    return {
        "skill_id": str(skill.get("id", "")),
        "skill_name": str(skill.get("name", "")),
        "workflow_kind": "table_analysis",
        "document": {
            "id": document["id"],
            "file_name": document["file_name"],
            "uploaded_at": document["uploaded_at"],
            "sheet_name": sheet_name,
        },
        "raw_shape": {"rows": int(raw_frame.shape[0]), "columns": int(raw_frame.shape[1])},
        "clean_shape": {"rows": profile["rows"], "columns": profile["columns"]},
        "column_names": profile["column_names"],
        "column_types": profile["column_types"],
        "missing_values": profile["missing_values"],
        "numeric_means": profile["numeric_means"],
        "cleaning_notes": cleaning_notes,
        "errors": [],
    }


def _run_workflow_with_execution(session_id: str, skill: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    workflow = dict(skill.get("workflow") or {})
    validated_workflow = _model_dump(TableAnalysisWorkflowInput(**workflow))
    skill_for_run = dict(skill)
    skill_for_run["workflow"] = validated_workflow
    idem_key = _idempotency_key(session_id, skill_for_run, validated_workflow)
    call_id = f"skill_{uuid.uuid4().hex[:12]}"
    max_retries = int(validated_workflow.get("max_retries") or 0)
    started = time.perf_counter()
    retry_count = 0
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            result = run_table_analysis_workflow(session_id, skill_for_run)
            timing_ms = (time.perf_counter() - started) * 1000
            execution = _execution_record(
                call_id=call_id,
                status="completed",
                retry_count=retry_count,
                timing_ms=timing_ms,
                idempotency_key=idem_key,
            )
            result["execution"] = execution
            validated_result = TableAnalysisWorkflowOutput(**result)
            return _model_dump(validated_result), execution
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                retry_count += 1
                continue
            break

    timing_ms = (time.perf_counter() - started) * 1000
    error_type = type(last_error).__name__ if last_error else "SkillWorkflowError"
    execution = _execution_record(
        call_id=call_id,
        status="failed",
        retry_count=retry_count,
        timing_ms=timing_ms,
        idempotency_key=idem_key,
        error_type=error_type,
    )
    if isinstance(last_error, ValidationError):
        raise SkillWorkflowError(f"Invalid workflow schema: {last_error}") from last_error
    raise SkillWorkflowError(str(last_error or "workflow failed")) from last_error


def format_workflow_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    blocks = [
        "Skill workflow execution results for this request. Use these computed results directly; do not recalculate or invent values."
    ]
    for index, result in enumerate(results, start=1):
        document = result.get("document") or {}
        block = [
            f"[Workflow {index}] {result.get('skill_name', '')} ({result.get('workflow_kind', '')})",
            f"File: {document.get('file_name', '')}",
            f"Sheet: {document.get('sheet_name', '')}",
            f"Raw shape: {result.get('raw_shape', {})}",
            f"Clean shape: {result.get('clean_shape', {})}",
            f"Column types: {result.get('column_types', {})}",
            f"Missing values: {result.get('missing_values', {})}",
            f"Numeric means: {result.get('numeric_means', {})}",
            f"Cleaning notes: {' '.join(result.get('cleaning_notes') or [])}",
        ]
        errors = result.get("errors") or []
        if errors:
            block.append(f"Errors: {errors}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def run_skill_workflows(session_id: str, skills: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    workflow_skills = [skill for skill in skills if str(skill.get("mode", "prompt")).casefold() == "workflow"]
    results: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []

    for skill in workflow_skills:
        workflow = dict(skill.get("workflow") or {})
        kind = str(workflow.get("kind") or "").strip().casefold()
        started_tool_call = {
            "tool": "run_skill_workflow",
            "display_name": "Skill workflow",
            "input": {"skill_id": skill.get("id"), "skill_name": skill.get("name"), "workflow": workflow},
            "status": "running",
            "output_preview": "",
            "error": "",
            "started_at": "",
            "ended_at": "",
            "execution": {},
        }
        try:
            if kind not in SUPPORTED_WORKFLOW_KINDS:
                raise SkillWorkflowError(f"Unsupported workflow kind: {kind or '<missing>'}.")
            result, execution = _run_workflow_with_execution(session_id, skill)
            results.append(result)
            started_tool_call["status"] = "completed"
            started_tool_call["execution"] = execution
            started_tool_call["output_preview"] = str(
                {
                    "file": (result.get("document") or {}).get("file_name"),
                    "clean_shape": result.get("clean_shape"),
                    "numeric_means": result.get("numeric_means"),
                    "execution": execution,
                }
            )[:1000]
        except Exception as exc:
            started_tool_call["status"] = "failed"
            started_tool_call["error"] = str(exc)
            started_tool_call["execution"] = started_tool_call.get("execution") or {}
            tool_calls.append(started_tool_call)
            raise SkillWorkflowError(str(exc)) from exc
        tool_calls.append(started_tool_call)

    return results, format_workflow_context(results), tool_calls
