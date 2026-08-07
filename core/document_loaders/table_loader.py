import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from langchain_core.documents import Document

from config import MAX_CHUNKS_PER_FILE


MIN_ROW_GROUP_SIZE = 50
MAX_TABLE_CHUNKS_PER_FILE = max(1, min(900, int(MAX_CHUNKS_PER_FILE) - 10))
MAX_CELL_CHARS = 160
MAX_SUMMARY_ROWS = 5
MAX_PROFILE_COLUMNS = 80


def _clean_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) > MAX_CELL_CHARS:
        return f"{text[:MAX_CELL_CHARS]}..."
    return text


def _stringify_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip() or f"column_{index + 1}" for index, column in enumerate(result.columns)]
    for column in result.columns:
        result[column] = result[column].map(_clean_value)
    return result.fillna("")


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    rows = frame.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _column_profile(frame: pd.DataFrame) -> str:
    pieces = []
    columns = list(frame.columns)
    for column in columns[:MAX_PROFILE_COLUMNS]:
        non_empty = int(frame[column].astype(str).str.strip().ne("").sum())
        samples = [item for item in frame[column].astype(str).head(3).tolist() if item]
        sample_text = ", ".join(samples)
        pieces.append(f"- {column}: non_empty={non_empty}; samples={sample_text}")
    if len(columns) > MAX_PROFILE_COLUMNS:
        pieces.append(f"- ... {len(columns) - MAX_PROFILE_COLUMNS} more columns omitted from profile")
    return "\n".join(pieces)


def _row_group_size(row_count: int, sheet_count: int) -> int:
    max_row_chunks = max(1, (MAX_TABLE_CHUNKS_PER_FILE - sheet_count) // max(sheet_count, 1))
    return max(MIN_ROW_GROUP_SIZE, math.ceil(max(row_count, 1) / max_row_chunks))


def _sheet_documents(file_name: str, sheet_name: str, raw_frame: pd.DataFrame, sheet_count: int) -> List[Document]:
    frame = _stringify_frame(raw_frame)
    docs: List[Document] = []
    columns = [str(column) for column in frame.columns]
    sheet_label = sheet_name or "Sheet1"

    summary_parts = [
        f"Table file: {file_name}",
        f"Sheet: {sheet_label}",
        f"Rows: {len(frame)}",
        f"Columns: {', '.join(columns)}",
        "Column profile:",
        _column_profile(frame),
    ]
    preview = _frame_to_markdown(frame.head(MAX_SUMMARY_ROWS))
    if preview:
        summary_parts.extend(["Preview:", preview])
    docs.append(
        Document(
            page_content="\n".join(summary_parts),
            metadata={
                "source": file_name,
                "file_type": Path(file_name).suffix.lower().lstrip("."),
                "chunk_kind": "table_summary",
                "pre_chunked": True,
                "sheet_name": sheet_label,
                "row_count": len(frame),
                "columns": columns,
            },
        )
    )

    row_group_size = _row_group_size(len(frame), sheet_count)
    for start in range(0, len(frame), row_group_size):
        end = min(start + row_group_size, len(frame))
        group = frame.iloc[start:end]
        markdown = _frame_to_markdown(group)
        if not markdown:
            continue
        docs.append(
            Document(
                page_content=(
                    f"Table file: {file_name}\n"
                    f"Sheet: {sheet_label}\n"
                    f"Rows {start + 1}-{end}\n"
                    f"{markdown}"
                ),
                metadata={
                    "source": file_name,
                    "file_type": Path(file_name).suffix.lower().lstrip("."),
                    "chunk_kind": "table_rows",
                    "pre_chunked": True,
                    "sheet_name": sheet_label,
                    "row_start": start + 1,
                    "row_end": end,
                    "row_group_size": row_group_size,
                    "columns": columns,
                },
            )
        )
    return docs


def _iter_excel_sheets(file_path: str) -> Iterable[tuple[str, pd.DataFrame]]:
    with pd.ExcelFile(file_path, engine="openpyxl") as workbook:
        for sheet_name in workbook.sheet_names:
            yield sheet_name, workbook.parse(sheet_name=sheet_name)


def load_table_documents(file_path: str, file_name: str) -> List[Document]:
    ext = Path(file_name).suffix.lower()
    if ext == ".csv":
        frames: Dict[str, pd.DataFrame] = {"CSV": pd.read_csv(file_path)}
    elif ext == ".xlsx":
        frames = dict(_iter_excel_sheets(file_path))
    elif ext == ".parquet":
        frames = {"Parquet": pd.read_parquet(file_path)}
    else:
        raise ValueError(f"Unsupported table file type: {ext}")

    docs: List[Document] = []
    sheet_count = max(len(frames), 1)
    for sheet_name, frame in frames.items():
        if frame.empty and not list(frame.columns):
            continue
        docs.extend(_sheet_documents(file_name, sheet_name, frame, sheet_count))
    if not docs:
        raise ValueError(f"{file_name} contains no readable table data.")
    return docs
