import json
import os
import sqlite3
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from config import DB_NAME

VALID_SKILL_MODES = {"prompt", "workflow"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"


def _connect():
    return sqlite3.connect(DB_NAME)


def _now() -> datetime:
    return datetime.now()


def _loads_list(raw: str) -> List[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dumps_list(value: Optional[List[str]]) -> str:
    return json.dumps(value or [], ensure_ascii=False)


def _loads_object(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _dumps_object(value: Optional[Dict[str, Any]]) -> str:
    if not isinstance(value, dict):
        return "{}"
    return json.dumps(value, ensure_ascii=False)


def _normalize_mode(value: Any) -> str:
    mode = str(value or "prompt").strip().casefold()
    return mode if mode in VALID_SKILL_MODES else "prompt"


def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _row_to_skill(row) -> Dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "trigger_terms": _loads_list(row[3]),
        "instruction": row[4],
        "examples": _loads_list(row[5]),
        "enabled": bool(row[6]),
        "created_at": row[7],
        "updated_at": row[8],
        "mode": _normalize_mode(row[9] if len(row) > 9 else "prompt"),
        "workflow": _loads_object(row[10] if len(row) > 10 else "{}"),
        "source": "database",
        "path": "",
        "slug": "",
        "frontmatter": {},
    }


def _skill_roots() -> List[Path]:
    configured = os.getenv("CHATBOT_SKILLS_DIR", "").strip()
    raw_roots = configured.split(os.pathsep) if configured else [str(DEFAULT_SKILLS_DIR)]
    return [Path(root).resolve() for root in raw_roots if root.strip()]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^\w.-]+", "-", value.strip().lower(), flags=re.UNICODE).strip("-")
    return slug or uuid.uuid4().hex[:12]


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _split_frontmatter(raw: str) -> tuple[Dict[str, Any], str]:
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text.strip()

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()

    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text.strip()

    frontmatter_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).strip()
    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        frontmatter = {}
    return frontmatter if isinstance(frontmatter, dict) else {}, body


def _skill_assets(skill_dir: Path) -> List[str]:
    assets = []
    for path in skill_dir.rglob("*"):
        if path.is_file() and path.name != "SKILL.md":
            assets.append(str(path.relative_to(skill_dir)).replace("\\", "/"))
    return sorted(assets)


def _load_filesystem_skill(skill_file: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = skill_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = skill_file.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None

    frontmatter, body = _split_frontmatter(raw)
    skill_dir = skill_file.parent
    slug = _slugify(str(frontmatter.get("slug") or skill_dir.name))
    name = str(frontmatter.get("name") or skill_dir.name).strip()
    description = str(frontmatter.get("description") or "").strip()
    trigger_terms = (
        _as_list(frontmatter.get("trigger_terms"))
        + _as_list(frontmatter.get("triggers"))
        + _as_list(frontmatter.get("aliases"))
    )
    examples = _as_list(frontmatter.get("examples"))
    instruction = body or str(frontmatter.get("instruction") or "").strip()
    mode = _normalize_mode(frontmatter.get("mode"))
    workflow = frontmatter.get("workflow") if isinstance(frontmatter.get("workflow"), dict) else {}
    enabled = bool(frontmatter.get("enabled", True))
    updated_at = datetime.fromtimestamp(skill_file.stat().st_mtime).isoformat(timespec="seconds")

    return {
        "id": f"fs:{slug}",
        "name": name,
        "description": description,
        "trigger_terms": list(dict.fromkeys([term for term in trigger_terms if term])),
        "instruction": instruction,
        "examples": examples,
        "enabled": enabled,
        "created_at": updated_at,
        "updated_at": updated_at,
        "mode": mode,
        "workflow": workflow,
        "source": "filesystem",
        "path": str(skill_file),
        "slug": slug,
        "frontmatter": dict(frontmatter),
        "assets": _skill_assets(skill_dir),
    }


def _list_filesystem_skills(enabled: Optional[bool] = None) -> List[Dict[str, Any]]:
    skills: List[Dict[str, Any]] = []
    for root in _skill_roots():
        if not root.exists():
            continue
        for skill_file in sorted(root.glob("*/SKILL.md")):
            skill = _load_filesystem_skill(skill_file)
            if not skill:
                continue
            if enabled is not None and bool(skill.get("enabled")) != enabled:
                continue
            skills.append(skill)
    return skills


def _list_database_skills(enabled: Optional[bool] = None) -> List[Dict[str, Any]]:
    init_skill_db()
    conn = _connect()
    try:
        if enabled is None:
            rows = conn.execute(
                """
                SELECT id, name, description, trigger_terms, instruction, examples, enabled, created_at, updated_at, mode, workflow
                FROM skills
                ORDER BY updated_at DESC
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, name, description, trigger_terms, instruction, examples, enabled, created_at, updated_at, mode, workflow
                FROM skills
                WHERE enabled = ?
                ORDER BY updated_at DESC
                """,
                (1 if enabled else 0,),
            ).fetchall()
    finally:
        conn.close()
    return [_row_to_skill(row) for row in rows]


def init_skill_db() -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                trigger_terms TEXT NOT NULL,
                instruction TEXT NOT NULL,
                examples TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                mode TEXT NOT NULL DEFAULT 'prompt',
                workflow TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        _ensure_column(conn, "skills", "mode", "mode TEXT NOT NULL DEFAULT 'prompt'")
        _ensure_column(conn, "skills", "workflow", "workflow TEXT NOT NULL DEFAULT '{}'")
        conn.commit()
    finally:
        conn.close()


def create_skill(
    *,
    name: str,
    description: str,
    trigger_terms: List[str],
    instruction: str,
    examples: Optional[List[str]] = None,
    enabled: bool = True,
    mode: str = "prompt",
    workflow: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    init_skill_db()
    skill_id = str(uuid.uuid4())
    now = _now()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO skills
                (id, name, description, trigger_terms, instruction, examples, enabled, created_at, updated_at, mode, workflow)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill_id,
                name.strip(),
                description.strip(),
                _dumps_list([term.strip() for term in trigger_terms if term.strip()]),
                instruction.strip(),
                _dumps_list([item.strip() for item in examples or [] if item.strip()]),
                1 if enabled else 0,
                now,
                now,
                _normalize_mode(mode),
                _dumps_object(workflow),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    skill = get_skill(skill_id)
    if skill is None:
        raise RuntimeError("Skill creation failed.")
    return skill


def list_skills(enabled: Optional[bool] = None) -> List[Dict[str, Any]]:
    skills = _list_database_skills(enabled=enabled) + _list_filesystem_skills(enabled=enabled)
    skills.sort(key=lambda item: (-_updated_sort_value(item), item.get("source", ""), item.get("name", "")))
    return skills


def get_skill(skill_id: str) -> Optional[Dict[str, Any]]:
    for skill in _list_filesystem_skills(enabled=None):
        identifiers = {
            str(skill.get("id", "")),
            str(skill.get("slug", "")),
            str(skill.get("name", "")),
        }
        if str(skill_id).strip() in identifiers:
            return skill

    init_skill_db()
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT id, name, description, trigger_terms, instruction, examples, enabled, created_at, updated_at, mode, workflow
            FROM skills
            WHERE id = ?
            """,
            (skill_id,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_skill(row) if row else None


def update_skill(skill_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if str(skill_id).startswith("fs:"):
        return None

    init_skill_db()
    allowed = {
        "name",
        "description",
        "trigger_terms",
        "instruction",
        "examples",
        "enabled",
        "mode",
        "workflow",
    }
    assignments = []
    values: List[Any] = []
    for key, value in updates.items():
        if key not in allowed:
            continue
        if key in {"trigger_terms", "examples"}:
            value = _dumps_list([str(item).strip() for item in value or [] if str(item).strip()])
        elif key == "enabled":
            value = 1 if bool(value) else 0
        elif key == "mode":
            value = _normalize_mode(value)
        elif key == "workflow":
            value = _dumps_object(value if isinstance(value, dict) else {})
        else:
            value = str(value).strip()
        assignments.append(f"{key} = ?")
        values.append(value)

    if not assignments:
        return get_skill(skill_id)

    assignments.append("updated_at = ?")
    values.append(_now())
    values.append(skill_id)

    conn = _connect()
    try:
        cursor = conn.execute(
            f"UPDATE skills SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None
    finally:
        conn.close()
    return get_skill(skill_id)


def delete_skill(skill_id: str) -> bool:
    if str(skill_id).startswith("fs:"):
        return False

    init_skill_db()
    conn = _connect()
    try:
        cursor = conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def replace_all_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    init_skill_db()
    conn = _connect()
    now = _now()
    try:
        conn.execute("DELETE FROM skills")
        for skill in skills:
            skill_id = str(skill.get("id") or uuid.uuid4())
            conn.execute(
                """
                INSERT INTO skills
                    (id, name, description, trigger_terms, instruction, examples, enabled, created_at, updated_at, mode, workflow)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    skill_id,
                    str(skill.get("name") or "").strip(),
                    str(skill.get("description") or "").strip(),
                    _dumps_list([str(term).strip() for term in skill.get("trigger_terms") or [] if str(term).strip()]),
                    str(skill.get("instruction") or "").strip(),
                    _dumps_list([str(item).strip() for item in skill.get("examples") or [] if str(item).strip()]),
                    1 if bool(skill.get("enabled", True)) else 0,
                    now,
                    now,
                    _normalize_mode(skill.get("mode")),
                    _dumps_object(skill.get("workflow") if isinstance(skill.get("workflow"), dict) else {}),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return list_skills()


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle) and needle in haystack


def _updated_sort_value(item: Dict[str, Any]) -> float:
    value = item.get("updated_at")
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


def match_skills(message: str, limit: int = 3) -> List[Dict[str, Any]]:
    query = message.casefold()
    scored = []
    for skill in list_skills(enabled=True):
        score = 0
        matched_terms = []
        name = str(skill.get("name", ""))
        slug = str(skill.get("slug", ""))
        direct_tokens = [f"/{slug.casefold()}"] if slug else []
        if name:
            direct_tokens.append(f"/{_slugify(name).casefold()}")
        for token in direct_tokens:
            if token and token in query:
                score += 20
                matched_terms.append(token)
        for term in skill["trigger_terms"]:
            normalized = term.casefold().strip()
            if _contains(query, normalized):
                score += 3
                matched_terms.append(term)
        if _contains(query, name.casefold()):
            score += 2
        if _contains(query, str(skill.get("description", "")).casefold()):
            score += 1
        if score > 0:
            item = dict(skill)
            item["match_score"] = score
            item["matched_terms"] = matched_terms
            scored.append(item)

    scored.sort(key=lambda item: (-item["match_score"], -_updated_sort_value(item), item["name"]))
    return scored[:limit]


_WORKFLOW_INTENT_TERMS = (
    "工作流",
    "workflow",
    "run_skill_workflow",
    "数据处理",
    "数据分析",
    "表格分析",
    "处理上传",
    "处理文件",
    "data_analysis",
    "data analysis",
    "csv",
    "xlsx",
    "excel",
)

_TABLE_WORKFLOW_TERMS = (
    "数据",
    "data",
    "analysis",
    "analyze",
    "表格",
    "table",
    "sheet",
    "dataframe",
    "workflow",
    "run_skill_workflow",
    "分析",
    "处理",
    "清洗",
    "均值",
    "平均",
    "统计",
    "缺失值",
    "csv",
    "xlsx",
    "excel",
)


def infer_workflow_skills(message: str, limit: int = 1) -> List[Dict[str, Any]]:
    """Fallback matcher for workflow intents that do not name a skill exactly."""
    query = message.casefold()
    if not any(term.casefold() in query for term in _WORKFLOW_INTENT_TERMS):
        return []
    if not any(term.casefold() in query for term in _TABLE_WORKFLOW_TERMS):
        return []

    candidates = []
    for skill in list_skills(enabled=True):
        if str(skill.get("mode", "prompt")).casefold() != "workflow":
            continue

        workflow = skill.get("workflow") or {}
        kind = str(workflow.get("kind") or "").casefold()
        score = 0
        if kind == "table_analysis" and any(term.casefold() in query for term in _TABLE_WORKFLOW_TERMS):
            score += 4
        if "skill" in query or "助手" in query:
            score += 1
        if _contains(query, str(skill.get("name", "")).casefold()):
            score += 3
        if score > 0:
            item = dict(skill)
            item["match_score"] = score
            item["matched_terms"] = ["workflow_intent"]
            candidates.append(item)

    candidates.sort(key=lambda item: (-item["match_score"], -_updated_sort_value(item), item["name"]))
    return candidates[:limit]


def match_skills_with_workflow_fallback(message: str, limit: int = 3) -> List[Dict[str, Any]]:
    matched = match_skills(message, limit=limit)
    if any(str(skill.get("mode", "prompt")).casefold() == "workflow" for skill in matched):
        return matched

    inferred = infer_workflow_skills(message, limit=1)
    existing_ids = {str(skill.get("id", "")) for skill in matched}
    for skill in inferred:
        if str(skill.get("id", "")) not in existing_ids:
            matched.insert(0, skill)
    return matched[:limit]


def summarize_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "id": str(skill.get("id", "")),
            "name": str(skill.get("name", "")),
            "description": str(skill.get("description", "")),
            "mode": str(skill.get("mode", "prompt")),
            "source": str(skill.get("source", "database")),
            "path": str(skill.get("path", "")),
        }
        for skill in skills
    ]
