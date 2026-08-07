import json
import re
from typing import Any, Dict, List, Optional, Tuple

from db.skill_manager import (
    create_skill,
    delete_skill,
    get_skill,
    list_skills,
    update_skill,
)


class SkillCommandError(ValueError):
    pass


def _normalize(text: str) -> str:
    return text.strip().casefold()


def _is_skill_message(message: str) -> bool:
    lowered = _normalize(message)
    return "skill" in lowered or "技能" in lowered or "工作流" in lowered


def _find_json_object(message: str) -> Optional[Dict[str, Any]]:
    start = message.find("{")
    end = message.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(message[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SkillCommandError(f"JSON 解析失败: {exc}") from exc
    if not isinstance(value, dict):
        raise SkillCommandError("JSON 内容必须是对象。")
    return value


def _workflow_summary(skill: Dict[str, Any]) -> str:
    workflow = skill.get("workflow") or {}
    kind = str(workflow.get("kind") or "").strip()
    if kind == "table_analysis":
        return "后端表格分析工作流：读取 CSV/XLSX，清洗空行空列，统计列信息、缺失值和数值均值。"
    return f"后端工作流：{kind}" if kind else ""


def _compact_skill(skill: Dict[str, Any]) -> str:
    enabled = "启用" if skill.get("enabled") else "禁用"
    mode = str(skill.get("mode", "prompt"))
    mode_label = "工作流" if mode == "workflow" else "提示词"
    source = str(skill.get("source", "database"))
    slug = str(skill.get("slug", ""))
    source_label = "文件包" if source == "filesystem" else "数据库"
    trigger_terms = "、".join(skill.get("trigger_terms") or []) or "无"
    examples = skill.get("examples") or []
    example_text = f"\n  示例: {examples[0]}" if examples else ""
    workflow_text = f"\n  工作流: {_workflow_summary(skill)}" if mode == "workflow" else ""
    direct_text = f"\n  直接调用: /{slug}" if slug else ""
    return (
        f"- {skill.get('name')} [{mode_label} / {enabled} / {source_label}]\n"
        f"  用途: {skill.get('description')}\n"
        f"  触发词: {trigger_terms}"
        f"{direct_text}"
        f"{example_text}"
        f"{workflow_text}"
    )


def _find_skill(identifier: str) -> Optional[Dict[str, Any]]:
    needle = _normalize(identifier)
    if not needle:
        return None

    skill = get_skill(identifier.strip())
    if skill:
        return skill

    for candidate in list_skills():
        names = [
            str(candidate.get("name", "")),
            str(candidate.get("id", "")),
            *[str(term) for term in candidate.get("trigger_terms") or []],
        ]
        if any(_normalize(name) == needle for name in names):
            return candidate
    for candidate in list_skills():
        if needle in _normalize(str(candidate.get("name", ""))):
            return candidate
    return None


def _identifier_after(message: str, verbs: List[str]) -> str:
    text = message.strip()
    for verb in verbs:
        pattern = re.compile(re.escape(verb), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            identifier = text[match.end() :].strip(" :：，,。")
            identifier = re.sub(r"^(skill|技能)\s*", "", identifier, flags=re.IGNORECASE).strip(" :：，,。")
            return identifier
    return ""


def _default_table_workflow_payload(name: str = "数据分析助手") -> Dict[str, Any]:
    return {
        "name": name,
        "description": "分析上传的 CSV/XLSX 表格，返回行列数、列类型、缺失值和数值列均值。",
        "trigger_terms": [name, "数据分析", "表格分析", "data_analysis_skill", "数据处理skill"],
        "instruction": "使用后端 table_analysis 工作流分析上传表格，不要输出代码或伪造运行结果。",
        "examples": ["上传 CSV 后输入：用数据分析助手处理上传的csv"],
        "enabled": True,
        "mode": "workflow",
        "workflow": {
            "kind": "table_analysis",
            "steps": ["load_latest_table", "clean_table", "profile_columns", "numeric_means", "summarize"],
        },
    }


def _create_skill_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "name" not in payload:
        raise SkillCommandError("创建 skill 需要 name 字段。")
    return create_skill(
        name=str(payload.get("name") or "").strip(),
        description=str(payload.get("description") or "").strip() or "用户创建的 skill。",
        trigger_terms=[str(item) for item in payload.get("trigger_terms") or []],
        instruction=str(payload.get("instruction") or "").strip() or "当请求相关时按这个 skill 的说明处理。",
        examples=[str(item) for item in payload.get("examples") or []],
        enabled=bool(payload.get("enabled", True)),
        mode=str(payload.get("mode") or "prompt"),
        workflow=payload.get("workflow") if isinstance(payload.get("workflow"), dict) else {},
    )


def _update_skill_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    identifier = str(payload.get("id") or payload.get("name") or "").strip()
    if not identifier:
        raise SkillCommandError("修改 skill 需要 id 或 name 字段。")
    skill = _find_skill(identifier)
    if not skill:
        raise SkillCommandError(f"没有找到 skill: {identifier}")

    updates = {key: value for key, value in payload.items() if key not in {"id"}}
    updated = update_skill(str(skill["id"]), updates)
    if not updated:
        raise SkillCommandError(f"修改失败，skill 不存在: {identifier}")
    return updated


def _list_skills_response() -> Tuple[str, Dict[str, Any]]:
    skills = list_skills()
    if not skills:
        return "当前没有 skill。", {"operation": "list", "skills": []}
    answer = "当前 skills:\n" + "\n".join(_compact_skill(skill) for skill in skills)
    return answer, {"operation": "list", "skills": skills}


def handle_skill_chat_command(message: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not _is_skill_message(message):
        return None

    lowered = _normalize(message)
    payload = _find_json_object(message)

    try:
        if "skill" in lowered and len(lowered) <= 40 and not any(
            term in lowered
            for term in [
                "data_analysis",
                "run_skill_workflow",
                "使用",
                "用",
                "处理",
                "分析",
                "创建",
                "新增",
                "修改",
                "更新",
                "启用",
                "禁用",
                "删除",
                "create",
                "add",
                "update",
                "enable",
                "disable",
                "delete",
            ]
        ):
            return _list_skills_response()

        if any(term in lowered for term in ["列出", "查看", "有哪些", "list", "show"]) and any(
            term in lowered for term in ["skill", "技能"]
        ):
            return _list_skills_response()

        if any(term in lowered for term in ["启用", "enable"]) and any(term in lowered for term in ["skill", "技能"]):
            identifier = _identifier_after(message, ["启用", "enable"])
            skill = _find_skill(identifier)
            if not skill:
                raise SkillCommandError(f"没有找到 skill: {identifier}")
            updated = update_skill(str(skill["id"]), {"enabled": True})
            return f"已启用 skill:\n{_compact_skill(updated or skill)}", {"operation": "enable", "skill": updated or skill}

        if any(term in lowered for term in ["禁用", "disable"]) and any(term in lowered for term in ["skill", "技能"]):
            identifier = _identifier_after(message, ["禁用", "disable"])
            skill = _find_skill(identifier)
            if not skill:
                raise SkillCommandError(f"没有找到 skill: {identifier}")
            updated = update_skill(str(skill["id"]), {"enabled": False})
            return f"已禁用 skill:\n{_compact_skill(updated or skill)}", {"operation": "disable", "skill": updated or skill}

        if any(term in lowered for term in ["删除", "delete", "remove"]) and any(term in lowered for term in ["skill", "技能"]):
            identifier = _identifier_after(message, ["删除", "delete", "remove"])
            skill = _find_skill(identifier)
            if not skill:
                raise SkillCommandError(f"没有找到 skill: {identifier}")
            delete_skill(str(skill["id"]))
            return f"已删除 skill: {skill.get('name')} ({skill.get('id')})", {"operation": "delete", "skill": skill}

        if any(term in lowered for term in ["修改", "更新", "update", "patch"]) and any(term in lowered for term in ["skill", "技能"]):
            if payload is None:
                return "请用 JSON 指定要修改的 skill，至少包含 id 或 name。", {"operation": "update_help"}
            skill = _update_skill_from_payload(payload)
            return f"已修改 skill:\n{_compact_skill(skill)}", {"operation": "update", "skill": skill}

        if any(term in lowered for term in ["创建", "新增", "create", "add"]) and any(term in lowered for term in ["skill", "技能"]):
            if payload is None and ("data_analysis" in lowered or "数据分析" in lowered or "数据处理" in lowered):
                payload = _default_table_workflow_payload()
            if payload is None:
                return (
                    "请用 JSON 指定要创建的 skill，例如：\n"
                    '创建skill {"name":"中文摘要助手","description":"把内容整理成简短中文要点","trigger_terms":["中文摘要"],'
                    '"instruction":"用中文提炼 3-5 个要点","mode":"prompt"}',
                    {"operation": "create_help"},
                )
            skill = _create_skill_from_payload(payload)
            return f"已创建 skill:\n{_compact_skill(skill)}", {"operation": "create", "skill": skill}
    except SkillCommandError as exc:
        return str(exc), {"operation": "error", "error": str(exc)}

    return None
