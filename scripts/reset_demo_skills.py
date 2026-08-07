import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from db.skill_manager import replace_all_skills  # noqa: E402


DEMO_SKILLS = [
    {
        "id": "demo_table_analysis",
        "name": "数据分析助手",
        "description": "分析上传的 CSV/XLSX 表格，返回行列数、列类型、缺失值和数值列均值。",
        "trigger_terms": ["数据分析助手", "数据分析", "表格分析", "数据处理", "data_analysis_skill"],
        "instruction": "使用后端 table_analysis 工作流分析上传表格，不要输出代码或伪造运行结果。",
        "examples": ["上传 CSV 后输入：用数据分析助手处理上传的csv"],
        "enabled": True,
        "mode": "workflow",
        "workflow": {
            "kind": "table_analysis",
            "steps": ["load_latest_table", "clean_table", "profile_columns", "numeric_means", "summarize"],
        },
    },
    {
        "id": "demo_chinese_summary",
        "name": "中文摘要助手",
        "description": "把用户提供的内容整理成简短、清晰的中文要点。",
        "trigger_terms": ["中文摘要", "总结一下", "提炼要点", "摘要助手", "摘要skill", "摘要 skill", "摘要"],
        "instruction": (
            "当用户要求中文摘要、总结或提炼要点时，用中文输出 3-5 条要点。"
            "保留关键事实和结论，不编造原文没有的信息。"
        ),
        "examples": ["请用中文摘要总结以下内容：..."],
        "enabled": True,
        "mode": "prompt",
        "workflow": {},
    },
    {
        "id": "demo_deep_document_analysis",
        "name": "文档深度分析助手",
        "description": "对上传文档做多步骤结构化分析，提炼主题、证据、亮点、风险和后续问题。",
        "trigger_terms": ["文档深度分析", "深度分析文档", "文档分析skill", "文档审阅", "结构化分析", "复杂文档处理"],
        "instruction": (
            "当用户要求对上传文档做深度分析、结构化分析、审阅或复杂文档处理时，按以下流程回答：\n"
            "1. 先用 1-2 句话说明文档核心主题。\n"
            "2. 提炼 3-6 个关键事实或论点；每一点都尽量基于检索到的文档证据。\n"
            "3. 单独列出文档中的亮点、风险/缺口、可追问问题。\n"
            "4. 如果文档像简历、项目材料或报告，补充适合的行动建议。\n"
            "5. 不要编造文档没有的信息；证据不足时明确说哪些部分无法确认。"
        ),
        "examples": [
            "请用文档深度分析助手分析我上传的 PDF",
            "对已经上传的文档做结构化分析，列出亮点和风险",
        ],
        "enabled": True,
        "mode": "prompt",
        "workflow": {},
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset demo skills to a small readable Chinese test set.")
    parser.add_argument("--yes", action="store_true", help="Confirm deleting all existing skills before inserting demo skills.")
    args = parser.parse_args()

    if not args.yes:
        print("This will delete all existing skills and insert two demo skills. Re-run with --yes to confirm.")
        return 2

    skills = replace_all_skills(DEMO_SKILLS)
    print(f"Reset complete. Current skills: {len(skills)}")
    for skill in skills:
        print(f"- {skill['name']} ({skill['mode']}, {'启用' if skill['enabled'] else '禁用'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
