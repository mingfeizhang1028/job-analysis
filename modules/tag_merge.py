from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
ALIAS_PATH = BASE_DIR / "data" / "llm_skill_alias.json"


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        value = value.strip()
        return [x.strip() for x in re.split(r"[，,；;、\n]+", value) if x.strip()] if value else []
    return []


def _normalize_by_alias(tags: List[str], alias_map: Dict[str, List[str]]) -> List[str]:
    reverse = {}
    for canonical, aliases in alias_map.items():
        reverse[canonical] = canonical
        for a in aliases:
            reverse[a] = canonical

    result = []
    seen = set()
    for tag in tags:
        norm = reverse.get(tag, tag)
        if norm and norm not in seen:
            result.append(norm)
            seen.add(norm)
    return result


DEFAULT_ALIAS_MAP = {
    "硬技能": {
        "AI工具使用": ["AI工具应用", "使用AI工具", "AI应用工具使用"],
        "AI流程搭建": ["AI工作流设计", "AI流程设计", "构建AI流程", "流程搭建"],
        "Prompt设计": ["Prompt Engineering", "提示词设计", "提示工程"],
        "产品运营": ["运营能力", "产品运营能力"],
        "用户体验分析": ["用户体验敏感度", "用户体验洞察", "UX分析"],
    },
    "软素质": {
        "沟通协作": ["沟通能力", "团队协作", "跨部门协作", "表达能力", "协调能力"],
        "结构化思维": ["结构化表达", "逻辑表达", "条理清晰"],
        "主动性": ["主动尝试", "主动探索", "积极主动"],
        "学习能力": ["快速学习", "学习意愿", "适应能力"],
        "问题解决能力": ["解决问题能力", "问题拆解能力"],
        "创新意识": ["创新思维", "创新能力"],
        "动手实践能力": ["实践能力", "动手能力"],
    },
    "业务职责": {
        "需求分析": ["业务需求分析", "需求拆解", "需求洞察"],
        "流程优化": ["流程设计", "流程改进", "流程搭建", "流程重塑"],
    },
    "行业场景": {
        "互联网": ["互联网行业", "互联网产品"],
        "AI应用": ["AIGC", "生成式AI", "大模型应用", "LLM应用"],
    },
}


def _load_alias_map() -> Dict[str, Dict[str, List[str]]]:
    if not ALIAS_PATH.exists():
        return DEFAULT_ALIAS_MAP
    try:
        data = json.loads(ALIAS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            merged = {**DEFAULT_ALIAS_MAP}
            for key, value in data.items():
                if isinstance(value, dict):
                    merged[key] = value
            return merged
    except Exception:
        pass
    return DEFAULT_ALIAS_MAP


def _dedup_keep_order(tags: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for tag in tags:
        if tag and tag not in seen:
            cleaned.append(tag)
            seen.add(tag)
    return cleaned


def merge_rule_and_llm_tags(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    alias_maps = _load_alias_map()

    merge_specs = [
        {
            "source_cols": ["硬技能标签", "规则必须技能", "规则加分技能", "规则工具栈", "LLM深度硬技能标签", "LLM深度工具栈", "LLM必须技能", "LLM加分技能", "LLM工具栈"],
            "final_col": "最终硬技能标签",
            "tag_type": "硬技能",
        },
        {
            "source_cols": ["软素质标签", "LLM深度软素质标签"],
            "final_col": "最终软素质标签",
            "tag_type": "软素质",
        },
        {
            "source_cols": ["业务职责标签", "规则岗位工作内容", "LLM深度业务职责标签", "LLM岗位工作内容", "LLM核心目标"],
            "final_col": "最终业务职责标签",
            "tag_type": "业务职责",
        },
        {
            "source_cols": ["行业场景标签", "规则所属行业", "LLM深度行业场景标签", "LLM深度岗位类型", "LLM所属行业", "LLM岗位类型"],
            "final_col": "最终行业场景标签",
            "tag_type": "行业场景",
        },
    ]

    for spec in merge_specs:
        final_values = []
        alias_map = alias_maps.get(spec["tag_type"], {})
        for _, row in result.iterrows():
            merged = []
            for col in spec["source_cols"]:
                merged.extend(_safe_list(row.get(col, [])))
            merged = _normalize_by_alias(merged, alias_map)
            final_values.append(_dedup_keep_order(merged))
        result[spec["final_col"]] = final_values

    all_final_values = []
    for _, row in result.iterrows():
        merged = []
        for col in ["最终硬技能标签", "最终软素质标签", "最终业务职责标签", "最终行业场景标签"]:
            merged.extend(_safe_list(row.get(col, [])))
        all_final_values.append(_dedup_keep_order(merged))
    result["最终全部标签"] = all_final_values

    return result
