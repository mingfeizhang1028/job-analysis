from __future__ import annotations

from typing import Optional

TAG_SOURCE_OPTIONS = ["最终标签", "词典标签", "LLM标签"]

TAG_SOURCE_COLUMN_MAP: dict[str, dict[str, str]] = {
    "词典标签": {
        "硬技能": "硬技能标签",
        "软素质": "软素质标签",
        "业务职责": "业务职责标签",
        "行业场景": "行业场景标签",
        "全部": "全部标签",
    },
    "最终标签": {
        "硬技能": "最终硬技能标签",
        "软素质": "最终软素质标签",
        "业务职责": "最终业务职责标签",
        "行业场景": "最终行业场景标签",
        "全部": "最终全部标签",
    },
    "LLM标签": {
        "业务职责": "LLM岗位工作内容",
        "行业场景": "LLM所属行业",
    },
}


def get_supported_tag_sources() -> list[str]:
    return list(TAG_SOURCE_OPTIONS)


def resolve_tag_column(tag_type: str, source_mode: str = "最终标签") -> Optional[str]:
    return TAG_SOURCE_COLUMN_MAP.get(source_mode, {}).get(tag_type)


def get_source_mode_hint(tag_type: str, source_mode: str) -> str:
    if source_mode == "最终标签":
        return "推荐优先使用最终融合标签，适合做统计、网络分析与后续人岗匹配。"
    if source_mode == "词典标签":
        return "词典标签适合检查规则覆盖情况，便于诊断 tag_dict / 规则抽取效果。"
    if source_mode == "LLM标签" and tag_type == "业务职责":
        return "当前业务职责的 LLM 来源使用“LLM岗位工作内容”字段，可能包含短句而不是归一化标签，建议优先参考最终标签。"
    if source_mode == "LLM标签" and tag_type == "行业场景":
        return "当前行业场景的 LLM 来源直接使用“LLM所属行业”字段，适合质量检查，但稳定性可能弱于最终标签。"
    if source_mode == "LLM标签":
        return "LLM标签适合检查模型抽取质量，不一定适合作为最终分析口径。"
    return ""
