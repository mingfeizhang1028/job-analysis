from __future__ import annotations

import re
from typing import Any

import pandas as pd


DETAIL_COL = "岗位详情"

RULE_JOB_CONTENT_COL = "规则岗位工作内容"
RULE_JOB_REQUIREMENTS_COL = "规则岗位要求"
RULE_JOB_BONUS_COL = "规则加分项"
RULE_JOB_INDUSTRY_COL = "规则所属行业"
RULE_JOB_TYPE_COL = "规则岗位类型"
RULE_JOB_GOAL_COL = "规则核心目标"
RULE_MUST_HAVE_SKILLS_COL = "规则必须技能"
RULE_NICE_TO_HAVE_SKILLS_COL = "规则加分技能"
RULE_TOOL_STACK_COL = "规则工具栈"

HARD_TAG_COL = "硬技能标签"
SOFT_TAG_COL = "软素质标签"
RESP_TAG_COL = "业务职责标签"
SCENE_TAG_COL = "行业场景标签"
ALL_TAG_COL = "全部标签"


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_safe_str(item) for item in value if _safe_str(item)]
    if isinstance(value, (tuple, set)):
        return [_safe_str(item) for item in value if _safe_str(item)]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[，,；;、\n]+", value) if item.strip()]
    return []


def _merge_list(*values: Any, max_items: int = 30) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _as_list(value):
            key = item.lower()
            if not item or key in seen:
                continue
            result.append(item)
            seen.add(key)
            if len(result) >= max_items:
                return result
    return result


def normalize_jd_text(text: Any) -> str:
    value = _safe_str(text)
    if not value:
        return ""

    value = value.replace("\xa0", " ").replace("\u3000", " ")
    value = re.sub(r"\s+", " ", value).strip()
    replacements = {
        "唾眠": "睡眠",
        "客来自户": "客户",
        "0pen": "Open",
        "0PEN": "Open",
        "八书": "飞书",
        "A+服务": "AI服务",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    value = re.sub(r"(?<![A-Za-z])A(?=(智能体|系统|对话|服务|项目|工具|应用|平台))", "AI", value)
    return value


def _section_text(text: str, start_markers: list[str], stop_markers: list[str]) -> str:
    if not text:
        return ""

    start = -1
    matched_marker = ""
    for marker in start_markers:
        idx = text.find(marker)
        if idx >= 0 and (start < 0 or idx < start):
            start = idx
            matched_marker = marker

    if start < 0:
        return ""

    content_start = start + len(matched_marker)
    end = len(text)
    for marker in stop_markers:
        idx = text.find(marker, content_start)
        if idx >= 0:
            end = min(end, idx)
    return text[content_start:end].strip(" ：:;；")


def split_jd_sections(text: Any) -> dict[str, str]:
    source = normalize_jd_text(text)
    return {
        "highlights": _section_text(source, ["岗位亮点", "职位亮点", "工作亮点"], ["岗位职责", "工作职责", "工作内容", "任职要求", "职位要求", "加分项", "我们提供"]),
        "responsibilities": _section_text(source, ["岗位职责", "工作职责", "工作内容", "职位职责"], ["任职要求", "职位要求", "岗位要求", "加分项", "优先", "我们提供"]),
        "requirements": _section_text(source, ["任职要求", "职位要求", "岗位要求"], ["加分项", "优先", "我们提供", "福利", "薪资"]),
        "bonus": _section_text(source, ["加分项", "优先", "者优先"], ["我们提供", "福利", "薪资"]),
        "offer": _section_text(source, ["我们提供", "你将获得", "岗位福利"], ["公司介绍"]),
        "all": source,
    }


def _contains(text: str, patterns: list[str]) -> bool:
    return any(pattern and pattern.lower() in text.lower() for pattern in patterns)


def _extract_by_patterns(text: str, specs: list[tuple[str, list[str]]]) -> list[str]:
    hits: list[str] = []
    for label, patterns in specs:
        if _contains(text, patterns):
            hits.append(label)
    return _merge_list(hits)


HARD_SPECS = [
    ("AI智能体", ["AI智能体", "智能体", "Agent"]),
    ("智能体搭建", ["智能体搭建", "搭建智能体", "智能体平台"]),
    ("智能体运营", ["智能体运营", "智能体高效运作"]),
    ("对话流程设计", ["对话流程设计", "对话系统", "对话流程"]),
    ("节点管理", ["节点管理"]),
    ("Prompt设计", ["Prompt", "提示词"]),
    ("FastGPT", ["FastGPT"]),
    ("OpenClaw", ["OpenClaw", "小龙虾"]),
    ("企业微信", ["企业微信"]),
    ("飞书", ["飞书"]),
    ("凡科网", ["凡科网"]),
    ("企业AI系统对接", ["企业AI系统对接", "系统对接", "AI系统对接"]),
    ("Python", ["Python"]),
    ("低代码工具", ["低代码"]),
    ("数据分析", ["数据分析"]),
    ("效果分析", ["效果优化", "输出报告", "改进建议"]),
    ("知识库建设", ["知识库建设", "知识库"]),
    ("流程自动化", ["流程自动化", "自动化"]),
]

SOFT_SPECS = [
    ("逻辑思维", ["逻辑思维"]),
    ("流程设计能力", ["流程设计"]),
    ("分析能力", ["分析能力", "数据分析能力"]),
    ("学习能力", ["学习", "快速积累"]),
    ("实战能力", ["实战", "落地实践"]),
    ("沟通协作", ["对接", "协助", "支持团队"]),
]

RESP_SPECS = [
    ("智能体搭建与优化", ["搭建并优化", "智能体搭建", "迭代优化"]),
    ("对话流程设计", ["对话流程设计"]),
    ("节点管理", ["节点管理"]),
    ("系统对接", ["对接企业微信", "系统对接", "企业AI系统对接"]),
    ("智能客服落地", ["智能客服", "客户咨询", "客户服务"]),
    ("智能销售落地", ["智能销售", "销售落地", "销售闭环"]),
    ("销售/服务流程自动化", ["销售/服务流程自动化", "流程自动化"]),
    ("客户引流", ["客户引流"]),
    ("线索分发", ["线索分发"]),
    ("客户跟进交付", ["跟进交付"]),
    ("数据分析与效果优化", ["数据分析与效果优化", "输出报告", "改进建议"]),
    ("团队培训", ["团队培训"]),
    ("知识库建设", ["知识库建设"]),
]

SCENE_SPECS = [
    ("AI应用", ["AI项目", "AI系统", "AI服务", "AI智能体"]),
    ("智能客服", ["智能客服", "客户咨询", "客户服务"]),
    ("智能销售", ["智能销售", "销售闭环", "销售系统", "销售落地"]),
    ("企业服务", ["企业微信", "飞书", "企业AI系统"]),
    ("健康服务", ["健康"]),
    ("睡眠行业", ["睡眠"]),
    ("门店运营", ["门店"]),
    ("线上项目", ["线上项目", "线上"]),
]

JOB_TYPE_SPECS = [
    ("AI产品/运营", ["智能体搭建", "智能体运营", "AI服务"]),
    ("智能客服/销售", ["智能客服", "智能销售", "销售闭环"]),
    ("数据分析", ["数据分析", "效果优化"]),
]


def extract_rule_jd_profile(text: Any) -> dict[str, list[str] | str]:
    sections = split_jd_sections(text)
    all_text = sections["all"]
    resp_text = sections["responsibilities"] or all_text
    req_text = sections["requirements"] or all_text
    bonus_text = " ".join([sections["bonus"], sections["offer"], sections["highlights"]]).strip()

    hard = _extract_by_patterns(all_text, HARD_SPECS)
    soft = _extract_by_patterns(req_text, SOFT_SPECS)
    responsibilities = _extract_by_patterns(resp_text, RESP_SPECS)
    requirements = _extract_by_patterns(req_text, HARD_SPECS + SOFT_SPECS)
    bonus = _extract_by_patterns(bonus_text, HARD_SPECS + RESP_SPECS + SCENE_SPECS) if bonus_text else []
    scenes = _extract_by_patterns(all_text, SCENE_SPECS)
    job_types = _extract_by_patterns(all_text, JOB_TYPE_SPECS)

    must_have = _extract_by_patterns(req_text, HARD_SPECS)
    nice_to_have = _extract_by_patterns(bonus_text, HARD_SPECS) if bonus_text else []
    tools = _extract_by_patterns(all_text, [
        ("FastGPT", ["FastGPT"]),
        ("OpenClaw", ["OpenClaw", "小龙虾"]),
        ("企业微信", ["企业微信"]),
        ("飞书", ["飞书"]),
        ("凡科网", ["凡科网"]),
        ("Python", ["Python"]),
        ("低代码工具", ["低代码"]),
    ])

    core_goal = ""
    if responsibilities:
        core_goal = "、".join(responsibilities[:3])
    elif scenes:
        core_goal = "、".join(scenes[:3])

    return {
        "hard_skills": hard,
        "soft_skills": soft,
        "responsibilities": responsibilities,
        "requirements": requirements,
        "bonus_points": bonus,
        "industry": scenes,
        "job_type": job_types,
        "core_goal": core_goal[:25],
        "must_have_skills": must_have,
        "nice_to_have_skills": nice_to_have,
        "tool_stack": tools,
    }


def apply_rule_jd_extraction(df: pd.DataFrame, detail_col: str = DETAIL_COL, overwrite: bool = False) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("apply_rule_jd_extraction expects a pandas.DataFrame")

    result = df.copy()
    if detail_col not in result.columns:
        result[detail_col] = ""

    for col in [
        RULE_JOB_CONTENT_COL,
        RULE_JOB_REQUIREMENTS_COL,
        RULE_JOB_BONUS_COL,
        RULE_JOB_INDUSTRY_COL,
        RULE_JOB_TYPE_COL,
        RULE_MUST_HAVE_SKILLS_COL,
        RULE_NICE_TO_HAVE_SKILLS_COL,
        RULE_TOOL_STACK_COL,
        HARD_TAG_COL,
        SOFT_TAG_COL,
        RESP_TAG_COL,
        SCENE_TAG_COL,
        ALL_TAG_COL,
    ]:
        if col not in result.columns:
            result[col] = [[] for _ in range(len(result))]
    if RULE_JOB_GOAL_COL not in result.columns:
        result[RULE_JOB_GOAL_COL] = ""

    for idx, row in result.iterrows():
        profile = extract_rule_jd_profile(row.get(detail_col, ""))
        if overwrite or not _as_list(row.get(RULE_JOB_CONTENT_COL, [])):
            result.at[idx, RULE_JOB_CONTENT_COL] = profile["responsibilities"]
        if overwrite or not _as_list(row.get(RULE_JOB_REQUIREMENTS_COL, [])):
            result.at[idx, RULE_JOB_REQUIREMENTS_COL] = profile["requirements"]
        if overwrite or not _as_list(row.get(RULE_JOB_BONUS_COL, [])):
            result.at[idx, RULE_JOB_BONUS_COL] = profile["bonus_points"]
        if overwrite or not _as_list(row.get(RULE_JOB_INDUSTRY_COL, [])):
            result.at[idx, RULE_JOB_INDUSTRY_COL] = profile["industry"]
        if overwrite or not _as_list(row.get(RULE_JOB_TYPE_COL, [])):
            result.at[idx, RULE_JOB_TYPE_COL] = profile["job_type"]
        if overwrite or not _safe_str(row.get(RULE_JOB_GOAL_COL, "")):
            result.at[idx, RULE_JOB_GOAL_COL] = profile["core_goal"]
        if overwrite or not _as_list(row.get(RULE_MUST_HAVE_SKILLS_COL, [])):
            result.at[idx, RULE_MUST_HAVE_SKILLS_COL] = profile["must_have_skills"]
        if overwrite or not _as_list(row.get(RULE_NICE_TO_HAVE_SKILLS_COL, [])):
            result.at[idx, RULE_NICE_TO_HAVE_SKILLS_COL] = profile["nice_to_have_skills"]
        if overwrite or not _as_list(row.get(RULE_TOOL_STACK_COL, [])):
            result.at[idx, RULE_TOOL_STACK_COL] = profile["tool_stack"]

        result.at[idx, HARD_TAG_COL] = _merge_list(row.get(HARD_TAG_COL, []), profile["hard_skills"])
        result.at[idx, SOFT_TAG_COL] = _merge_list(row.get(SOFT_TAG_COL, []), profile["soft_skills"])
        result.at[idx, RESP_TAG_COL] = _merge_list(row.get(RESP_TAG_COL, []), profile["responsibilities"])
        result.at[idx, SCENE_TAG_COL] = _merge_list(row.get(SCENE_TAG_COL, []), profile["industry"])
        result.at[idx, ALL_TAG_COL] = _merge_list(
            result.at[idx, HARD_TAG_COL],
            result.at[idx, SOFT_TAG_COL],
            result.at[idx, RESP_TAG_COL],
            result.at[idx, SCENE_TAG_COL],
        )

    return result
