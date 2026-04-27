from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pandas as pd


def _flatten(values: pd.Series) -> List[str]:
    items: List[str] = []
    for value in values:
        if isinstance(value, list):
            items.extend([str(x).strip() for x in value if str(x).strip()])
        elif isinstance(value, str) and value.strip():
            items.append(value.strip())
    return items


def build_career_strategy_summary(match_df: pd.DataFrame, candidate_profile: Dict[str, Any]) -> Dict[str, Any]:
    if match_df is None or match_df.empty:
        return {
            "best_fit_roles": [],
            "top_skill_gaps": [],
            "resume_optimization_tips": [],
            "application_strategy": [],
        }

    top_df = match_df.head(20).copy()
    role_col = "职位名称_norm" if "职位名称_norm" in top_df.columns else "职位名称_raw"
    best_fit_roles = top_df[role_col].fillna("未知岗位").astype(str).value_counts().head(10).index.tolist()

    gap_counter = Counter(_flatten(top_df.get("缺失技能", pd.Series(dtype=object))))
    top_skill_gaps = [name for name, _ in gap_counter.most_common(10)]

    candidate_roles = candidate_profile.get("target_roles", []) or []
    candidate_projects = candidate_profile.get("project_tags", []) or []
    candidate_internships = candidate_profile.get("internship_tags", []) or []
    candidate_skills = (candidate_profile.get("hard_skills", []) or []) + (candidate_profile.get("tool_stack", []) or [])

    tips: List[str] = []
    if top_skill_gaps:
        tips.append(f"把这些高频缺口优先补进简历且给出证据：{'、'.join(top_skill_gaps[:5])}。")
    if candidate_projects:
        tips.append(f"把最相关的项目放到首页上半部分，优先写：{'；'.join(candidate_projects[:2])}。")
    else:
        tips.append("当前项目证据偏弱，建议补充1-2段可量化项目，写清场景、动作、结果。")
    if candidate_internships:
        tips.append("把实习经历改写为‘业务目标-你的动作-结果指标’三段式，而不是只写工作内容。")
    if not candidate_roles:
        tips.append("在简历标题或开头摘要中明确写出目标岗位，如‘数据分析/AI产品/产品经理’。")

    strategy: List[str] = []
    top_recommended = match_df[match_df.get("匹配结论", "") .isin(["强烈推荐", "推荐投递"])].head(15)
    if not top_recommended.empty:
        companies = top_recommended.get("企业名称_norm", pd.Series(dtype=object)).dropna().astype(str).head(5).tolist()
        roles = top_recommended.get(role_col, pd.Series(dtype=object)).dropna().astype(str).head(5).tolist()
        if roles:
            strategy.append(f"优先投递当前更匹配的岗位：{'、'.join(roles[:5])}。")
        if companies:
            strategy.append(f"可优先关注这些公司/组织的同类岗位：{'、'.join(companies[:5])}。")
    if top_skill_gaps:
        strategy.append(f"先补最常出现的缺口技能：{'、'.join(top_skill_gaps[:3])}，再扩展投递范围。")
    if candidate_skills:
        strategy.append(f"把你已有能力在简历里前置：{'、'.join(candidate_skills[:6])}。")
    else:
        strategy.append("先完成基础技能盘点，再开始大规模海投，否则匹配率会一直偏低。")

    return {
        "best_fit_roles": best_fit_roles,
        "top_skill_gaps": top_skill_gaps,
        "resume_optimization_tips": tips,
        "application_strategy": strategy,
    }
