from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


MATCH_LEVEL_THRESHOLDS = [
    (85, "强烈推荐"),
    (70, "推荐投递"),
    (55, "可尝试"),
    (40, "谨慎投递"),
    (0, "暂不推荐"),
]


SKILL_ALIAS = {
    "powerbi": "Power BI",
    "power bi": "Power BI",
    "tableau": "Tableau",
    "sql": "SQL",
    "python": "Python",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "ab测试": "A/B测试",
    "a/b test": "A/B测试",
    "prompt": "Prompt Engineering",
    "prompt engineering": "Prompt Engineering",
    "figma": "Figma",
    "axure": "Axure",
    "prd": "PRD",
}


def _normalize_token(token: str) -> str:
    token = str(token or "").strip()
    if not token:
        return ""
    lowered = token.lower()
    return SKILL_ALIAS.get(lowered, token)


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        text = value.replace("；", ",").replace(";", ",").replace("、", ",").replace("/", ",")
        items = [x.strip() for x in re_split(text)]
    else:
        return []

    cleaned = []
    seen = set()
    for item in items:
        s = _normalize_token(item)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    return cleaned


def re_split(text: str) -> List[str]:
    import re

    return re.split(r"[,\n]+", text)


def _contains_any(text: str, keywords: List[str]) -> bool:
    text = str(text or "")
    return any(k in text for k in keywords)


def _find_partial_matches(candidate_items: List[str], jd_items: List[str]) -> Tuple[List[str], List[str]]:
    candidate_clean = [_normalize_token(x) for x in candidate_items]
    matched = []
    missing = []
    for jd in jd_items:
        jd_norm = _normalize_token(jd)
        found = False
        for cand in candidate_clean:
            if not cand or not jd_norm:
                continue
            cand_l = cand.lower()
            jd_l = jd_norm.lower()
            if cand_l == jd_l or cand_l in jd_l or jd_l in cand_l:
                found = True
                break
        if found:
            matched.append(jd_norm)
        else:
            missing.append(jd_norm)
    return matched, missing


def _score_overlap(candidate_items: List[str], jd_items: List[str], full_score: int) -> Tuple[int, List[str], List[str]]:
    jd_list = _to_list(jd_items)
    candidate_list = _to_list(candidate_items)
    if not jd_list:
        return full_score, [], []

    matched, missing = _find_partial_matches(candidate_list, jd_list)
    score = int(full_score * len(matched) / max(len(jd_list), 1))
    return score, matched, missing


def score_degree_match(candidate_degree: str, jd_degree: str) -> int:
    candidate_degree = str(candidate_degree or "")
    jd_degree = str(jd_degree or "")
    if not jd_degree:
        return 8
    if not candidate_degree:
        return 4

    rank = {"大专": 1, "本科": 2, "硕士": 3, "博士": 4}
    candidate_rank = max([v for k, v in rank.items() if k in candidate_degree] or [0])
    jd_rank = max([v for k, v in rank.items() if k in jd_degree] or [0])
    return 10 if candidate_rank >= jd_rank else 4


def score_experience_match(candidate_level: str, jd_experience: str) -> int:
    candidate_level = str(candidate_level or "应届")
    jd_experience = str(jd_experience or "")
    if not jd_experience:
        return 10
    if _contains_any(jd_experience, ["不限", "应届", "校招", "0-1", "1年以下"]):
        return 15
    if _contains_any(jd_experience, ["1-3年", "1年以上", "2年", "3年"]):
        return 8 if "应届" in candidate_level else 12
    if _contains_any(jd_experience, ["3年以上", "5年", "资深", "高级"]):
        return 2
    return 8


def score_direction_match(candidate_roles: List[str], row: pd.Series) -> int:
    text = " ".join([
        str(row.get("职位类别", "")),
        str(row.get("职位方向", "")),
        str(row.get("职位名称_norm", row.get("职位名称_raw", ""))),
        " ".join(_to_list(row.get("LLM岗位类型", []))),
    ]).lower()
    roles = [x.lower() for x in _to_list(candidate_roles)]
    if not roles:
        return 10
    matched = sum(1 for role in roles if role and role in text)
    return min(20, 8 + matched * 6) if matched else 6


def score_city_match(candidate_cities: List[str], job_city: str) -> int:
    cities = _to_list(candidate_cities)
    job_city = str(job_city or "")
    if not cities:
        return 3
    return 5 if any(city in job_city or job_city in city for city in cities if city and job_city) else 1


def score_fresh_grad_fit(row: pd.Series) -> int:
    experience = str(row.get("LLM经验要求", row.get("经验要求", "")))
    jd_text = str(row.get("岗位详情", ""))
    seniority = str(row.get("LLM资历级别", ""))
    fresh_flag = str(row.get("LLM应届友好度", ""))

    if _contains_any(fresh_flag + seniority + experience + jd_text, ["应届", "校招", "2025届", "2026届", "接受应届", "不限经验"]):
        return 10
    if _contains_any(experience, ["1-3年", "1年以上", "2年", "3年"]):
        return 5
    if _contains_any(experience + seniority, ["3年以上", "资深", "高级", "专家"]):
        return 1
    return 6


def get_match_level(score: int) -> str:
    for threshold, label in MATCH_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "暂不推荐"


def build_match_reason(row: pd.Series, matched_skills: List[str], missing_skills: List[str], fresh_score: int, direction_score: int) -> str:
    reasons = []
    title = str(row.get("职位名称_norm", row.get("职位名称_raw", "该岗位")))
    if matched_skills:
        reasons.append(f"你与{title}已匹配的关键词有：{'、'.join(matched_skills[:5])}")
    if missing_skills:
        reasons.append(f"当前最关键缺口是：{'、'.join(missing_skills[:5])}")
    if fresh_score >= 8:
        reasons.append("该岗位对应届生相对友好")
    elif fresh_score <= 3:
        reasons.append("该岗位经验门槛偏高")
    if direction_score >= 14:
        reasons.append("岗位方向与你当前目标较一致")
    return "；".join(reasons)


def _collect_jd_hard_skills(row: pd.Series) -> List[str]:
    items = []
    for col in ["最终硬技能标签", "LLM必须技能", "LLM加分项", "LLM岗位要求"]:
        items.extend(_to_list(row.get(col, [])))
    return _to_list(items)


def _collect_jd_soft_skills(row: pd.Series) -> List[str]:
    items = []
    for col in ["最终软素质标签"]:
        items.extend(_to_list(row.get(col, [])))
    return _to_list(items)


def match_jobs_with_candidate(jobs_df: pd.DataFrame, candidate_profile: Dict[str, Any]) -> pd.DataFrame:
    if jobs_df is None or jobs_df.empty:
        return pd.DataFrame()

    result = jobs_df.copy()
    hard_candidate = _to_list(candidate_profile.get("hard_skills", [])) + _to_list(candidate_profile.get("tool_stack", []))
    soft_candidate = _to_list(candidate_profile.get("soft_skills", []))

    records = []
    for idx, row in result.iterrows():
        jd_hard = _collect_jd_hard_skills(row)
        jd_soft = _collect_jd_soft_skills(row)

        hard_score, matched_hard, missing_hard = _score_overlap(hard_candidate, jd_hard, 30)
        soft_score, matched_soft, missing_soft = _score_overlap(soft_candidate, jd_soft, 10)
        direction_score = score_direction_match(candidate_profile.get("target_roles", []), row)
        degree_score = score_degree_match(candidate_profile.get("degree", ""), row.get("LLM学历要求", row.get("学历要求", "")))
        exp_score = score_experience_match(candidate_profile.get("seniority_level", "应届"), row.get("LLM经验要求", row.get("经验要求", "")))
        city_score = score_city_match(candidate_profile.get("target_cities", []), row.get("所在地区", row.get("工作地点", "")))
        fresh_score = score_fresh_grad_fit(row)

        overall_score = hard_score + soft_score + direction_score + degree_score + exp_score + city_score + fresh_score
        matched_skills = _to_list(matched_hard + matched_soft)
        missing_skills = _to_list(missing_hard + missing_soft)

        records.append({
            "_index": idx,
            "匹配总分": overall_score,
            "硬技能匹配分": hard_score,
            "软技能匹配分": soft_score,
            "方向匹配分": direction_score,
            "学历匹配分": degree_score,
            "经验匹配分": exp_score,
            "城市匹配分": city_score,
            "应届友好度分": fresh_score,
            "匹配技能": matched_skills[:10],
            "缺失技能": missing_skills[:10],
            "匹配结论": get_match_level(overall_score),
            "匹配说明": build_match_reason(row, matched_skills, missing_skills, fresh_score, direction_score),
        })

    score_df = pd.DataFrame(records).set_index("_index")
    result = result.join(score_df)
    result = result.sort_values(["匹配总分", "应届友好度分"], ascending=[False, False]).reset_index(drop=True)
    return result
