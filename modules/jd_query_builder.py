from __future__ import annotations

from typing import Any, Dict, List


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip() for x in value.replace("；", "、").replace(";", "、").split("、") if x.strip()]
    return []


def build_jd_query_from_row(row) -> Dict[str, Any]:
    title = str(row.get("职位名称_norm", row.get("职位名称_raw", ""))).strip()
    company = str(row.get("企业名称_norm", row.get("企业名称_raw", ""))).strip()
    city = str(row.get("所在地区", row.get("工作地点", ""))).strip()
    hard_skills = []
    for col in ["最终硬技能标签", "LLM必须技能", "LLM加分项"]:
        hard_skills.extend(_to_list(row.get(col, [])))
    requirements = _to_list(row.get("LLM岗位要求", []))
    exp = str(row.get("LLM经验要求", row.get("经验要求", ""))).strip()
    degree = str(row.get("LLM学历要求", row.get("学历要求", ""))).strip()

    query_parts = [title, company, city, exp, degree]
    query_parts.extend(hard_skills[:10])
    query_parts.extend(requirements[:8])
    query_text = "；".join([x for x in query_parts if x])

    return {
        "title": title,
        "company": company,
        "city": city,
        "hard_skills": list(dict.fromkeys([x for x in hard_skills if x]))[:15],
        "requirements": list(dict.fromkeys([x for x in requirements if x]))[:12],
        "experience_requirement": exp,
        "degree_requirement": degree,
        "query_text": query_text,
    }
