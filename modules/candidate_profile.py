from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_CANDIDATE_PROFILE = {
    "candidate_name": "",
    "degree": "",
    "major": "",
    "graduation_year": "",
    "target_roles": [],
    "target_cities": [],
    "hard_skills": [],
    "soft_skills": [],
    "project_tags": [],
    "internship_tags": [],
    "tool_stack": [],
    "industry_tags": [],
    "strengths": [],
    "weaknesses": [],
    "seniority_level": "应届",
    "education": [],
    "projects": [],
    "internships": [],
    "competitions": [],
    "research": [],
    "evidence": {},
}


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [part.strip() for part in value.replace("；", "，").split("，")]
    else:
        items = list(value) if hasattr(value, "__iter__") else []

    cleaned = []
    seen = set()
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    return cleaned


def _pick_first_nonempty(*values: Any) -> str:
    for value in values:
        s = str(value).strip() if value is not None else ""
        if s:
            return s
    return ""


def build_candidate_profile(resume_struct: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(resume_struct, dict):
        return DEFAULT_CANDIDATE_PROFILE.copy()

    basic_info = resume_struct.get("basic_info", {}) if isinstance(resume_struct.get("basic_info"), dict) else {}
    skills = resume_struct.get("skills", {}) if isinstance(resume_struct.get("skills"), dict) else {}
    experiences = resume_struct.get("experiences", {}) if isinstance(resume_struct.get("experiences"), dict) else {}
    education = resume_struct.get("education", []) if isinstance(resume_struct.get("education"), list) else []
    evidence = resume_struct.get("evidence", {}) if isinstance(resume_struct.get("evidence"), dict) else {}

    first_edu = education[0] if education and isinstance(education[0], dict) else {}

    degree = _pick_first_nonempty(
        basic_info.get("degree", ""),
        first_edu.get("degree", ""),
        resume_struct.get("degree", ""),
    )
    major = _pick_first_nonempty(
        basic_info.get("major", ""),
        first_edu.get("major", ""),
        resume_struct.get("major", ""),
    )
    graduation_year = _pick_first_nonempty(
        basic_info.get("graduation_year", ""),
        first_edu.get("end_year", ""),
        resume_struct.get("graduation_year", ""),
    )
    seniority_level = _pick_first_nonempty(
        basic_info.get("seniority_level", ""),
        resume_struct.get("seniority_level", ""),
        "应届",
    )

    projects = _to_list(experiences.get("projects", resume_struct.get("projects", [])))
    internships = _to_list(experiences.get("internships", resume_struct.get("internships", [])))
    competitions = _to_list(experiences.get("competitions", resume_struct.get("competitions", [])))
    research = _to_list(experiences.get("research", resume_struct.get("research", [])))

    return {
        "candidate_name": _pick_first_nonempty(resume_struct.get("candidate_name", "")),
        "degree": degree,
        "major": major,
        "graduation_year": graduation_year,
        "target_roles": _to_list(basic_info.get("target_roles", resume_struct.get("target_roles", []))),
        "target_cities": _to_list(basic_info.get("target_cities", resume_struct.get("target_cities", []))),
        "hard_skills": _to_list(skills.get("hard_skills", resume_struct.get("hard_skills", []))),
        "soft_skills": _to_list(skills.get("soft_skills", resume_struct.get("soft_skills", []))),
        "project_tags": projects,
        "internship_tags": internships,
        "tool_stack": _to_list(skills.get("tools", resume_struct.get("tools", []))),
        "industry_tags": _to_list(resume_struct.get("industry_tags", [])),
        "strengths": _to_list(resume_struct.get("strengths", [])),
        "weaknesses": _to_list(resume_struct.get("weaknesses", [])),
        "seniority_level": seniority_level,
        "education": education,
        "projects": projects,
        "internships": internships,
        "competitions": competitions,
        "research": research,
        "evidence": evidence,
    }
