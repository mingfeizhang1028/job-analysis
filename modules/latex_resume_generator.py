from __future__ import annotations

from typing import Any, Dict, List


def _bullets_from_results(results: List[Dict[str, Any]], limit: int = 4) -> List[str]:
    bullets = []
    for item in results[:limit]:
        text = str(item.get("chunk_text", "")).replace("\n", " ").strip()
        if text:
            bullets.append(text[:180])
    return bullets


def build_resume_plan(candidate_profile: Dict[str, Any], job_query: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    results = evidence.get("results", []) or []
    hard_skills = job_query.get("hard_skills", []) or []
    summary_parts = []
    if candidate_profile.get("degree"):
        summary_parts.append(f"{candidate_profile.get('degree')}背景")
    if candidate_profile.get("major"):
        summary_parts.append(candidate_profile.get("major"))
    if candidate_profile.get("target_roles"):
        summary_parts.append(f"目标方向：{'/'.join(candidate_profile.get('target_roles', [])[:3])}")
    if hard_skills:
        summary_parts.append(f"重点突出：{'、'.join(hard_skills[:5])}")

    return {
        "target_role": job_query.get("title", ""),
        "target_company": job_query.get("company", ""),
        "summary": "；".join([x for x in summary_parts if x]),
        "selected_evidence_bullets": _bullets_from_results(results),
        "emphasis_keywords": hard_skills[:8],
    }
