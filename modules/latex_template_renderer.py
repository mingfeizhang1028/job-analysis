from __future__ import annotations

from typing import Any, Dict, List


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    result = str(text or "")
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def render_resume_tex(plan: Dict[str, Any], candidate_profile: Dict[str, Any]) -> str:
    name = escape_latex(candidate_profile.get("candidate_name", "Candidate"))
    summary = escape_latex(plan.get("summary", ""))
    degree = escape_latex(candidate_profile.get("degree", ""))
    major = escape_latex(candidate_profile.get("major", ""))
    target_role = escape_latex(plan.get("target_role", ""))
    bullets = plan.get("selected_evidence_bullets", []) or []
    bullet_tex = "\n".join([f"\\item {escape_latex(b)}" for b in bullets]) or "\\item 待补充与目标岗位最相关的项目/实习证据"
    keywords = "、".join(plan.get("emphasis_keywords", []) or [])

    return f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage[UTF8]{{ctex}}
\\begin{{document}}
\\begin{{center}}
    {{\\LARGE {name}}}\\\\
    目标岗位：{target_role}
\\end{{center}}

\\section*{{教育背景}}
{degree} {major}

\\section*{{个人概述}}
{summary}

\\section*{{与岗位最相关的证据}}
\\begin{{itemize}}
{bullet_tex}
\\end{{itemize}}

\\section*{{关键词}}
{escape_latex(keywords)}

\\end{{document}}
"""
