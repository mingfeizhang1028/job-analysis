from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from modules.llm_cache import make_cache_key, load_json_cache, save_json_cache
from modules.llm_client import call_ollama_generate, call_openai_compatible_generate

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_CACHE_NAME = "resume_struct_cache.json"


DEFAULT_RESUME_STRUCT = {
    "basic_info": {
        "degree": "",
        "major": "",
        "graduation_year": "",
        "target_roles": [],
        "target_cities": [],
        "seniority_level": "应届",
    },
    "education": [],
    "skills": {
        "hard_skills": [],
        "soft_skills": [],
        "tools": [],
        "languages": [],
    },
    "experiences": {
        "internships": [],
        "projects": [],
        "competitions": [],
        "research": [],
    },
    "industry_tags": [],
    "strengths": [],
    "weaknesses": [],
    "evidence": {
        "degree_evidence": [],
        "major_evidence": [],
        "skill_evidence": [],
        "project_evidence": [],
        "internship_evidence": [],
    },
    "candidate_name": "",
}

FALLBACK_ROLE_KEYWORDS = [
    "AI产品经理", "产品经理", "数据分析师", "数据分析", "商业分析", "用户研究", "增长运营", "内容运营",
    "运营", "算法工程师", "机器学习", "NLP", "后端开发", "前端开发", "测试开发", "项目经理",
]
FALLBACK_SKILL_KEYWORDS = [
    "Python", "SQL", "Excel", "Power BI", "Tableau", "Pandas", "NumPy", "Scikit-learn", "PyTorch",
    "TensorFlow", "A/B测试", "数据分析", "用户研究", "竞品分析", "需求分析", "Axure", "Figma",
    "PRD", "Prompt Engineering", "机器学习", "深度学习", "Java", "C++", "MySQL", "PostgreSQL",
]
FALLBACK_SOFT_KEYWORDS = [
    "沟通协作", "学习能力", "逻辑思维", "自驱力", "执行力", "责任心", "团队合作", "抗压能力",
]
FALLBACK_TOOL_KEYWORDS = [
    "Excel", "Power BI", "Tableau", "Axure", "Figma", "Xmind", "Visio", "Jira", "Confluence",
    "Notion", "Git", "GitHub", "MySQL", "PostgreSQL", "Docker", "Linux", "MATLAB", "SPSS",
    "Python", "SQL", "Pandas", "NumPy", "PyTorch", "TensorFlow",
]
FALLBACK_CITY_KEYWORDS = [
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都", "武汉", "西安",
]
DEGREE_PATTERNS = ["博士", "硕士", "研究生", "本科", "学士", "大专"]
MAJOR_HINT_WORDS = [
    "计算机", "软件工程", "数据科学", "人工智能", "信息管理", "信息系统", "统计学", "数学", "经济学",
    "金融", "市场营销", "工商管理", "电子商务", "自动化", "电子信息", "通信工程",
]
ROLE_TOOLS_HINTS = {
    "产品": ["Axure", "Figma", "Xmind", "Visio", "Jira", "Confluence", "Excel"],
    "运营": ["Excel", "Power BI", "Tableau", "SQL", "Python"],
    "数据": ["Excel", "SQL", "Python", "Pandas", "Power BI", "Tableau"],
    "开发": ["Git", "GitHub", "Docker", "Linux", "MySQL", "PostgreSQL"],
    "算法": ["Python", "PyTorch", "TensorFlow", "NumPy", "Pandas"],
}


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_list(value: Any, max_items: int = 12, max_len: int = 50) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(r"[，,；;、\n]+", value)
    elif isinstance(value, list):
        items = value
    else:
        return []

    cleaned = []
    seen = set()
    for item in items:
        s = _safe_str(item).strip(" -•[]【】()（）\"'")
        if not s or len(s) > max_len:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _normalize_text(value: Any, max_len: int = 40) -> str:
    s = _safe_str(value)
    return s[:max_len] if s else ""


def _extract_json_text(text: str) -> str:
    text = _safe_str(text)
    if not text:
        return ""
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _try_parse_json_loose(text: str) -> Dict[str, Any]:
    text = _extract_json_text(text)
    if not text:
        return {}

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _keyword_hits(text: str, keywords: List[str], limit: int = 10) -> List[str]:
    hits = []
    lowered = text.lower()
    for keyword in keywords:
        if keyword.lower() in lowered and keyword not in hits:
            hits.append(keyword)
        if len(hits) >= limit:
            break
    return hits


def _extract_degree(text: str) -> str:
    for degree in DEGREE_PATTERNS:
        if degree in text:
            if degree == "学士":
                return "本科"
            if degree == "研究生":
                return "硕士"
            return degree
    return ""


def _extract_graduation_year(text: str) -> str:
    matches = re.findall(r"(20\d{2})", text)
    if not matches:
        return ""
    years = [int(y) for y in matches if 2018 <= int(y) <= 2035]
    if not years:
        return ""
    preferred = [y for y in years if y >= 2024]
    return str(min(preferred) if preferred else max(years))


def _extract_major(text: str) -> str:
    patterns = [
        r"专业[：:\s]*([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,30})",
        r"院系[：:\s]*([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,30})",
        r"([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,30}专业)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if len(value) <= 30:
                return value
    for word in MAJOR_HINT_WORDS:
        if word in text:
            return word
    return ""


def _extract_name(text: str, lines: List[str]) -> str:
    patterns = [
        r"姓名[：:\s]*([\u4e00-\u9fa5·]{2,10})",
        r"^([\u4e00-\u9fa5·]{2,6})$",
    ]
    for line in lines[:8]:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                name = match.group(1).strip()
                if 2 <= len(name) <= 10:
                    return name
    return ""


def _extract_education_entries(lines: List[str], text: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    edu_keywords = ["大学", "学院", "学校", "本科", "硕士", "博士", "研究生", "学士", "教育经历", "学历"]
    section_active = False
    for line in lines:
        compact = line.replace(" ", "")
        if any(k in compact for k in ["教育经历", "教育背景", "学历背景"]):
            section_active = True
            continue
        if section_active and any(k in compact for k in ["工作经历", "实习经历", "项目经历", "项目经验", "校园经历", "技能", "专业技能"]):
            section_active = False
        if not section_active and not any(k in line for k in edu_keywords):
            continue

        school_match = re.search(r"([\u4e00-\u9fa5A-Za-z]{2,40}(?:大学|学院|学校))", line)
        school = school_match.group(1).strip() if school_match else ""
        degree = _extract_degree(line) or (_extract_degree(text) if section_active else "")
        major = _extract_major(line) or ""
        years = re.findall(r"(20\d{2})", line)
        start_year = years[0] if len(years) >= 1 else ""
        end_year = years[1] if len(years) >= 2 else (years[0] if len(years) == 1 and int(years[0]) >= 2020 else "")
        if school or degree or major:
            entries.append({
                "school": school,
                "degree": degree,
                "major": major,
                "start_year": start_year,
                "end_year": end_year,
            })
        if len(entries) >= 4:
            break
    return entries


def _extract_section_items(lines: List[str], start_keywords: List[str], stop_keywords: List[str], limit: int = 8) -> List[str]:
    items: List[str] = []
    active = False
    buffer = ""
    for line in lines:
        compact = line.replace(" ", "")
        if any(k in compact for k in start_keywords):
            active = True
            tail = re.sub("|".join(map(re.escape, start_keywords)), "", line).strip(" ：:-")
            if tail:
                buffer = tail
            continue
        if active and any(k in compact for k in stop_keywords):
            if buffer:
                items.append(buffer.strip())
            active = False
            buffer = ""
            continue
        if not active:
            continue
        cleaned = line.strip(" -•·\t|")
        if not cleaned:
            continue
        is_new_record = bool(
            re.match(r"^(20\d{2}|19\d{2}|\d{4})", cleaned)
            or re.search(r"(20\d{2})\s*[-至~/]\s*(20\d{2}|今|至今)", cleaned)
            or any(token in cleaned for token in ["有限公司", "公司", "集团", "科技", "银行", "大学", "学院", "项目", "系统", "平台", "课题"])
        )
        if is_new_record:
            if buffer:
                items.append(buffer.strip())
            buffer = cleaned
        else:
            buffer = f"{buffer}；{cleaned}" if buffer else cleaned
        if len(items) >= limit:
            break
    if active and buffer and len(items) < limit:
        items.append(buffer.strip())
    return _normalize_list(items, max_items=limit, max_len=220)


def _infer_target_roles(text: str, projects: List[str], internships: List[str], hard_skills: List[str]) -> List[str]:
    roles = _keyword_hits(text, FALLBACK_ROLE_KEYWORDS, limit=6)
    combined = "\n".join(projects + internships)
    if any(k in combined for k in ["产品", "需求", "用户", "PRD", "原型"]):
        roles.extend([r for r in ["产品经理", "AI产品经理"] if r not in roles])
    if any(k in combined for k in ["运营", "增长", "内容", "活动"]):
        roles.extend([r for r in ["增长运营", "内容运营", "运营"] if r not in roles])
    if any(k in combined for k in ["数据分析", "报表", "指标", "SQL", "可视化"]) or any(k in hard_skills for k in ["SQL", "Python", "数据分析"]):
        roles.extend([r for r in ["数据分析师", "数据分析", "商业分析"] if r not in roles])
    if any(k in combined for k in ["模型", "机器学习", "深度学习", "NLP"]) or any(k in hard_skills for k in ["PyTorch", "TensorFlow", "机器学习"]):
        roles.extend([r for r in ["算法工程师", "机器学习", "NLP"] if r not in roles])
    return roles[:6]


def _infer_tools(text: str, hard_skills: List[str], target_roles: List[str]) -> List[str]:
    tools = _keyword_hits(text, FALLBACK_TOOL_KEYWORDS, limit=12)
    for role in target_roles:
        for key, hint_tools in ROLE_TOOLS_HINTS.items():
            if key in role:
                for tool in hint_tools:
                    if tool in hard_skills and tool not in tools:
                        tools.append(tool)
    return tools[:12]


def _infer_soft_skills(text: str, internships: List[str], projects: List[str]) -> List[str]:
    soft_skills = _keyword_hits(text, FALLBACK_SOFT_KEYWORDS, limit=8)
    combined = "\n".join(internships + projects)
    heuristic_pairs = {
        "沟通协作": ["沟通", "协作", "跨部门", "对接"],
        "学习能力": ["自学", "快速学习", "学习能力"],
        "逻辑思维": ["分析", "拆解", "逻辑"],
        "执行力": ["推进", "落地", "执行"],
        "责任心": ["负责", "owner", "主导"],
        "团队合作": ["团队", "合作"],
    }
    for label, keys in heuristic_pairs.items():
        if label in soft_skills:
            continue
        if any(k in combined for k in keys):
            soft_skills.append(label)
    return soft_skills[:8]


def _normalize_resume_struct(obj: Dict[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(DEFAULT_RESUME_STRUCT, ensure_ascii=False))

    basic_info = obj.get("basic_info", {}) if isinstance(obj.get("basic_info"), dict) else {}
    skills = obj.get("skills", {}) if isinstance(obj.get("skills"), dict) else {}
    experiences = obj.get("experiences", {}) if isinstance(obj.get("experiences"), dict) else {}
    evidence = obj.get("evidence", {}) if isinstance(obj.get("evidence"), dict) else {}
    education = obj.get("education", []) if isinstance(obj.get("education"), list) else []

    result["candidate_name"] = _normalize_text(obj.get("candidate_name", ""), max_len=30)
    result["basic_info"] = {
        "degree": _normalize_text(basic_info.get("degree", obj.get("degree", "")), max_len=10),
        "major": _normalize_text(basic_info.get("major", obj.get("major", "")), max_len=30),
        "graduation_year": _normalize_text(basic_info.get("graduation_year", obj.get("graduation_year", "")), max_len=10),
        "target_roles": _normalize_list(basic_info.get("target_roles", obj.get("target_roles", [])), max_items=8, max_len=30),
        "target_cities": _normalize_list(basic_info.get("target_cities", obj.get("target_cities", [])), max_items=8, max_len=20),
        "seniority_level": _normalize_text(basic_info.get("seniority_level", obj.get("seniority_level", "应届")), max_len=10) or "应届",
    }
    result["skills"] = {
        "hard_skills": _normalize_list(skills.get("hard_skills", obj.get("hard_skills", [])), max_items=15, max_len=30),
        "soft_skills": _normalize_list(skills.get("soft_skills", obj.get("soft_skills", [])), max_items=12, max_len=30),
        "tools": _normalize_list(skills.get("tools", obj.get("tools", [])), max_items=12, max_len=30),
        "languages": _normalize_list(skills.get("languages", []), max_items=8, max_len=20),
    }
    result["experiences"] = {
        "internships": _normalize_list(experiences.get("internships", obj.get("internships", [])), max_items=8, max_len=100),
        "projects": _normalize_list(experiences.get("projects", obj.get("projects", [])), max_items=8, max_len=100),
        "competitions": _normalize_list(experiences.get("competitions", []), max_items=8, max_len=80),
        "research": _normalize_list(experiences.get("research", []), max_items=8, max_len=80),
    }
    cleaned_education = []
    for item in education[:4]:
        if not isinstance(item, dict):
            continue
        cleaned_education.append({
            "school": _normalize_text(item.get("school", ""), max_len=40),
            "degree": _normalize_text(item.get("degree", ""), max_len=10),
            "major": _normalize_text(item.get("major", ""), max_len=30),
            "start_year": _normalize_text(item.get("start_year", ""), max_len=10),
            "end_year": _normalize_text(item.get("end_year", ""), max_len=10),
        })
    result["education"] = cleaned_education
    result["industry_tags"] = _normalize_list(obj.get("industry_tags", []), max_items=8, max_len=20)
    result["strengths"] = _normalize_list(obj.get("strengths", []), max_items=8, max_len=30)
    result["weaknesses"] = _normalize_list(obj.get("weaknesses", []), max_items=8, max_len=30)
    result["evidence"] = {
        "degree_evidence": _normalize_list(evidence.get("degree_evidence", []), max_items=6, max_len=80),
        "major_evidence": _normalize_list(evidence.get("major_evidence", []), max_items=6, max_len=80),
        "skill_evidence": _normalize_list(evidence.get("skill_evidence", []), max_items=10, max_len=80),
        "project_evidence": _normalize_list(evidence.get("project_evidence", []), max_items=8, max_len=100),
        "internship_evidence": _normalize_list(evidence.get("internship_evidence", []), max_items=8, max_len=100),
    }
    return result


def heuristic_resume_structuring(resume_text: str) -> Dict[str, Any]:
    text = _safe_str(resume_text)
    if not text:
        return _normalize_resume_struct(DEFAULT_RESUME_STRUCT)

    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    degree = _extract_degree(text)
    major = _extract_major(text)
    graduation_year = _extract_graduation_year(text)
    candidate_name = _extract_name(text, lines)

    section_stop_words = ["教育经历", "教育背景", "工作经历", "实习经历", "项目经历", "项目经验", "校园经历", "竞赛经历", "科研经历", "技能", "专业技能", "荣誉", "证书", "自我评价"]
    project_lines = _extract_section_items(
        lines,
        start_keywords=["项目经历", "项目经验", "项目实践", "项目", "Project"],
        stop_keywords=[k for k in section_stop_words if k not in ["项目经历", "项目经验"]],
        limit=8,
    )
    internship_lines = _extract_section_items(
        lines,
        start_keywords=["工作经历", "实习经历", "实习经验", "工作经验"],
        stop_keywords=[k for k in section_stop_words if k not in ["工作经历", "实习经历"]],
        limit=8,
    )
    if not project_lines:
        project_lines = [line for line in lines if any(k in line for k in ["项目", "Project", "课题", "系统", "平台"])][:6]
    if not internship_lines:
        internship_lines = [line for line in lines if any(k in line for k in ["实习", "有限公司", "科技", "公司", "运营", "产品", "数据", "助理", "专员"])][:6]
    education_entries = _extract_education_entries(lines, text)

    hard_skills = _keyword_hits(text, FALLBACK_SKILL_KEYWORDS, limit=12)
    target_roles = _infer_target_roles(text, project_lines, internship_lines, hard_skills)
    target_cities = _keyword_hits(text, FALLBACK_CITY_KEYWORDS, limit=5)
    tool_lines = _infer_tools(text, hard_skills, target_roles)
    soft_skills = _infer_soft_skills(text, internship_lines, project_lines)

    seniority_level = "应届"
    if any(k in text for k in ["2024届", "2025届", "2026届", "应届", "校招", "毕业生"]):
        seniority_level = "应届"
    elif any(k in text for k in ["1年", "2年", "3年", "初级"]):
        seniority_level = "初级"

    if not degree and education_entries:
        degree = _safe_str(education_entries[0].get("degree", ""))
    if not major and education_entries:
        major = _safe_str(education_entries[0].get("major", ""))
    if not graduation_year and education_entries:
        graduation_year = _safe_str(education_entries[0].get("end_year", ""))

    fallback = {
        "basic_info": {
            "degree": degree,
            "major": major,
            "graduation_year": graduation_year,
            "target_roles": target_roles,
            "target_cities": target_cities,
            "seniority_level": seniority_level,
        },
        "education": education_entries,
        "skills": {
            "hard_skills": hard_skills,
            "soft_skills": soft_skills,
            "tools": tool_lines,
            "languages": [],
        },
        "experiences": {
            "internships": internship_lines,
            "projects": project_lines,
            "competitions": [],
            "research": [],
        },
        "industry_tags": [],
        "strengths": soft_skills[:4],
        "weaknesses": [],
        "evidence": {
            "degree_evidence": [degree] if degree else [],
            "major_evidence": [major] if major else [],
            "skill_evidence": (hard_skills + tool_lines)[:8],
            "project_evidence": project_lines[:4],
            "internship_evidence": internship_lines[:4],
        },
        "candidate_name": candidate_name,
    }
    return _normalize_resume_struct(fallback)


def _is_meaningful_resume_struct(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    basic = data.get("basic_info", {}) if isinstance(data.get("basic_info"), dict) else {}
    skills = data.get("skills", {}) if isinstance(data.get("skills"), dict) else {}
    exp = data.get("experiences", {}) if isinstance(data.get("experiences"), dict) else {}
    education = data.get("education", []) if isinstance(data.get("education"), list) else []
    candidate_name = _safe_str(data.get("candidate_name", ""))

    checks = [
        candidate_name,
        basic.get("degree"),
        basic.get("major"),
        basic.get("graduation_year"),
        basic.get("target_roles"),
        basic.get("target_cities"),
        skills.get("hard_skills"),
        skills.get("tools"),
        exp.get("projects"),
        exp.get("internships"),
        education,
    ]
    meaningful_count = sum(1 for item in checks if bool(item))
    return meaningful_count >= 2


def _build_prompt(resume_text: str, max_chars: int = 12000) -> str:
    text = _safe_str(resume_text)[:max_chars]
    return f"""
你是中文简历结构化抽取助手。请只根据简历原文抽取信息，输出一个合法 JSON 对象。

硬性规则：
1. 只输出 JSON，不要 Markdown，不要解释。
2. 不要编造；原文没有依据时填 "" 或 []。
3. 必须按栏目归类，严禁错位：
   - education 只放学校/学历/专业/起止年份。
   - internships 只放公司、组织、实习、工作经历。
   - projects 只放项目、课程项目、产品/系统/平台项目。
   - competitions 只放竞赛/比赛/奖项经历。
   - research 只放科研/论文/课题经历。
4. 如果原文有“教育经历/教育背景/工作经历/实习经历/项目经历/专业技能”等标题，必须按标题后的内容归入对应字段。
5. internships/projects 数组元素可以是 1-2 句经历摘要，保留名称、角色、动作、成果；不要只输出关键词。
6. skills 中只放能力/工具，不要放学校、公司、项目名称。
7. basic_info.degree/major/graduation_year 优先来自 education 中最高或最近学历。
8. seniority_level 只能是 应届、初级、中级、高级、不明确。

输出 JSON schema：
{{
  "candidate_name": "",
  "basic_info": {{
    "degree": "",
    "major": "",
    "graduation_year": "",
    "target_roles": [],
    "target_cities": [],
    "seniority_level": "不明确"
  }},
  "education": [{{"school": "", "degree": "", "major": "", "start_year": "", "end_year": ""}}],
  "skills": {{"hard_skills": [], "soft_skills": [], "tools": [], "languages": []}},
  "experiences": {{"internships": [], "projects": [], "competitions": [], "research": []}},
  "industry_tags": [],
  "strengths": [],
  "weaknesses": [],
  "evidence": {{"degree_evidence": [], "major_evidence": [], "skill_evidence": [], "project_evidence": [], "internship_evidence": []}}
}}

简历原文：
{text}
""".strip()


def _merge_unique(primary_list: Any, fallback_list: Any, max_items: int = 12) -> List[str]:
    merged: List[str] = []
    seen = set()
    for source in [primary_list, fallback_list]:
        for item in _normalize_list(source, max_items=max_items):
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max_items:
                return merged
    return merged


def _merge_with_fallback(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(primary, ensure_ascii=False))
    p_basic = result.get("basic_info", {})
    f_basic = fallback.get("basic_info", {})
    for key in ["degree", "major", "graduation_year", "seniority_level"]:
        if not p_basic.get(key) and f_basic.get(key):
            p_basic[key] = f_basic.get(key)
    p_basic["target_roles"] = _merge_unique(p_basic.get("target_roles", []), f_basic.get("target_roles", []), max_items=8)
    p_basic["target_cities"] = _merge_unique(p_basic.get("target_cities", []), f_basic.get("target_cities", []), max_items=8)
    result["basic_info"] = p_basic

    p_skills = result.get("skills", {})
    f_skills = fallback.get("skills", {})
    for key in ["hard_skills", "soft_skills", "tools", "languages"]:
        p_skills[key] = _merge_unique(p_skills.get(key, []), f_skills.get(key, []), max_items=15 if key == "hard_skills" else 12)
    result["skills"] = p_skills

    p_exp = result.get("experiences", {})
    f_exp = fallback.get("experiences", {})
    for key in ["internships", "projects", "competitions", "research"]:
        p_exp[key] = _merge_unique(p_exp.get(key, []), f_exp.get(key, []), max_items=8)
    result["experiences"] = p_exp

    if not result.get("education"):
        result["education"] = fallback.get("education", [])
    if not result.get("strengths"):
        result["strengths"] = fallback.get("strengths", [])
    if not result.get("industry_tags"):
        result["industry_tags"] = fallback.get("industry_tags", [])
    if not result.get("candidate_name") and fallback.get("candidate_name"):
        result["candidate_name"] = fallback.get("candidate_name")

    p_evidence = result.get("evidence", {})
    f_evidence = fallback.get("evidence", {})
    for key in ["degree_evidence", "major_evidence", "skill_evidence", "project_evidence", "internship_evidence"]:
        p_evidence[key] = _merge_unique(p_evidence.get(key, []), f_evidence.get(key, []), max_items=10)
    result["evidence"] = p_evidence
    return _normalize_resume_struct(result)


def call_resume_structuring(
    resume_text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
    temperature: float = 0.1,
    enable_remote_fallback: bool = False,
    remote_base_url: str = "",
    remote_api_key: str = "",
    remote_model: str = "",
    return_debug: bool = False,
) -> Tuple[Dict[str, Any], str] | Tuple[Dict[str, Any], str, Dict[str, Any]]:
    debug_info: Dict[str, Any] = {
        "provider_used": "ollama",
        "ollama_error": "",
        "ollama_raw_output": "",
        "ollama_parsed": {},
        "fallback_struct": {},
        "merged_struct": {},
        "remote_error": "",
        "remote_raw_output": "",
        "remote_parsed": {},
        "final_source": "",
    }
    if not _safe_str(resume_text):
        result = _normalize_resume_struct(DEFAULT_RESUME_STRUCT)
        if return_debug:
            debug_info["final_source"] = "empty"
            return result, "简历内容为空", debug_info
        return result, "简历内容为空"

    prompt = _build_prompt(resume_text, max_chars=12000)
    fallback = heuristic_resume_structuring(resume_text)
    debug_info["fallback_struct"] = fallback

    raw_output, err = call_ollama_generate(
        prompt=prompt,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        temperature=temperature,
        num_predict=2200,
    )
    debug_info["ollama_error"] = err
    debug_info["ollama_raw_output"] = raw_output[:6000] if raw_output else ""
    if not err:
        parsed = _try_parse_json_loose(raw_output)
        normalized = _normalize_resume_struct(parsed)
        merged = _merge_with_fallback(normalized, fallback)
        debug_info["ollama_parsed"] = normalized
        debug_info["merged_struct"] = merged
        if _is_meaningful_resume_struct(merged):
            debug_info["final_source"] = "ollama+fallback_merge"
            if return_debug:
                return merged, "", debug_info
            return merged, ""

    if enable_remote_fallback:
        debug_info["provider_used"] = "ollama+remote"
        remote_output, remote_err = call_openai_compatible_generate(
            prompt=prompt,
            model=remote_model,
            base_url=remote_base_url,
            api_key=remote_api_key,
            timeout=timeout,
            temperature=0.1,
        )
        debug_info["remote_error"] = remote_err
        debug_info["remote_raw_output"] = remote_output[:6000] if remote_output else ""
        if not remote_err:
            parsed = _try_parse_json_loose(remote_output)
            normalized = _normalize_resume_struct(parsed)
            merged = _merge_with_fallback(normalized, fallback)
            debug_info["remote_parsed"] = normalized
            debug_info["merged_struct"] = merged
            if _is_meaningful_resume_struct(merged):
                debug_info["final_source"] = "remote+fallback_merge"
                if return_debug:
                    return merged, "", debug_info
                return merged, ""
            debug_info["final_source"] = "fallback_only"
            if return_debug:
                return fallback, "远程LLM结果质量不足，已回退到规则解析", debug_info
            return fallback, "远程LLM结果质量不足，已回退到规则解析"
        debug_info["final_source"] = "fallback_only"
        message = f"本地LLM失败：{err or '结果为空'}；远程LLM失败：{remote_err}；已回退到规则解析"
        if return_debug:
            return fallback, message, debug_info
        return fallback, message

    debug_info["final_source"] = "fallback_only"
    message = f"LLM结构化结果质量不足，已自动回退到规则解析（原因：{err or '结果为空或字段缺失'}）"
    if return_debug:
        return fallback, message, debug_info
    return fallback, message


def structure_resume_text(
    resume_text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    overwrite: bool = False,
    use_cache: bool = True,
    cache_name: str = DEFAULT_CACHE_NAME,
    enable_remote_fallback: bool = False,
    remote_base_url: str = "",
    remote_api_key: str = "",
    remote_model: str = "",
    return_debug: bool = False,
) -> Tuple[Dict[str, Any], str] | Tuple[Dict[str, Any], str, Dict[str, Any]]:
    cache = load_json_cache(cache_name) if use_cache else {}
    cache_changed = False
    cache_key = make_cache_key(
        task="resume_structuring",
        model=f"{model}|remote:{remote_model if enable_remote_fallback else 'off'}",
        jd_text=resume_text,
        version="v3",
    )

    if use_cache and cache_key in cache and not overwrite:
        cached = cache[cache_key]
        resume_struct = cached.get("resume_struct", _normalize_resume_struct(DEFAULT_RESUME_STRUCT))
        error = cached.get("error", "")
        debug = cached.get("debug", {})
        if return_debug:
            return resume_struct, error, debug
        return resume_struct, error

    result = call_resume_structuring(
        resume_text=resume_text,
        model=model,
        ollama_url=ollama_url,
        enable_remote_fallback=enable_remote_fallback,
        remote_base_url=remote_base_url,
        remote_api_key=remote_api_key,
        remote_model=remote_model,
        return_debug=return_debug,
    )
    if return_debug:
        resume_struct, error, debug = result
    else:
        resume_struct, error = result
        debug = {}

    if use_cache:
        cache[cache_key] = {
            "resume_struct": resume_struct,
            "error": error,
            "debug": debug,
        }
        cache_changed = True

    if use_cache and cache_changed:
        save_json_cache(cache_name, cache)

    if return_debug:
        return resume_struct, error, debug
    return resume_struct, error
