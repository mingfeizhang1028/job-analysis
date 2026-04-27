from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from modules.llm_cache import make_cache_key, load_json_cache, save_json_cache
from modules.llm_client import call_ollama_generate, call_openai_compatible_generate


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"

LLM_JOB_CONTENT_COL = "LLM岗位工作内容"
LLM_JOB_REQUIREMENTS_COL = "LLM岗位要求"
LLM_JOB_BONUS_COL = "LLM加分项"
LLM_JOB_INDUSTRY_COL = "LLM所属行业"
LLM_JOB_TYPE_COL = "LLM岗位类型"
LLM_JOB_GOAL_COL = "LLM核心目标"
LLM_DEGREE_REQ_COL = "LLM学历要求"
LLM_EXPERIENCE_REQ_COL = "LLM经验要求"
LLM_FRESH_GRAD_COL = "LLM应届友好度"
LLM_SENIORITY_COL = "LLM资历级别"
LLM_MUST_HAVE_SKILLS_COL = "LLM必须技能"
LLM_NICE_TO_HAVE_SKILLS_COL = "LLM加分技能"
LLM_TOOL_STACK_COL = "LLM工具栈"
LLM_JOB_STRUCT_ERROR_COL = "LLM结构化提取错误"


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _build_jd_text(
    row: pd.Series,
    title_cols: Optional[List[str]] = None,
    detail_col: str = "岗位详情",
    max_chars: int = 4000,
) -> str:
    if title_cols is None:
        title_cols = ["职位名称_norm", "职位名称_raw", "职位名称", "企业名称_norm", "企业名称_raw"]

    parts = []

    for col in title_cols:
        if col in row.index:
            val = _safe_str(row.get(col))
            if val:
                parts.append(f"【{col}】{val}")

    if detail_col in row.index:
        detail = _safe_str(row.get(detail_col))
        if detail:
            parts.append(f"【{detail_col}】{detail}")

    text = "\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _build_prompt(jd_text: str) -> str:
    return f"""
你是招聘JD结构化抽取助手。任务：只根据JD原文抽取岗位信息，输出可用于统计分析的短标签。

严格要求：
1. 只输出一个合法 JSON 对象，不要 Markdown，不要解释。
2. 不要编造 JD 没有依据的信息。
3. 数组元素必须是短标签或短短语，不要输出长句。
4. 标签优先使用中文，常见英文技术名词可保留，如 Python、SQL、RAG、LLM、Excel。
5. 不要把福利、薪资、地点、公司宣传语当成岗位能力。
6. 同义表达要收敛：
   - 沟通能力/团队协作/跨部门沟通 -> 沟通协作
   - 逻辑能力/结构化表达/条理清晰 -> 结构化思维
   - 提示词工程/Prompt Engineering/提示词设计 -> Prompt设计
   - AI工具应用/使用AI工具/AIGC工具 -> AI工具使用
   - 流程设计/流程改进/流程搭建 -> 流程优化
7. 如果无法判断，数组返回 []，行业返回 ["未知"]，字符串返回 ""。

字段定义：
- job_content：入职后主要做什么，3-6个短语。
- job_requirements：候选人核心要求，3-8个短语。
- bonus_points：优先/加分条件，0-5个短语。
- industry：所属行业/业务领域，1-3个标签，不明确返回 ["未知"]。
- job_type：岗位类型，1-3个标签，如 产品、运营、数据、技术、市场、销售、设计、职能、管理。
- core_goal：岗位核心目标，不超过20字。
- degree_requirement：学历要求，如 本科及以上；未提及返回 ""。
- experience_requirement：经验要求，如 应届、1-3年、3年以上；未提及返回 ""。
- fresh_grad_friendly：只能是 是、否、不明确。
- seniority_level：只能是 应届、初级、中级、高级、不明确。
- must_have_skills：必须技能，3-8个短标签。
- nice_to_have_skills：加分技能，0-5个短标签。
- tool_stack：工具/平台/技术栈，0-8个短标签。

输出 JSON 格式：
{{
  "job_content": [],
  "job_requirements": [],
  "bonus_points": [],
  "industry": [],
  "job_type": [],
  "core_goal": "",
  "degree_requirement": "",
  "experience_requirement": "",
  "fresh_grad_friendly": "不明确",
  "seniority_level": "不明确",
  "must_have_skills": [],
  "nice_to_have_skills": [],
  "tool_stack": []
}}

JD：
{jd_text}
""".strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "job_content": [],
            "job_requirements": [],
            "bonus_points": [],
            "industry": ["未知"],
            "job_type": [],
            "core_goal": "",
            "degree_requirement": "",
            "experience_requirement": "",
            "fresh_grad_friendly": "不明确",
            "seniority_level": "",
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "tool_stack": [],
        }

    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "job_content": [],
        "job_requirements": [],
        "bonus_points": [],
        "industry": ["未知"],
        "job_type": [],
        "core_goal": "",
        "degree_requirement": "",
        "experience_requirement": "",
        "fresh_grad_friendly": "不明确",
        "seniority_level": "",
        "must_have_skills": [],
        "nice_to_have_skills": [],
        "tool_stack": [],
    }


def _normalize_list(value: Any, max_items: int = 10, max_len: int = 40) -> List[str]:
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
        s = _safe_str(item)
        s = s.strip(" -•[]【】()（）\"'")
        if not s:
            continue
        if len(s) > max_len:
            continue

        key = s.lower()
        if key not in seen:
            cleaned.append(s)
            seen.add(key)

        if len(cleaned) >= max_items:
            break

    return cleaned


def _normalize_text(value: Any, max_len: int = 25) -> str:
    s = _safe_str(value)
    if not s:
        return ""
    if len(s) > max_len:
        s = s[:max_len]
    return s


def call_ollama_jd_structuring(
    jd_text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
    temperature: float = 0.1,
    provider: str = "local",
    remote_enabled: bool = False,
    remote_model: str = "",
    remote_base_url: str = "",
    remote_api_key: str = "",
) -> Tuple[Dict[str, Any], str]:
    empty_result = {
        "job_content": [],
        "job_requirements": [],
        "bonus_points": [],
        "industry": ["未知"],
        "job_type": [],
        "core_goal": "",
        "degree_requirement": "",
        "experience_requirement": "",
        "fresh_grad_friendly": "不明确",
        "seniority_level": "",
        "must_have_skills": [],
        "nice_to_have_skills": [],
        "tool_stack": [],
    }
    if not jd_text.strip():
        return empty_result, ""

    prompt = _build_prompt(jd_text)
    provider = str(provider or "local").lower()

    fallback_used = False
    if provider == "remote":
        raw_output, remote_error = call_openai_compatible_generate(
            prompt=prompt,
            model=remote_model,
            base_url=remote_base_url,
            api_key=remote_api_key,
            timeout=timeout,
            temperature=temperature,
        )
        if remote_error:
            return empty_result, remote_error
        error = ""
    else:
        raw_output, error = call_ollama_generate(
            prompt=prompt,
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
            temperature=temperature,
            num_predict=1024,
        )

        if error and remote_enabled:
            raw_output, remote_error = call_openai_compatible_generate(
                prompt=prompt,
                model=remote_model,
                base_url=remote_base_url,
                api_key=remote_api_key,
                timeout=timeout,
                temperature=temperature,
            )
            if remote_error:
                return empty_result, f"local_failed: {error}; remote_failed: {remote_error}"
            fallback_used = True
            error = ""

    if error:
        return empty_result, error

    obj = _extract_json_object(raw_output)
    result = {
        "job_content": _normalize_list(obj.get("job_content", []), max_items=8, max_len=40),
        "job_requirements": _normalize_list(obj.get("job_requirements", []), max_items=10, max_len=40),
        "bonus_points": _normalize_list(obj.get("bonus_points", []), max_items=8, max_len=40),
        "industry": _normalize_list(obj.get("industry", ["未知"]), max_items=3, max_len=20) or ["未知"],
        "job_type": _normalize_list(obj.get("job_type", []), max_items=3, max_len=20),
        "core_goal": _normalize_text(obj.get("core_goal", ""), max_len=25),
        "degree_requirement": _normalize_text(obj.get("degree_requirement", ""), max_len=20),
        "experience_requirement": _normalize_text(obj.get("experience_requirement", ""), max_len=20),
        "fresh_grad_friendly": _normalize_text(obj.get("fresh_grad_friendly", "不明确"), max_len=10) or "不明确",
        "seniority_level": _normalize_text(obj.get("seniority_level", ""), max_len=10),
        "must_have_skills": _normalize_list(obj.get("must_have_skills", []), max_items=8, max_len=30),
        "nice_to_have_skills": _normalize_list(obj.get("nice_to_have_skills", []), max_items=6, max_len=30),
        "tool_stack": _normalize_list(obj.get("tool_stack", []), max_items=8, max_len=30),
    }

    meaningful = bool(
        result["job_content"] or result["job_requirements"] or result["must_have_skills"] or result["tool_stack"]
    )
    if not meaningful and remote_enabled and not fallback_used:
        raw_output, remote_error = call_openai_compatible_generate(
            prompt=prompt,
            model=remote_model,
            base_url=remote_base_url,
            api_key=remote_api_key,
            timeout=timeout,
            temperature=temperature,
        )
        if not remote_error:
            obj = _extract_json_object(raw_output)
            result = {
                "job_content": _normalize_list(obj.get("job_content", []), max_items=8, max_len=40),
                "job_requirements": _normalize_list(obj.get("job_requirements", []), max_items=10, max_len=40),
                "bonus_points": _normalize_list(obj.get("bonus_points", []), max_items=8, max_len=40),
                "industry": _normalize_list(obj.get("industry", ["未知"]), max_items=3, max_len=20) or ["未知"],
                "job_type": _normalize_list(obj.get("job_type", []), max_items=3, max_len=20),
                "core_goal": _normalize_text(obj.get("core_goal", ""), max_len=25),
                "degree_requirement": _normalize_text(obj.get("degree_requirement", ""), max_len=20),
                "experience_requirement": _normalize_text(obj.get("experience_requirement", ""), max_len=20),
                "fresh_grad_friendly": _normalize_text(obj.get("fresh_grad_friendly", "不明确"), max_len=10) or "不明确",
                "seniority_level": _normalize_text(obj.get("seniority_level", ""), max_len=10),
                "must_have_skills": _normalize_list(obj.get("must_have_skills", []), max_items=8, max_len=30),
                "nice_to_have_skills": _normalize_list(obj.get("nice_to_have_skills", []), max_items=6, max_len=30),
                "tool_stack": _normalize_list(obj.get("tool_stack", []), max_items=8, max_len=30),
            }
            fallback_used = True
        else:
            return empty_result, f"local_empty_result; remote_failed: {remote_error}"

    if fallback_used:
        return result, "remote_fallback_used"
    if not meaningful:
        return result, "local_empty_result"
    return result, ""


def apply_llm_jd_structuring(
    df: pd.DataFrame,
    detail_col: str = "岗位详情",
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_chars: int = 4000,
    sleep_seconds: float = 0.0,
    limit: Optional[int] = None,
    overwrite: bool = False,
    use_cache: bool = True,
    cache_name: str = "jd_struct_cache.json",
    provider: str = "local",
    remote_enabled: bool = False,
    remote_model: str = "",
    remote_base_url: str = "",
    remote_api_key: str = "",
) -> pd.DataFrame:
    result = df.copy()

    for col, default in [
        (LLM_JOB_CONTENT_COL, [[] for _ in range(len(result))]),
        (LLM_JOB_REQUIREMENTS_COL, [[] for _ in range(len(result))]),
        (LLM_JOB_BONUS_COL, [[] for _ in range(len(result))]),
        (LLM_JOB_INDUSTRY_COL, [[] for _ in range(len(result))]),
        (LLM_JOB_TYPE_COL, [[] for _ in range(len(result))]),
        (LLM_JOB_GOAL_COL, ""),
        (LLM_DEGREE_REQ_COL, ""),
        (LLM_EXPERIENCE_REQ_COL, ""),
        (LLM_FRESH_GRAD_COL, "不明确"),
        (LLM_SENIORITY_COL, ""),
        (LLM_MUST_HAVE_SKILLS_COL, [[] for _ in range(len(result))]),
        (LLM_NICE_TO_HAVE_SKILLS_COL, [[] for _ in range(len(result))]),
        (LLM_TOOL_STACK_COL, [[] for _ in range(len(result))]),
        (LLM_JOB_STRUCT_ERROR_COL, ""),
    ]:
        if col not in result.columns:
            result[col] = default

    indices = list(result.index)
    if limit is not None:
        indices = indices[:limit]

    total = len(indices)
    cache = load_json_cache(cache_name) if use_cache else {}
    cache_changed = False

    for i, idx in enumerate(indices, start=1):
        if not overwrite:
            existing = result.at[idx, LLM_JOB_CONTENT_COL]
            if isinstance(existing, list) and len(existing) > 0:
                continue

        row = result.loc[idx]
        jd_text = _build_jd_text(
            row,
            detail_col=detail_col,
            max_chars=max_chars,
        )

        cache_key = make_cache_key(
            task="jd_structuring",
            model=model,
            jd_text=jd_text,
            version="v1",
        )

        if use_cache and cache_key in cache and not overwrite:
            cached = cache[cache_key]
            struct_result = cached.get("struct_result", {
                "job_content": [],
                "job_requirements": [],
                "bonus_points": [],
                "industry": ["未知"],
                "job_type": [],
                "core_goal": "",
                "degree_requirement": "",
                "experience_requirement": "",
                "fresh_grad_friendly": "不明确",
                "seniority_level": "",
                "must_have_skills": [],
                "nice_to_have_skills": [],
                "tool_stack": [],
            })
            error = cached.get("error", "")
        else:
            struct_result, error = call_ollama_jd_structuring(
                jd_text=jd_text,
                model=model,
                ollama_url=ollama_url,
                provider=provider,
                remote_enabled=remote_enabled,
                remote_model=remote_model,
                remote_base_url=remote_base_url,
                remote_api_key=remote_api_key,
            )

            if use_cache:
                cache[cache_key] = {
                    "struct_result": struct_result,
                    "error": error,
                }
                cache_changed = True

        result.at[idx, LLM_JOB_CONTENT_COL] = struct_result.get("job_content", [])
        result.at[idx, LLM_JOB_REQUIREMENTS_COL] = struct_result.get("job_requirements", [])
        result.at[idx, LLM_JOB_BONUS_COL] = struct_result.get("bonus_points", [])
        result.at[idx, LLM_JOB_INDUSTRY_COL] = struct_result.get("industry", ["未知"])
        result.at[idx, LLM_JOB_TYPE_COL] = struct_result.get("job_type", [])
        result.at[idx, LLM_JOB_GOAL_COL] = struct_result.get("core_goal", "")
        result.at[idx, LLM_DEGREE_REQ_COL] = struct_result.get("degree_requirement", "")
        result.at[idx, LLM_EXPERIENCE_REQ_COL] = struct_result.get("experience_requirement", "")
        result.at[idx, LLM_FRESH_GRAD_COL] = struct_result.get("fresh_grad_friendly", "不明确")
        result.at[idx, LLM_SENIORITY_COL] = struct_result.get("seniority_level", "")
        result.at[idx, LLM_MUST_HAVE_SKILLS_COL] = struct_result.get("must_have_skills", [])
        result.at[idx, LLM_NICE_TO_HAVE_SKILLS_COL] = struct_result.get("nice_to_have_skills", [])
        result.at[idx, LLM_TOOL_STACK_COL] = struct_result.get("tool_stack", [])
        result.at[idx, LLM_JOB_STRUCT_ERROR_COL] = error

        if sleep_seconds > 0 and i < total:
            time.sleep(sleep_seconds)

    if use_cache and cache_changed:
        save_json_cache(cache_name, cache)

    return result
