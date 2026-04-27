# modules/llm_skill_extraction.py

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



def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _build_jd_text(
    row: pd.Series,
    title_cols: Optional[List[str]] = None,
    detail_col: str = "岗位详情",
    max_chars: int = 3000,
) -> str:
    """
    构造给 LLM 的 JD 文本。
    默认会优先拼接职位名称和岗位详情。
    """
    if title_cols is None:
        title_cols = ["职位名称_norm", "职位名称_raw", "职位名称"]

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
你是招聘JD技能标签抽取助手。任务：从JD中抽取硬技能和软素质标签，用于统计分析。

严格要求：
1. 只输出合法 JSON，不要 Markdown，不要解释。
2. 不要编造 JD 没有依据的能力。
3. 标签必须短，优先名词或短语，不要长句。
4. 每类最多 10 个。
5. 不要输出学历、年限、薪资、地点、福利。
6. 同义表达必须收敛：
   - 沟通能力/团队协作/跨部门沟通 -> 沟通协作
   - 逻辑能力/结构化表达/条理清晰 -> 结构化思维
   - 提示词工程/Prompt Engineering/提示词设计 -> Prompt设计
   - AI工具应用/使用AI工具/AIGC工具 -> AI工具使用
   - 流程设计/流程改进/流程搭建 -> 流程优化

硬技能定义：
工具、方法、专业能力、业务技能、技术栈，例如 Python、SQL、Excel、数据分析、用户研究、产品设计、项目管理、市场调研、AI工具使用、Prompt设计、RAG、LLM。

软素质定义：
行为能力、协作能力、思维方式、职业素养，例如 沟通协作、结构化思维、学习能力、责任心、主动性、抗压能力、问题解决能力、创新意识。

输出 JSON：
{{
  "hard_skills": [],
  "soft_skills": []
}}

JD：
{jd_text}
""".strip()

def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    尽量从模型输出中提取 JSON 对象。
    Ollama 有时会在 JSON 前后加解释，这里做容错。
    """
    if not text:
        return {"hard_skills": [], "soft_skills": []}

    text = text.strip()

    # 去掉可能的 markdown 包裹
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # 尝试截取第一个 {...}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"hard_skills": [], "soft_skills": []}


def _normalize_skill_list(value: Any, max_items: int = 12) -> List[str]:
    """
    清洗模型返回的技能列表。
    """
    if value is None:
        return []

    if isinstance(value, str):
        # 兼容模型返回 "Python, SQL, 沟通能力"
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

        # 过滤太长的句子，避免把解释当标签
        if len(s) > 30:
            continue

        key = s.lower()
        if key not in seen:
            cleaned.append(s)
            seen.add(key)

        if len(cleaned) >= max_items:
            break

    return cleaned


def call_ollama_skill_extraction(
    jd_text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
    temperature: float = 0.1,
    remote_enabled: bool = False,
    remote_model: str = "",
    remote_base_url: str = "",
    remote_api_key: str = "",
) -> Tuple[List[str], List[str], str]:
    """
    调用本地模型；当本地失败且已开启远程兜底时，尝试远程模型。
    返回：hard_skills, soft_skills, error_message
    """
    if not jd_text.strip():
        return [], [], ""

    prompt = _build_prompt(jd_text)

    raw_output, error = call_ollama_generate(
        prompt=prompt,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        temperature=temperature,
        num_predict=512,
    )

    fallback_used = False
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
            return [], [], f"local_failed: {error}; remote_failed: {remote_error}"
        fallback_used = True
        error = ""

    if error:
        return [], [], error

    obj = _extract_json_object(raw_output)
    hard_skills = _normalize_skill_list(obj.get("hard_skills", []))
    soft_skills = _normalize_skill_list(obj.get("soft_skills", []))

    if not hard_skills and not soft_skills and remote_enabled and not fallback_used:
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
            hard_skills = _normalize_skill_list(obj.get("hard_skills", []))
            soft_skills = _normalize_skill_list(obj.get("soft_skills", []))
            fallback_used = True
        elif not hard_skills and not soft_skills:
            return [], [], f"local_empty_result; remote_failed: {remote_error}"

    if fallback_used:
        return hard_skills, soft_skills, "remote_fallback_used"
    return hard_skills, soft_skills, ""


def _merge_into_list_field(existing: Any, additions: List[str]) -> List[str]:
    base = _normalize_skill_list(existing, max_items=100) if existing is not None else []
    cleaned_additions = _normalize_skill_list(additions, max_items=100)
    merged: List[str] = []
    seen: set[str] = set()
    for item in base + cleaned_additions:
        key = _safe_str(item).lower()
        value = _safe_str(item)
        if not value or key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged



def apply_llm_skill_extraction(
    df: pd.DataFrame,
    detail_col: str = "岗位详情",
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_chars: int = 3000,
    sleep_seconds: float = 0.0,
    limit: Optional[int] = None,
    overwrite: bool = False,
    use_cache: bool = True,
    cache_name: str = "skill_cache.json",
    remote_enabled: bool = False,
    remote_model: str = "",
    remote_base_url: str = "",
    remote_api_key: str = "",
) -> pd.DataFrame:
    """
    对 DataFrame 每一行调用本地 Ollama，执行技能/素质识别流程。

    当前策略：
    - 不再写入独立的 LLM硬技能标签 / LLM软技能标签 / LLM技能识别错误 列
    - 若识别出新的技能或素质标签，则直接合并进原有规则标签列：
      * 硬技能 -> 硬技能标签
      * 软素质 -> 软素质标签
    - overwrite=False 时，仅在原标签列为空时跳过；否则允许在原标签基础上补充新标签
    """
    result = df.copy()

    if "硬技能标签" not in result.columns:
        result["硬技能标签"] = [[] for _ in range(len(result))]
    if "软素质标签" not in result.columns:
        result["软素质标签"] = [[] for _ in range(len(result))]

    indices = list(result.index)
    if limit is not None:
        indices = indices[:limit]

    cache = load_json_cache(cache_name) if use_cache else {}
    cache_changed = False

    for idx in indices:
        row = result.loc[idx]

        existing_hard = _normalize_skill_list(row.get("硬技能标签", []), max_items=100)
        existing_soft = _normalize_skill_list(row.get("软素质标签", []), max_items=100)
        if not overwrite and existing_hard and existing_soft:
            continue

        jd_text = _build_jd_text(
            row,
            detail_col=detail_col,
            max_chars=max_chars,
        )

        cache_key = make_cache_key(
            task="skill_extraction",
            model=model,
            jd_text=jd_text,
            version="v2_merge_into_rule_tags",
        )

        if use_cache and cache_key in cache and not overwrite:
            cached = cache[cache_key]
            hard_skills = cached.get("hard_skills", [])
            soft_skills = cached.get("soft_skills", [])
        else:
            hard_skills, soft_skills, _error = call_ollama_skill_extraction(
                jd_text=jd_text,
                model=model,
                ollama_url=ollama_url,
                remote_enabled=remote_enabled,
                remote_model=remote_model,
                remote_base_url=remote_base_url,
                remote_api_key=remote_api_key,
            )

            if use_cache:
                cache[cache_key] = {
                    "hard_skills": hard_skills,
                    "soft_skills": soft_skills,
                }
                cache_changed = True

        result.at[idx, "硬技能标签"] = _merge_into_list_field(result.at[idx, "硬技能标签"], hard_skills)
        result.at[idx, "软素质标签"] = _merge_into_list_field(result.at[idx, "软素质标签"], soft_skills)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if use_cache and cache_changed:
        save_json_cache(cache_name, cache)

    return result


__all__ = ["apply_llm_skill_extraction", "call_ollama_skill_extraction"]

