from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import pandas as pd

from modules.llm_cache import load_json_cache, make_cache_key, save_json_cache
from modules.llm_client import call_ollama_generate, call_openai_compatible_generate


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"

LLM_DEEP_HARD_COL = "LLM深度硬技能标签"
LLM_DEEP_SOFT_COL = "LLM深度软素质标签"
LLM_DEEP_RESP_COL = "LLM深度业务职责标签"
LLM_DEEP_SCENE_COL = "LLM深度行业场景标签"
LLM_DEEP_REQ_COL = "LLM深度岗位要求"
LLM_DEEP_BONUS_COL = "LLM深度加分项"
LLM_DEEP_TOOL_COL = "LLM深度工具栈"
LLM_DEEP_TYPE_COL = "LLM深度岗位类型"
LLM_DEEP_ERROR_COL = "LLM深度标签错误"


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _normalize_list(value: Any, max_items: int = 18, max_len: int = 32) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(r"[，,；;、\n]+", value)
    elif isinstance(value, list):
        items = value
    else:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _safe_str(item).strip(" -[]【】()（）\"'")
        if not text or len(text) > max_len:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= max_items:
            break
    return result


def _merge_list(existing: Any, additions: Any, max_items: int = 40) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for source in [existing, additions]:
        for item in _normalize_list(source, max_items=100, max_len=40):
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
            if len(result) >= max_items:
                return result
    return result


def _build_jd_text(row: pd.Series, detail_col: str, max_chars: int = 5000) -> str:
    title_cols = ["职位名称_norm", "职位名称_raw", "职位名称", "企业名称_norm", "企业名称_raw", "企业名称"]
    context_cols = [
        "规则岗位工作内容",
        "规则岗位要求",
        "规则加分项",
        "规则工具栈",
        "硬技能标签",
        "软素质标签",
        "业务职责标签",
        "行业场景标签",
    ]
    parts: list[str] = []
    for col in title_cols + context_cols:
        if col in row.index:
            value = row.get(col)
            if isinstance(value, list):
                value = "，".join(str(item) for item in value)
            value_str = _safe_str(value)
            if value_str:
                parts.append(f"【{col}】{value_str}")
    detail = _safe_str(row.get(detail_col, ""))
    if detail:
        parts.append(f"【岗位详情】{detail}")
    text = "\n".join(parts).strip()
    return text[:max_chars]


def _build_prompt(jd_text: str) -> str:
    return f"""
你是招聘 JD 深度标签分析助手。请只基于 JD 原文和已有候选标签，抽取可用于岗位画像、能力匹配和后续统计的高质量短标签。

要求：
1. 只输出一个合法 JSON 对象，不要 Markdown，不要解释。
2. 不要只抽泛词。优先保留具体工具、平台、系统、流程、业务动作和落地场景。
3. 标签应为短名词或短动宾短语，避免长句；每个标签不超过 14 个中文字符，常见英文工具名可保留。
4. 不要把福利、薪资、地点、公司宣传语当成能力标签。
5. 对明显 OCR/抓取错误要按语义修正，例如：
   - A智能体 / A系统 / A服务 -> AI智能体 / AI系统 / AI服务
   - 0penClaw -> OpenClaw
   - 八书 -> 飞书
   - 唾眠 -> 睡眠
6. “业务职责”只放入岗位要做的事情，不要混入候选人要求。
7. “岗位要求”放入候选人需要具备的能力、经验、工具或思维方式。
8. “加分项”只放 JD 中明确优先、加分、我们提供中与能力成长或稀缺经验相关的内容；没有依据则返回 []。
9. 行业场景要具体，例如 智能客服、智能销售、企业服务、睡眠行业、门店运营、线上项目。

JSON 字段：
{{
  "hard_skills": [],
  "soft_skills": [],
  "responsibilities": [],
  "industry_scenes": [],
  "job_requirements": [],
  "bonus_points": [],
  "tool_stack": [],
  "job_type": []
}}

字段含义：
- hard_skills：工具、平台、技术、方法、专业能力，如 FastGPT、OpenClaw、企业微信、飞书、Python、低代码工具、数据分析、知识库建设、流程自动化。
- soft_skills：思维和行为能力，如 逻辑思维、流程设计能力、分析能力、沟通协作、学习能力、实战能力。
- responsibilities：岗位职责动作，如 智能体搭建与优化、对话流程设计、节点管理、系统对接、智能客服落地、智能销售落地、客户引流、线索分发、数据分析与效果优化。
- industry_scenes：行业或业务场景，如 AI应用、智能客服、智能销售、企业服务、健康服务、睡眠行业、门店运营、线上项目。
- job_requirements：候选人要求，可以混合技能和素质，但必须是 JD 中要求具备的。
- bonus_points：明确优先或加分的技能、经验、平台或业务背景。
- tool_stack：工具/平台/系统名。
- job_type：岗位类型或方向，如 AI产品/运营、智能客服/销售、数据分析。

JD：
{jd_text}
""".strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
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
    return {}


def call_llm_tag_refinement(
    jd_text: str,
    *,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
    temperature: float = 0.1,
    remote_enabled: bool = False,
    remote_model: str = "",
    remote_base_url: str = "",
    remote_api_key: str = "",
) -> tuple[dict[str, list[str]], str]:
    empty = {
        "hard_skills": [],
        "soft_skills": [],
        "responsibilities": [],
        "industry_scenes": [],
        "job_requirements": [],
        "bonus_points": [],
        "tool_stack": [],
        "job_type": [],
    }
    if not jd_text.strip():
        return empty, ""

    prompt = _build_prompt(jd_text)
    raw_output, error = call_ollama_generate(
        prompt=prompt,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        temperature=temperature,
        num_predict=1200,
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
            return empty, f"local_failed: {error}; remote_failed: {remote_error}"
        fallback_used = True
        error = ""

    if error:
        return empty, error

    obj = _extract_json_object(raw_output)
    result = {
        "hard_skills": _normalize_list(obj.get("hard_skills", []), max_items=18),
        "soft_skills": _normalize_list(obj.get("soft_skills", []), max_items=12),
        "responsibilities": _normalize_list(obj.get("responsibilities", []), max_items=18),
        "industry_scenes": _normalize_list(obj.get("industry_scenes", []), max_items=12),
        "job_requirements": _normalize_list(obj.get("job_requirements", []), max_items=18),
        "bonus_points": _normalize_list(obj.get("bonus_points", []), max_items=12),
        "tool_stack": _normalize_list(obj.get("tool_stack", []), max_items=12),
        "job_type": _normalize_list(obj.get("job_type", []), max_items=6),
    }
    if fallback_used:
        return result, "remote_fallback_used"
    if not any(result.values()):
        return result, "empty_result"
    return result, ""


def apply_llm_tag_refinement(
    df: pd.DataFrame,
    detail_col: str = "岗位详情",
    *,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_chars: int = 5000,
    sleep_seconds: float = 0.0,
    limit: Optional[int] = None,
    overwrite: bool = False,
    use_cache: bool = True,
    cache_name: str = "tag_refinement_cache.json",
    remote_enabled: bool = False,
    remote_model: str = "",
    remote_base_url: str = "",
    remote_api_key: str = "",
) -> pd.DataFrame:
    result = df.copy()

    list_cols = [
        LLM_DEEP_HARD_COL,
        LLM_DEEP_SOFT_COL,
        LLM_DEEP_RESP_COL,
        LLM_DEEP_SCENE_COL,
        LLM_DEEP_REQ_COL,
        LLM_DEEP_BONUS_COL,
        LLM_DEEP_TOOL_COL,
        LLM_DEEP_TYPE_COL,
    ]
    for col in list_cols:
        if col not in result.columns:
            result[col] = [[] for _ in range(len(result))]
    if LLM_DEEP_ERROR_COL not in result.columns:
        result[LLM_DEEP_ERROR_COL] = ""

    for col in ["硬技能标签", "软素质标签", "业务职责标签", "行业场景标签"]:
        if col not in result.columns:
            result[col] = [[] for _ in range(len(result))]

    indices = list(result.index)
    if limit is not None:
        indices = indices[:limit]

    cache = load_json_cache(cache_name) if use_cache else {}
    cache_changed = False

    for idx in indices:
        if not overwrite and _normalize_list(result.at[idx, LLM_DEEP_HARD_COL]):
            continue

        row = result.loc[idx]
        jd_text = _build_jd_text(row, detail_col=detail_col, max_chars=max_chars)
        cache_key = make_cache_key(
            task="tag_refinement",
            model=model,
            jd_text=jd_text,
            version="v1_deep_tags",
        )

        if use_cache and cache_key in cache and not overwrite:
            payload = cache[cache_key].get("result", {})
            error = cache[cache_key].get("error", "")
        else:
            payload, error = call_llm_tag_refinement(
                jd_text,
                model=model,
                ollama_url=ollama_url,
                remote_enabled=remote_enabled,
                remote_model=remote_model,
                remote_base_url=remote_base_url,
                remote_api_key=remote_api_key,
            )
            if use_cache:
                cache[cache_key] = {"result": payload, "error": error}
                cache_changed = True

        result.at[idx, LLM_DEEP_HARD_COL] = payload.get("hard_skills", [])
        result.at[idx, LLM_DEEP_SOFT_COL] = payload.get("soft_skills", [])
        result.at[idx, LLM_DEEP_RESP_COL] = payload.get("responsibilities", [])
        result.at[idx, LLM_DEEP_SCENE_COL] = payload.get("industry_scenes", [])
        result.at[idx, LLM_DEEP_REQ_COL] = payload.get("job_requirements", [])
        result.at[idx, LLM_DEEP_BONUS_COL] = payload.get("bonus_points", [])
        result.at[idx, LLM_DEEP_TOOL_COL] = payload.get("tool_stack", [])
        result.at[idx, LLM_DEEP_TYPE_COL] = payload.get("job_type", [])
        result.at[idx, LLM_DEEP_ERROR_COL] = error

        result.at[idx, "硬技能标签"] = _merge_list(
            result.at[idx, "硬技能标签"],
            payload.get("hard_skills", []) + payload.get("tool_stack", []),
        )
        result.at[idx, "软素质标签"] = _merge_list(result.at[idx, "软素质标签"], payload.get("soft_skills", []))
        result.at[idx, "业务职责标签"] = _merge_list(result.at[idx, "业务职责标签"], payload.get("responsibilities", []))
        result.at[idx, "行业场景标签"] = _merge_list(
            result.at[idx, "行业场景标签"],
            payload.get("industry_scenes", []) + payload.get("job_type", []),
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if use_cache and cache_changed:
        save_json_cache(cache_name, cache)

    return result

