from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from modules.llm_client import call_ollama_generate, call_openai_compatible_generate


NETWORK_ADVICE_SCHEMA_HINT = {
    "summary": "一句话概括当前岗位关系网络体现的市场结构",
    "recommended_directions": [
        {
            "direction": "建议投入的岗位/能力方向",
            "reason": "推荐原因，必须引用统计信号",
            "priority": "高/中/低",
        }
    ],
    "focus_first": ["资源有限时最先投入的能力或组合"],
    "core_capabilities": ["必须补齐的核心能力"],
    "bonus_capabilities": ["差异化加分能力"],
    "avoid_low_value_effort": ["暂不建议优先投入的方向"],
    "risk_notes": ["风险或误判提醒"],
    "resume_hints": ["简历表达建议"],
    "interview_hints": ["面试准备建议"],
}


def _df_to_records(df: Any, limit: int = 12) -> list[dict[str, Any]]:
    if isinstance(df, pd.DataFrame) and not df.empty:
        safe_df = df.head(limit).copy()
        return json.loads(safe_df.to_json(orient="records", force_ascii=False))
    return []


def build_network_payload_for_llm(insight_payload: dict[str, Any], *, compact: bool = True) -> dict[str, Any]:
    overview = insight_payload.get("overview", {}) or {}
    payload = {
        "overview": overview,
        "role_distribution": _df_to_records(insight_payload.get("role_distribution"), 10),
        "tag_top": _df_to_records(insight_payload.get("tag_top"), 15 if compact else 25),
        "priority_table": _df_to_records(insight_payload.get("priority_table"), 15 if compact else 25),
        "high_value_combinations": _df_to_records(insight_payload.get("high_value_combinations"), 12 if compact else 25),
        "community_summary": _df_to_records(insight_payload.get("community_summary"), 8 if compact else 15),
        "meta": insight_payload.get("meta", {}) or {},
    }
    return payload


def build_local_network_prompt(insight_payload: dict[str, Any]) -> str:
    payload = build_network_payload_for_llm(insight_payload, compact=True)
    return f"""
你是求职分析助手。请基于下面的岗位关系网络统计结果，给出简洁、务实的求职方向解释。

要求：
1. 只依据输入数据，不要编造岗位事实。
2. 重点解释：主流岗位方向、核心能力、高价值组合、应该优先补什么。
3. 输出 JSON，不要 Markdown，不要额外解释。
4. JSON 字段必须尽量符合这个结构：{json.dumps(NETWORK_ADVICE_SCHEMA_HINT, ensure_ascii=False)}

岗位关系网络统计：
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def build_remote_network_prompt(insight_payload: dict[str, Any]) -> str:
    payload = build_network_payload_for_llm(insight_payload, compact=False)
    return f"""
你是一名资深求职策略顾问，擅长从岗位关系网络、标签共现、岗位方向占比中判断候选人的投入方向。

请基于下面结构化统计数据，输出更高质量的职业努力方向建议。

分析目标：
- 判断当前岗位样本中哪些方向更值得优先投入。
- 区分“必修核心能力”“方向强化能力”“差异化加分能力”“低价值投入”。
- 给出简历优化与面试准备建议。
- 结论必须引用输入中的统计信号，例如覆盖率、优先级、共现组合、主流方向占比、主题簇。

输出要求：
1. 只输出 JSON，不要 Markdown。
2. 不要编造不存在的数据。
3. 建议要可执行，避免空泛。
4. JSON 字段必须尽量符合这个结构：{json.dumps(NETWORK_ADVICE_SCHEMA_HINT, ensure_ascii=False)}

输入数据：
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def _strip_code_fence(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_network_advice(text: str) -> dict[str, Any]:
    raw = _strip_code_fence(text)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"summary": str(data)}
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else {"summary": str(data)}
            except Exception:
                pass
    return {"summary": raw}


def fallback_network_advice(insight_payload: dict[str, Any], reason: str = "") -> dict[str, Any]:
    overview = insight_payload.get("overview", {}) or {}
    priority_df = insight_payload.get("priority_table")
    combo_df = insight_payload.get("high_value_combinations")
    role_df = insight_payload.get("role_distribution")

    focus_first: list[str] = []
    core_capabilities: list[str] = []
    if isinstance(priority_df, pd.DataFrame) and not priority_df.empty:
        focus_first = priority_df.head(5)["标签"].astype(str).tolist() if "标签" in priority_df.columns else []
        if "建议层级" in priority_df.columns and "标签" in priority_df.columns:
            core_capabilities = priority_df[priority_df["建议层级"].isin(["必修核心", "方向强化"])].head(8)["标签"].astype(str).tolist()

    combo_items: list[str] = []
    if isinstance(combo_df, pd.DataFrame) and not combo_df.empty:
        for _, row in combo_df.head(5).iterrows():
            combo_items.append(f"{row.get('标签A', '')} + {row.get('标签B', '')}")

    main_direction = overview.get("main_direction") or "当前样本中的高频方向"
    ratio = overview.get("main_direction_ratio", 0)
    summary = f"当前样本主流方向为“{main_direction}”（占比约 {ratio:.1%}），建议优先围绕高覆盖、高连接能力建立求职叙事。"

    recommended = []
    if isinstance(role_df, pd.DataFrame) and not role_df.empty:
        for _, row in role_df.head(3).iterrows():
            recommended.append({
                "direction": str(row.get("岗位方向", "")),
                "reason": f"该方向在样本中岗位数为 {row.get('岗位数', 0)}，占比约 {float(row.get('占比', 0)):.1%}。",
                "priority": "高" if float(row.get("占比", 0)) >= 0.2 else "中",
            })

    return {
        "summary": summary,
        "recommended_directions": recommended,
        "focus_first": focus_first,
        "core_capabilities": core_capabilities or focus_first[:5],
        "bonus_capabilities": combo_items,
        "avoid_low_value_effort": ["低覆盖且与核心节点连接弱的孤立标签，暂不建议作为主投入方向。"],
        "risk_notes": [reason or "当前为统计兜底建议，未调用 LLM；请结合具体岗位样本复核。"],
        "resume_hints": ["简历中优先围绕高频能力组合组织项目经历，而不是仅罗列工具名。"],
        "interview_hints": ["准备能说明业务场景、方法、指标结果的案例，以对应高频职责和标签组合。"],
        "model_used": "fallback",
        "raw_text": "",
    }


def get_network_advice(
    insight_payload: dict[str, Any],
    llm_config: dict[str, Any],
    mode: str = "auto",
) -> dict[str, Any]:
    mode = str(mode or "auto").lower()
    if mode in {"off", "none", "仅统计"}:
        return fallback_network_advice(insight_payload, reason="未启用 LLM。")

    provider = str(llm_config.get("provider") or llm_config.get("model_type") or "local").lower()
    use_remote = mode in {"deep", "remote", "深度策略"} or provider == "remote"

    if mode in {"quick", "local", "快速洞察"}:
        use_remote = False

    if use_remote:
        if not bool(llm_config.get("remote_enabled", False)) and provider != "remote":
            return fallback_network_advice(insight_payload, reason="远程模型未启用，已回退到统计建议。")
        prompt = build_remote_network_prompt(insight_payload)
        text, err = call_openai_compatible_generate(
            prompt,
            model=str(llm_config.get("remote_model") or llm_config.get("model") or "gpt-5.4"),
            base_url=str(llm_config.get("remote_base_url") or llm_config.get("url") or ""),
            api_key=str(llm_config.get("remote_api_key") or ""),
            timeout=180,
            temperature=0.1,
        )
        model_used = str(llm_config.get("remote_model") or llm_config.get("model") or "remote")
    else:
        prompt = build_local_network_prompt(insight_payload)
        text, err = call_ollama_generate(
            prompt,
            model=str(llm_config.get("local_model") or llm_config.get("model") or "qwen3:8b"),
            ollama_url=str(llm_config.get("local_url") or llm_config.get("url") or "http://localhost:11434/api/generate"),
            timeout=180,
            temperature=0.1,
            num_predict=1600,
        )
        model_used = str(llm_config.get("local_model") or llm_config.get("model") or "qwen3:8b")

    if err or not text.strip():
        return fallback_network_advice(insight_payload, reason=f"LLM 调用失败：{err or '空响应'}")

    parsed = parse_network_advice(text)
    if not parsed:
        return fallback_network_advice(insight_payload, reason="LLM 返回内容无法解析。")
    parsed.setdefault("summary", "")
    parsed.setdefault("recommended_directions", [])
    parsed.setdefault("focus_first", [])
    parsed.setdefault("core_capabilities", [])
    parsed.setdefault("bonus_capabilities", [])
    parsed.setdefault("avoid_low_value_effort", [])
    parsed.setdefault("risk_notes", [])
    parsed.setdefault("resume_hints", [])
    parsed.setdefault("interview_hints", [])
    parsed["model_used"] = model_used
    parsed["raw_text"] = text
    return parsed
