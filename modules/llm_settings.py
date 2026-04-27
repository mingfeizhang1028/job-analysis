from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = BASE_DIR / "data" / "llm_settings.json"


DEFAULT_LLM_SETTINGS: Dict[str, Any] = {
    "llm_skill_extraction_enabled": False,
    "llm_skill_extraction_limit": 20,
    "llm_skill_extraction_overwrite": False,
    "llm_jd_structuring_enabled": False,
    "llm_jd_structuring_limit": 20,
    "llm_jd_structuring_overwrite": False,
    "llm_tag_refinement_enabled": False,
    "llm_tag_refinement_limit": 20,
    "llm_tag_refinement_overwrite": False,
    "jd_skill_provider": "local",
    "jd_skill_local_model": "qwen3:8b",
    "jd_skill_local_url": "http://localhost:11434/api/generate",
    "jd_skill_remote_enabled": False,
    "jd_skill_remote_model": "gpt-5.4",
    "jd_skill_remote_base_url": "",
    "jd_skill_remote_api_key": "",
    "jd_struct_provider": "local",
    "jd_struct_local_model": "qwen3:8b",
    "jd_struct_local_url": "http://localhost:11434/api/generate",
    "jd_struct_remote_enabled": False,
    "jd_struct_remote_model": "gpt-5.4",
    "jd_struct_remote_base_url": "",
    "jd_struct_remote_api_key": "",
    "resume_struct_provider": "local",
    "resume_struct_local_model": "qwen3:8b",
    "resume_struct_local_url": "http://localhost:11434/api/generate",
    "resume_struct_remote_enabled": False,
    "resume_struct_remote_model": "gpt-5.4",
    "resume_struct_remote_base_url": "",
    "resume_struct_remote_api_key": "",
    "kb_embedding_provider": "local",
    "kb_embedding_local_model": "qwen3-embedding:8b",
    "kb_embedding_local_url": "http://localhost:11434/api/embeddings",
    "kb_embedding_remote_enabled": False,
    "kb_embedding_remote_model": "text-embedding-3-small",
    "kb_embedding_remote_base_url": "",
    "kb_embedding_remote_api_key": "",
    "evidence_embedding_provider": "local",
    "evidence_embedding_local_model": "qwen3-embedding:8b",
    "evidence_embedding_local_url": "http://localhost:11434/api/embeddings",
    "evidence_embedding_remote_enabled": False,
    "evidence_embedding_remote_model": "text-embedding-3-small",
    "evidence_embedding_remote_base_url": "",
    "evidence_embedding_remote_api_key": "",
    "network_advice_provider": "local",
    "network_advice_local_model": "qwen3:8b",
    "network_advice_local_url": "http://localhost:11434/api/generate",
    "network_advice_remote_enabled": False,
    "network_advice_remote_model": "gpt-5.4",
    "network_advice_remote_base_url": "",
    "network_advice_remote_api_key": "",
}


TASK_LABELS = {
    "jd_skill": "JD 技能识别（硬技能 / 软技能）",
    "jd_struct": "JD 结构化提取（工作内容 / 要求 / 加分项等）",
    "resume_struct": "简历结构化",
    "kb_embedding": "知识库入库 Embedding",
    "evidence_embedding": "证据召回 Embedding",
    "network_advice": "岗位关系网络洞察建议",
}


def get_default_llm_settings() -> Dict[str, Any]:
    return DEFAULT_LLM_SETTINGS.copy()


def get_llm_settings() -> Dict[str, Any]:
    settings = st.session_state.get("llm_settings")
    if not isinstance(settings, dict):
        settings = get_default_llm_settings()
        if SETTINGS_PATH.exists():
            try:
                loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    settings.update(loaded)
            except Exception:
                pass
        st.session_state["llm_settings"] = settings
    merged = get_default_llm_settings()
    merged.update(settings)
    st.session_state["llm_settings"] = merged
    return merged


def save_llm_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    merged = get_default_llm_settings()
    merged.update(settings or {})
    st.session_state["llm_settings"] = merged
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return merged


def _render_task_provider_editor(task_key: str, current: Dict[str, Any], provider_help: str = "") -> Dict[str, Any]:
    task_label = TASK_LABELS.get(task_key, task_key)
    provider_key = f"{task_key}_provider"
    local_model_key = f"{task_key}_local_model"
    local_url_key = f"{task_key}_local_url"
    remote_enabled_key = f"{task_key}_remote_enabled"
    remote_model_key = f"{task_key}_remote_model"
    remote_base_url_key = f"{task_key}_remote_base_url"
    remote_api_key_key = f"{task_key}_remote_api_key"

    provider = st.selectbox(
        "模型类型",
        ["local", "remote"],
        index=0 if str(current.get(provider_key, "local")) != "remote" else 1,
        format_func=lambda x: "本地模型" if x == "local" else "远程模型",
        key=f"llm_settings_{provider_key}",
        help=provider_help or f"为“{task_label}”单独选择模型来源。",
    )

    local_col, remote_col = st.columns(2)
    with local_col:
        st.markdown("**本地模型配置**")
        local_model = st.text_input(
            "本地模型名称",
            value=str(current.get(local_model_key, DEFAULT_LLM_SETTINGS[local_model_key])),
            key=f"llm_settings_{local_model_key}",
        )
        local_url = st.text_input(
            "本地接口 URL",
            value=str(current.get(local_url_key, DEFAULT_LLM_SETTINGS[local_url_key])),
            key=f"llm_settings_{local_url_key}",
        )

    with remote_col:
        st.markdown("**远程模型配置**")
        remote_enabled = st.checkbox(
            "允许使用远程模型",
            value=bool(current.get(remote_enabled_key, False)),
            key=f"llm_settings_{remote_enabled_key}",
        )
        remote_model = st.text_input(
            "远程模型名称",
            value=str(current.get(remote_model_key, DEFAULT_LLM_SETTINGS[remote_model_key])),
            key=f"llm_settings_{remote_model_key}",
        )
        remote_base_url = st.text_input(
            "远程 Base URL",
            value=str(current.get(remote_base_url_key, "")),
            key=f"llm_settings_{remote_base_url_key}",
        )
        remote_api_key = st.text_input(
            "远程 API Key",
            value=str(current.get(remote_api_key_key, "")),
            type="password",
            key=f"llm_settings_{remote_api_key_key}",
        )

    return {
        provider_key: provider,
        local_model_key: local_model,
        local_url_key: local_url,
        remote_enabled_key: remote_enabled,
        remote_model_key: remote_model,
        remote_base_url_key: remote_base_url,
        remote_api_key_key: remote_api_key,
    }


def render_skill_analysis_settings_panel(in_sidebar: bool = False) -> Dict[str, Any]:
    """Render page-local settings for JD skill/tag analysis."""
    current = get_llm_settings()
    target = st.sidebar if in_sidebar else st
    updated: Dict[str, Any] = {}

    from app_core import STAT_METRIC_OPTIONS, TAG_TYPE_OPTIONS
    from modules.tag_source_resolver import get_supported_tag_sources

    tag_type = target.selectbox(
        "关键词/标签类型",
        TAG_TYPE_OPTIONS,
        key="skill_analysis_tag_type",
    )
    stat_metric = target.selectbox(
        "统计口径",
        STAT_METRIC_OPTIONS,
        index=0,
        key="skill_analysis_stat_metric",
    )
    source_options = get_supported_tag_sources()
    source_mode = target.selectbox(
        "标签来源",
        source_options,
        index=0,
        key="skill_analysis_source_mode",
    )

    with target.expander("LLM 技能/素质补充识别配置", expanded=False):
        updated["llm_skill_extraction_enabled"] = target.checkbox(
            "启用 JD 技能/素质补充识别（合并进现有标签列）",
            value=bool(current.get("llm_skill_extraction_enabled", False)),
            key="page_llm_skill_extraction_enabled",
        )
        updated["llm_skill_extraction_limit"] = target.number_input(
            "JD 技能/素质补充识别处理条数限制，0 表示全部",
            min_value=0,
            value=int(current.get("llm_skill_extraction_limit", 20)),
            step=10,
            key="page_llm_skill_extraction_limit",
        )
        updated["llm_skill_extraction_overwrite"] = target.checkbox(
            "覆盖已有 JD 技能/素质补充结果",
            value=bool(current.get("llm_skill_extraction_overwrite", False)),
            key="page_llm_skill_extraction_overwrite",
        )
        updated.update(_render_task_provider_editor("jd_skill", current))

    save_llm_settings({**current, **updated})
    return {
        "tag_type": tag_type,
        "stat_metric": stat_metric,
        "source_mode": source_mode,
        **updated,
    }


_TASK_ALIAS = {
    "skill_extraction": "jd_skill",
    "jd_skill_extraction": "jd_skill",
    "jd_skill": "jd_skill",
    "jd_structuring": "jd_struct",
    "jd_struct": "jd_struct",
    "resume_structuring": "resume_struct",
    "resume_struct": "resume_struct",
    "candidate_embedding": "kb_embedding",
    "kb_embedding": "kb_embedding",
    "evidence_embedding": "evidence_embedding",
    "network_advice": "network_advice",
    "network_insight": "network_advice",
}


def get_task_llm_config(task_key: str) -> Dict[str, Any]:
    """Return normalized model config for a specific LLM / embedding task.

    This is a compatibility helper used by page modules. It accepts both old
    task names such as ``resume_structuring`` / ``candidate_embedding`` and the
    current internal keys such as ``resume_struct`` / ``kb_embedding``.
    """
    settings = get_llm_settings()
    normalized = _TASK_ALIAS.get(task_key, task_key)

    provider = settings.get(f"{normalized}_provider", "local")
    local_model = settings.get(
        f"{normalized}_local_model",
        DEFAULT_LLM_SETTINGS.get(f"{normalized}_local_model", ""),
    )
    local_url = settings.get(
        f"{normalized}_local_url",
        DEFAULT_LLM_SETTINGS.get(f"{normalized}_local_url", ""),
    )
    remote_enabled = bool(settings.get(f"{normalized}_remote_enabled", False))
    remote_model = settings.get(
        f"{normalized}_remote_model",
        DEFAULT_LLM_SETTINGS.get(f"{normalized}_remote_model", "gpt-5.4"),
    )
    remote_base_url = settings.get(f"{normalized}_remote_base_url", "")
    remote_api_key = settings.get(f"{normalized}_remote_api_key", "")

    return {
        "task_key": normalized,
        "provider": provider,
        "model_type": provider,
        "local_model": local_model,
        "local_url": local_url,
        "model": local_model if provider != "remote" else remote_model,
        "url": local_url if provider != "remote" else remote_base_url,
        "remote_enabled": remote_enabled,
        "remote_model": remote_model,
        "remote_base_url": remote_base_url,
        "remote_api_key": remote_api_key,
        "embedding_model": local_model,
        "embedding_url": local_url,
    }


def _render_task_settings_panel(task_keys: list[str], title: str = "模型配置", include_switches: bool = False, in_sidebar: bool = False) -> Dict[str, Any]:
    current = get_llm_settings()
    target = st.sidebar if in_sidebar else st
    updated: Dict[str, Any] = {}
    target.subheader(title)
    if include_switches:
        updated["llm_skill_extraction_enabled"] = target.checkbox(
            "启用 JD 技能识别（硬技能 / 软技能）",
            value=bool(current.get("llm_skill_extraction_enabled", False)),
            key=f"{title}_llm_skill_extraction_enabled",
        )
        updated["llm_skill_extraction_limit"] = target.number_input(
            "JD 技能/素质补充识别处理条数限制，0 表示全部",
            min_value=0,
            value=int(current.get("llm_skill_extraction_limit", 20)),
            step=10,
            key=f"{title}_llm_skill_extraction_limit",
        )
        updated["llm_skill_extraction_overwrite"] = target.checkbox(
            "覆盖已有 JD 技能/素质补充结果",
            value=bool(current.get("llm_skill_extraction_overwrite", False)),
            key=f"{title}_llm_skill_extraction_overwrite",
        )
        target.markdown("---")
        updated["llm_tag_refinement_enabled"] = target.checkbox(
            "启用 JD 深度标签增强（职责 / 要求 / 加分项 / 工具栈 / 场景）",
            value=bool(current.get("llm_tag_refinement_enabled", False)),
            key=f"{title}_llm_tag_refinement_enabled",
        )
        updated["llm_tag_refinement_limit"] = target.number_input(
            "JD 深度标签增强处理条数限制，0 表示全部",
            min_value=0,
            value=int(current.get("llm_tag_refinement_limit", 20)),
            step=10,
            key=f"{title}_llm_tag_refinement_limit",
        )
        updated["llm_tag_refinement_overwrite"] = target.checkbox(
            "覆盖已有 JD 深度标签增强结果",
            value=bool(current.get("llm_tag_refinement_overwrite", False)),
            key=f"{title}_llm_tag_refinement_overwrite",
        )
        target.markdown("---")
        updated["llm_jd_structuring_enabled"] = target.checkbox(
            "启用 JD 结构化提取（工作内容 / 要求 / 加分项等）",
            value=bool(current.get("llm_jd_structuring_enabled", False)),
            key=f"{title}_llm_jd_structuring_enabled",
        )
        updated["llm_jd_structuring_limit"] = target.number_input(
            "JD 结构化处理条数限制，0 表示全部",
            min_value=0,
            value=int(current.get("llm_jd_structuring_limit", 20)),
            step=10,
            key=f"{title}_llm_jd_structuring_limit",
        )
        updated["llm_jd_structuring_overwrite"] = target.checkbox(
            "覆盖已有 JD 结构化结果",
            value=bool(current.get("llm_jd_structuring_overwrite", False)),
            key=f"{title}_llm_jd_structuring_overwrite",
        )
    for task_key in task_keys:
        with target.expander(TASK_LABELS[task_key], expanded=False):
            updated.update(_render_task_provider_editor(task_key, current))
    return save_llm_settings({**current, **updated})


def render_page_task_llm_settings(task_keys: list[str], title: str = "模型配置", include_switches: bool = False) -> Dict[str, Any]:
    return _render_task_settings_panel(task_keys=task_keys, title=title, include_switches=include_switches, in_sidebar=False)


def render_llm_settings_panel() -> Dict[str, Any]:
    st.sidebar.header("LLM / Embedding 配置")
    st.sidebar.caption("仅保留全局默认配置；与页面强相关的模型设置会放到对应页面顶部。")
    return _render_task_settings_panel(
        task_keys=["jd_struct"],
        title="全局默认模型配置",
        include_switches=False,
        in_sidebar=True,
    )


def build_processing_llm_overrides() -> Dict[str, Any]:
    settings = get_llm_settings()
    return {
        "enable_llm_skills": bool(settings.get("llm_skill_extraction_enabled", False)),
        "llm_model": settings.get("jd_skill_local_model", DEFAULT_LLM_SETTINGS["jd_skill_local_model"]),
        "ollama_url": settings.get("jd_skill_local_url", DEFAULT_LLM_SETTINGS["jd_skill_local_url"]),
        "llm_limit": int(settings.get("llm_skill_extraction_limit", 20)),
        "llm_overwrite": bool(settings.get("llm_skill_extraction_overwrite", False)),
        "enable_llm_jd_struct": bool(settings.get("llm_jd_structuring_enabled", False)),
        "llm_jd_limit": int(settings.get("llm_jd_structuring_limit", 20)),
        "llm_jd_overwrite": bool(settings.get("llm_jd_structuring_overwrite", False)),
        "enable_llm_tag_refinement": bool(settings.get("llm_tag_refinement_enabled", False)),
        "llm_tag_refinement_limit": int(settings.get("llm_tag_refinement_limit", 20)),
        "llm_tag_refinement_overwrite": bool(settings.get("llm_tag_refinement_overwrite", False)),
        "jd_skill_provider": settings.get("jd_skill_provider", "local"),
        "jd_skill_local_model": settings.get("jd_skill_local_model", DEFAULT_LLM_SETTINGS["jd_skill_local_model"]),
        "jd_skill_local_url": settings.get("jd_skill_local_url", DEFAULT_LLM_SETTINGS["jd_skill_local_url"]),
        "jd_skill_remote_enabled": bool(settings.get("jd_skill_remote_enabled", False)),
        "jd_skill_remote_model": settings.get("jd_skill_remote_model", DEFAULT_LLM_SETTINGS["jd_skill_remote_model"]),
        "jd_skill_remote_base_url": settings.get("jd_skill_remote_base_url", ""),
        "jd_skill_remote_api_key": settings.get("jd_skill_remote_api_key", ""),
        "skill_extraction_llm": {
            "provider": settings.get("jd_skill_provider", "local"),
            "local_model": settings.get("jd_skill_local_model", DEFAULT_LLM_SETTINGS["jd_skill_local_model"]),
            "local_url": settings.get("jd_skill_local_url", DEFAULT_LLM_SETTINGS["jd_skill_local_url"]),
            "remote_enabled": bool(settings.get("jd_skill_remote_enabled", False)),
            "remote_model": settings.get("jd_skill_remote_model", DEFAULT_LLM_SETTINGS["jd_skill_remote_model"]),
            "remote_base_url": settings.get("jd_skill_remote_base_url", ""),
            "remote_api_key": settings.get("jd_skill_remote_api_key", ""),
        },
        "jd_struct_provider": settings.get("jd_struct_provider", "local"),
        "jd_struct_local_model": settings.get("jd_struct_local_model", DEFAULT_LLM_SETTINGS["jd_struct_local_model"]),
        "jd_struct_local_url": settings.get("jd_struct_local_url", DEFAULT_LLM_SETTINGS["jd_struct_local_url"]),
        "jd_struct_remote_enabled": bool(settings.get("jd_struct_remote_enabled", False)),
        "jd_struct_remote_model": settings.get("jd_struct_remote_model", DEFAULT_LLM_SETTINGS["jd_struct_remote_model"]),
        "jd_struct_remote_base_url": settings.get("jd_struct_remote_base_url", ""),
        "jd_struct_remote_api_key": settings.get("jd_struct_remote_api_key", ""),
        "jd_structuring_llm": {
            "provider": settings.get("jd_struct_provider", "local"),
            "local_model": settings.get("jd_struct_local_model", DEFAULT_LLM_SETTINGS["jd_struct_local_model"]),
            "local_url": settings.get("jd_struct_local_url", DEFAULT_LLM_SETTINGS["jd_struct_local_url"]),
            "remote_enabled": bool(settings.get("jd_struct_remote_enabled", False)),
            "remote_model": settings.get("jd_struct_remote_model", DEFAULT_LLM_SETTINGS["jd_struct_remote_model"]),
            "remote_base_url": settings.get("jd_struct_remote_base_url", ""),
            "remote_api_key": settings.get("jd_struct_remote_api_key", ""),
        },
        "resume_struct_provider": settings.get("resume_struct_provider", "local"),
        "resume_struct_local_model": settings.get("resume_struct_local_model", DEFAULT_LLM_SETTINGS["resume_struct_local_model"]),
        "resume_struct_local_url": settings.get("resume_struct_local_url", DEFAULT_LLM_SETTINGS["resume_struct_local_url"]),
        "resume_struct_remote_enabled": bool(settings.get("resume_struct_remote_enabled", False)),
        "resume_struct_remote_model": settings.get("resume_struct_remote_model", DEFAULT_LLM_SETTINGS["resume_struct_remote_model"]),
        "resume_struct_remote_base_url": settings.get("resume_struct_remote_base_url", ""),
        "resume_struct_remote_api_key": settings.get("resume_struct_remote_api_key", ""),
        "embedding_model": settings.get("kb_embedding_local_model", DEFAULT_LLM_SETTINGS["kb_embedding_local_model"]),
        "embedding_url": settings.get("kb_embedding_local_url", DEFAULT_LLM_SETTINGS["kb_embedding_local_url"]),
        "kb_embedding_provider": settings.get("kb_embedding_provider", "local"),
        "kb_embedding_local_model": settings.get("kb_embedding_local_model", DEFAULT_LLM_SETTINGS["kb_embedding_local_model"]),
        "kb_embedding_local_url": settings.get("kb_embedding_local_url", DEFAULT_LLM_SETTINGS["kb_embedding_local_url"]),
        "kb_embedding_remote_enabled": bool(settings.get("kb_embedding_remote_enabled", False)),
        "kb_embedding_remote_model": settings.get("kb_embedding_remote_model", DEFAULT_LLM_SETTINGS["kb_embedding_remote_model"]),
        "kb_embedding_remote_base_url": settings.get("kb_embedding_remote_base_url", ""),
        "kb_embedding_remote_api_key": settings.get("kb_embedding_remote_api_key", ""),
        "evidence_embedding_provider": settings.get("evidence_embedding_provider", "local"),
        "evidence_embedding_local_model": settings.get("evidence_embedding_local_model", DEFAULT_LLM_SETTINGS["evidence_embedding_local_model"]),
        "evidence_embedding_local_url": settings.get("evidence_embedding_local_url", DEFAULT_LLM_SETTINGS["evidence_embedding_local_url"]),
        "evidence_embedding_remote_enabled": bool(settings.get("evidence_embedding_remote_enabled", False)),
        "evidence_embedding_remote_model": settings.get("evidence_embedding_remote_model", DEFAULT_LLM_SETTINGS["evidence_embedding_remote_model"]),
        "evidence_embedding_remote_base_url": settings.get("evidence_embedding_remote_base_url", ""),
        "evidence_embedding_remote_api_key": settings.get("evidence_embedding_remote_api_key", ""),
    }
