import hashlib
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(page_title="招聘岗位分析工作台", layout="wide")

import pandas as pd

from app_core import (
    load_data,
    process_data,
    render_sidebar_processing_options,
    render_sidebar_data_source,
    apply_global_filters,
)
from modules.llm_settings import build_processing_llm_overrides
from modules.boss_capture import load_captured_jobs, merge_captured_jobs

from page_views.dashboard_page import render_dashboard_page
from page_views.job_profile_page import render_job_profile_page
from page_views.network_page import render_network_page
from page_views.detail_page import render_detail_page
from page_views.job_workspace_page import render_job_workspace_page
from page_views.latex_resume_page import render_latex_resume_page
from page_views import capture_page as capture_page_view


PAGE_OPTIONS = {
    "总览与去重": "dashboard",
    "岗位采集": "capture",
    "岗位与标签分析": "job_profile",
    "岗位关系网络": "network",
    "岗位明细查询": "detail",
    "求职工作台": "resume_match",
}

CACHE_DIR = Path("data/app_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESS_CACHE_VERSION = "v4_llm_tag_refinement"
DEFAULT_DATASET_LABEL = "default_jobs"
CURRENT_DATASET_KEY_STATE = "current_dataset_cache_key"
CURRENT_PROCESS_KEY_STATE = "current_process_cache_key"
CURRENT_PROCESS_META_STATE = "current_process_cache_meta"
CURRENT_PROCESS_DF_STATE = "current_process_df"
CURRENT_PROCESS_SOURCE_STATE = "current_process_source"
CURRENT_FORCE_REBUILD_TOKEN_STATE = "current_force_rebuild_token"


def _json_default(value: Any):
    return str(value)


def _serialize_processing_options(options: dict) -> dict:
    serializable = {}
    for key, value in options.items():
        if isinstance(value, dict):
            serializable[key] = _serialize_processing_options(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            serializable[key] = value
        else:
            serializable[key] = str(value)
    return serializable


def _build_uploaded_file_signature(uploaded_file) -> dict:
    if uploaded_file is None:
        default_path = Path("data/jobs.xlsx")
        exists = default_path.exists()
        stat = default_path.stat() if exists else None
        return {
            "source_type": "default_file",
            "dataset_label": DEFAULT_DATASET_LABEL,
            "path": str(default_path),
            "exists": exists,
            "size": stat.st_size if stat else None,
            "mtime": int(stat.st_mtime) if stat else None,
        }

    file_name = getattr(uploaded_file, "name", "uploaded.xlsx")
    file_bytes = uploaded_file.getvalue()
    return {
        "source_type": "uploaded_file",
        "dataset_label": file_name,
        "file_name": file_name,
        "size": len(file_bytes),
        "content_hash": hashlib.md5(file_bytes).hexdigest(),
    }


def _build_dataset_cache_key(uploaded_file) -> str:
    payload = _build_uploaded_file_signature(uploaded_file)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _build_data_signature(df_raw: pd.DataFrame) -> str:
    if df_raw is None or df_raw.empty:
        return "empty"
    payload = {
        "rows": int(len(df_raw)),
        "columns": list(df_raw.columns),
        "hash": hashlib.md5(
            pd.util.hash_pandas_object(df_raw.fillna(""), index=True).values.tobytes()
        ).hexdigest(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _build_process_cache_key(df_raw: pd.DataFrame, processing_options: dict) -> str:
    payload = {
        "version": PROCESS_CACHE_VERSION,
        "data_signature": _build_data_signature(df_raw),
        "options": _serialize_processing_options(processing_options),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _get_process_cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"processed_{cache_key}.parquet"


def _get_process_meta_path(cache_key: str) -> Path:
    return CACHE_DIR / f"processed_{cache_key}.meta.json"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_processed_cache(cache_key: str) -> pd.DataFrame | None:
    cache_path = _get_process_cache_path(cache_key)
    if not cache_path.exists():
        return None
    try:
        return pd.read_parquet(cache_path)
    except Exception:
        return None


def load_processed_cache_meta(cache_key: str) -> dict:
    return _read_json(_get_process_meta_path(cache_key))


def save_processed_cache(cache_key: str, df: pd.DataFrame, metadata: dict | None = None) -> None:
    cache_path = _get_process_cache_path(cache_key)
    meta_path = _get_process_meta_path(cache_key)
    try:
        df.to_parquet(cache_path, index=False)
        payload = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "cache_key": cache_key,
            "cache_version": PROCESS_CACHE_VERSION,
        }
        if metadata:
            payload.update(metadata)
        meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    except Exception:
        pass


def delete_processed_cache(cache_key: str) -> None:
    for path in [_get_process_cache_path(cache_key), _get_process_meta_path(cache_key)]:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def _build_dataset_label(uploaded_file) -> str:
    if uploaded_file is None:
        return DEFAULT_DATASET_LABEL
    return str(getattr(uploaded_file, "name", DEFAULT_DATASET_LABEL))


def _build_process_run_reason(force_rebuild: bool, cache_exists: bool, same_session_cache: bool) -> str:
    if force_rebuild:
        return "用户手动触发重算"
    if same_session_cache:
        return "命中当前会话缓存"
    if cache_exists:
        return "命中磁盘缓存"
    return "首次处理或缓存缺失"


def render_sidebar_cache_options() -> dict:
    st.sidebar.header("运行与缓存控制")
    st.sidebar.caption("默认优先复用已有处理结果；只有你明确要求时才重新全量计算。")
    force_rebuild = st.sidebar.button("重新计算当前数据", type="primary", use_container_width=True)
    clear_current_cache = st.sidebar.button("删除当前数据缓存", use_container_width=True)
    clear_enriched_state = st.sidebar.button("清除当前会话增强结果", use_container_width=True)
    reuse_manual_enriched = st.sidebar.checkbox("优先复用当前会话中的手工增强结果", value=True)
    return {
        "force_rebuild": force_rebuild,
        "clear_current_cache": clear_current_cache,
        "clear_enriched_state": clear_enriched_state,
        "reuse_manual_enriched": reuse_manual_enriched,
    }


def render_sidebar_navigation() -> str:
    st.sidebar.header("页面导航")
    labels = list(PAGE_OPTIONS.keys())
    default_label = st.session_state.get("sidebar_page_label", labels[0])
    default_index = labels.index(default_label) if default_label in labels else 0
    page_label = st.sidebar.radio("选择页面", labels, index=default_index)
    st.session_state["sidebar_page_label"] = page_label
    return PAGE_OPTIONS[page_label]


def render_current_page(
    page_key: str,
    df_raw: pd.DataFrame,
    filtered: pd.DataFrame,
    processing_options: dict,
):
    if page_key == "dashboard":
        render_dashboard_page(df_raw, filtered)
    elif page_key == "capture":
        render_capture_page(df_raw, processing_options)
    elif page_key == "job_profile":
        render_job_profile_page(filtered)
    elif page_key == "network":
        render_network_page(filtered)
    elif page_key == "detail":
        render_detail_page(filtered)
    elif page_key == "resume_match":
        render_job_workspace_page(filtered, processing_options)
    else:
        st.error(f"未知页面：{page_key}")


def render_capture_page(df_raw: pd.DataFrame | None = None, processing_options: dict | None = None):
    importlib.reload(capture_page_view)
    capture_page_view.render_capture_page(df_raw, processing_options)


def _handle_cache_actions(cache_options: dict, process_cache_key: str) -> bool:
    action_taken = False

    if cache_options["clear_current_cache"]:
        delete_processed_cache(process_cache_key)
        if st.session_state.get(CURRENT_PROCESS_KEY_STATE) == process_cache_key:
            st.session_state.pop(CURRENT_PROCESS_DF_STATE, None)
            st.session_state.pop(CURRENT_PROCESS_META_STATE, None)
            st.session_state.pop(CURRENT_PROCESS_SOURCE_STATE, None)
        st.sidebar.success("当前数据对应的处理缓存已删除。")
        action_taken = True

    if cache_options["clear_enriched_state"]:
        st.session_state.pop("llm_enriched_df", None)
        st.sidebar.success("当前会话中的手工增强结果已清除。")
        action_taken = True

    return action_taken


def _get_processed_dataframe(df_raw: pd.DataFrame, processing_options: dict, uploaded_file, cache_options: dict) -> tuple[pd.DataFrame, dict]:
    dataset_key = _build_dataset_cache_key(uploaded_file)
    dataset_label = _build_dataset_label(uploaded_file)
    process_cache_key = _build_process_cache_key(df_raw, processing_options)

    force_rebuild = bool(cache_options.get("force_rebuild"))
    if force_rebuild:
        st.session_state[CURRENT_FORCE_REBUILD_TOKEN_STATE] = datetime.now().strftime("%Y%m%d%H%M%S%f")

    same_session_dataset = st.session_state.get(CURRENT_DATASET_KEY_STATE) == dataset_key
    same_session_process = st.session_state.get(CURRENT_PROCESS_KEY_STATE) == process_cache_key

    if _handle_cache_actions(cache_options, process_cache_key):
        force_rebuild = force_rebuild or False

    cached_df = None
    source = "new"
    meta = load_processed_cache_meta(process_cache_key)

    if not force_rebuild and same_session_dataset and same_session_process:
        session_df = st.session_state.get(CURRENT_PROCESS_DF_STATE)
        if isinstance(session_df, pd.DataFrame) and not session_df.empty:
            cached_df = session_df.copy()
            source = "session"

    if cached_df is None and not force_rebuild:
        disk_df = load_processed_cache(process_cache_key)
        if isinstance(disk_df, pd.DataFrame) and not disk_df.empty:
            cached_df = disk_df.copy()
            source = "disk"

    if cached_df is None:
        reason = _build_process_run_reason(force_rebuild, bool(meta), False)
        with st.spinner(f"正在处理数据：{reason}..."):
            cached_df = process_data(
                df_raw,
                enable_norm=processing_options["enable_norm"],
                enable_tags=processing_options["enable_tags"],
                enable_dedup=processing_options["enable_dedup"],
                dedup_threshold=processing_options["dedup_threshold"],
                enable_llm_skills=processing_options["enable_llm_skills"],
                llm_model=processing_options["llm_model"],
                ollama_url=processing_options["ollama_url"],
                llm_limit=processing_options["llm_limit"],
                llm_overwrite=processing_options["llm_overwrite"],
                enable_llm_jd_struct=processing_options["enable_llm_jd_struct"],
                llm_jd_limit=processing_options["llm_jd_limit"],
                llm_jd_overwrite=processing_options["llm_jd_overwrite"],
                enable_llm_tag_refinement=processing_options.get("enable_llm_tag_refinement", False),
                llm_tag_refinement_limit=processing_options.get("llm_tag_refinement_limit", 20),
                llm_tag_refinement_overwrite=processing_options.get("llm_tag_refinement_overwrite", False),
                skill_extraction_llm=processing_options.get("skill_extraction_llm"),
                jd_structuring_llm=processing_options.get("jd_structuring_llm"),
            )
        source = "recomputed"
        save_processed_cache(
            process_cache_key,
            cached_df,
            metadata={
                "dataset_key": dataset_key,
                "dataset_label": dataset_label,
                "processing_options": _serialize_processing_options(processing_options),
                "rebuild_reason": "manual" if force_rebuild else "missing_cache",
            },
        )
        meta = load_processed_cache_meta(process_cache_key)
    else:
        if not meta:
            meta = {
                "dataset_key": dataset_key,
                "dataset_label": dataset_label,
                "processing_options": _serialize_processing_options(processing_options),
                "rows": int(len(cached_df)),
            }

    st.session_state[CURRENT_DATASET_KEY_STATE] = dataset_key
    st.session_state[CURRENT_PROCESS_KEY_STATE] = process_cache_key
    st.session_state[CURRENT_PROCESS_META_STATE] = meta
    st.session_state[CURRENT_PROCESS_DF_STATE] = cached_df.copy()
    st.session_state[CURRENT_PROCESS_SOURCE_STATE] = source

    runtime_info = {
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "process_cache_key": process_cache_key,
        "source": source,
        "meta": meta,
        "force_rebuild": force_rebuild,
    }
    return cached_df, runtime_info


def _resolve_final_df(processed_df: pd.DataFrame, cache_options: dict) -> tuple[pd.DataFrame, str]:
    final_df = processed_df.copy()
    final_source = st.session_state.get(CURRENT_PROCESS_SOURCE_STATE, "unknown")

    if cache_options.get("reuse_manual_enriched", True):
        enriched_df = st.session_state.get("llm_enriched_df")
        if isinstance(enriched_df, pd.DataFrame) and not enriched_df.empty and len(enriched_df) == len(processed_df):
            final_df = enriched_df.copy()
            final_source = "session_enriched"

    return final_df, final_source


def _render_cache_status(df_raw: pd.DataFrame, df: pd.DataFrame, filtered: pd.DataFrame, runtime_info: dict, final_source: str) -> None:
    meta = runtime_info.get("meta") or {}
    source_map = {
        "session": "当前会话缓存",
        "disk": "磁盘缓存",
        "recomputed": "本次重新计算",
        "session_enriched": "当前会话手工增强结果",
        "unknown": "未知来源",
    }
    st.sidebar.markdown("---")
    st.sidebar.caption(f"数据集：{runtime_info.get('dataset_label', DEFAULT_DATASET_LABEL)}")
    st.sidebar.caption(f"处理结果来源：{source_map.get(runtime_info.get('source'), runtime_info.get('source'))}")
    st.sidebar.caption(f"当前展示来源：{source_map.get(final_source, final_source)}")
    if meta.get("created_at"):
        st.sidebar.caption(f"缓存生成时间：{meta.get('created_at')}")
    st.sidebar.caption(f"原始数据量：{len(df_raw)}")
    st.sidebar.caption(f"处理后数据量：{len(df)}")
    st.sidebar.caption(f"筛选后数据量：{len(filtered)}")
    dedup_count = len(filtered)
    if "duplicate_keep" in filtered.columns:
        dedup_count = int((filtered["duplicate_keep"] == True).sum())
    elif "is_duplicate" in filtered.columns:
        dedup_count = int((filtered["is_duplicate"] != True).sum())
    st.sidebar.caption(f"去重后数据量：{dedup_count}")
    with st.sidebar.expander("查看缓存详情", expanded=False):
        st.json(
            {
                "dataset_key": runtime_info.get("dataset_key"),
                "process_cache_key": runtime_info.get("process_cache_key"),
                "cache_version": meta.get("cache_version", PROCESS_CACHE_VERSION),
                "rows": meta.get("rows", len(df)),
                "created_at": meta.get("created_at", ""),
            }
        )


def main():
    default_page = st.session_state.pop("navigate_to_page", None)
    if default_page and default_page in PAGE_OPTIONS.values():
        reverse_map = {v: k for k, v in PAGE_OPTIONS.items()}
        st.session_state["sidebar_page_label"] = reverse_map[default_page]

    page_key = render_sidebar_navigation()
    st.title("招聘岗位分析工作台")

    with st.sidebar:
        st.markdown("---")

    processing_options = render_sidebar_processing_options()
    processing_options.update(build_processing_llm_overrides())
    uploaded_file = render_sidebar_data_source()
    cache_options = render_sidebar_cache_options()

    df_raw = load_data(uploaded_file)
    if st.session_state.get("boss_captured_jobs_enabled", False):
        captured_jobs = load_captured_jobs()
        if isinstance(captured_jobs, pd.DataFrame) and not captured_jobs.empty:
            df_raw = merge_captured_jobs(df_raw, captured_jobs)
            st.sidebar.caption(f"BOSS 采集岗位已加入分析：{len(captured_jobs)} 条")
    if df_raw is None or df_raw.empty:
        st.warning("暂无可用数据，请先检查数据源。")
        return

    if page_key == "capture":
        render_capture_page(df_raw, processing_options)
        return

    processed_df, runtime_info = _get_processed_dataframe(
        df_raw=df_raw,
        processing_options=processing_options,
        uploaded_file=uploaded_file,
        cache_options=cache_options,
    )

    df, final_source = _resolve_final_df(processed_df, cache_options)
    filtered = apply_global_filters(df)
    _render_cache_status(df_raw, df, filtered, runtime_info, final_source)

    render_current_page(
        page_key=page_key,
        df_raw=df_raw,
        filtered=filtered,
        processing_options=processing_options,
    )


if __name__ == "__main__":
    main()
