import io

import pandas as pd
import streamlit as st

from app_core import (
    format_ratio_column,
    get_keyword_stats_by_mode_v2,
    render_company_mapping_tab,
    render_job_category_tab,
    render_job_mapping_tab,
    render_keyword_chart,
    render_keyword_job_lookup,
    render_keyword_stats_table,
    render_llm_skill_preview,
    safe_show_dataframe,
)
from modules.llm_checkpoint import (
    list_llm_checkpoints,
    load_llm_checkpoint,
    load_llm_checkpoint_meta,
    save_llm_checkpoint,
)
from modules.llm_jd_structuring import apply_llm_jd_structuring
from modules.llm_settings import get_task_llm_config, render_skill_analysis_settings_panel
from modules.llm_skill_extraction import apply_llm_skill_extraction
from modules.llm_tag_refinement import apply_llm_tag_refinement
from modules.tag_merge import merge_rule_and_llm_tags
from modules.tag_source_resolver import get_source_mode_hint, get_supported_tag_sources
from modules.job_evidence_analysis import (
    build_evidence_overview,
    build_fresh_friendly_jobs,
    build_job_direction_opportunity_table,
    build_skill_combination_table,
    build_tag_evidence_examples,
    build_tag_evidence_table,
)
from utils.page_helpers import ensure_page_data, get_deduped_df


DIAG_FIELD_META = [
    ("词典标签", "硬技能", "硬技能标签"),
    ("词典标签", "软素质", "软素质标签"),
    ("词典标签", "业务职责", "业务职责标签"),
    ("词典标签", "行业场景", "行业场景标签"),
    ("规则结构化", "业务职责", "规则岗位工作内容"),
    ("规则结构化", "岗位要求", "规则岗位要求"),
    ("规则结构化", "加分项", "规则加分项"),
    ("规则结构化", "行业场景", "规则所属行业"),
    ("规则结构化", "岗位类型", "规则岗位类型"),
    ("规则结构化", "必须技能", "规则必须技能"),
    ("规则结构化", "加分技能", "规则加分技能"),
    ("规则结构化", "工具栈", "规则工具栈"),
    ("LLM深度增强", "硬技能", "LLM深度硬技能标签"),
    ("LLM深度增强", "软素质", "LLM深度软素质标签"),
    ("LLM深度增强", "业务职责", "LLM深度业务职责标签"),
    ("LLM深度增强", "行业场景", "LLM深度行业场景标签"),
    ("LLM深度增强", "岗位要求", "LLM深度岗位要求"),
    ("LLM深度增强", "加分项", "LLM深度加分项"),
    ("LLM深度增强", "工具栈", "LLM深度工具栈"),
    ("LLM深度增强", "岗位类型", "LLM深度岗位类型"),
    ("LLM标签", "业务职责", "LLM岗位工作内容"),
    ("LLM标签", "岗位要求", "LLM岗位要求"),
    ("LLM标签", "加分项", "LLM加分项"),
    ("LLM标签", "行业场景", "LLM所属行业"),
    ("LLM标签", "岗位类型", "LLM岗位类型"),
    ("LLM标签", "必须技能", "LLM必须技能"),
    ("LLM标签", "加分技能", "LLM加分技能"),
    ("LLM标签", "工具栈", "LLM工具栈"),
    ("最终标签", "硬技能", "最终硬技能标签"),
    ("最终标签", "软素质", "最终软素质标签"),
    ("最终标签", "业务职责", "最终业务职责标签"),
    ("最终标签", "行业场景", "最终行业场景标签"),
    ("最终标签", "全部", "最终全部标签"),
]


def _count_non_empty_list_rows(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0

    def has_items(value) -> bool:
        if isinstance(value, (list, tuple, set)):
            return len(value) > 0
        if hasattr(value, "tolist"):
            try:
                return len(value.tolist()) > 0
            except Exception:
                return False
        return False

    return int(df[col].apply(has_items).sum())


def _same_job_scope(left: pd.DataFrame | None, right: pd.DataFrame) -> bool:
    if not isinstance(left, pd.DataFrame) or left.empty or right.empty:
        return False
    if "job_id" not in left.columns or "job_id" not in right.columns:
        return len(left) == len(right)
    return set(left["job_id"].dropna().astype(str)) == set(right["job_id"].dropna().astype(str))


def _render_llm_job_runner(base_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("LLM 岗位理解与标签增强")
    working_df = st.session_state.get("llm_enriched_df")
    if not _same_job_scope(working_df, base_df):
        working_df = base_df.copy()

    total = max(len(working_df), 1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("JD结构化完成", f"{_count_non_empty_list_rows(working_df, 'LLM岗位工作内容')}/{len(working_df)}")
    col2.metric("最终硬技能标签", f"{_count_non_empty_list_rows(working_df, '最终硬技能标签')}/{len(working_df)}")
    col3.metric("最终职责标签", f"{_count_non_empty_list_rows(working_df, '最终业务职责标签')}/{len(working_df)}")
    col4.metric("最终行业标签", f"{_count_non_empty_list_rows(working_df, '最终行业场景标签')}/{len(working_df)}")
    st.caption(
        f"当前会话数据行数：{len(working_df)}，最终硬技能覆盖率：{_count_non_empty_list_rows(working_df, '最终硬技能标签') / total:.1%}，JD结构完成率：{_count_non_empty_list_rows(working_df, 'LLM岗位工作内容') / total:.1%}"
    )
    st.info("推荐流程：先用 20 条测试；确认结果后继续运行。默认不覆盖已有结果，因此会自动跳过已完成岗位；可先保存复原点，后续恢复后继续。")

    skill_cfg = get_task_llm_config("jd_skill")
    struct_cfg = get_task_llm_config("jd_struct")

    task_col1, task_col2, task_col3 = st.columns([1, 1, 1])
    run_skill = task_col1.checkbox("技能/素质标签", value=True, key="job_profile_run_skill")
    run_struct = task_col2.checkbox("职责/要求/加分项/行业结构化", value=True, key="job_profile_run_struct")
    run_refine = task_col3.checkbox("深度标签增强", value=True, key="job_profile_run_refine")
    refresh_tags = st.checkbox("刷新最终融合标签", value=True, key="job_profile_refresh_tags")
    st.caption("深度标签增强会调用本地/远程大模型，批量补全硬技能、职责、要求、加分项、工具栈和场景；最终融合标签会用于关键词统计与关系网络分析。")

    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns([1, 1, 1, 1])
    limit = cfg_col1.number_input("本次处理条数", min_value=0, value=20, step=10, help="0 表示处理全部")
    overwrite = cfg_col2.checkbox("覆盖已有结果", value=False, key="job_profile_llm_overwrite")
    auto_save = cfg_col3.checkbox("完成后自动保存复原点", value=True, key="job_profile_llm_auto_save")
    checkpoint_name = cfg_col4.text_input("复原点名称", value="llm_enriched", key="job_profile_checkpoint_name")

    with st.expander("当前执行模型", expanded=False):
        st.write(f"技能识别：{skill_cfg.get('provider', 'local')} / {skill_cfg.get('model', '')}")
        st.write(f"JD结构化：{struct_cfg.get('provider', 'local')} / {struct_cfg.get('model', '')}")
        if skill_cfg.get("provider") == "remote" or struct_cfg.get("provider") == "remote":
            st.caption("当前任务已切换为远程模型；如需节约成本，请在模型配置中切回本地模型。")
        else:
            st.caption("当前默认优先使用本地模型；远程兜底逻辑将作为下一步增强。")

    action_col1, action_col2, action_col3 = st.columns([1.2, 1, 1.2])
    run_clicked = action_col1.button("开始/继续 LLM 岗位理解", type="primary", key="job_profile_run_local_llm")
    save_clicked = action_col2.button("保存当前结果为复原点", key="job_profile_save_checkpoint")

    checkpoints = list_llm_checkpoints()
    checkpoint_labels = ["不加载"]
    checkpoint_map = {"不加载": None}
    for p in checkpoints:
        meta = load_llm_checkpoint_meta(p)
        label = f"{p.name} | {meta.get('created_at', '未知时间')} | {meta.get('rows', '?')}行"
        checkpoint_labels.append(label)
        checkpoint_map[label] = p
    selected_checkpoint = action_col3.selectbox("加载复原点", checkpoint_labels, key="job_profile_checkpoint_select")
    load_clicked = st.button("从复原点恢复", key="job_profile_load_checkpoint", disabled=selected_checkpoint == "不加载")

    if run_clicked:
        updated = working_df.copy()
        run_limit = None if int(limit) == 0 else int(limit)

        if run_skill:
            with st.spinner("正在执行 LLM 技能识别..."):
                updated = apply_llm_skill_extraction(
                    updated,
                    model=str(skill_cfg.get("local_model") or skill_cfg.get("model") or "qwen3:8b"),
                    ollama_url=str(skill_cfg.get("local_url") or skill_cfg.get("url") or "http://localhost:11434/api/generate"),
                    limit=run_limit,
                    overwrite=overwrite,
                    remote_enabled=bool(skill_cfg.get("remote_enabled", False)),
                    remote_model=str(skill_cfg.get("remote_model") or ""),
                    remote_base_url=str(skill_cfg.get("remote_base_url") or ""),
                    remote_api_key=str(skill_cfg.get("remote_api_key") or ""),
                )
        if run_struct:
            with st.spinner("正在执行 LLM JD 结构化..."):
                updated = apply_llm_jd_structuring(
                    updated,
                    model=str(struct_cfg.get("local_model") or struct_cfg.get("model") or "qwen3:8b"),
                    ollama_url=str(struct_cfg.get("local_url") or struct_cfg.get("url") or "http://localhost:11434/api/generate"),
                    limit=run_limit,
                    overwrite=overwrite,
                    remote_enabled=bool(struct_cfg.get("remote_enabled", False)),
                    remote_model=str(struct_cfg.get("remote_model") or ""),
                    remote_base_url=str(struct_cfg.get("remote_base_url") or ""),
                    remote_api_key=str(struct_cfg.get("remote_api_key") or ""),
                )
        if run_refine:
            with st.spinner("正在执行 LLM 深度标签增强..."):
                updated = apply_llm_tag_refinement(
                    updated,
                    model=str(skill_cfg.get("local_model") or skill_cfg.get("model") or "qwen3:8b"),
                    ollama_url=str(skill_cfg.get("local_url") or skill_cfg.get("url") or "http://localhost:11434/api/generate"),
                    limit=run_limit,
                    overwrite=overwrite,
                    remote_enabled=bool(skill_cfg.get("remote_enabled", False)),
                    remote_model=str(skill_cfg.get("remote_model") or ""),
                    remote_base_url=str(skill_cfg.get("remote_base_url") or ""),
                    remote_api_key=str(skill_cfg.get("remote_api_key") or ""),
                )
        if refresh_tags:
            updated = merge_rule_and_llm_tags(updated)

        st.session_state["llm_enriched_df"] = updated
        struct_error_count = int((updated.get("LLM结构化提取错误", pd.Series([""] * len(updated))).fillna("") != "").sum()) if len(updated) else 0
        summary_msg = f"LLM 识别已完成：结构化错误 {struct_error_count} 条。"
        if auto_save:
            path = save_llm_checkpoint(
                updated,
                metadata={"source": "job_profile_page", "rows": int(len(updated)), "auto_saved": True},
                name=checkpoint_name,
            )
            st.success(f"{summary_msg} 已自动保存复原点：{path.name}")
        else:
            st.success(summary_msg)
        st.rerun()

    if save_clicked:
        target_df = working_df.copy()
        if refresh_tags and "最终全部标签" not in target_df.columns:
            target_df = merge_rule_and_llm_tags(target_df)
        path = save_llm_checkpoint(
            target_df,
            metadata={"source": "job_profile_page", "rows": int(len(target_df))},
            name=checkpoint_name,
        )
        st.success(f"已保存复原点：{path.name}")

    if load_clicked and selected_checkpoint != "不加载":
        checkpoint_path = checkpoint_map.get(selected_checkpoint)
        if checkpoint_path is not None:
            restored = load_llm_checkpoint(checkpoint_path)
            st.session_state["llm_enriched_df"] = restored
            meta = load_llm_checkpoint_meta(checkpoint_path)
            st.success(f"已恢复复原点：{checkpoint_path.name}（{meta.get('created_at', '未知时间')}）")
            st.rerun()

    export_df = st.session_state.get("llm_enriched_df", working_df).copy()
    export_buffer = io.BytesIO()
    export_df.to_excel(export_buffer, index=False)
    st.download_button(
        "下载增强后的Excel",
        data=export_buffer.getvalue(),
        file_name="llm_enriched_jobs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="job_profile_download_enriched_excel",
    )

    return st.session_state.get("llm_enriched_df", working_df)


def _build_tag_diagnosis_df(dedup_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = max(len(dedup_df), 1)
    for source_type, tag_dim, col in DIAG_FIELD_META:
        if col not in dedup_df.columns:
            continue
        non_empty = _count_non_empty_list_rows(dedup_df, col)
        coverage = non_empty / total
        if coverage < 0.3:
            suggestion = "覆盖率较低，建议检查规则词典或运行 LLM 增强。"
        elif coverage < 0.7:
            suggestion = "覆盖率中等，可继续结合 LLM 增强或检查融合规则。"
        else:
            suggestion = "覆盖率较好，可直接用于后续分析。"
        rows.append(
            {
                "来源类型": source_type,
                "标签维度": tag_dim,
                "字段": col,
                "非空行数": non_empty,
                "总行数": int(len(dedup_df)),
                "覆盖率": coverage,
                "建议": suggestion,
            }
        )
    return pd.DataFrame(rows)


def _render_job_profile_overview(dedup_df: pd.DataFrame):
    st.caption("公司页面 / 岗位页面优先复用招聘原链接；岗位名称支持跳转到岗位明细查询页。")
    render_company_mapping_tab(dedup_df)
    st.markdown("---")
    render_job_mapping_tab(dedup_df)
    st.markdown("---")
    render_job_category_tab(dedup_df)


def _render_tag_analysis_panel(dedup_df: pd.DataFrame):
    with st.container(border=True):
        st.subheader("标签分析设置")
        page_options = render_skill_analysis_settings_panel(in_sidebar=False)
        tag_type = page_options["tag_type"]
        stat_metric = page_options["stat_metric"]
        source_mode = page_options["source_mode"]

    hint = get_source_mode_hint(tag_type, source_mode)
    if hint:
        st.caption(hint)

    if tag_type in ["硬技能", "软素质"]:
        render_llm_skill_preview(dedup_df)

    stats = get_keyword_stats_by_mode_v2(dedup_df, tag_type, source_mode)
    if stats is None or stats.empty:
        st.info("当前来源下暂无关键词统计数据")
        return

    render_keyword_chart(stats, tag_type, stat_metric)
    render_keyword_stats_table(stats, tag_type)
    render_keyword_job_lookup(dedup_df, tag_type, stats, source_mode=source_mode)


def _render_tag_diagnosis_panel(dedup_df: pd.DataFrame):
    st.caption("用于检查词典标签、LLM 标签、最终标签的覆盖情况，并辅助判断下一步是补规则还是跑 LLM。")
    diag_df = _build_tag_diagnosis_df(dedup_df)
    if diag_df.empty:
        st.info("暂无可诊断标签字段")
        return

    metric_cols = st.columns(3)
    final_avg = diag_df[diag_df["来源类型"] == "最终标签"]["覆盖率"].mean()
    dict_avg = diag_df[diag_df["来源类型"] == "词典标签"]["覆盖率"].mean()
    llm_avg = diag_df[diag_df["来源类型"] == "LLM标签"]["覆盖率"].mean()
    metric_cols[0].metric("最终标签平均覆盖率", f"{0 if pd.isna(final_avg) else final_avg:.1%}")
    metric_cols[1].metric("词典标签平均覆盖率", f"{0 if pd.isna(dict_avg) else dict_avg:.1%}")
    metric_cols[2].metric("LLM字段平均覆盖率", f"{0 if pd.isna(llm_avg) else llm_avg:.1%}")

    low_coverage = diag_df[diag_df["覆盖率"] < 0.3]
    if not low_coverage.empty:
        st.warning(f"当前有 {len(low_coverage)} 个字段覆盖率低于 30%，建议优先检查词典规则或运行 LLM 增强。")
    else:
        st.success("当前主要标签字段覆盖率整体正常。")

    safe_show_dataframe(format_ratio_column(diag_df, "覆盖率"), hide_index=True)


def _format_evidence_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    percent_cols = [
        "覆盖率",
        "公司覆盖率",
        "岗位方向覆盖率",
        "必备字段命中率",
        "职责字段命中率",
        "加分字段命中率",
        "标题命中率",
        "证据片段覆盖率",
        "共现覆盖率",
    ]
    for col in percent_cols:
        if col in result.columns:
            result[col] = result[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    return result


def _render_evidence_analysis_panel(dedup_df: pd.DataFrame):
    st.caption("把标签统计升级为可解释的求职依据：同时给出覆盖率、公司覆盖、必备/加分语气、应届友好度和 JD 原文证据。")

    c1, c2, c3 = st.columns([1, 1, 1])
    tag_type = c1.selectbox(
        "分析能力类型",
        ["全部", "硬技能", "软素质", "业务职责", "行业场景"],
        key="job_evidence_tag_type",
    )
    source_options = get_supported_tag_sources()
    source_mode = c2.selectbox(
        "标签来源",
        source_options,
        index=0,
        key="job_evidence_source_mode",
    )
    top_n = c3.slider("展示 TopN", min_value=10, max_value=80, value=30, step=10, key="job_evidence_top_n")

    tag_table = build_tag_evidence_table(dedup_df, tag_type=tag_type, source_mode=source_mode)
    if tag_table.empty:
        st.info("当前标签来源下暂无可分析数据。建议先运行规则标签抽取或 LLM 增强，并优先选择“最终标签”。")
        return

    overview = build_evidence_overview(tag_table, dedup_df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("岗位样本数", overview["岗位样本数"])
    m2.metric("高置信能力", overview["高置信能力数"])
    m3.metric("核心门槛", overview["核心门槛数"])
    m4.metric("重点强化", overview["重点强化数"])
    m5.metric("加分项", overview["加分项数"])

    high_value = tag_table.head(top_n).copy()
    display_cols = [
        "标签", "求职优先级", "置信等级", "重要性得分", "覆盖岗位数", "覆盖率",
        "覆盖公司数", "公司覆盖率", "岗位方向覆盖数", "必备字段命中率",
        "职责字段命中率", "加分字段命中率", "平均应届友好分", "求职建议",
    ]
    st.subheader("高价值能力与求职优先级")
    st.dataframe(
        _format_evidence_percent_columns(high_value[[c for c in display_cols if c in high_value.columns]]),
        use_container_width=True,
        hide_index=True,
    )

    selected_tag = st.selectbox("查看某个能力的 JD 证据", high_value["标签"].tolist(), key="job_evidence_selected_tag")
    examples = build_tag_evidence_examples(
        dedup_df,
        selected_tag,
        tag_type=tag_type,
        source_mode=source_mode,
        limit=8,
    )
    st.subheader(f"「{selected_tag}」为什么值得准备")
    if examples.empty:
        st.info("暂无可展示的 JD 原文证据。")
    else:
        st.dataframe(examples, use_container_width=True, hide_index=True)

    combo_col, role_col = st.columns(2)
    with combo_col:
        st.subheader("高价值能力组合")
        combo_df = build_skill_combination_table(
            dedup_df,
            tag_type=tag_type,
            source_mode=source_mode,
            min_support=2,
            top_n=20,
        )
        if combo_df.empty:
            st.info("暂无稳定共现组合。")
        else:
            st.dataframe(_format_evidence_percent_columns(combo_df), use_container_width=True, hide_index=True)

    with role_col:
        st.subheader("岗位方向投递优先级")
        opportunity_df = build_job_direction_opportunity_table(dedup_df, tag_table=tag_table, top_n=20)
        if opportunity_df.empty:
            st.info("暂无岗位方向数据。")
        else:
            st.dataframe(opportunity_df, use_container_width=True, hide_index=True)

    st.subheader("对应届生更友好的岗位样例")
    fresh_df = build_fresh_friendly_jobs(dedup_df, top_n=30)
    if fresh_df.empty:
        st.info("暂无可判断的岗位样例。")
    else:
        st.dataframe(fresh_df, use_container_width=True, hide_index=True)


def render_job_profile_page(df):
    st.header("岗位与标签分析")

    if not ensure_page_data(df):
        return

    dedup_df = get_deduped_df(df)
    if not ensure_page_data(dedup_df, "当前去重后无可展示数据"):
        return

    working_df = st.session_state.get("llm_enriched_df")
    active_df = working_df.copy() if _same_job_scope(working_df, dedup_df) else dedup_df.copy()

    llm_enabled = st.session_state.get("job_profile_show_llm_tools", False)
    if not llm_enabled:
        st.info("当前页面默认聚焦岗位画像与标签分析，LLM增强区已折叠。若需生成 LLM 技能、职责、行业等字段，请展开下方 LLM 操作区。")
        llm_enabled = st.toggle("展开 LLM 增强操作区", value=False, key="job_profile_show_llm_tools")
    else:
        llm_enabled = st.toggle("展开 LLM 增强操作区", value=True, key="job_profile_show_llm_tools")

    tabs = ["岗位画像", "标签分析", "求职依据", "标签诊断"]
    if llm_enabled:
        tabs.append("LLM增强")
    rendered_tabs = st.tabs(tabs)

    with rendered_tabs[0]:
        _render_job_profile_overview(active_df)

    with rendered_tabs[1]:
        _render_tag_analysis_panel(active_df)

    with rendered_tabs[2]:
        _render_evidence_analysis_panel(active_df)

    with rendered_tabs[3]:
        _render_tag_diagnosis_panel(active_df)

    if llm_enabled and len(rendered_tabs) > 4:
        with rendered_tabs[4]:
            updated_df = _render_llm_job_runner(dedup_df)
            if isinstance(updated_df, pd.DataFrame) and not updated_df.empty:
                active_df = updated_df.copy()
                st.caption("本次增强结果已写入当前会话。切回“标签分析 / 标签诊断”即可查看更新后的统计与覆盖情况。")
    else:
        with st.expander("LLM 字段为空时如何处理", expanded=False):
            st.markdown("""
- `LLM岗位工作内容`、`LLM所属行业`、`LLM岗位要求` 等来自“职责/要求/加分项/行业结构化”任务。
- 如果这些列为空，通常不是字段被废弃，而是当前会话尚未运行 LLM 增强，或运行失败。
- 打开上方“展开 LLM 增强操作区”后即可看到“开始/继续 LLM 岗位理解”按钮。
            """)
