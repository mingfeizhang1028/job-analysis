from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.resume_loader import load_resume_text, supported_resume_types
from modules.llm_resume_structuring import structure_resume_text
from modules.candidate_profile import build_candidate_profile
from modules.job_resume_matching import match_jobs_with_candidate
from modules.career_fit_analysis import build_career_strategy_summary
from modules.llm_settings import get_task_llm_config, render_page_task_llm_settings
from app_core import build_job_external_link


def render_career_strategy_page(filtered_jobs: pd.DataFrame, processing_options: dict):
    st.header("求职策略建议")
    st.caption("基于简历和当前岗位池，输出更适合你的岗位方向、关键缺口与投递建议。")

    resume_text_input = st.text_area("粘贴简历文本", height=220, key="career_resume_text")
    resume_file = st.file_uploader(
        "或上传简历文件",
        type=supported_resume_types(),
        key="career_strategy_upload",
        help="当前支持 txt / md / docx / pdf",
    )
    overwrite_cache = st.checkbox("忽略简历缓存，重新解析", value=False, key="career_strategy_overwrite")
    enable_remote = st.checkbox("本地模型失败时启用远程增强", value=False, key="career_strategy_remote")

    if not st.button("生成求职策略", type="primary"):
        st.info("请输入简历后点击“生成求职策略”。")
        return

    resume_text = load_resume_text(resume_text_input, resume_file)
    if not resume_text:
        st.warning("未读取到简历内容，请检查输入。")
        return

    with st.spinner("正在分析简历与岗位市场..."):
        resume_llm = get_task_llm_config("resume_structuring")
        remote_enabled = bool(resume_llm.get("model_type") == "remote" and resume_llm.get("remote_enabled", False)) or (bool(resume_llm.get("remote_enabled", False)) and enable_remote)
        resume_struct, resume_error, resume_debug = structure_resume_text(
            resume_text=resume_text,
            model=str(resume_llm.get("local_model") or "qwen3:8b"),
            ollama_url=str(resume_llm.get("local_url") or "http://localhost:11434/api/generate"),
            overwrite=overwrite_cache,
            enable_remote_fallback=remote_enabled,
            remote_base_url=str(resume_llm.get("remote_base_url") or ""),
            remote_api_key=str(resume_llm.get("remote_api_key") or ""),
            remote_model=str(resume_llm.get("remote_model") or "gpt-5.4"),
            return_debug=True,
        )
        candidate_profile = build_candidate_profile(resume_struct)
        match_df = match_jobs_with_candidate(filtered_jobs, candidate_profile)
        summary = build_career_strategy_summary(match_df, candidate_profile)

    if resume_error:
        with st.expander("简历解析调试信息", expanded=False):
            st.warning(resume_error)
            st.caption(f"最终解析来源：{resume_debug.get('final_source', '-')}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("最适合的岗位方向")
        for item in summary.get("best_fit_roles", []):
            st.write(f"- {item}")

        st.subheader("高频技能缺口")
        if summary.get("top_skill_gaps"):
            for item in summary.get("top_skill_gaps", []):
                st.write(f"- {item}")
        else:
            st.write("- 请先在‘我的求职匹配’中选择具体岗位查看专属缺口。")

    with col2:
        st.subheader("简历优化建议")
        for item in summary.get("resume_optimization_tips", []):
            st.write(f"- {item}")

        st.subheader("投递策略建议")
        for item in summary.get("application_strategy", []):
            st.write(f"- {item}")

    if not match_df.empty:
        st.subheader("推荐投递岗位 Top 10")
        city_col = "所在地区" if "所在地区" in match_df.columns else "工作地点"
        preview_df = match_df.head(10).copy()
        preview_df["岗位链接"] = preview_df.apply(build_job_external_link, axis=1)
        show_cols = [
            c for c in [
                "职位名称_norm", "企业名称_norm", city_col, "匹配总分", "匹配结论", "缺失技能", "匹配说明", "岗位链接"
            ] if c in preview_df.columns
        ]
        st.dataframe(preview_df[show_cols], use_container_width=True, hide_index=True)

        selected_strategy_job = st.selectbox(
            "选择一个推荐岗位查看外部链接",
            preview_df.apply(lambda row: f"{row.get('职位名称_norm', row.get('职位名称_raw', '未知岗位'))}｜{row.get('企业名称_norm', row.get('企业名称_raw', '未知公司'))}｜{row.get('匹配总分', 0)}分", axis=1).tolist(),
            key="career_strategy_job_link",
        )
        selected_idx = preview_df.apply(lambda row: f"{row.get('职位名称_norm', row.get('职位名称_raw', '未知岗位'))}｜{row.get('企业名称_norm', row.get('企业名称_raw', '未知公司'))}｜{row.get('匹配总分', 0)}分", axis=1).tolist().index(selected_strategy_job)
        selected_row = preview_df.iloc[selected_idx]
        selected_link = selected_row.get("岗位链接", "")
        if selected_link:
            st.markdown(f"[打开该岗位原网页]({selected_link})")
�位链接", "")
        if selected_link:
            st.markdown(f"[打开该岗位原网页]({selected_link})")
