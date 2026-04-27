from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.resume_loader import load_resume_text_with_meta, supported_resume_types
from modules.llm_resume_structuring import structure_resume_text
from modules.candidate_profile import build_candidate_profile
from modules.job_resume_matching import match_jobs_with_candidate
from modules.llm_settings import get_task_llm_config, render_page_task_llm_settings
from modules.resume_match_cache import (
    build_jobs_fingerprint,
    load_resume_match_cache,
    make_resume_cache_key,
    save_resume_match_cache,
)
from app_core import build_job_external_link


def _safe_show_candidate_profile(profile: dict):
    st.subheader("候选人画像")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("学历：", profile.get("degree", "") or "-")
        st.write("专业：", profile.get("major", "") or "-")
        st.write("毕业年份：", profile.get("graduation_year", "") or "-")
    with col2:
        st.write("目标岗位：", "、".join(profile.get("target_roles", [])) or "-")
        st.write("目标城市：", "、".join(profile.get("target_cities", [])) or "-")
        st.write("资历级别：", profile.get("seniority_level", "应届") or "应届")
    with col3:
        st.write("硬技能：", "、".join(profile.get("hard_skills", [])) or "-")
        st.write("软技能：", "、".join(profile.get("soft_skills", [])) or "-")
        st.write("工具栈：", "、".join(profile.get("tool_stack", [])) or "-")

    edu_items = profile.get("education", []) or []
    if edu_items:
        st.markdown("### 教育经历")
        for item in edu_items:
            if not isinstance(item, dict):
                continue
            school = item.get("school", "") or "未知学校"
            degree = item.get("degree", "")
            major = item.get("major", "")
            years = "-".join([x for x in [item.get("start_year", ""), item.get("end_year", "")] if x])
            st.write(f"- {school}｜{degree or '-'}｜{major or '-'}｜{years or '-'}")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.markdown("### 工作 / 实习经历")
        internships = profile.get("internships", []) or profile.get("internship_tags", []) or []
        if internships:
            for item in internships[:8]:
                st.write(f"- {item}")
        else:
            st.write("- 暂无")
    with exp_col2:
        st.markdown("### 项目经历")
        projects = profile.get("projects", []) or profile.get("project_tags", []) or []
        if projects:
            for item in projects[:8]:
                st.write(f"- {item}")
        else:
            st.write("- 暂无")

    extra_col1, extra_col2 = st.columns(2)
    with extra_col1:
        st.markdown("### 竞赛经历")
        competitions = profile.get("competitions", []) or []
        if competitions:
            for item in competitions[:6]:
                st.write(f"- {item}")
        else:
            st.write("- 暂无")
    with extra_col2:
        st.markdown("### 科研经历")
        research = profile.get("research", []) or []
        if research:
            for item in research[:6]:
                st.write(f"- {item}")
        else:
            st.write("- 暂无")

    with st.expander("查看完整画像", expanded=False):
        st.json(profile)


def render_resume_match_page(filtered_jobs: pd.DataFrame, processing_options: dict):
    st.header("岗位匹配分析")
    st.caption("基于岗位数据/JD 和候选人资料，计算匹配度、定位差距，并按优先级展示推荐岗位。")

    with st.container(border=True):
        render_page_task_llm_settings(["resume_struct"], title="本页模型配置", include_switches=False)

    resume_text_input = st.text_area(
        "粘贴简历文本",
        height=220,
        placeholder="可直接粘贴简历全文，建议包含教育、项目、实习、技能等信息",
    )
    resume_file = st.file_uploader(
        "或上传简历文件",
        type=supported_resume_types(),
        key="resume_match_upload",
        help="当前支持 txt / md / docx / pdf",
    )

    overwrite_cache = st.checkbox("忽略简历缓存，重新解析", value=False, key="resume_match_overwrite")
    enable_remote = st.checkbox("本地模型失败时启用远程增强", value=False, key="resume_match_remote")

    run_match = st.button("开始岗位匹配分析", type="primary")
    if st.button("清除当前匹配结果", key="resume_match_clear_result"):
        st.session_state.pop("resume_match_result", None)
        st.success("已清除当前会话中的匹配结果。")
        st.rerun()

    cached_result = st.session_state.get("resume_match_result")
    if not run_match and not cached_result:
        st.info("请先输入简历内容，再点击“开始岗位匹配分析”。")
        return

    if run_match:
        resume_text, resume_load_meta = load_resume_text_with_meta(resume_text_input, resume_file)
        if not resume_text:
            st.warning("未读取到简历内容，请检查文本输入或上传文件。")
            with st.expander("简历读取调试信息", expanded=True):
                st.json(resume_load_meta)
            return

        with st.spinner("正在解析简历..."):
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
        with st.spinner("正在计算岗位匹配分..."):
            match_df = match_jobs_with_candidate(filtered_jobs, candidate_profile)

        resume_text_preview = resume_text[:3000]
        st.session_state["resume_match_result"] = {
            "resume_struct": resume_struct,
            "resume_error": resume_error,
            "resume_debug": resume_debug,
            "resume_load_meta": resume_load_meta,
            "resume_text_preview": resume_text_preview,
            "candidate_profile": candidate_profile,
            "match_df": match_df,
        }
    else:
        resume_struct = cached_result.get("resume_struct", {})
        resume_error = cached_result.get("resume_error", "")
        resume_debug = cached_result.get("resume_debug", {})
        resume_load_meta = cached_result.get("resume_load_meta", {})
        resume_text_preview = cached_result.get("resume_text_preview", "")
        candidate_profile = cached_result.get("candidate_profile", {})
        match_df = cached_result.get("match_df", pd.DataFrame())

    with st.expander("简历解析调试信息", expanded=False):
        if resume_error:
            st.warning(resume_error)
        if resume_load_meta:
            st.write("文件读取信息：")
            st.json(resume_load_meta)
        st.caption(f"最终解析来源：{resume_debug.get('final_source', '-')}")
        if resume_debug.get("ollama_error"):
            st.write("Ollama错误：", resume_debug.get("ollama_error"))
        st.write("规则回退结果：")
        st.json(resume_debug.get("fallback_struct", {}))
        if resume_debug.get("ollama_parsed"):
            st.write("LLM解析结果：")
            st.json(resume_debug.get("ollama_parsed", {}))
        if resume_text_preview:
            st.write("简历原文预览：")
            st.text_area("resume_text_preview", resume_text_preview, height=220, disabled=True, label_visibility="collapsed")

    _safe_show_candidate_profile(candidate_profile)

    if match_df.empty:
        st.info("当前没有可展示的匹配结果。")
        return

    st.subheader("岗位优先级排序 Top 20")
    city_col = "所在地区" if "所在地区" in match_df.columns else "工作地点"
    exp_col = "LLM经验要求" if "LLM经验要求" in match_df.columns else "经验要求"
    degree_col = "LLM学历要求" if "LLM学历要求" in match_df.columns else "学历要求"
    preview_df = match_df.copy()
    preview_df["岗位链接"] = preview_df.apply(build_job_external_link, axis=1)
    summary_cols = [
        "职位名称_norm",
        "企业名称_norm",
        city_col,
        exp_col,
        degree_col,
        "匹配总分",
        "匹配结论",
        "应届友好度分",
        "匹配技能",
        "缺失技能",
        "匹配说明",
        "岗位链接",
    ]
    show_cols = [c for c in summary_cols if c in preview_df.columns]
    st.dataframe(preview_df[show_cols].head(20), use_container_width=True, hide_index=True)

    st.subheader("选择目标岗位，查看专属缺口")
    top_detail_df = preview_df.head(30).copy()
    option_labels = []
    for i, row in top_detail_df.iterrows():
        title = row.get("职位名称_norm", row.get("职位名称_raw", "未知岗位"))
        company = row.get("企业名称_norm", row.get("企业名称_raw", "未知公司"))
        option_labels.append(f"{i + 1}. {title}｜{company}｜{row.get('匹配总分', 0)}分")

    if not option_labels:
        st.info("暂无可查看的岗位详情。")
        return

    selected_label = st.selectbox("选择一个岗位查看详情", option_labels)
    selected_idx = option_labels.index(selected_label)
    selected_row = top_detail_df.iloc[selected_idx]

    st.markdown("### 该岗位的技能缺口")
    missing_skills = selected_row.get("缺失技能", [])
    matched_skills = selected_row.get("匹配技能", [])
    col_gap1, col_gap2 = st.columns(2)
    with col_gap1:
        st.write("已匹配技能：", "、".join(matched_skills) or "-")
    with col_gap2:
        st.write("缺失技能：", "、".join(missing_skills) or "-")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 岗位信息")
        st.write("职位：", selected_row.get("职位名称_norm", selected_row.get("职位名称_raw", "")))
        st.write("公司：", selected_row.get("企业名称_norm", selected_row.get("企业名称_raw", "")))
        st.write("地区：", selected_row.get("所在地区", selected_row.get("工作地点", "")))
        st.write("经验要求：", selected_row.get("LLM经验要求", selected_row.get("经验要求", "")))
        st.write("学历要求：", selected_row.get("LLM学历要求", selected_row.get("学历要求", "")))
        st.write("匹配结论：", selected_row.get("匹配结论", ""))
        st.write("匹配总分：", selected_row.get("匹配总分", 0))
        if selected_row.get("岗位链接", ""):
            st.markdown(f"[打开该岗位原网页]({selected_row.get('岗位链接', '')})")

    with col2:
        st.markdown("### 匹配分析")
        st.write("匹配说明：", selected_row.get("匹配说明", ""))
        jd_requirements = selected_row.get("LLM岗位要求", [])
        st.write(
            "JD岗位要求：",
            "、".join(jd_requirements) if isinstance(jd_requirements, list) else str(jd_requirements or ""),
        )

    with st.expander("查看岗位原文", expanded=False):
        st.write(selected_row.get("岗位详情", ""))
