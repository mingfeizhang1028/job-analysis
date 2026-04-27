from __future__ import annotations

import streamlit as st
import pandas as pd

from modules.resume_loader import load_resume_text, supported_resume_types
from modules.llm_resume_structuring import structure_resume_text
from modules.candidate_profile import build_candidate_profile
from modules.candidate_evidence_retrieval import retrieve_evidence_for_job
from modules.jd_query_builder import build_jd_query_from_row
from modules.latex_resume_generator import build_resume_plan
from modules.latex_template_renderer import render_resume_tex
from modules.llm_settings import get_task_llm_config, render_page_task_llm_settings
from app_core import build_job_external_link


def render_latex_resume_page(filtered_jobs: pd.DataFrame, processing_options: dict):
    st.header("LaTeX 定制简历生成")
    st.caption("基于目标 JD 和候选人知识库，生成岗位定制版 LaTeX 简历草稿。")

    if filtered_jobs is None or filtered_jobs.empty:
        st.warning("当前没有可用岗位数据。")
        return

    with st.container(border=True):
        render_page_task_llm_settings(["resume_struct", "evidence_embedding"], title="本页模型配置", include_switches=False)

    resume_text_input = st.text_area("粘贴基础简历文本（用于补充候选人画像）", height=180)
    resume_file = st.file_uploader(
        "或上传基础简历",
        type=supported_resume_types(),
        key="latex_resume_upload",
    )

    top_df = filtered_jobs.head(50).copy()
    options = []
    for i, row in top_df.iterrows():
        title = row.get("职位名称_norm", row.get("职位名称_raw", "未知岗位"))
        company = row.get("企业名称_norm", row.get("企业名称_raw", "未知公司"))
        options.append(f"{i + 1}. {title}｜{company}")

    selected_label = st.selectbox("选择目标岗位", options)
    selected_idx = options.index(selected_label)
    selected_row = top_df.iloc[selected_idx]
    selected_link = build_job_external_link(selected_row)
    if selected_link:
        st.markdown(f"[打开目标岗位原网页]({selected_link})")

    if not st.button("生成 LaTeX 简历", type="primary"):
        st.info("请先选择岗位，再点击按钮生成简历。")
        return

    resume_text = load_resume_text(resume_text_input, resume_file)
    resume_llm = get_task_llm_config("resume_structuring")
    embedding_cfg = get_task_llm_config("candidate_embedding")
    resume_struct, _, _ = structure_resume_text(
        resume_text=resume_text,
        model=str(resume_llm.get("local_model") or "qwen3:8b"),
        ollama_url=str(resume_llm.get("local_url") or "http://localhost:11434/api/generate"),
        enable_remote_fallback=bool(resume_llm.get("model_type") == "remote" and resume_llm.get("remote_enabled", False)),
        remote_base_url=str(resume_llm.get("remote_base_url") or ""),
        remote_api_key=str(resume_llm.get("remote_api_key") or ""),
        remote_model=str(resume_llm.get("remote_model") or "gpt-4o-mini"),
        return_debug=True,
    )
    candidate_profile = build_candidate_profile(resume_struct)
    jd_query = build_jd_query_from_row(selected_row)
    evidence = retrieve_evidence_for_job(
        selected_row,
        top_k=6,
        model=str(embedding_cfg.get("local_model") or processing_options.get("embedding_model", "qwen3-embedding:8b")),
        ollama_url=str(embedding_cfg.get("local_url") or processing_options.get("embedding_url", "http://localhost:11434/api/embeddings")),
    )
    plan = build_resume_plan(candidate_profile, jd_query, evidence)
    tex = render_resume_tex(plan, candidate_profile)

    st.subheader("简历内容规划")
    st.json(plan)

    st.subheader("召回证据片段")
    for item in evidence.get("results", []):
        st.markdown(f"**[{item.get('chunk_type', 'other')}] {item.get('title', '')}**  相似度={item.get('score', 0)}")
        st.write(item.get("chunk_text", "")[:400])

    st.subheader("LaTeX 简历源码")
    st.code(tex, language="latex")

    st.download_button(
        label="下载 .tex 文件",
        data=tex.encode("utf-8"),
        file_name="tailored_resume.tex",
        mime="text/plain",
    )
