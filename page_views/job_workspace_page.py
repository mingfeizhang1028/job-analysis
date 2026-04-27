from __future__ import annotations

import pandas as pd
import streamlit as st

from page_views.candidate_kb_page import render_candidate_kb_page
from page_views.resume_match_page import render_resume_match_page
from page_views.resume_rewrite_assistant_page import render_resume_rewrite_assistant_page


WORKSPACE_TABS = [
    {
        "label": "候选人资料库",
        "summary": "上传/解析简历、项目经历、作品集、个人背景资料，形成候选人知识库。",
        "input": "简历、项目、实习、竞赛、科研、作品集、背景材料",
        "output": "结构化候选人资料、语义片段、可检索证据库",
    },
    {
        "label": "岗位匹配分析",
        "summary": "基于岗位数据/JD 和候选人知识库，做匹配度、差距、证据引用、优先级排序。",
        "input": "当前岗位池、目标 JD、候选人画像与知识库材料",
        "output": "匹配分、技能差距、岗位优先级、可追溯分析依据",
    },
    {
        "label": "简历改写助手",
        "summary": "针对选中的岗位/JD，调用候选人知识库和企业/JD 信息，生成定制化简历改写建议。",
        "input": "目标 JD、企业资料、原始简历、候选人知识库证据",
        "output": "匹配建议、经历改写、项目表述、质检结论",
    },
]


def _render_workspace_intro(tab_meta: dict) -> None:
    st.subheader(tab_meta["label"])
    st.caption(tab_meta["summary"])
    input_col, output_col = st.columns(2)
    with input_col:
        st.markdown(f"**输入**：{tab_meta['input']}")
    with output_col:
        st.markdown(f"**输出**：{tab_meta['output']}")
    st.divider()


def render_job_workspace_page(filtered_jobs: pd.DataFrame, processing_options: dict) -> None:
    st.header("求职工作台")
    st.caption("从候选人资料沉淀，到岗位匹配分析，再到面向目标 JD 的简历改写，串联完整求职准备流程。")

    kb_tab, match_tab, rewrite_tab = st.tabs([item["label"] for item in WORKSPACE_TABS])

    with kb_tab:
        _render_workspace_intro(WORKSPACE_TABS[0])
        render_candidate_kb_page(processing_options)

    with match_tab:
        _render_workspace_intro(WORKSPACE_TABS[1])
        render_resume_match_page(filtered_jobs, processing_options)

    with rewrite_tab:
        _render_workspace_intro(WORKSPACE_TABS[2])
        render_resume_rewrite_assistant_page()
