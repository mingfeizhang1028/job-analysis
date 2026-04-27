import streamlit as st
import pandas as pd

from app_core import (
    duplicate_summary,
    metric_cards,
    render_overview_basic_distribution,
    render_overview_job_structure,
    render_dedup_page as render_dedup_page_impl,
)
from utils.page_helpers import ensure_page_data, get_deduped_df
from utils.keyword_helpers import render_keyword_top_chart_v2


def render_dashboard_keyword_summary(dedup_df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_keyword_top_chart_v2(dedup_df, "硬技能", "硬技能 Top10", source_mode="词典标签")
    with col2:
        render_keyword_top_chart_v2(dedup_df, "软素质", "软素质 Top10", source_mode="词典标签")
    with col3:
        render_keyword_top_chart_v2(dedup_df, "业务职责", "业务职责 Top10", source_mode="词典标签")
    with col4:
        render_keyword_top_chart_v2(dedup_df, "行业场景", "行业场景 Top10", source_mode="词典标签")


def render_dashboard_page(df_raw, df):
    st.header("总览与去重")

    if not ensure_page_data(df):
        return

    summary = duplicate_summary(df)
    metric_cards(summary)

    dedup_df = get_deduped_df(df)
    if not ensure_page_data(dedup_df, "当前去重后无可展示数据"):
        return

    tab1, tab2 = st.tabs(["市场总览", "岗位去重检查"])

    with tab1:
        st.markdown("### 核心市场概览")
        render_overview_basic_distribution(dedup_df)

        st.markdown("### 岗位结构概览")
        render_overview_job_structure(dedup_df)

        st.markdown("### 能力要求摘要")
        render_dashboard_keyword_summary(dedup_df)

    with tab2:
        st.caption("默认先显示重复组列表；仅在需要时展开查看组内岗位详情。")
        render_dedup_page_impl(df, show_summary=False)
