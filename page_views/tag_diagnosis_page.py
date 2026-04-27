import streamlit as st

from page_views.job_profile_page import _render_tag_diagnosis_panel
from utils.page_helpers import ensure_page_data, get_deduped_df


def render_tag_diagnosis_page(df, tag_type=None, stat_metric=None):
    st.header("能力标签诊断")

    if not ensure_page_data(df):
        return

    dedup_df = get_deduped_df(df)
    if not ensure_page_data(dedup_df, "当前去重后无可展示数据"):
        return

    st.caption("当前诊断页已复用岗位与标签分析页面中的“标签诊断”能力，聚焦覆盖率和诊断建议。")
    _render_tag_diagnosis_panel(dedup_df)
