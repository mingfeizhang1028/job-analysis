import streamlit as st
import pandas as pd
import plotly.express as px

from app_core import get_keyword_stats_by_mode_v2


def render_keyword_top_chart_v2(
    df: pd.DataFrame,
    tag_type: str,
    title: str,
    source_mode: str = "词典标签",
):
    stat = get_keyword_stats_by_mode_v2(df, tag_type, source_mode=source_mode).head(10)

    if stat is None or stat.empty:
        st.info(f"{tag_type}暂无数据")
        return

    fig = px.bar(stat, x="标签", y="覆盖岗位数", title=title)
    st.plotly_chart(fig, use_container_width=True)
