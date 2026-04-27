import streamlit as st

from app_core import render_dedup_page as render_dedup_page_impl


def render_dedup_page(df):
    st.header("岗位去重检查")
    render_dedup_page_impl(df)
