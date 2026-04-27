import streamlit as st

from app_core import render_detail_page as render_detail_page_impl


def render_detail_page(df):
    st.header("岗位明细查询")
    render_detail_page_impl(df)
