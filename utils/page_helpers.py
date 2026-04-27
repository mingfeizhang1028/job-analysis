import streamlit as st
import pandas as pd


def ensure_page_data(df: pd.DataFrame, empty_msg: str = "当前无可展示数据") -> bool:
    if df is None or df.empty:
        st.info(empty_msg)
        return False
    return True


def get_deduped_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if "duplicate_keep" in df.columns:
        return df[df["duplicate_keep"] == True].copy()

    if "is_duplicate" in df.columns:
        return df[df["is_duplicate"] != True].copy()

    return df.copy()
