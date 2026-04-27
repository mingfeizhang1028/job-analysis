from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.normalization import apply_normalization
from modules.tag_extraction import apply_tag_extraction
from modules.jd_rule_extraction import apply_rule_jd_extraction
from modules.llm_skill_extraction import apply_llm_skill_extraction
from modules.llm_tag_refinement import apply_llm_tag_refinement
from modules.llm_jd_structuring import apply_llm_jd_structuring
from modules.tag_merge import merge_rule_and_llm_tags
from modules.deduplication import deduplicate_similar_jobs, duplicate_summary
from modules.keyword_analysis import get_keyword_stats_by_mode, get_jobs_by_keyword
from modules.tag_source_resolver import resolve_tag_column
from modules.network_analysis import (
    detect_communities,
    get_all_edges,
    get_jobs_by_edge_pair,
    get_jobs_by_node_label,
    get_network_by_dimension_v2,
    get_network_summary,
    get_top_edges,
    get_top_nodes,
)
from modules.network_viz import (
    build_highlighted_graph_by_edge,
    build_highlighted_graph_by_node,
    build_subgraph_for_edge,
    build_subgraph_for_node,
    render_pyvis_network,
)

DEFAULT_DATA_PATH = "data/jobs.xlsx"
TAG_TYPE_OPTIONS = ["硬技能", "软素质", "业务职责", "行业场景", "全部"]
STAT_METRIC_OPTIONS = ["覆盖率", "覆盖岗位数", "词频"]


def load_data(file) -> pd.DataFrame:
    try:
        if file is not None:
            return pd.read_excel(file)
        return pd.read_excel(DEFAULT_DATA_PATH)
    except Exception as e:
        st.error(f"数据读取失败：{e}")
        return pd.DataFrame()


def ensure_dedup_default_fields(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "duplicate_keep" not in result.columns:
        result["duplicate_keep"] = True
    if "is_duplicate" not in result.columns:
        result["is_duplicate"] = False
    if "duplicate_group_id" not in result.columns:
        result["duplicate_group_id"] = None
    return result


def process_data(
    df: pd.DataFrame,
    enable_norm: bool,
    enable_tags: bool,
    enable_dedup: bool,
    dedup_threshold: float,
    enable_llm_skills: bool,
    enable_llm_jd_struct: bool,
    llm_jd_limit: int,
    llm_jd_overwrite: bool,
    llm_model: str,
    ollama_url: str,
    llm_limit: int,
    llm_overwrite: bool,
    enable_llm_tag_refinement: bool = False,
    llm_tag_refinement_limit: int = 20,
    llm_tag_refinement_overwrite: bool = False,
    skill_extraction_llm: dict | None = None,
    jd_structuring_llm: dict | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()
    if "job_id" not in result.columns:
        result["job_id"] = [f"job_{i}" for i in range(len(result))]

    if enable_norm:
        try:
            result = apply_normalization(result)
        except Exception as e:
            st.warning(f"标准化执行失败，已跳过：{e}")

    if enable_tags:
        try:
            result = apply_tag_extraction(result)
        except Exception as e:
            st.warning(f"标签抽取失败，已跳过：{e}")

    try:
        result = apply_rule_jd_extraction(result, detail_col="岗位详情")
    except Exception as e:
        st.warning(f"JD规则结构化失败，已跳过：{e}")

    skill_extraction_llm = skill_extraction_llm or {}
    jd_structuring_llm = jd_structuring_llm or {}

    if enable_llm_skills:
        try:
            result = apply_llm_skill_extraction(
                result,
                detail_col="岗位详情",
                model=str(skill_extraction_llm.get("local_model") or llm_model),
                ollama_url=str(skill_extraction_llm.get("local_url") or ollama_url),
                limit=None if llm_limit == 0 else int(llm_limit),
                overwrite=llm_overwrite,
            )
        except Exception as e:
            st.warning(f"LLM 技能识别失败，已跳过：{e}")

    if enable_llm_jd_struct:
        try:
            result = apply_llm_jd_structuring(
                result,
                detail_col="岗位详情",
                model=str(jd_structuring_llm.get("local_model") or llm_model),
                ollama_url=str(jd_structuring_llm.get("local_url") or ollama_url),
                limit=None if llm_jd_limit == 0 else int(llm_jd_limit),
                overwrite=llm_jd_overwrite,
            )
        except Exception:
            pass

    if enable_llm_tag_refinement:
        try:
            result = apply_llm_tag_refinement(
                result,
                detail_col="岗位详情",
                model=str(skill_extraction_llm.get("local_model") or llm_model),
                ollama_url=str(skill_extraction_llm.get("local_url") or ollama_url),
                limit=None if llm_tag_refinement_limit == 0 else int(llm_tag_refinement_limit),
                overwrite=llm_tag_refinement_overwrite,
                remote_enabled=bool(skill_extraction_llm.get("remote_enabled", False)),
                remote_model=str(skill_extraction_llm.get("remote_model") or ""),
                remote_base_url=str(skill_extraction_llm.get("remote_base_url") or ""),
                remote_api_key=str(skill_extraction_llm.get("remote_api_key") or ""),
            )
        except Exception as e:
            st.warning(f"LLM 深度标签增强失败，已跳过：{e}")

    try:
        result = merge_rule_and_llm_tags(result)
    except Exception as e:
        st.warning(f"最终标签合并失败，已跳过：{e}")

    if enable_dedup:
        try:
            result = deduplicate_similar_jobs(result, threshold=dedup_threshold)
        except Exception as e:
            st.warning(f"去重失败，已回退为不去重状态：{e}")
            result = ensure_dedup_default_fields(result)
    else:
        result = ensure_dedup_default_fields(result)

    return result


def render_sidebar_analysis_options() -> dict:
    st.sidebar.header("分析模式")
    tag_type = st.sidebar.selectbox("关键词/标签类型", TAG_TYPE_OPTIONS)
    stat_metric = st.sidebar.selectbox("统计口径", STAT_METRIC_OPTIONS, index=0)
    return {"tag_type": tag_type, "stat_metric": stat_metric}


def render_sidebar_processing_options() -> dict:
    st.sidebar.header("数据处理选项")
    enable_norm = st.sidebar.checkbox("启用公司/职位标准化", value=True)
    enable_tags = st.sidebar.checkbox("启用标签抽取", value=True)
    enable_dedup = st.sidebar.checkbox("启用相似 JD 去重", value=True)
    dedup_threshold = st.sidebar.slider("去重相似度阈值", 0.80, 0.99, 0.90, 0.01)

    return {
        "enable_norm": enable_norm,
        "enable_tags": enable_tags,
        "enable_dedup": enable_dedup,
        "dedup_threshold": dedup_threshold,
    }


def render_sidebar_data_source():
    st.sidebar.header("数据源")
    return st.sidebar.file_uploader("上传 Excel 文件", type=["xlsx", "xls"])


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("全局筛选")
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    if "所在地区" in filtered.columns:
        city_options = sorted(filtered["所在地区"].dropna().astype(str).unique().tolist())
        selected_cities = st.sidebar.multiselect("地区", city_options)
        if selected_cities:
            filtered = filtered[filtered["所在地区"].astype(str).isin(selected_cities)]

    if "企业名称_norm" in filtered.columns:
        company_options = sorted(filtered["企业名称_norm"].dropna().astype(str).unique().tolist())
        selected_companies = st.sidebar.multiselect("标准化公司", company_options)
        if selected_companies:
            filtered = filtered[filtered["企业名称_norm"].astype(str).isin(selected_companies)]

    if "职位类别" in filtered.columns:
        cat_options = sorted(filtered["职位类别"].dropna().astype(str).unique().tolist())
        selected_cats = st.sidebar.multiselect("职位类别", cat_options)
        if selected_cats:
            filtered = filtered[filtered["职位类别"].astype(str).isin(selected_cats)]

    if "职位名称_norm" in filtered.columns:
        title_options = sorted(filtered["职位名称_norm"].dropna().astype(str).unique().tolist())
        selected_titles = st.sidebar.multiselect("标准化职位", title_options)
        if selected_titles:
            filtered = filtered[filtered["职位名称_norm"].astype(str).isin(selected_titles)]

    return filtered


def metric_cards(summary: dict):
    if not summary:
        st.info("暂无指标数据")
        return
    cols = st.columns(len(summary))
    for col, (k, v) in zip(cols, summary.items()):
        col.metric(k, v)


def safe_show_dataframe(df: pd.DataFrame, cols=None, message: str = "暂无数据", hide_index: bool = False):
    if df is None or df.empty:
        st.info(message)
        return
    show_df = df.copy()
    if cols is not None:
        show_cols = [c for c in cols if c in show_df.columns]
        if not show_cols:
            st.info("暂无可展示字段")
            return
        show_df = show_df[show_cols]
    st.dataframe(show_df, use_container_width=True, hide_index=hide_index)


def build_job_external_link(row: pd.Series) -> str:
    for col in ["详情链接", "岗位链接", "职位链接", "url", "URL", "link", "Link"]:
        if col in row.index:
            value = row.get(col, "")
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    return ""


def render_job_preview_list(df: pd.DataFrame, limit: int = 20, show_keep_flag: bool = False):
    if df is None or df.empty:
        st.info("暂无岗位详情")
        return
    for _, row in df.head(limit).iterrows():
        title = row.get("职位名称_norm", row.get("职位名称_raw", ""))
        company = row.get("企业名称_norm", row.get("企业名称_raw", ""))
        prefix = ""
        if show_keep_flag:
            prefix = "保留｜" if row.get("duplicate_keep", False) else "重复删除｜"
        st.markdown(f"### {prefix}{title}｜{company}")
        link = build_job_external_link(row)
        if link:
            st.markdown(f"[查看原始岗位网页]({link})")
        st.write(row.get("岗位详情", ""))
        st.markdown("---")


def format_ratio_column(df: pd.DataFrame, col: str = "覆盖率") -> pd.DataFrame:
    result = df.copy()
    if col in result.columns:
        result[col] = result[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return result


def render_overview_basic_distribution(dedup_df: pd.DataFrame):
    col1, col2 = st.columns(2)
    with col1:
        if "所在地区" in dedup_df.columns:
            city_stat = dedup_df["所在地区"].fillna("未知").value_counts().head(15).reset_index()
            city_stat.columns = ["地区", "岗位数"]
            fig = px.bar(city_stat, x="地区", y="岗位数", title="城市岗位分布 Top15")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("缺少字段：所在地区")
    with col2:
        if "企业名称_norm" in dedup_df.columns:
            company_stat = dedup_df["企业名称_norm"].fillna("未知").value_counts().head(15).reset_index()
            company_stat.columns = ["公司", "岗位数"]
            fig = px.bar(company_stat, x="公司", y="岗位数", title="标准化公司招聘 Top15")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("缺少字段：企业名称_norm")


def render_overview_job_structure(dedup_df: pd.DataFrame):
    col1, col2 = st.columns(2)
    with col1:
        if "职位类别" in dedup_df.columns:
            cat_stat = dedup_df["职位类别"].fillna("未知").value_counts().reset_index()
            cat_stat.columns = ["职位类别", "岗位数"]
            fig = px.pie(cat_stat, names="职位类别", values="岗位数", title="职位类别分布")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("缺少字段：职位类别")
    with col2:
        if "职位名称_norm" in dedup_df.columns:
            title_stat = dedup_df["职位名称_norm"].fillna("未知").value_counts().head(20).reset_index()
            title_stat.columns = ["标准化职位", "岗位数"]
            fig = px.bar(title_stat, x="标准化职位", y="岗位数", title="标准化职位 Top20")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("缺少字段：职位名称_norm")


def render_company_mapping_tab(dedup_df: pd.DataFrame):
    if {"企业名称_norm", "企业名称_raw"}.issubset(dedup_df.columns):
        mapping = dedup_df.groupby("企业名称_norm").agg(
            原始公司名数量=("企业名称_raw", "nunique"),
            岗位数=("job_id", "nunique"),
            原始名称示例=("企业名称_raw", lambda x: "、".join(list(pd.Series(x).dropna().astype(str).unique())[:5])),
        ).reset_index().sort_values("岗位数", ascending=False)
        link_map = dedup_df.groupby("企业名称_norm").apply(
            lambda g: next((build_job_external_link(row) for _, row in g.iterrows() if build_job_external_link(row)), "")
        ).to_dict()
        mapping["公司页面"] = mapping["企业名称_norm"].map(link_map)
        safe_show_dataframe(mapping)
    else:
        st.info("缺少公司标准化字段，请启用标准化后查看。")


def render_job_mapping_tab(dedup_df: pd.DataFrame):
    if {"职位名称_norm", "职位名称_raw"}.issubset(dedup_df.columns):
        mapping = dedup_df.groupby("职位名称_norm").agg(
            原始职位名数量=("职位名称_raw", "nunique"),
            岗位数=("job_id", "nunique"),
            原始名称示例=("职位名称_raw", lambda x: "、".join(list(pd.Series(x).dropna().astype(str).unique())[:5])),
        ).reset_index().sort_values("岗位数", ascending=False)
        link_map = dedup_df.groupby("职位名称_norm").apply(
            lambda g: next((build_job_external_link(row) for _, row in g.iterrows() if build_job_external_link(row)), "")
        ).to_dict()
        mapping["岗位页面"] = mapping["职位名称_norm"].map(link_map)
        safe_show_dataframe(mapping)
    else:
        st.info("缺少职位标准化字段，请启用标准化后查看。")


def render_job_category_tab(dedup_df: pd.DataFrame):
    if "职位类别" not in dedup_df.columns:
        st.info("缺少字段：职位类别")
        return
    stat = dedup_df["职位类别"].fillna("未知").value_counts().reset_index()
    stat.columns = ["职位类别", "岗位数"]
    stat["占比"] = stat["岗位数"] / stat["岗位数"].sum()
    safe_show_dataframe(stat)
    fig = px.bar(stat, x="职位类别", y="岗位数", title="职位类别岗位数")
    st.plotly_chart(fig, use_container_width=True)


def normalize_tag_value(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in re.split(r"[，,；;、\n]+", value) if str(x).strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if hasattr(value, "tolist"):
        try:
            return [str(x).strip() for x in value.tolist() if str(x).strip()]
        except Exception:
            return []
    return []


def build_keyword_stats_from_list_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return pd.DataFrame(columns=["标签", "词频", "覆盖岗位数", "覆盖率"])
    total_jobs = df["job_id"].nunique() if "job_id" in df.columns else len(df)
    rows = []
    for _, row in df.iterrows():
        job_id = row.get("job_id", None)
        tags = normalize_tag_value(row.get(col, []))
        for tag in tags:
            rows.append({"job_id": job_id, "标签": tag})
    if not rows:
        return pd.DataFrame(columns=["标签", "词频", "覆盖岗位数", "覆盖率"])
    temp = pd.DataFrame(rows)
    stats = temp.groupby("标签").agg(词频=("标签", "count"), 覆盖岗位数=("job_id", "nunique")).reset_index()
    stats["覆盖率"] = stats["覆盖岗位数"] / max(total_jobs, 1)
    return stats.sort_values(["覆盖岗位数", "词频", "标签"], ascending=[False, False, True]).reset_index(drop=True)


def get_keyword_stats_by_mode_v2(df: pd.DataFrame, tag_type: str, source_mode: str = "最终标签") -> pd.DataFrame:
    target_col = resolve_tag_column(tag_type, source_mode)
    if target_col:
        return build_keyword_stats_from_list_column(df, target_col)
    if source_mode == "词典标签":
        return get_keyword_stats_by_mode(df, tag_type)
    return pd.DataFrame(columns=["标签", "词频", "覆盖岗位数", "覆盖率"])


def render_llm_skill_preview(dedup_df: pd.DataFrame):
    return


def render_keyword_stats_table(stats: pd.DataFrame, tag_type: str):
    stats_display = format_ratio_column(stats, "覆盖率")
    st.subheader(f"{tag_type}关键词统计")
    safe_show_dataframe(stats_display)


def render_keyword_chart(stats: pd.DataFrame, tag_type: str, stat_metric: str):
    metric_col = {"词频": "词频", "覆盖岗位数": "覆盖岗位数", "覆盖率": "覆盖率"}.get(stat_metric, "覆盖岗位数")
    plot_data = stats.head(30).copy()
    y_col = metric_col
    if metric_col == "覆盖率":
        plot_data["覆盖率数值"] = plot_data["覆盖率"]
        y_col = "覆盖率数值"
    if y_col not in plot_data.columns:
        st.info(f"缺少统计字段：{y_col}")
        return
    fig = px.bar(plot_data, x="标签", y=y_col, title=f"{tag_type} Top30 - {stat_metric}")
    st.plotly_chart(fig, use_container_width=True)


def render_keyword_job_lookup(dedup_df: pd.DataFrame, tag_type: str, stats: pd.DataFrame, source_mode: str = "最终标签"):
    st.subheader("关键词反查岗位")
    if "标签" not in stats.columns:
        st.info("缺少字段：标签")
        return
    target_col = resolve_tag_column(tag_type, source_mode)
    if not target_col:
        st.info("当前标签来源不支持该类型的岗位反查")
        return
    if target_col not in dedup_df.columns:
        st.info(f"缺少字段：{target_col}")
        return
    keyword_options = stats["标签"].tolist()
    if not keyword_options:
        st.info("暂无可反查关键词")
        return
    selected_keyword = st.selectbox("选择关键词", keyword_options)
    matched = dedup_df[dedup_df[target_col].apply(lambda x: selected_keyword in normalize_tag_value(x))].copy()
    if "岗位详情" in matched.columns:
        from modules.tag_extraction import keyword_context
        matched["命中片段"] = matched["岗位详情"].apply(lambda x: keyword_context(x, selected_keyword, window=50))
    else:
        matched["命中片段"] = ""
    st.write(f"包含关键词「{selected_keyword}」的岗位数：{len(matched)}")
    show_cols = ["职位名称_norm", "企业名称_norm", "所在地区", "经验要求", "学历要求", "命中片段"]
    safe_show_dataframe(matched, cols=show_cols, message="暂无命中岗位")
    with st.expander("查看岗位详情"):
        render_job_preview_list(matched, limit=20)


def render_dedup_page(df, show_summary: bool = True):
    if df is None or df.empty:
        st.info("当前无可展示数据")
        return
    if show_summary:
        summary = duplicate_summary(df)
        metric_cards(summary)
    if "duplicate_group_id" not in df.columns:
        st.info("尚未启用去重")
        return
    dup_df = df[df["duplicate_group_id"].notna()].copy()
    if dup_df.empty:
        st.success("未发现疑似重复岗位。")
        return
    group_stat = build_duplicate_group_stat(dup_df)
    st.subheader("重复组列表")
    safe_show_dataframe(group_stat)
    if "duplicate_group_id" not in group_stat.columns or group_stat.empty:
        st.info("暂无重复组")
        return
    selected_group = st.selectbox("选择重复组", group_stat["duplicate_group_id"].tolist())
    detail = dup_df[dup_df["duplicate_group_id"] == selected_group]
    render_dedup_group_detail(selected_group, detail)


def build_duplicate_group_stat(dup_df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {"组内岗位数": ("job_id", "nunique")}
    if "企业名称_norm" in dup_df.columns:
        agg_dict["企业名称"] = ("企业名称_norm", "first")
    if "职位名称_norm" in dup_df.columns:
        agg_dict["职位名称"] = ("职位名称_norm", "first")
    if "max_similarity_in_group" in dup_df.columns:
        agg_dict["最高相似度"] = ("max_similarity_in_group", "max")
    group_stat = dup_df.groupby("duplicate_group_id").agg(**agg_dict).reset_index()
    if "最高相似度" in group_stat.columns:
        group_stat = group_stat.sort_values("最高相似度", ascending=False)
    return group_stat


def render_dedup_group_detail(selected_group, detail: pd.DataFrame):
    st.subheader(f"重复组详情：{selected_group}")
    show_cols = ["job_id", "职位名称_norm", "企业名称_norm", "所在地区", "duplicate_keep", "is_duplicate", "max_similarity_in_group", "duplicate_reason"]
    safe_show_dataframe(detail, cols=show_cols)
    with st.expander("查看组内岗位详情"):
        render_job_preview_list(detail, limit=50, show_keep_flag=True)


def render_network_page(df):
    if df is None or df.empty:
        st.info("当前无可展示数据")
        return
    st.caption("建议优先使用标签共现网络查看能力组合关系。")
    dedup_df = df[df["duplicate_keep"] == True].copy() if "duplicate_keep" in df.columns else df.copy()
    if dedup_df.empty:
        st.info("当前去重后无可展示数据")
        return
    params = render_network_controls()
    G = get_network_by_dimension_v2(
        dedup_df,
        dimension=params["dimension"],
        network_type=params["network_type"],
        top_n=params["top_n"],
        min_edge_weight=params["min_edge_weight"],
        similarity_threshold=params["similarity_threshold"],
    )
    summary = get_network_summary(G)
    render_network_metrics(summary)
    if G.number_of_nodes() == 0:
        st.warning("当前筛选条件下没有生成网络。")
        return
    top_nodes = get_top_nodes(G, top_n=10)
    top_edges = get_top_edges(G, top_n=10)
    all_edges = get_all_edges(G)
    linkage_state = render_network_linkage_controls(top_nodes, top_edges)
    G_view = build_network_view_graph(G, linkage_state)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("关系网络图")
        render_pyvis_network(G_view, height="760px", physics_enabled=False, random_seed=42)
    with right:
        render_network_focus_panel(G, dedup_df, params, linkage_state, top_nodes, top_edges)
    render_network_communities(G)
    render_network_edges_table(all_edges)


def render_network_controls() -> dict:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dimension = st.selectbox("关系维度", ["技术关系", "素质关系", "业务职责关系", "行业场景关系", "综合标签关系"], index=0)
    with c2:
        network_type = st.selectbox("网络类型", ["标签共现网络", "岗位相似网络", "公司能力画像网络"], index=0)
    with c3:
        top_n = st.slider("节点数量 TopN", min_value=10, max_value=150, value=50, step=10)
    with c4:
        if network_type == "岗位相似网络":
            similarity_threshold = st.slider("岗位相似度阈值", min_value=0.05, max_value=1.00, value=0.25, step=0.05)
            min_edge_weight = 1
        else:
            min_edge_weight = st.slider("最小边权重/共现次数", min_value=1, max_value=20, value=2, step=1)
            similarity_threshold = 0.25
    return {
        "dimension": dimension,
        "network_type": network_type,
        "top_n": top_n,
        "min_edge_weight": min_edge_weight,
        "similarity_threshold": similarity_threshold,
    }


def render_network_metrics(summary: dict):
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("节点数", summary.get("节点数", 0))
    m2.metric("边数", summary.get("边数", 0))
    m3.metric("平均度", summary.get("平均度", 0))
    m4.metric("连通分量数", summary.get("连通分量数", 0))
    m5.metric("网络密度", summary.get("网络密度", 0))


def render_network_linkage_controls(top_nodes: pd.DataFrame, top_edges: pd.DataFrame) -> dict:
    st.subheader("图谱联动分析")
    l1, l2, l3 = st.columns([1, 2, 1])
    selected_node_id = None
    selected_edge = None
    with l1:
        highlight_mode = st.radio("高亮模式", ["无", "核心节点", "强关联组合"], index=0)
    with l2:
        if highlight_mode == "核心节点":
            if top_nodes is not None and not top_nodes.empty:
                node_options = top_nodes.apply(lambda r: f"{r['节点']} ｜ 加权度={r['加权度']} ｜ 连接数={r['连接数']}", axis=1).tolist()
                selected_node_text = st.selectbox("选择核心节点", node_options)
                selected_row = top_nodes.iloc[node_options.index(selected_node_text)]
                selected_node_id = selected_row["节点ID"]
            else:
                st.info("暂无核心节点可选")
        elif highlight_mode == "强关联组合":
            if top_edges is not None and not top_edges.empty:
                edge_options = top_edges.apply(lambda r: f"{r['节点A']}  <->  {r['节点B']} ｜ 权重={r['权重']}", axis=1).tolist()
                selected_edge_text = st.selectbox("选择强关联组合", edge_options)
                selected_row = top_edges.iloc[edge_options.index(selected_edge_text)]
                selected_edge = (selected_row["source_id"], selected_row["target_id"])
            else:
                st.info("暂无强关联组合可选")
    with l3:
        only_focus_subgraph = st.checkbox("仅显示相关子图", value=False)
    return {
        "highlight_mode": highlight_mode,
        "selected_node_id": selected_node_id,
        "selected_edge": selected_edge,
        "only_focus_subgraph": only_focus_subgraph,
    }


def build_network_view_graph(G, linkage_state: dict):
    G_view = G
    if linkage_state["highlight_mode"] == "核心节点" and linkage_state["selected_node_id"] is not None:
        if linkage_state["only_focus_subgraph"]:
            G_view = build_subgraph_for_node(G, linkage_state["selected_node_id"])
        G_view = build_highlighted_graph_by_node(G_view, linkage_state["selected_node_id"])
    elif linkage_state["highlight_mode"] == "强关联组合" and linkage_state["selected_edge"] is not None:
        source_id, target_id = linkage_state["selected_edge"]
        if linkage_state["only_focus_subgraph"]:
            G_view = build_subgraph_for_edge(G, source_id, target_id)
        G_view = build_highlighted_graph_by_edge(G_view, source_id, target_id)
    return G_view


def render_network_focus_panel(G, dedup_df: pd.DataFrame, params: dict, linkage_state: dict, top_nodes: pd.DataFrame, top_edges: pd.DataFrame):
    st.subheader("当前聚焦详情")
    mode = linkage_state["highlight_mode"]
    if mode == "无":
        show_nodes = format_ratio_column(top_nodes.copy() if top_nodes is not None else pd.DataFrame(), "覆盖率")
        safe_show_dataframe(show_nodes, cols=["节点", "加权度", "连接数", "出现次数", "覆盖岗位数", "覆盖率"], hide_index=True)
        safe_show_dataframe(top_edges, cols=["节点A", "节点B", "权重", "关系"], hide_index=True)
    elif mode == "核心节点":
        render_network_node_focus_panel(G, dedup_df, params["dimension"], params["network_type"], linkage_state["selected_node_id"])
    elif mode == "强关联组合":
        render_network_edge_focus_panel(G, dedup_df, params["dimension"], params["network_type"], linkage_state["selected_edge"])


def render_network_node_focus_panel(G, dedup_df: pd.DataFrame, dimension: str, network_type: str, selected_node_id):
    if selected_node_id is None or selected_node_id not in G.nodes:
        st.info("请选择一个核心节点")
        return
    attrs = G.nodes[selected_node_id]
    node_label = attrs.get("label", selected_node_id)
    st.markdown(f"### 节点：{node_label}")
    render_network_node_neighbors(G, selected_node_id)
    if network_type == "标签共现网络":
        matched_jobs = get_jobs_by_node_label(dedup_df, dimension, node_label)
        safe_show_dataframe(matched_jobs, cols=["职位名称_norm", "企业名称_norm", "所在地区", "经验要求", "学历要求"], hide_index=True)
        with st.expander("查看岗位详情"):
            render_job_preview_list(matched_jobs, limit=15)


def render_network_node_neighbors(G, selected_node_id):
    neighbors = []
    for neighbor in G.neighbors(selected_node_id):
        edge_data = G.get_edge_data(selected_node_id, neighbor)
        neighbors.append({
            "关联节点": G.nodes[neighbor].get("label", neighbor),
            "关系权重": edge_data.get("weight", 1),
            "关系说明": edge_data.get("relation", ""),
        })
    if neighbors:
        nb_df = pd.DataFrame(neighbors).sort_values("关系权重", ascending=False)
        safe_show_dataframe(nb_df, hide_index=True)


def render_network_edge_focus_panel(G, dedup_df: pd.DataFrame, dimension: str, network_type: str, selected_edge):
    if selected_edge is None:
        st.info("请选择一个强关联组合")
        return
    u, v = selected_edge
    if u not in G.nodes or v not in G.nodes or not G.has_edge(u, v):
        st.info("所选关联组合不在当前图中")
        return
    edge_data = G.get_edge_data(u, v)
    node_a_label = G.nodes[u].get("label", u)
    node_b_label = G.nodes[v].get("label", v)
    st.markdown(f"### 组合：{node_a_label} ↔ {node_b_label}")
    st.info(edge_data.get("title", "暂无说明"))
    if network_type == "标签共现网络":
        matched_jobs = get_jobs_by_edge_pair(dedup_df, dimension, node_a_label, node_b_label).copy()
        matched_jobs["岗位页面"] = matched_jobs.apply(build_job_external_link, axis=1)
        safe_show_dataframe(
            matched_jobs,
            cols=["职位名称_norm", "企业名称_norm", "所在地区", "经验要求", "学历要求", "岗位页面"],
            hide_index=True,
            column_config={"岗位页面": st.column_config.LinkColumn("岗位页面", display_text="打开网页")},
        )


def render_network_communities(G):
    st.subheader("主题簇 / 社区发现")
    communities = detect_communities(G)
    if communities is None or communities.empty:
        st.info("暂无社区结果")
        return
    community_summary = communities.groupby("社区ID").agg(节点数=("节点", "count"), 代表节点=("节点", lambda x: "、".join(list(x)[:10]))).reset_index().sort_values("节点数", ascending=False)
    safe_show_dataframe(community_summary, hide_index=True)


def render_network_edges_table(all_edges: pd.DataFrame):
    st.subheader("关系边明细")
    safe_show_dataframe(all_edges, cols=["节点A", "节点B", "权重", "关系", "说明"], message="暂无边明细", hide_index=True)


def render_detail_page(df):
    if df is None or df.empty:
        st.info("当前无可展示数据")
        return
    dedup_df = df[df["duplicate_keep"] == True].copy() if "duplicate_keep" in df.columns else df.copy()
    if dedup_df.empty:
        st.info("当前去重后无可展示数据")
        return
    search = st.text_input("搜索关键词", "")
    result = apply_detail_search(dedup_df, search)
    st.write(f"当前岗位数：{len(result)}")
    show_cols = [
        "job_id", "职位名称_norm", "职位名称_raw", "职位类别", "职位方向", "企业名称_norm", "企业名称_raw",
        "所在地区", "经验要求", "学历要求", "硬技能标签", "软素质标签",
        "规则岗位工作内容", "规则岗位要求", "规则加分项", "规则所属行业", "规则岗位类型",
        "LLM岗位工作内容", "LLM岗位要求", "LLM加分项", "LLM所属行业", "LLM岗位类型",
        "LLM核心目标", "LLM结构化提取错误", "业务职责标签", "行业场景标签", "is_duplicate", "duplicate_keep",
    ]
    safe_show_dataframe(result, cols=show_cols)
    with st.expander("查看岗位详情样例"):
        render_job_preview_list(result, limit=20)


def apply_detail_search(df: pd.DataFrame, search: str) -> pd.DataFrame:
    result = df.copy()
    if not search:
        return result
    mask = pd.Series(False, index=result.index)
    for col in ["岗位详情", "职位名称_norm", "职位名称_raw", "企业名称_norm", "企业名称_raw"]:
        if col in result.columns:
            mask = mask | result[col].fillna("").astype(str).str.contains(search, case=False, na=False)
    return result[mask]
