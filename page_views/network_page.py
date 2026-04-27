from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from modules.network_analysis import (
    CROSS_DIMENSION_STOP_TAGS,
    MANUAL_NETWORK_STOP_TAGS,
    analyze_dimension_tag_overlap,
    detect_communities,
    find_best_available_tag_col,
    get_preferred_tag_columns,
    get_tag_column_coverage,
    infer_high_coverage_tags,
    inspect_candidate_tag_columns,
    merge_dimension_tag_columns,
    normalize_tag_columns_inplace,
    get_all_edges,
    get_jobs_by_edge_pair,
    get_jobs_by_node_label,
    get_network_by_dimension_v2,
    get_network_summary,
    get_top_edges,
    get_top_nodes,
)
from modules.llm_network_advisor import get_network_advice
from modules.llm_settings import get_task_llm_config, render_page_task_llm_settings
from modules.network_insight import build_network_insight_payload
from modules.network_viz import (
    build_highlighted_graph_by_edge,
    build_highlighted_graph_by_node,
    build_subgraph_for_edge,
    build_subgraph_for_node,
    render_pyvis_network,
)
from app_core import (
    build_job_external_link,
    format_ratio_column,
    render_job_preview_list,
    safe_show_dataframe,
)


NETWORK_VALUE_GUIDE = {
    "岗位-标签洞察网络": {
        "定位": "核心决策网络",
        "价值": "直接回答不同岗位方向分别需要哪些能力，是当前页面里最适合做求职方向判断的网络。",
        "适合": ["判断主攻岗位方向", "看某类岗位的高频能力要求", "决定简历应突出哪些标签"],
        "不适合": ["观察具体岗位之间的迁移路径", "判断公司个体差异"],
    },
    "标签共现网络": {
        "定位": "能力组合网络",
        "价值": "用于发现哪些技能、职责、素质经常成组出现，适合规划学习路线和整理简历关键词组合。",
        "适合": ["看能力组合", "找核心标签簇", "准备面试中的复合能力案例"],
        "不适合": ["直接判断具体岗位名称优先级", "判断目标公司偏好"],
    },
    "岗位相似网络": {
        "定位": "扩展分析网络",
        "价值": "用于发现名称不同但要求相近的岗位，适合扩展可投岗位池，而不是做第一优先决策。",
        "适合": ["找相邻岗位", "扩展投递范围", "识别可迁移岗位"],
        "不适合": ["直接判断该学什么", "作为主图长期观察"],
    },
    "公司能力画像网络": {
        "定位": "定向研究网络",
        "价值": "用于看公司偏好哪些能力，适合面试前做目标公司研究，对通用求职方向判断价值次于前两类网络。",
        "适合": ["定向投递前研究公司偏好", "比较公司需求差异", "准备面试侧重点"],
        "不适合": ["判断市场主流学习方向", "作为通用能力规划主图"],
    },
}
from utils.page_helpers import get_deduped_df


DEFAULT_THEME = "tech_steady"
DEFAULT_LAYOUT_MODE = "force"
DEFAULT_HIGH_COVERAGE_THRESHOLD = 0.99
DEFAULT_MANUAL_STOP_TAGS = set(MANUAL_NETWORK_STOP_TAGS) | set(CROSS_DIMENSION_STOP_TAGS)
DEFAULT_STOP_TAGS_TEXT = "、".join(sorted(DEFAULT_MANUAL_STOP_TAGS))


def render_network_page(df: pd.DataFrame):
    st.header("岗位关系网络")
    st.caption("通过岗位关系网络、标签共现结构与岗位样本证据，辅助判断更值得投入的求职方向。岗位相似网络会优先复用岗位明细中的所在地区/工作城市和真实业务行业/场景字段（如医疗、电商、直播等）；岗位方向会作为独立补充信息显示，不再混作行业。")

    if df is None or df.empty:
        st.info("当前无可展示数据")
        return

    source_df = st.session_state.get("llm_enriched_df")
    if isinstance(source_df, pd.DataFrame) and not source_df.empty and len(source_df) == len(df):
        working_df = source_df.copy()
        data_source_label = "当前会话增强结果"
    else:
        working_df = df.copy()
        data_source_label = "当前筛选数据"

    dedup_df = get_deduped_df(working_df)
    if dedup_df.empty:
        st.info("当前去重后无可展示数据")
        return

    custom_stop_tags = parse_network_custom_stop_tags()
    effective_manual_stop_tags = set(DEFAULT_MANUAL_STOP_TAGS) | set(custom_stop_tags)

    dedup_df = normalize_tag_columns_inplace(dedup_df, stop_tags=effective_manual_stop_tags)
    tag_source_columns = [
        "最终硬技能标签", "硬技能标签",
        "最终软素质标签", "软素质标签",
        "最终业务职责标签", "业务职责标签",
        "最终行业场景标签", "行业场景标签",
        "最终全部标签", "全部标签",
    ]
    high_coverage_tags_df = infer_high_coverage_tags(
        dedup_df,
        tag_source_columns,
        coverage_threshold=DEFAULT_HIGH_COVERAGE_THRESHOLD,
    )
    auto_stop_tags = set(high_coverage_tags_df["标签"].tolist()) if high_coverage_tags_df is not None and not high_coverage_tags_df.empty else set()
    effective_stop_tags = set(effective_manual_stop_tags) | auto_stop_tags

    dedup_df = merge_dimension_tag_columns(
        dedup_df,
        ["最终硬技能标签", "硬技能标签"],
        "网络_技术标签",
        stop_tags=effective_stop_tags,
    )
    dedup_df = merge_dimension_tag_columns(
        dedup_df,
        ["最终软素质标签", "软素质标签"],
        "网络_素质标签",
        stop_tags=effective_stop_tags,
    )
    dedup_df = merge_dimension_tag_columns(
        dedup_df,
        ["最终业务职责标签", "业务职责标签"],
        "网络_业务职责标签",
        stop_tags=effective_stop_tags,
    )
    dedup_df = merge_dimension_tag_columns(
        dedup_df,
        ["最终行业场景标签", "行业场景标签"],
        "网络_行业场景标签",
        stop_tags=effective_stop_tags,
    )
    dedup_df = merge_dimension_tag_columns(
        dedup_df,
        [
            "最终硬技能标签",
            "硬技能标签",
            "最终软素质标签",
            "软素质标签",
            "最终业务职责标签",
            "业务职责标签",
            "最终行业场景标签",
            "行业场景标签",
            "最终全部标签",
            "全部标签",
        ],
        "网络_综合标签",
        stop_tags=effective_stop_tags,
    )

    st.caption(f"当前网络分析数据源：{data_source_label}｜样本数：{len(dedup_df)}")
    st.markdown("### 图谱配置")
    params = render_network_filters(custom_stop_tags=custom_stop_tags)
    dimension_network_map = {
        "技术关系": "网络_技术标签",
        "素质关系": "网络_素质标签",
        "业务职责关系": "网络_业务职责标签",
        "行业场景关系": "网络_行业场景标签",
        "综合标签关系": "网络_综合标签",
    }
    preferred_network_col = dimension_network_map.get(params["dimension"])
    chosen_tag_col, tag_diag = find_best_available_tag_col(dedup_df, params["dimension"])
    if preferred_network_col and preferred_network_col in dedup_df.columns:
        preferred_cov = get_tag_column_coverage(dedup_df, [preferred_network_col])
        if not preferred_cov.empty and int(preferred_cov.iloc[0]["非空行数"]) > 0:
            chosen_tag_col = preferred_network_col
            tag_diag = {
                **tag_diag,
                "chosen_col": preferred_network_col,
                "preferred_network_col": preferred_network_col,
            }
    params["resolved_tag_col"] = chosen_tag_col
    params["tag_diagnostics"] = tag_diag
    params["candidate_tag_columns"] = inspect_candidate_tag_columns(dedup_df)
    params["high_coverage_tags_df"] = high_coverage_tags_df
    params["effective_manual_stop_tags"] = sorted(effective_manual_stop_tags)
    params["effective_stop_tags"] = sorted(effective_stop_tags)
    params["tag_coverage_df"] = get_tag_column_coverage(
        dedup_df,
        columns=get_preferred_tag_columns() + list(dimension_network_map.values()),
    )
    params["dimension_overlap"] = analyze_dimension_tag_overlap(dedup_df)

    graph = get_network_by_dimension_v2(
        dedup_df,
        dimension=params["dimension"],
        network_type=params["network_type"],
        top_n=params["top_n"],
        min_edge_weight=params["min_edge_weight"],
        similarity_threshold=params["similarity_threshold"],
    )

    summary = get_network_summary(graph)
    insight_payload = build_network_insight_payload(dedup_df, graph, tag_col=chosen_tag_col)

    if graph.number_of_nodes() == 0:
        st.warning("当前筛选条件下没有生成网络。请优先尝试：关系维度切换为“综合标签关系”、最小边权重设为 1、岗位相似度阈值降低到 0.08，或切换到“岗位-标签洞察网络”。")
        return
    if graph.number_of_edges() == 0:
        st.info("当前图只有节点、没有连线。系统会继续展示节点；如需形成网络，请降低阈值或切换到“岗位-标签洞察网络 / 综合标签关系”。")

    top_nodes = get_top_nodes(graph, top_n=10)
    top_edges = get_top_edges(graph, top_n=10)
    all_edges = get_all_edges(graph)
    linkage_state = render_network_linkage_controls(top_nodes, top_edges)
    view_graph = build_network_view_graph(graph, linkage_state, theme=params["theme"])

    render_network_graph_section(view_graph, params)

    render_network_overview(summary, insight_payload, params)
    render_network_insight_panel(graph, dedup_df, params, linkage_state, top_nodes, top_edges, insight_payload)
    render_network_detail_tabs(graph, dedup_df, params, insight_payload)
    render_network_llm_settings_section()
    render_network_blacklist_settings()
    render_network_debug_section(params, insight_payload, all_edges)



def parse_network_custom_stop_tags() -> set[str]:
    custom_stop_tags_text = st.session_state.get("network_custom_stop_tags_text", "")
    return {
        token.strip()
        for token in re.split(r"[\n,，;；、\s]+", custom_stop_tags_text or "")
        if token and token.strip()
    }


def render_network_filters(custom_stop_tags: set[str] | None = None) -> dict:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dimension = st.selectbox("关系维度", ["综合标签关系", "技术关系", "业务职责关系", "行业场景关系", "素质关系"], index=0)
    with c2:
        network_type = st.selectbox("网络类型", ["岗位-标签洞察网络", "标签共现网络", "岗位相似网络", "公司能力画像网络"], index=0)
    with c3:
        top_n = st.slider("节点数量 TopN", min_value=10, max_value=150, value=60, step=10)
    with c4:
        llm_mode = st.selectbox("LLM建议", ["关闭", "快速洞察", "深度策略"], index=0)

    c5, c6, c7 = st.columns(3)
    with c5:
        theme = st.selectbox("图谱主题", ["tech_steady", "fresh_mint", "midnight_blue", "dark_glow", "default"], index=0)
    with c6:
        allowed_layouts = ["force", "layered"] if network_type in ["岗位-标签洞察网络", "公司能力画像网络"] else ["force"]
        layout_mode = st.selectbox("布局模式", allowed_layouts, index=0)
    with c7:
        if network_type == "岗位相似网络":
            similarity_threshold = st.slider("岗位相似度阈值", min_value=0.01, max_value=0.80, value=0.08, step=0.01)
            min_edge_weight = 1
        else:
            default_min_weight = 1 if network_type in ["标签共现网络", "公司能力画像网络", "岗位-标签洞察网络"] else 2
            min_edge_weight = st.slider("最小边权重/共现次数", min_value=1, max_value=20, value=default_min_weight, step=1)
            similarity_threshold = 0.08

    return {
        "dimension": dimension,
        "network_type": network_type,
        "top_n": top_n,
        "min_edge_weight": min_edge_weight,
        "similarity_threshold": similarity_threshold,
        "theme": theme or DEFAULT_THEME,
        "layout_mode": layout_mode or DEFAULT_LAYOUT_MODE,
        "llm_mode": llm_mode,
        "custom_stop_tags": custom_stop_tags or set(),
    }


def render_network_blacklist_settings():
    with st.expander("标签黑名单设置", expanded=False):
        st.caption("用于屏蔽 AI、产品、平台等过泛标签；修改后页面会自动重算。默认已包含：" + DEFAULT_STOP_TAGS_TEXT)
        if "network_custom_stop_tags_text" not in st.session_state:
            st.session_state["network_custom_stop_tags_text"] = ""
        custom_stop_tags_text = st.text_area(
            "追加停用标签",
            help="多个标签可用中文顿号、逗号、分号、空格或换行分隔，例如：AI、产品、平台",
            key="network_custom_stop_tags_text",
            height=90,
        )
        parsed_custom_stop_tags = {
            token.strip()
            for token in re.split(r"[\n,，;；、\s]+", custom_stop_tags_text or "")
            if token and token.strip()
        }
        if parsed_custom_stop_tags:
            st.caption("本次追加停用标签：" + "、".join(sorted(parsed_custom_stop_tags)))
        else:
            st.caption("当前未追加自定义停用标签")


def render_network_llm_settings_section():
    render_page_task_llm_settings(["network_advice"], title="岗位关系网络 LLM 配置", include_switches=False)



def render_network_value_guide(network_type: str):
    info = NETWORK_VALUE_GUIDE.get(network_type)
    if not info:
        return
    with st.expander("查看当前网络的求职价值说明", expanded=False):
        st.markdown(f"**定位：** {info['定位']}")
        st.markdown(f"**作用：** {info['价值']}")
        st.markdown("**适合回答：**")
        for item in info["适合"]:
            st.markdown(f"- {item}")
        st.markdown("**不适合回答：**")
        for item in info["不适合"]:
            st.markdown(f"- {item}")


def render_network_overview(summary: dict, insight_payload: dict, params: dict):
    overview = insight_payload.get("overview", {})
    priority_df = insight_payload.get("priority_table")
    combo_df = insight_payload.get("high_value_combinations")

    top_ability = "-"
    top_level = "-"
    if isinstance(priority_df, pd.DataFrame) and not priority_df.empty:
        top_ability = str(priority_df.iloc[0].get("标签", "-"))
        top_level = str(priority_df.iloc[0].get("建议层级", "-"))

    top_combo = "-"
    if isinstance(combo_df, pd.DataFrame) and not combo_df.empty:
        first_combo = combo_df.iloc[0]
        top_combo = f"{first_combo.get('标签A', '')} + {first_combo.get('标签B', '')}"

    st.subheader("求职决策摘要")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("主流方向", overview.get("main_direction", "-"))
    m2.metric("方向占比", f"{overview.get('main_direction_ratio', 0):.1%}")
    m3.metric("优先能力", top_ability)
    m4.metric("能力层级", top_level)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.info(
            f"建议先围绕「{overview.get('main_direction', '当前主流方向')}」整理简历叙事，"
            f"优先补强「{top_ability}」，并尝试把「{top_combo}」写成同一段项目经历中的方法链路。"
        )
    with c2:
        st.caption(
            f"当前图谱：{params.get('network_type')} / {params.get('dimension')}。"
            f"已生成 {summary.get('节点数', 0)} 个节点、{summary.get('边数', 0)} 条关系。"
        )

    render_network_value_guide(params.get("network_type", ""))



def render_network_graph_section(graph, params: dict):
    layout_mode = params.get("layout_mode", DEFAULT_LAYOUT_MODE)
    st.subheader("关系网络图")
    st.caption("节点越大，代表覆盖岗位越多或连接越强；拖拽节点可查看能力、岗位方向与公司画像之间的真实关联。")
    render_pyvis_network(
        graph,
        height="760px",
        physics_enabled=layout_mode == "force",
        random_seed=42,
        theme=params.get("theme", DEFAULT_THEME),
        layout_mode=layout_mode,
        show_buttons=True,
        keep_physics_after_stabilization=layout_mode == "force",
    )



def render_network_insight_panel(
    graph,
    dedup_df: pd.DataFrame,
    params: dict,
    linkage_state: dict,
    top_nodes: pd.DataFrame,
    top_edges: pd.DataFrame,
    insight_payload: dict,
):
    st.subheader("当前洞察")

    mode = linkage_state["highlight_mode"]
    if mode == "无":
        priority_df = insight_payload.get("priority_table")
        if priority_df is not None and not priority_df.empty:
            st.markdown("**优先投入能力**")
            st.dataframe(priority_df.head(10), use_container_width=True, hide_index=True)

        combo_df = insight_payload.get("high_value_combinations")
        if combo_df is not None and not combo_df.empty:
            st.markdown("**高价值能力组合**")
            st.dataframe(combo_df.head(8), use_container_width=True, hide_index=True)

        community_df = insight_payload.get("community_summary")
        if community_df is not None and not community_df.empty:
            st.markdown("**主题簇概览**")
            st.dataframe(community_df.head(8), use_container_width=True, hide_index=True)

        render_network_llm_advice_block(insight_payload, params.get("llm_mode", "关闭"))

        st.markdown("**核心节点概览**")
        show_nodes = format_ratio_column(top_nodes.copy() if top_nodes is not None else pd.DataFrame(), "覆盖率")
        safe_show_dataframe(show_nodes, cols=["节点", "加权度", "连接数", "出现次数", "覆盖岗位数", "覆盖率"], hide_index=True)

        st.markdown("**强关联组合概览**")
        safe_show_dataframe(top_edges, cols=["节点A", "节点B", "权重", "关系"], hide_index=True)
    elif mode == "核心节点":
        render_network_node_focus_panel(graph, dedup_df, params["dimension"], params["network_type"], linkage_state["selected_node_id"])
    elif mode == "强关联组合":
        render_network_edge_focus_panel(graph, dedup_df, params["dimension"], params["network_type"], linkage_state["selected_edge"])



def _render_list_items(items, empty_text: str = "暂无"):
    if isinstance(items, list) and items:
        for item in items[:8]:
            if isinstance(item, dict):
                title = item.get("direction") or item.get("name") or item.get("title") or "建议"
                reason = item.get("reason") or item.get("priority") or ""
                st.markdown(f"- **{title}**：{reason}")
            else:
                st.markdown(f"- {item}")
    else:
        st.caption(empty_text)


def render_network_llm_advice_block(insight_payload: dict, llm_mode: str):
    st.markdown("**LLM 洞察建议**")
    if llm_mode == "关闭":
        st.caption("当前未启用 LLM。可在顶部将“LLM建议”切换为“快速洞察”或“深度策略”。")
        return

    mode_map = {"快速洞察": "quick", "深度策略": "deep"}
    mode = mode_map.get(llm_mode, "quick")
    cache_key = f"network_advice_{llm_mode}_{hash(str(insight_payload.get('overview', {})) + str(insight_payload.get('meta', {})))}"

    if cache_key not in st.session_state:
        if st.button(f"生成{llm_mode}", key=f"btn_{cache_key}"):
            with st.spinner("正在生成岗位关系网络洞察建议..."):
                llm_config = get_task_llm_config("network_advice")
                st.session_state[cache_key] = get_network_advice(insight_payload, llm_config, mode=mode)
        else:
            st.caption("点击按钮后，将基于当前网络统计结果生成建议。")
            return

    advice = st.session_state.get(cache_key) or {}
    if advice.get("summary"):
        st.info(advice.get("summary"))
    st.caption(f"模型：{advice.get('model_used', '-')}")

    with st.expander("推荐方向", expanded=True):
        _render_list_items(advice.get("recommended_directions"), "暂无推荐方向")
    with st.expander("优先投入能力", expanded=True):
        _render_list_items(advice.get("focus_first") or advice.get("core_capabilities"), "暂无优先能力")
    with st.expander("简历 / 面试提示", expanded=False):
        st.markdown("**简历建议**")
        _render_list_items(advice.get("resume_hints"), "暂无简历建议")
        st.markdown("**面试建议**")
        _render_list_items(advice.get("interview_hints"), "暂无面试建议")
    with st.expander("风险与低价值投入提醒", expanded=False):
        _render_list_items(advice.get("risk_notes"), "暂无风险提醒")
        _render_list_items(advice.get("avoid_low_value_effort"), "暂无低价值投入提醒")


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



def build_network_view_graph(graph, linkage_state: dict, theme: str = DEFAULT_THEME):
    view_graph = graph
    if linkage_state["highlight_mode"] == "核心节点" and linkage_state["selected_node_id"] is not None:
        if linkage_state["only_focus_subgraph"]:
            view_graph = build_subgraph_for_node(graph, linkage_state["selected_node_id"])
        view_graph = build_highlighted_graph_by_node(view_graph, linkage_state["selected_node_id"], theme=theme)
    elif linkage_state["highlight_mode"] == "强关联组合" and linkage_state["selected_edge"] is not None:
        source_id, target_id = linkage_state["selected_edge"]
        if linkage_state["only_focus_subgraph"]:
            view_graph = build_subgraph_for_edge(graph, source_id, target_id)
        view_graph = build_highlighted_graph_by_edge(view_graph, source_id, target_id, theme=theme)
    return view_graph



def render_network_node_focus_panel(graph, dedup_df: pd.DataFrame, dimension: str, network_type: str, selected_node_id):
    if selected_node_id is None or selected_node_id not in graph.nodes:
        st.info("请选择一个核心节点")
        return
    attrs = graph.nodes[selected_node_id]
    node_label = attrs.get("label", selected_node_id)
    node_type = attrs.get("node_type", "")
    st.markdown(f"### 节点：{node_label}")
    st.caption(f"节点类型：{node_type or '-'}")
    render_network_node_neighbors(graph, selected_node_id)

    matched_jobs = get_jobs_by_node_label(dedup_df, dimension, node_label).copy()
    if matched_jobs is None or matched_jobs.empty:
        st.info("当前节点暂无可反查岗位")
        return

    matched_jobs["岗位页面"] = matched_jobs.apply(build_job_external_link, axis=1)
    safe_show_dataframe(
        matched_jobs,
        cols=["职位名称_norm", "企业名称_norm", "所在地区", "经验要求", "学历要求", "岗位页面"],
        hide_index=True,
        column_config={"岗位页面": st.column_config.LinkColumn("岗位页面", display_text="打开网页")},
    )
    with st.expander("查看岗位详情"):
        render_job_preview_list(matched_jobs, limit=15)



def render_network_node_neighbors(graph, selected_node_id):
    neighbors = []
    for neighbor in graph.neighbors(selected_node_id):
        edge_data = graph.get_edge_data(selected_node_id, neighbor)
        neighbors.append({
            "关联节点": graph.nodes[neighbor].get("label", neighbor),
            "关系权重": edge_data.get("weight", 1),
            "关系说明": edge_data.get("relation", ""),
        })
    if neighbors:
        nb_df = pd.DataFrame(neighbors).sort_values("关系权重", ascending=False)
        safe_show_dataframe(nb_df, hide_index=True)



def render_network_edge_focus_panel(graph, dedup_df: pd.DataFrame, dimension: str, network_type: str, selected_edge):
    if selected_edge is None:
        st.info("请选择一个强关联组合")
        return
    u, v = selected_edge
    if u not in graph.nodes or v not in graph.nodes or not graph.has_edge(u, v):
        st.info("所选关联组合不在当前图中")
        return
    edge_data = graph.get_edge_data(u, v)
    node_a_label = graph.nodes[u].get("label", u)
    node_b_label = graph.nodes[v].get("label", v)
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



def render_network_detail_tabs(graph, dedup_df: pd.DataFrame, params: dict, insight_payload: dict):
    tab1, tab2, tab3 = st.tabs(["标签优先级", "能力组合", "主题簇"])
    with tab1:
        priority_df = insight_payload.get("priority_table")
        if priority_df is not None and not priority_df.empty:
            show_df = priority_df.copy()
            if "覆盖率" in show_df.columns:
                show_df = format_ratio_column(show_df, "覆盖率")
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无优先级结果")
    with tab2:
        combo_df = insight_payload.get("high_value_combinations")
        if combo_df is not None and not combo_df.empty:
            st.dataframe(combo_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无能力组合结果")
    with tab3:
        render_network_communities(graph)


def render_network_debug_section(params: dict, insight_payload: dict, all_edges: pd.DataFrame):
    with st.expander("数据质量与调试信息", expanded=False):
        st.caption("这里保留字段覆盖、标签来源、边明细等排查信息；日常求职判断优先看上方摘要、图谱和优先级表。")
        tab1, tab2, tab3, tab4 = st.tabs(["字段覆盖", "标签来源", "关系边明细", "岗位方向分布"])

        with tab1:
            manual_stop_tags = params.get("effective_manual_stop_tags") or []
            effective_stop_tags = params.get("effective_stop_tags") or []
            st.caption(f"当前手动停用标签：{'、'.join(manual_stop_tags) if manual_stop_tags else '无'}")
            st.caption(f"当前实际剔除标签：{'、'.join(effective_stop_tags) if effective_stop_tags else '无'}")

            tag_coverage_df = params.get("tag_coverage_df")
            if isinstance(tag_coverage_df, pd.DataFrame) and not tag_coverage_df.empty:
                safe_show_dataframe(tag_coverage_df, hide_index=True)
            else:
                st.info("暂无字段覆盖数据")

            high_coverage_tags_df = params.get("high_coverage_tags_df")
            if isinstance(high_coverage_tags_df, pd.DataFrame) and not high_coverage_tags_df.empty:
                st.markdown("**自动剔除的超高覆盖标签**")
                show_df = high_coverage_tags_df.copy()
                if "覆盖率" in show_df.columns:
                    show_df = format_ratio_column(show_df, "覆盖率")
                safe_show_dataframe(show_df, hide_index=True)

        with tab2:
            tag_diagnostics = params.get("tag_diagnostics") or {}
            column_stats = tag_diagnostics.get("column_stats")
            st.caption(f"当前实际用于分析的标签列：{params.get('resolved_tag_col') or '未识别'}")
            if isinstance(column_stats, pd.DataFrame) and not column_stats.empty:
                st.markdown("**预设标签列诊断**")
                safe_show_dataframe(column_stats, hide_index=True)

            candidate_tag_columns = params.get("candidate_tag_columns")
            if isinstance(candidate_tag_columns, pd.DataFrame) and not candidate_tag_columns.empty:
                st.markdown("**候选标签字段扫描**")
                safe_show_dataframe(candidate_tag_columns, hide_index=True)

            dimension_overlap = params.get("dimension_overlap")
            if isinstance(dimension_overlap, dict) and dimension_overlap:
                st.markdown("**不同维度标签重叠诊断**")
                for name, table in dimension_overlap.items():
                    if isinstance(table, pd.DataFrame) and not table.empty:
                        st.markdown(f"**{name}**")
                        safe_show_dataframe(table, hide_index=True)

        with tab3:
            if isinstance(all_edges, pd.DataFrame) and not all_edges.empty:
                safe_show_dataframe(all_edges, hide_index=True)
            else:
                st.info("暂无关系边明细")

        with tab4:
            role_distribution = insight_payload.get("role_distribution")
            if isinstance(role_distribution, pd.DataFrame) and not role_distribution.empty:
                show_df = role_distribution.copy()
                if "占比" in show_df.columns:
                    show_df = format_ratio_column(show_df, "占比")
                st.dataframe(show_df, use_container_width=True, hide_index=True)
            else:
                st.info("暂无岗位方向分布")



def render_network_communities(graph):
    st.subheader("主题簇 / 社区发现")
    communities = detect_communities(graph)
    if communities is None or communities.empty:
        st.info("暂无社区结果")
        return
    safe_show_dataframe(communities, hide_index=True)
