"""
[MODULE_SPEC]
module_id: modules.network_viz
module_path: modules/network_viz.py
module_name: 网络图可视化与高亮联动模块
module_type: network_visualization
layer: 展示辅助层 / 可视化层

responsibility:
  - 将 network_analysis.py 生成的 NetworkX 图对象渲染为 Streamlit 中可交互网络图。
  - 提供节点高亮、边高亮、局部子图提取和展示级图过滤能力。
  - 本模块负责“画图”和“可视化表现”，不负责业务构图逻辑与边权定义。

notes:
  - 网络结构和业务语义应在 modules.network_analysis.py 中定义。
  - 本模块应尽量返回 Graph 副本，不修改原始 Graph。
[/MODULE_SPEC]
"""

from __future__ import annotations

import copy
import os
import tempfile
from math import log1p

import networkx as nx
import streamlit.components.v1 as components


DEFAULT_NODE_COLOR = "#9ecae1"
DEFAULT_EDGE_COLOR = "#d9d9d9"

HIGHLIGHT_NODE_COLOR = "#e63946"
NEIGHBOR_NODE_COLOR = "#f4a261"
DIM_NODE_COLOR = "#d3d3d3"

HIGHLIGHT_EDGE_COLOR = "#e63946"
NEIGHBOR_EDGE_COLOR = "#f4a261"
DIM_EDGE_COLOR = "#e5e5e5"

DEFAULT_HEIGHT = "760px"
DEFAULT_WIDTH = "100%"
DEFAULT_BGCOLOR = "#ffffff"
DEFAULT_FONT_COLOR = "#333333"
DEFAULT_THEME = "default"
DEFAULT_LAYOUT_MODE = "force"


THEME_MAP = {
    "default": {
        "bgcolor": "#ffffff",
        "font_color": "#333333",
        "node_color": "#9ecae1",
        "edge_color": "#d9d9d9",
        "highlight_node_color": "#e63946",
        "neighbor_node_color": "#f4a261",
        "dim_node_color": "#d3d3d3",
        "highlight_edge_color": "#e63946",
        "neighbor_edge_color": "#f4a261",
        "dim_edge_color": "#e5e5e5",
        "panel_border": "#e5e7eb",
        "panel_bg": "#fafafa",
        "panel_text": "#666666",
        "shadow_color": "rgba(0,0,0,0.10)",
    },
    "dark_glow": {
        "bgcolor": "#0B1020",
        "font_color": "#E8ECF3",
        "node_color": "#7AA2FF",
        "edge_color": "rgba(180,210,255,0.20)",
        "highlight_node_color": "#8B5CF6",
        "neighbor_node_color": "#22D3EE",
        "dim_node_color": "rgba(255,255,255,0.20)",
        "highlight_edge_color": "rgba(139,92,246,0.90)",
        "neighbor_edge_color": "rgba(34,211,238,0.60)",
        "dim_edge_color": "rgba(255,255,255,0.08)",
        "panel_border": "rgba(148,163,184,0.18)",
        "panel_bg": "linear-gradient(135deg, #11182b 0%, #0b1020 100%)",
        "panel_text": "#A8B3C7",
        "shadow_color": "rgba(15,23,42,0.35)",
    },
    "tech_steady": {
        "bgcolor": "#07111F",
        "font_color": "#DCE7F7",
        "node_color": "#4DA3FF",
        "edge_color": "rgba(88,166,255,0.34)",
        "highlight_node_color": "#66E3FF",
        "neighbor_node_color": "#22D3A6",
        "dim_node_color": "rgba(148,163,184,0.28)",
        "highlight_edge_color": "rgba(102,227,255,0.92)",
        "neighbor_edge_color": "rgba(34,211,166,0.68)",
        "dim_edge_color": "rgba(148,163,184,0.12)",
        "panel_border": "rgba(88,166,255,0.24)",
        "panel_bg": "radial-gradient(circle at 20% 20%, #10243f 0%, #07111f 58%, #050914 100%)",
        "panel_text": "#AFC1DA",
        "shadow_color": "rgba(2,8,23,0.38)",
        "node_type_colors": {
            "role": "#60A5FA",
            "岗位方向": "#60A5FA",
            "job_type": "#60A5FA",
            "responsibility": "#2DD4BF",
            "职责": "#2DD4BF",
            "skill": "#FBBF24",
            "tag": "#FBBF24",
            "company": "#A78BFA",
            "job": "#93C5FD"
        },
    },
    "fresh_mint": {
        "bgcolor": "#F6FFFC",
        "font_color": "#24433E",
        "node_color": "#4DB6AC",
        "edge_color": "rgba(64,120,112,0.30)",
        "highlight_node_color": "#2563EB",
        "neighbor_node_color": "#10B981",
        "dim_node_color": "rgba(148,163,184,0.28)",
        "highlight_edge_color": "rgba(37,99,235,0.78)",
        "neighbor_edge_color": "rgba(16,185,129,0.58)",
        "dim_edge_color": "rgba(148,163,184,0.16)",
        "panel_border": "rgba(45,212,191,0.30)",
        "panel_bg": "linear-gradient(135deg, #ffffff 0%, #ecfdf5 100%)",
        "panel_text": "#49645F",
        "shadow_color": "rgba(15,118,110,0.12)",
        "node_type_colors": {
            "role": "#3B82F6",
            "岗位方向": "#3B82F6",
            "job_type": "#3B82F6",
            "responsibility": "#14B8A6",
            "职责": "#14B8A6",
            "skill": "#F59E0B",
            "tag": "#F59E0B",
            "company": "#8B5CF6",
            "job": "#38BDF8"
        },
    },
    "midnight_blue": {
        "bgcolor": "#10192D",
        "font_color": "#E6EDF7",
        "node_color": "#5B8DEF",
        "edge_color": "rgba(148,163,184,0.24)",
        "highlight_node_color": "#F59E0B",
        "neighbor_node_color": "#38BDF8",
        "dim_node_color": "rgba(203,213,225,0.18)",
        "highlight_edge_color": "rgba(245,158,11,0.85)",
        "neighbor_edge_color": "rgba(56,189,248,0.55)",
        "dim_edge_color": "rgba(203,213,225,0.10)",
        "panel_border": "rgba(148,163,184,0.22)",
        "panel_bg": "linear-gradient(135deg, #16233d 0%, #10192d 100%)",
        "panel_text": "#B8C3D9",
        "shadow_color": "rgba(2,6,23,0.30)",
    },
}


NODE_TYPE_COLOR_MAP = {
    "role": "#7C9BFF",
    "岗位方向": "#7C9BFF",
    "job_type": "#7C9BFF",
    "responsibility": "#34D3B4",
    "职责": "#34D3B4",
    "skill": "#F6C760",
    "tag": "#F6C760",
    "company": "#F472B6",
    "job": "#93C5FD",
}


def _get_theme(theme: str | None) -> dict:
    if not theme:
        return THEME_MAP[DEFAULT_THEME]
    return THEME_MAP.get(theme, THEME_MAP[DEFAULT_THEME])


def _get_color_by_node_type(attrs: dict, theme: dict) -> str:
    node_type = str(attrs.get("node_type", "")).strip()
    theme_type_colors = theme.get("node_type_colors", {}) if isinstance(theme, dict) else {}
    return theme_type_colors.get(node_type) or NODE_TYPE_COLOR_MAP.get(node_type, theme.get("node_color", DEFAULT_NODE_COLOR))


def _is_empty_graph(G: nx.Graph | None) -> bool:
    return G is None or G.number_of_nodes() == 0


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int_from_height(height: str, default: int = 760) -> int:
    if not isinstance(height, str):
        return default

    text = height.strip().lower()
    if text.endswith("px"):
        text = text[:-2].strip()

    try:
        return int(float(text))
    except Exception:
        return default


def _html_message_box(message: str, *, height: int = 160, theme: str | None = None) -> None:
    theme_cfg = _get_theme(theme)
    html = f"""
    <div style="
        height:{height-20}px;
        display:flex;
        align-items:center;
        justify-content:center;
        border:1px solid {theme_cfg.get('panel_border', '#e5e7eb')};
        border-radius:12px;
        background:{theme_cfg.get('panel_bg', '#fafafa')};
        color:{theme_cfg.get('panel_text', '#666666')};
        box-shadow:0 12px 28px {theme_cfg.get('shadow_color', 'rgba(0,0,0,0.10)')};
        font-family:Arial, sans-serif;
        font-size:15px;
        text-align:center;
        padding:10px;">
        {message}
    </div>
    """
    components.html(html, height=height)


def _get_node_base_size(attrs: dict, default: float = 10.0) -> float:
    return _safe_float(attrs.get("size", default), default)


def _compute_visual_node_size(attrs: dict) -> float:
    base_size = _get_node_base_size(attrs, 10.0)
    node_type = str(attrs.get("node_type", "")).strip()
    count = _safe_float(attrs.get("count", attrs.get("coverage_jobs", 0)), 0)
    coverage_jobs = _safe_float(attrs.get("coverage_jobs", attrs.get("count", 0)), 0)
    weighted_degree = _safe_float(attrs.get("weighted_degree", 0), 0)
    size_score = _safe_float(attrs.get("size_score", 0), 0)

    if node_type in {"tag", "skill", "职责", "responsibility"}:
        signal = max(count, coverage_jobs, 0)
        visual_size = 10 + log1p(signal) * 9.5 + log1p(max(weighted_degree, 0)) * 1.6
    elif node_type in {"role", "岗位方向", "job_type", "company"}:
        signal = max(coverage_jobs, count, 0)
        visual_size = 14 + log1p(signal) * 7.2 + log1p(max(weighted_degree, 0)) * 1.8
    elif node_type == "job":
        signal = max(count, 0)
        visual_size = 12 + log1p(signal) * 4.0 + log1p(max(weighted_degree, 0)) * 2.0
    else:
        signal = max(count, coverage_jobs, 0)
        visual_size = 10 + log1p(signal) * 6.2 + log1p(max(weighted_degree, 0)) * 2.2 + size_score * 2.0

    visual_size = max(base_size * 0.9, visual_size)
    return min(visual_size, 56)


def _copy_graph(G: nx.Graph) -> nx.Graph:
    return copy.deepcopy(G)


def filter_graph_for_visualization(
    G: nx.Graph,
    *,
    min_edge_weight: float | None = None,
    remove_isolates: bool = False,
    max_nodes: int | None = None,
) -> nx.Graph:
    """
    在渲染前对图做展示级过滤，不改变业务层图定义。
    """
    if _is_empty_graph(G):
        return nx.Graph()

    H = G.copy()

    if min_edge_weight is not None:
        edges_to_remove = []
        for u, v, attrs in H.edges(data=True):
            weight = _safe_float(attrs.get("weight", 1), 1)
            if weight < min_edge_weight:
                edges_to_remove.append((u, v))
        H.remove_edges_from(edges_to_remove)

    if remove_isolates:
        isolates = list(nx.isolates(H))
        H.remove_nodes_from(isolates)

    if max_nodes is not None and H.number_of_nodes() > max_nodes:
        degree_rank = sorted(H.degree(weight="weight"), key=lambda x: x[1], reverse=True)
        keep_nodes = {node for node, _ in degree_rank[:max_nodes]}
        H = H.subgraph(keep_nodes).copy()

    return H


def build_highlighted_graph_by_node(G: nx.Graph, selected_node: str, *, theme: str | None = None) -> nx.Graph:
    """
    高亮单个节点及其一阶邻居。
    返回 Graph 副本，不修改原图。
    """
    H = _copy_graph(G)
    theme_cfg = _get_theme(theme)

    if _is_empty_graph(H) or selected_node not in H.nodes:
        return H

    neighbors = set(H.neighbors(selected_node))

    for node in H.nodes:
        attrs = H.nodes[node]
        base_size = _get_node_base_size(attrs)

        attrs["base_color"] = attrs.get("color", _get_color_by_node_type(attrs, theme_cfg))
        attrs["color"] = theme_cfg.get("dim_node_color", DIM_NODE_COLOR)
        attrs["size"] = base_size * 0.9

        if node == selected_node:
            attrs["color"] = theme_cfg.get("highlight_node_color", HIGHLIGHT_NODE_COLOR)
            attrs["size"] = base_size * 1.8
        elif node in neighbors:
            attrs["color"] = theme_cfg.get("neighbor_node_color", NEIGHBOR_NODE_COLOR)
            attrs["size"] = base_size * 1.25

    for u, v, attrs in H.edges(data=True):
        attrs["base_color"] = attrs.get("color", theme_cfg.get("edge_color", DEFAULT_EDGE_COLOR))
        attrs["color"] = theme_cfg.get("dim_edge_color", DIM_EDGE_COLOR)
        attrs["width"] = max(1.0, _safe_float(attrs.get("width", 1), 1) * 0.75)

        if u == selected_node or v == selected_node:
            attrs["color"] = theme_cfg.get("highlight_edge_color", HIGHLIGHT_EDGE_COLOR)
            attrs["width"] = max(4.0, _safe_float(attrs.get("width", 1), 1) * 1.8)
        elif u in neighbors and v in neighbors:
            attrs["color"] = theme_cfg.get("neighbor_edge_color", NEIGHBOR_EDGE_COLOR)
            attrs["width"] = max(2.0, _safe_float(attrs.get("width", 1), 1) * 1.1)

    return H


def build_highlighted_graph_by_edge(G: nx.Graph, source: str, target: str, *, theme: str | None = None) -> nx.Graph:
    """
    高亮单条边及其两端节点，并次高亮共同邻居。
    返回 Graph 副本，不修改原图。
    """
    H = _copy_graph(G)
    theme_cfg = _get_theme(theme)

    if _is_empty_graph(H) or not H.has_edge(source, target):
        return H

    focus_nodes = {source, target}
    shared_neighbors = set(H.neighbors(source)).intersection(set(H.neighbors(target)))

    for node in H.nodes:
        attrs = H.nodes[node]
        base_size = _get_node_base_size(attrs)

        attrs["base_color"] = attrs.get("color", _get_color_by_node_type(attrs, theme_cfg))
        attrs["color"] = theme_cfg.get("dim_node_color", DIM_NODE_COLOR)
        attrs["size"] = base_size * 0.9

        if node in focus_nodes:
            attrs["color"] = theme_cfg.get("highlight_node_color", HIGHLIGHT_NODE_COLOR)
            attrs["size"] = base_size * 1.8
        elif node in shared_neighbors:
            attrs["color"] = theme_cfg.get("neighbor_node_color", NEIGHBOR_NODE_COLOR)
            attrs["size"] = base_size * 1.25

    for u, v, attrs in H.edges(data=True):
        attrs["base_color"] = attrs.get("color", theme_cfg.get("edge_color", DEFAULT_EDGE_COLOR))
        attrs["color"] = theme_cfg.get("dim_edge_color", DIM_EDGE_COLOR)
        attrs["width"] = max(1.0, _safe_float(attrs.get("width", 1), 1) * 0.75)

        if (u == source and v == target) or (u == target and v == source):
            attrs["color"] = theme_cfg.get("highlight_edge_color", HIGHLIGHT_EDGE_COLOR)
            attrs["width"] = max(5.0, _safe_float(attrs.get("width", 1), 1) * 2.0)
        elif u in focus_nodes or v in focus_nodes:
            attrs["color"] = theme_cfg.get("neighbor_edge_color", NEIGHBOR_EDGE_COLOR)
            attrs["width"] = max(2.5, _safe_float(attrs.get("width", 1), 1) * 1.15)

    return H


def build_subgraph_for_node(G: nx.Graph, selected_node: str) -> nx.Graph:
    """
    只保留选中节点及其一阶邻居。
    返回 Graph 副本。
    """
    if _is_empty_graph(G) or selected_node not in G.nodes:
        return G.copy() if G is not None else nx.Graph()

    nodes = {selected_node} | set(G.neighbors(selected_node))
    return G.subgraph(nodes).copy()


def build_subgraph_for_edge(G: nx.Graph, source: str, target: str) -> nx.Graph:
    """
    只保留选中边、两端节点及其共同邻居。
    返回 Graph 副本。
    """
    if _is_empty_graph(G) or not G.has_edge(source, target):
        return G.copy() if G is not None else nx.Graph()

    shared_neighbors = set(G.neighbors(source)).intersection(set(G.neighbors(target)))
    nodes = {source, target} | shared_neighbors
    return G.subgraph(nodes).copy()


def focus_graph_on_node(
    G: nx.Graph,
    selected_node: str,
    *,
    subgraph_only: bool = False,
    theme: str | None = None,
) -> nx.Graph:
    """
    聚焦某个节点：
    - subgraph_only=False: 保留全图，仅高亮
    - subgraph_only=True: 裁剪到一阶邻域后再高亮
    """
    base = build_subgraph_for_node(G, selected_node) if subgraph_only else G
    return build_highlighted_graph_by_node(base, selected_node, theme=theme)


def focus_graph_on_edge(
    G: nx.Graph,
    source: str,
    target: str,
    *,
    subgraph_only: bool = False,
    theme: str | None = None,
) -> nx.Graph:
    """
    聚焦某条边：
    - subgraph_only=False: 保留全图，仅高亮
    - subgraph_only=True: 裁剪到局部子图后再高亮
    """
    base = build_subgraph_for_edge(G, source, target) if subgraph_only else G
    return build_highlighted_graph_by_edge(base, source, target, theme=theme)


def _assign_hierarchical_levels(G: nx.Graph) -> int:
    level_map = {
        "role": 0,
        "岗位方向": 0,
        "job_type": 0,
        "company": 0,
        "job": 0,
        "responsibility": 1,
        "职责": 1,
        "tag": 1,
        "skill": 1,
    }
    levels: set[int] = set()
    for _, attrs in G.nodes(data=True):
        node_type = str(attrs.get("node_type", "")).strip()
        attrs["level"] = level_map.get(node_type, attrs.get("level", 1))
        levels.add(int(attrs["level"]))
    return len(levels)



def graph_to_pyvis_html(
    G: nx.Graph,
    *,
    height: str = DEFAULT_HEIGHT,
    width: str = DEFAULT_WIDTH,
    bgcolor: str | None = None,
    font_color: str | None = None,
    theme: str = DEFAULT_THEME,
    layout_mode: str = DEFAULT_LAYOUT_MODE,
    physics_enabled: bool = False,
    random_seed: int = 42,
    show_buttons: bool = False,
    keep_physics_after_stabilization: bool = False,
) -> str:
    """
    将 NetworkX Graph 转换为 PyVis HTML 字符串。
    不直接渲染到 Streamlit。
    """
    if _is_empty_graph(G):
        return """
        <div style="padding:24px;text-align:center;color:#666;font-family:Arial,sans-serif;">
            暂无网络图数据
        </div>
        """

    try:
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError("未安装 pyvis，请先执行 pip install pyvis") from e

    theme_cfg = _get_theme(theme)
    bgcolor = bgcolor or theme_cfg.get("bgcolor", DEFAULT_BGCOLOR)
    font_color = font_color or theme_cfg.get("font_color", DEFAULT_FONT_COLOR)

    requested_layered = str(layout_mode).lower() == "layered"
    level_count = _assign_hierarchical_levels(G) if requested_layered else 0
    hierarchical_enabled = requested_layered and level_count >= 2
    force_fallback = requested_layered and not hierarchical_enabled

    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        directed=False,
    )

    physics_active = bool(physics_enabled or str(layout_mode).lower() == "force" or force_fallback)
    if physics_active:
        net.barnes_hut(
            gravity=-36000,
            central_gravity=0.10,
            spring_length=185,
            spring_strength=0.032,
            damping=0.14,
            overlap=0.42,
        )
    else:
        net.toggle_physics(False)

    for node, attrs in G.nodes(data=True):
        if str(theme).lower() in {"tech_steady", "fresh_mint", "midnight_blue", "dark_glow"} and not attrs.get("is_highlighted"):
            node_color = _get_color_by_node_type(attrs, theme_cfg)
        else:
            node_color = attrs.get("color") or _get_color_by_node_type(attrs, theme_cfg)
        title = str(attrs.get("title", attrs.get("label", node)))
        node_type = attrs.get("node_type", attrs.get("group", "未分类"))
        coverage = attrs.get("coverage", attrs.get("count", ""))
        weighted_degree = attrs.get("weighted_degree", "")
        degree = attrs.get("degree", "")
        if title == str(attrs.get("label", node)):
            title = (
                f"<b>{attrs.get('label', node)}</b><br>"
                f"类型：{node_type}<br>"
                f"覆盖/计数：{coverage if coverage != '' else '暂无'}<br>"
                f"连接度：{degree if degree != '' else '暂无'}<br>"
                f"加权连接度：{weighted_degree if weighted_degree != '' else '暂无'}"
            )

        node_kwargs = dict(
            label=str(attrs.get("label", node)),
            title=title,
            size=_compute_visual_node_size(attrs),
            color={"background": node_color, "border": theme_cfg.get("highlight_node_color", node_color), "highlight": {"background": theme_cfg.get("highlight_node_color", node_color), "border": "#FFFFFF"}},
            borderWidth=1.4,
            group=attrs.get("group", node_type),
        )
        if hierarchical_enabled:
            node_kwargs["level"] = int(attrs.get("level", 1))

        net.add_node(
            str(node),
            **node_kwargs,
        )

    for source, target, attrs in G.edges(data=True):
        weight = _safe_float(attrs.get("weight", 1), 1)
        width_value = _safe_float(
            attrs.get("width", max(1.0, min(weight, 8.0))),
            max(1.0, min(weight, 8.0)),
        )
        edge_title = str(attrs.get("title", "")) or f"关联强度：{round(weight, 3)}"

        edge_color = attrs.get("color") or theme_cfg.get("edge_color", DEFAULT_EDGE_COLOR)
        net.add_edge(
            str(source),
            str(target),
            title=edge_title,
            width=max(1.2, width_value),
            color={"color": edge_color, "highlight": theme_cfg.get("highlight_edge_color", edge_color), "hover": theme_cfg.get("neighbor_edge_color", edge_color), "inherit": False},
        )

    smooth_type = "cubicBezier" if hierarchical_enabled else "dynamic"

    options = f"""
    var options = {{
      "layout": {{
        "randomSeed": {int(random_seed)},
        "improvedLayout": true,
        "hierarchical": {{
          "enabled": {str(hierarchical_enabled).lower()},
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 180,
          "nodeSpacing": 160,
          "treeSpacing": 220
        }}
      }},
      "nodes": {{
        "shape": "dot",
        "scaling": {{"min": 8, "max": 42}},
        "borderWidth": 1.5,
        "borderWidthSelected": 2.5,
        "shadow": {{
          "enabled": true,
          "color": "rgba(0,0,0,0.22)",
          "size": 12,
          "x": 0,
          "y": 4
        }},
        "font": {{
          "size": 48,
          "face": "Arial",
          "color": "{font_color}"
        }}
      }},
      "edges": {{
        "smooth": {{
          "enabled": true,
          "type": "{smooth_type}",
          "roundness": 0.18
        }},
        "color": {{
          "inherit": false
        }},
        "selectionWidth": 1.6,
        "hoverWidth": 1.2
      }},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": false,
        "navigationButtons": {str(show_buttons).lower()},
        "keyboard": true,
        "multiselect": false
      }},
      "physics": {{
        "enabled": {str(physics_active).lower()},
        "stabilization": {{
          "enabled": true,
          "iterations": 180,
          "updateInterval": 25,
          "fit": true
        }},
        "barnesHut": {{
          "gravitationalConstant": -36000,
          "centralGravity": 0.10,
          "springLength": 185,
          "springConstant": 0.032,
          "damping": 0.14,
          "avoidOverlap": 0.52
        }},
        "minVelocity": 0.55
      }}
    }}
    """

    net.set_options(options)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = tmp_file.name

    try:
        net.save_graph(path)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

    if physics_active and not keep_physics_after_stabilization:
        html = html.replace(
            "network.setOptions(options);",
            "network.setOptions(options);\n"
            "network.once('stabilizationIterationsDone', function () {\n"
            "  network.setOptions({physics: false});\n"
            "});",
        )

    return html


def render_pyvis_network(
    G: nx.Graph,
    height: str = DEFAULT_HEIGHT,
    width: str = DEFAULT_WIDTH,
    bgcolor: str | None = None,
    font_color: str | None = None,
    *,
    theme: str = DEFAULT_THEME,
    layout_mode: str = DEFAULT_LAYOUT_MODE,
    physics_enabled: bool = False,
    random_seed: int = 42,
    show_buttons: bool = False,
    keep_physics_after_stabilization: bool = False,
):
    """
    在 Streamlit 中渲染 PyVis 网络图。
    对空图、缺少 pyvis、HTML 生成失败做容错处理。
    """
    if _is_empty_graph(G):
        _html_message_box("暂无网络图数据", height=180, theme=theme)
        return

    try:
        html = graph_to_pyvis_html(
            G,
            height=height,
            width=width,
            bgcolor=bgcolor,
            font_color=font_color,
            theme=theme,
            layout_mode=layout_mode,
            physics_enabled=physics_enabled,
            random_seed=random_seed,
            show_buttons=show_buttons,
            keep_physics_after_stabilization=keep_physics_after_stabilization,
        )
        height_int = _safe_int_from_height(height, default=760)
        components.html(html, height=height_int, scrolling=True)
    except ImportError as e:
        _html_message_box(str(e), height=180, theme=theme)
    except Exception as e:
        _html_message_box(f"网络图渲染失败：{e}", height=200, theme=theme)
