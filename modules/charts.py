from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from pyvis.network import Network
from wordcloud import WordCloud


DEFAULT_FIG_HEIGHT = 420
DEFAULT_HORIZONTAL_FIG_HEIGHT = 500
DEFAULT_HEATMAP_HEIGHT = 500


def _empty_plotly_figure(title: str = "暂无数据"):
    fig = px.scatter(title=title)
    fig.update_traces(visible=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=320,
        annotations=[
            {
                "text": title,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16},
                "x": 0.5,
                "y": 0.5,
            }
        ],
    )
    return fig


def _validate_columns(df: pd.DataFrame, required_cols: list[str]) -> tuple[bool, list[str]]:
    if not isinstance(df, pd.DataFrame):
        return False, required_cols

    missing = [col for col in required_cols if col not in df.columns]
    return len(missing) == 0, missing


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    *,
    height: int = DEFAULT_FIG_HEIGHT,
    color: str | None = None,
    text_auto: bool = True,
):
    """
    绘制基础柱状图。

    返回：
    - Plotly Figure
    """
    ok, missing = _validate_columns(df, [x, y])
    if not ok or df.empty:
        return _empty_plotly_figure(f"{title}（暂无可用数据）")

    fig = px.bar(df, x=x, y=y, title=title, text_auto=text_auto, color=color)
    fig.update_layout(height=height)
    return fig


def plot_horizontal_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    *,
    height: int = DEFAULT_HORIZONTAL_FIG_HEIGHT,
    color: str | None = None,
    text_auto: bool = True,
    categoryorder: str = "total ascending",
):
    """
    绘制横向柱状图。

    返回：
    - Plotly Figure
    """
    ok, missing = _validate_columns(df, [x, y])
    if not ok or df.empty:
        return _empty_plotly_figure(f"{title}（暂无可用数据）")

    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation="h",
        title=title,
        text_auto=text_auto,
        color=color,
    )
    fig.update_layout(height=height, yaxis={"categoryorder": categoryorder})
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    index_col: str,
    title: str,
    *,
    height: int = DEFAULT_HEATMAP_HEIGHT,
    color_continuous_scale: str = "Blues",
):
    """
    绘制热力图。

    约定：
    - index_col 作为行索引
    - 除 index_col 外，仅保留数值列作为热力图值
    """
    ok, missing = _validate_columns(df, [index_col])
    if not ok or df.empty:
        return _empty_plotly_figure(f"{title}（暂无可用数据）")

    temp = df.copy()
    numeric_cols = [col for col in temp.columns if col != index_col and pd.api.types.is_numeric_dtype(temp[col])]

    if not numeric_cols:
        return _empty_plotly_figure(f"{title}（无可用数值列）")

    temp = temp[[index_col] + numeric_cols].set_index(index_col)

    if temp.empty:
        return _empty_plotly_figure(f"{title}（暂无可用数据）")

    fig = px.imshow(
        temp,
        text_auto=True,
        aspect="auto",
        title=title,
        color_continuous_scale=color_continuous_scale,
    )
    fig.update_layout(height=height)
    return fig


def generate_wordcloud(
    freq_dict: dict[str, Any],
    *,
    width: int = 1000,
    height: int = 500,
    background_color: str = "white",
    font_path: str | None = "simhei.ttf",
    max_words: int = 150,
):
    """
    生成词云图。

    返回：
    - matplotlib Figure
    - 若词频为空，则返回 None

    说明：
    - 若 font_path 不存在，自动降级为 None，让 WordCloud 使用默认字体。
    - 如果环境没有中文字体，中文显示可能不正常，建议部署时显式传入字体文件。
    """
    if not isinstance(freq_dict, dict) or not freq_dict:
        return None

    clean_freq = {}
    for k, v in freq_dict.items():
        key = str(k).strip()
        if not key:
            continue
        try:
            value = float(v)
        except (TypeError, ValueError):
            continue
        if value > 0:
            clean_freq[key] = value

    if not clean_freq:
        return None

    actual_font_path = font_path
    if actual_font_path:
        if not Path(actual_font_path).exists():
            actual_font_path = None

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        font_path=actual_font_path,
        max_words=max_words,
    ).generate_from_frequencies(clean_freq)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()

    return fig


def build_pyvis_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    output_html: str = "outputs/network.html",
    *,
    node_id_col: str = "job_id",
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    title_fields: list[str] | None = None,
    label_fields: list[str] | None = None,
    height: str = "700px",
    width: str = "100%",
    bgcolor: str = "#ffffff",
    font_color: str = "black",
):
    """
    构建简单 PyVis 网络图并输出为 HTML。

    参数：
    - nodes_df: 节点表，默认至少包含 job_id
    - edges_df: 边表，默认至少包含 source / target
    - output_html: 输出 html 路径
    - title_fields: tooltip 展示字段列表
    - label_fields: 节点 label 拼接字段列表

    返回：
    - 输出 html 路径字符串
    """
    if title_fields is None:
        title_fields = ["职位名称", "企业名称", "所在地区", "经验要求", "学历要求"]

    if label_fields is None:
        label_fields = ["职位名称", "企业名称"]

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        notebook=False,
    )

    # 校验节点表
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty or node_id_col not in nodes_df.columns:
        net.save_graph(str(output_path))
        return str(output_path)

    # 添加节点
    for _, row in nodes_df.iterrows():
        node_id = row.get(node_id_col)
        if pd.isna(node_id):
            continue

        label_parts = []
        for field in label_fields:
            if field in nodes_df.columns:
                value = row.get(field, "")
                if not pd.isna(value) and str(value).strip():
                    label_parts.append(str(value).strip())

        label = " | ".join(label_parts) if label_parts else str(node_id)

        title_parts = []
        for field in title_fields:
            if field in nodes_df.columns:
                value = row.get(field, "")
                value_str = "" if pd.isna(value) else str(value).strip()
                title_parts.append(f"{field}: {value_str}")

        title = "<br>".join(title_parts) if title_parts else label

        net.add_node(str(node_id), label=label, title=title)

    # 添加边
    if isinstance(edges_df, pd.DataFrame) and not edges_df.empty:
        required_edge_cols = [source_col, target_col]
        ok, missing = _validate_columns(edges_df, required_edge_cols)

        if ok:
            for _, row in edges_df.iterrows():
                source = row.get(source_col)
                target = row.get(target_col)

                if pd.isna(source) or pd.isna(target):
                    continue

                weight = row.get(weight_col, 1.0)
                try:
                    weight = float(weight)
                except (TypeError, ValueError):
                    weight = 1.0

                net.add_edge(
                    str(source),
                    str(target),
                    value=weight,
                    title=f"权重: {weight:.3f}",
                )

    net.force_atlas_2based()
    net.save_graph(str(output_path))
    return str(output_path)
