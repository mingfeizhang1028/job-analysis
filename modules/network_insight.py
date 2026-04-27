from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import networkx as nx
import pandas as pd

from modules.network_analysis import _safe_tags, detect_communities


DEFAULT_ROLE_COL_CANDIDATES = ["职位方向", "LLM岗位类型", "职位类别"]
DEFAULT_TAG_COL_CANDIDATES = ["最终全部标签", "全部标签"]


def _pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def summarize_role_distribution(df: pd.DataFrame, role_col: str | None = None, top_n: int = 12) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["岗位方向", "岗位数", "占比"])

    role_col = role_col or _pick_existing_column(df, DEFAULT_ROLE_COL_CANDIDATES)
    if not role_col or role_col not in df.columns:
        return pd.DataFrame(columns=["岗位方向", "岗位数", "占比"])

    series = df[role_col].fillna("未知方向").astype(str).str.strip().replace("", "未知方向")
    stat = series.value_counts().head(top_n).reset_index()
    stat.columns = ["岗位方向", "岗位数"]
    total = max(len(df), 1)
    stat["占比"] = stat["岗位数"] / total
    return stat


def summarize_tag_frequency(df: pd.DataFrame, tag_col: str, top_n: int = 20) -> pd.DataFrame:
    if df is None or df.empty or tag_col not in df.columns:
        return pd.DataFrame(columns=["标签", "出现次数", "覆盖岗位数", "覆盖率"])

    tag_counter = Counter()
    tag_job_counter: dict[str, set[str]] = {}

    for idx, row in df.iterrows():
        job_id = str(row.get("job_id", idx))
        tags = set(_safe_tags(row.get(tag_col, [])))
        for tag in tags:
            tag_counter[tag] += 1
            tag_job_counter.setdefault(tag, set()).add(job_id)

    if not tag_counter:
        return pd.DataFrame(columns=["标签", "出现次数", "覆盖岗位数", "覆盖率"])

    total_jobs = max(df["job_id"].nunique() if "job_id" in df.columns else len(df), 1)
    rows = []
    for tag, count in tag_counter.most_common(top_n):
        coverage_jobs = len(tag_job_counter.get(tag, set()))
        rows.append({
            "标签": tag,
            "出现次数": int(count),
            "覆盖岗位数": int(coverage_jobs),
            "覆盖率": coverage_jobs / total_jobs,
        })
    return pd.DataFrame(rows)


def extract_high_value_combinations(
    df: pd.DataFrame,
    tag_col: str,
    min_support: int = 3,
    top_n: int = 20,
) -> pd.DataFrame:
    if df is None or df.empty or tag_col not in df.columns:
        return pd.DataFrame(columns=["标签A", "标签B", "共现岗位数"])

    pair_counter = Counter()
    for _, row in df.iterrows():
        tags = sorted(set(_safe_tags(row.get(tag_col, []))))
        if len(tags) < 2:
            continue
        for a, b in combinations(tags, 2):
            pair_counter[(a, b)] += 1

    rows = []
    for (a, b), count in pair_counter.most_common():
        if count < min_support:
            continue
        rows.append({
            "标签A": a,
            "标签B": b,
            "共现岗位数": int(count),
        })
        if len(rows) >= top_n:
            break

    return pd.DataFrame(rows)


def compute_effort_priority_scores(
    graph: nx.Graph,
    tag_top_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if graph is None or graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["标签", "连接数", "加权度", "覆盖率", "优先级得分", "建议层级"])

    weighted_degree = dict(graph.degree(weight="weight"))
    degree = dict(graph.degree())

    coverage_map = {}
    if tag_top_df is not None and not tag_top_df.empty and "标签" in tag_top_df.columns:
        coverage_map = dict(zip(tag_top_df["标签"], tag_top_df.get("覆盖率", 0)))

    rows = []
    for node, attrs in graph.nodes(data=True):
        label = attrs.get("label", node)
        node_type = attrs.get("node_type")
        if node_type not in (None, "tag", "skill", "职责", "responsibility"):
            continue

        conn = degree.get(node, 0)
        weight_degree = float(weighted_degree.get(node, 0))
        coverage = float(coverage_map.get(label, attrs.get("coverage", 0) or 0))
        score = round(weight_degree * 0.55 + conn * 0.25 + coverage * 100 * 0.20, 3)

        if score >= 18:
            level = "必修核心"
        elif score >= 10:
            level = "方向强化"
        elif score >= 6:
            level = "加分项"
        else:
            level = "观察项"

        rows.append({
            "标签": label,
            "连接数": conn,
            "加权度": round(weight_degree, 3),
            "覆盖率": coverage,
            "优先级得分": score,
            "建议层级": level,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["优先级得分", "加权度"], ascending=False).reset_index(drop=True)


def summarize_communities_simple(graph: nx.Graph) -> pd.DataFrame:
    if graph is None or graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["社区ID", "节点数", "代表节点"])

    try:
        comm_df = detect_communities(graph)
    except Exception:
        return pd.DataFrame(columns=["社区ID", "节点数", "代表节点"])

    if comm_df is None or comm_df.empty:
        return pd.DataFrame(columns=["社区ID", "节点数", "代表节点"])

    summary = (
        comm_df.groupby("社区ID")
        .agg(
            节点数=("节点", "count"),
            代表节点=("节点", lambda x: "、".join(list(x)[:8])),
        )
        .reset_index()
        .sort_values("节点数", ascending=False)
    )
    return summary


def build_network_insight_payload(
    df: pd.DataFrame,
    graph: nx.Graph,
    *,
    role_col: str | None = None,
    tag_col: str | None = None,
) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "overview": {},
            "role_distribution": pd.DataFrame(),
            "tag_top": pd.DataFrame(),
            "high_value_combinations": pd.DataFrame(),
            "priority_table": pd.DataFrame(),
            "community_summary": pd.DataFrame(),
        }

    tag_col = tag_col or _pick_existing_column(df, DEFAULT_TAG_COL_CANDIDATES)
    role_distribution = summarize_role_distribution(df, role_col=role_col)
    tag_top = summarize_tag_frequency(df, tag_col=tag_col, top_n=20) if tag_col else pd.DataFrame()
    combinations_df = extract_high_value_combinations(df, tag_col=tag_col, min_support=3, top_n=20) if tag_col else pd.DataFrame()
    priority_table = compute_effort_priority_scores(graph, tag_top_df=tag_top)
    community_summary = summarize_communities_simple(graph)

    overview = {
        "job_count": int(df["job_id"].nunique() if "job_id" in df.columns else len(df)),
        "node_count": int(graph.number_of_nodes() if graph is not None else 0),
        "edge_count": int(graph.number_of_edges() if graph is not None else 0),
        "main_direction": role_distribution.iloc[0]["岗位方向"] if not role_distribution.empty else "",
        "main_direction_ratio": float(role_distribution.iloc[0]["占比"]) if not role_distribution.empty else 0.0,
    }

    return {
        "overview": overview,
        "role_distribution": role_distribution,
        "tag_top": tag_top,
        "high_value_combinations": combinations_df,
        "priority_table": priority_table,
        "community_summary": community_summary,
        "meta": {
            "role_col": role_col,
            "tag_col": tag_col,
        },
    }
