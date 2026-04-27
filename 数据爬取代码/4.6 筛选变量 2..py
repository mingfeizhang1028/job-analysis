from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:22:15 2026

@author: Rosem
"""

# -*- coding: utf-8 -*-
"""
CFPS 变量协调：聚类归并版
方法：TF-IDF + 层次聚类
输入：variable_cards_enriched.csv
输出：
    1. variable_cluster_map.csv
    2. concept_clusters_summary.csv
    3. cluster_review_template.xlsx

Python 3.13
"""


import ast
import gc
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 0. 全局配置
# =========================
SEED = 20250406
random.seed(SEED)
np.random.seed(SEED)

# ===== 路径配置 =====
INPUT_FILE = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\harmonization_outputs\variable_cards_enriched.csv")
OUTPUT_DIR = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\cluster_harmonization_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VAR_MAP = OUTPUT_DIR / "variable_cluster_map.csv"
OUTPUT_CLUSTER_SUMMARY = OUTPUT_DIR / "concept_clusters_summary.csv"
OUTPUT_REVIEW_XLSX = OUTPUT_DIR / "cluster_review_template.xlsx"

# ===== 聚类配置 =====
MIN_GROUP_SIZE_TO_CLUSTER = 2

# 层次聚类距离阈值
# 距离 = 1 - cosine_similarity
# 阈值越小，聚类越严格；建议 0.30~0.55 之间调试
DISTANCE_THRESHOLD = 0.42

# 过大的组可限制最大处理量，避免极端组过慢
MAX_ROWS_PER_PREGROUP = 5000

# TF-IDF 参数
TFIDF_ANALYZER = "char_wb"      # 推荐 char_wb，对中文短标签更稳
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True
TFIDF_MAX_FEATURES = 30000

# 是否按预分组聚类
USE_PRE_GROUPING = True

# 是否把年份信息写入文本
INCLUDE_YEAR_IN_TEXT = False

# 是否在 concept_label 中加入 anchor 年份
INCLUDE_ANCHOR_YEAR_IN_LABEL = False


# =========================
# 1. 通用工具函数
# =========================
def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = x.replace("\u3000", " ")
    return x


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_label_text(x: str) -> str:
    x = normalize_text(x)
    x = re.sub(r"[，。；：、“”‘’（）()【】\[\]\-_/,:;!?]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def try_parse_list(x) -> List[str]:
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [safe_str(i) for i in x if safe_str(i)]
    s = str(x).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [safe_str(i) for i in val if safe_str(i)]
    except Exception:
        pass
    return [i.strip() for i in re.split(r"[;,|/]+", s) if i.strip()]


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_column(df: pd.DataFrame, col: str, default=""):
    if col not in df.columns:
        df[col] = default
    return df


def print_df_info(df: pd.DataFrame, title: str):
    print("=" * 80)
    print(title)
    print(f"行数：{len(df):,}")
    print(f"列数：{len(df.columns)}")
    print("列名：")
    print(list(df.columns))


# =========================
# 2. 字段识别
# =========================
def detect_columns(df: pd.DataFrame) -> dict:
    """
    尽量自动识别常见字段名。
    你可按自己的 variable_cards_enriched.csv 实际列名再微调。
    """
    colmap = {}

    colmap["var_name"] = first_existing_column(df, [
        "var_name", "variable_name", "name", "var", "变量名"
    ])
    colmap["var_label"] = first_existing_column(df, [
        "var_label", "variable_label", "label", "question_label", "变量标签", "题目", "题干"
    ])
    colmap["year"] = first_existing_column(df, [
        "year", "wave", "survey_year", "年份"
    ])
    colmap["source_file"] = first_existing_column(df, [
        "source_file", "file_name", "filename", "file", "问卷文件", "数据文件"
    ])
    colmap["module"] = first_existing_column(df, [
        "module", "file_type", "dataset_type", "survey_module", "模块", "库别"
    ])
    colmap["question_text"] = first_existing_column(df, [
        "question_text", "question", "qtext", "题干", "问题文本"
    ])
    colmap["value_labels"] = first_existing_column(df, [
        "value_labels", "value_label", "codes", "编码说明", "取值标签"
    ])
    colmap["var_type"] = first_existing_column(df, [
        "var_type", "type", "dtype", "变量类型"
    ])

    return colmap


# =========================
# 3. 预处理与文本构造
# =========================
def build_text_for_grouping(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    df = df.copy()

    for key in ["var_name", "var_label", "year", "source_file", "module", "question_text", "value_labels", "var_type"]:
        if colmap.get(key) is None:
            df[key] = ""
            colmap[key] = key
        else:
            df[key] = df[colmap[key]]

    df["var_name_std"] = df["var_name"].map(normalize_text)
    df["var_label_std"] = df["var_label"].map(clean_label_text)
    df["source_file_std"] = df["source_file"].map(normalize_text)
    df["module_std"] = df["module"].map(normalize_text)
    df["question_text_std"] = df["question_text"].map(clean_label_text)
    df["value_labels_std"] = df["value_labels"].map(clean_label_text)
    df["var_type_std"] = df["var_type"].map(normalize_text)
    df["year_std"] = df["year"].map(safe_str)

    # 用于概念归并的主文本
    text_parts = [
        df["var_name_std"],
        df["var_label_std"],
        df["question_text_std"],
        df["value_labels_std"],
        df["var_type_std"],
    ]
    if INCLUDE_YEAR_IN_TEXT:
        text_parts.append(df["year_std"])

    df["text_for_grouping"] = (
        text_parts[0].fillna("") + " | " +
        text_parts[1].fillna("") + " | " +
        text_parts[2].fillna("") + " | " +
        text_parts[3].fillna("") + " | " +
        text_parts[4].fillna("")
    )

    if INCLUDE_YEAR_IN_TEXT:
        df["text_for_grouping"] = df["text_for_grouping"] + " | " + df["year_std"].fillna("")

    # 预分组键：优先 module，其次 source_file，再退化为 all
    if USE_PRE_GROUPING:
        df["pre_group"] = df["module_std"].replace("", np.nan)
        df["pre_group"] = df["pre_group"].fillna(df["source_file_std"].replace("", np.nan))
        df["pre_group"] = df["pre_group"].fillna("all_variables")
    else:
        df["pre_group"] = "all_variables"

    # 附加一个粗略主题前缀，可帮助后续人工审查
    df["name_prefix"] = df["var_name_std"].map(extract_name_prefix)

    return df


def extract_name_prefix(name: str) -> str:
    if not name:
        return ""
    m = re.match(r"([a-zA-Z]+)", name)
    if m:
        return m.group(1)
    m2 = re.match(r"([\u4e00-\u9fa5]{1,4})", name)
    if m2:
        return m2.group(1)
    return name[:4]


# =========================
# 4. 聚类核心函数
# =========================
def cluster_one_group(
    gdf: pd.DataFrame,
    distance_threshold: float = 0.42
) -> pd.DataFrame:
    """
    对单个预分组内的变量做聚类。
    """
    gdf = gdf.copy().reset_index(drop=True)

    if len(gdf) == 1:
        gdf["cluster_local"] = 1
        gdf["cluster_confidence_local"] = 1.0
        return gdf

    texts = gdf["text_for_grouping"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        max_features=TFIDF_MAX_FEATURES
    )
    X = vectorizer.fit_transform(texts)

    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 1.0)

    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 1.0)

    # scipy linkage 需要 condensed distance matrix
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    gdf["cluster_local"] = labels

    # 计算每个样本在簇内的平均相似度，作为局部置信度
    cluster_conf = []
    for idx in range(len(gdf)):
        cid = labels[idx]
        members = np.where(labels == cid)[0]
        if len(members) == 1:
            cluster_conf.append(1.0)
        else:
            avg_sim = float(sim[idx, members].mean())
            cluster_conf.append(round(avg_sim, 6))

    gdf["cluster_confidence_local"] = cluster_conf
    return gdf


# =========================
# 5. 代表变量、簇标签与汇总
# =========================
def choose_anchor_var(cdf: pd.DataFrame) -> pd.Series:
    """
    为簇选择代表变量。
    优先规则：
    1. 标签较完整
    2. 文本长度较长
    3. 年份较新
    """
    temp = cdf.copy()

    temp["label_len"] = temp["var_label_std"].fillna("").map(len)
    temp["text_len"] = temp["text_for_grouping"].fillna("").map(len)

    def parse_year(y):
        try:
            return int(float(str(y)))
        except Exception:
            return -9999

    temp["year_num"] = temp["year_std"].map(parse_year)

    temp = temp.sort_values(
        by=["label_len", "text_len", "year_num"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return temp.iloc[0]


def infer_concept_label(cdf: pd.DataFrame, anchor_row: pd.Series) -> str:
    """
    给簇生成概念标签。
    优先使用 anchor 的变量标签，否则退回变量名。
    """
    label = safe_str(anchor_row.get("var_label", ""))
    var_name = safe_str(anchor_row.get("var_name", ""))
    year = safe_str(anchor_row.get("year", ""))

    label = clean_label_text(label)
    if not label:
        label = var_name

    if INCLUDE_ANCHOR_YEAR_IN_LABEL and year:
        return f"{label}（anchor:{year}）"
    return label


def summarize_cluster(cdf: pd.DataFrame, cluster_id: str) -> dict:
    anchor = choose_anchor_var(cdf)
    concept_label = infer_concept_label(cdf, anchor)

    years = sorted([safe_str(y) for y in cdf["year"].tolist() if safe_str(y)])
    years_covered = ", ".join(sorted(set(years)))

    member_vars = [
        f"{safe_str(r['var_name'])}[{safe_str(r['year'])}]"
        for _, r in cdf.iterrows()
    ]
    member_vars = sorted(member_vars)

    avg_conf = float(cdf["cluster_confidence_local"].mean()) if "cluster_confidence_local" in cdf.columns else np.nan

    return {
        "cluster_id": cluster_id,
        "cluster_size": len(cdf),
        "anchor_var": safe_str(anchor.get("var_name", "")),
        "anchor_year": safe_str(anchor.get("year", "")),
        "anchor_label": safe_str(anchor.get("var_label", "")),
        "concept_label": concept_label,
        "pre_group": safe_str(anchor.get("pre_group", "")),
        "name_prefix_mode": most_common_nonempty(cdf["name_prefix"].tolist()),
        "years_covered": years_covered,
        "member_vars": " | ".join(member_vars),
        "avg_cluster_confidence": round(avg_conf, 6) if pd.notna(avg_conf) else np.nan
    }


def most_common_nonempty(values: List[str]) -> str:
    values = [v for v in values if safe_str(v)]
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


# =========================
# 6. 主流程
# =========================
def run_clustering_pipeline(input_file: Path):
    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件：{input_file}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    print_df_info(df, "原始输入文件信息")

    colmap = detect_columns(df)
    print("=" * 80)
    print("自动识别字段：")
    print(json.dumps(colmap, ensure_ascii=False, indent=2))

    if colmap["var_name"] is None:
        raise ValueError("无法识别变量名字段，请检查输入文件列名。")
    if colmap["var_label"] is None:
        print("[提示] 未识别到变量标签字段，将仅依赖变量名等信息。")
    if colmap["year"] is None:
        print("[提示] 未识别到年份字段。")
    if colmap["source_file"] is None and colmap["module"] is None:
        print("[提示] 未识别到 source_file/module，预分组会退化为 all_variables。")

    df = build_text_for_grouping(df, colmap)

    # 保留原字段的标准名副本，方便后续输出
    keep_base_cols = [
        "var_name", "var_label", "year", "source_file", "module",
        "question_text", "value_labels", "var_type",
        "var_name_std", "var_label_std", "source_file_std", "module_std",
        "question_text_std", "value_labels_std", "var_type_std",
        "year_std", "text_for_grouping", "pre_group", "name_prefix"
    ]
    for c in keep_base_cols:
        ensure_column(df, c, "")

    results = []
    summary_records = []

    pre_groups = df["pre_group"].fillna("all_variables").astype(str).unique().tolist()
    print("=" * 80)
    print(f"预分组数量：{len(pre_groups)}")

    global_cluster_counter = 0

    for g in pre_groups:
        gdf = df[df["pre_group"] == g].copy()

        if len(gdf) == 0:
            continue

        if len(gdf) > MAX_ROWS_PER_PREGROUP:
            print(f"[警告] 预分组 {g} 含 {len(gdf):,} 行，超过上限 {MAX_ROWS_PER_PREGROUP}，将截断处理。")
            gdf = gdf.head(MAX_ROWS_PER_PREGROUP).copy()

        print("-" * 80)
        print(f"[开始] 预分组：{g}，样本数：{len(gdf):,}")

        if len(gdf) < MIN_GROUP_SIZE_TO_CLUSTER:
            gdf["cluster_local"] = 1
            gdf["cluster_confidence_local"] = 1.0
        else:
            gdf = cluster_one_group(gdf, distance_threshold=DISTANCE_THRESHOLD)

        local_clusters = sorted(gdf["cluster_local"].unique().tolist())

        for lc in local_clusters:
            cdf = gdf[gdf["cluster_local"] == lc].copy()
            global_cluster_counter += 1
            cluster_id = f"C{global_cluster_counter:06d}"

            anchor = choose_anchor_var(cdf)
            concept_label = infer_concept_label(cdf, anchor)

            cdf["cluster_id"] = cluster_id
            cdf["anchor_var"] = safe_str(anchor.get("var_name", ""))
            cdf["anchor_year"] = safe_str(anchor.get("year", ""))
            cdf["concept_label"] = concept_label
            cdf["cluster_size"] = len(cdf)
            cdf["cluster_confidence"] = cdf["cluster_confidence_local"]

            results.append(cdf)

            summary_records.append(summarize_cluster(cdf, cluster_id))

        print(f"[完成] 预分组：{g}，生成簇数：{len(local_clusters)}")

        del gdf
        gc.collect()

    if not results:
        raise ValueError("未生成任何聚类结果，请检查输入数据或参数设置。")

    final_map = pd.concat(results, ignore_index=True)
    cluster_summary = pd.DataFrame(summary_records)

    # 排序
    final_map = final_map.sort_values(
        by=["pre_group", "cluster_id", "year_std", "var_name_std"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    cluster_summary = cluster_summary.sort_values(
        by=["pre_group", "cluster_size", "avg_cluster_confidence"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    # 输出变量映射表
    var_map_cols = [
        "cluster_id", "concept_label", "anchor_var", "anchor_year",
        "cluster_size", "cluster_confidence", "pre_group",
        "var_name", "var_label", "year", "source_file", "module",
        "question_text", "value_labels", "var_type",
        "name_prefix", "text_for_grouping"
    ]
    var_map_cols = [c for c in var_map_cols if c in final_map.columns]
    final_map[var_map_cols].to_csv(OUTPUT_VAR_MAP, index=False, encoding="utf-8-sig")

    # 输出簇汇总表
    summary_cols = [
        "cluster_id", "concept_label", "cluster_size", "avg_cluster_confidence",
        "anchor_var", "anchor_year", "anchor_label",
        "pre_group", "name_prefix_mode", "years_covered", "member_vars"
    ]
    summary_cols = [c for c in summary_cols if c in cluster_summary.columns]
    cluster_summary[summary_cols].to_csv(OUTPUT_CLUSTER_SUMMARY, index=False, encoding="utf-8-sig")

    # 输出 Excel 审核模板
    with pd.ExcelWriter(OUTPUT_REVIEW_XLSX, engine="openpyxl") as writer:
        cluster_summary[summary_cols].to_excel(writer, sheet_name="cluster_summary", index=False)
        final_map[var_map_cols].to_excel(writer, sheet_name="variable_cluster_map", index=False)

        # 低置信度簇单独导出，便于人工复核
        low_conf_clusters = cluster_summary[
            cluster_summary["avg_cluster_confidence"].fillna(0) < 0.60
        ].copy()
        low_conf_clusters.to_excel(writer, sheet_name="low_conf_clusters", index=False)

        large_clusters = cluster_summary[
            cluster_summary["cluster_size"] >= 8
        ].copy()
        large_clusters.to_excel(writer, sheet_name="large_clusters", index=False)

    print("=" * 80)
    print("[完成] 输出文件：")
    print(f"1. {OUTPUT_VAR_MAP}")
    print(f"2. {OUTPUT_CLUSTER_SUMMARY}")
    print(f"3. {OUTPUT_REVIEW_XLSX}")
    print("=" * 80)
    print(f"变量总数：{len(final_map):,}")
    print(f"概念簇总数：{cluster_summary['cluster_id'].nunique():,}")
    print(f"平均簇大小：{round(cluster_summary['cluster_size'].mean(), 3)}")


if __name__ == "__main__":
    run_clustering_pipeline(INPUT_FILE)