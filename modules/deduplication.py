"""
[MODULE_SPEC]
module_id: modules.deduplication
module_path: modules/deduplication.py
module_name: 招聘岗位去重模块
module_type: data_deduplication
layer: 数据处理层

responsibility:
  - 负责识别和移除重复招聘岗位记录。
  - 降低重复抓取、重复发布、相似岗位重复计数对统计分析的影响。
  - 为后续标签分析、薪资分析、公司分析和关系网络分析提供更接近真实岗位结构的数据。

notes:
  - 精确去重优先依赖预处理后的标准化字段。
  - 模糊去重仅作为补充，不应替代 preprocessing.py 的字段标准化职责。
[/MODULE_SPEC]
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DedupMode = Literal["conservative", "standard", "aggressive"]
KeepStrategy = Literal["longest_detail", "latest_scrape_time", "latest_then_longest"]


def clean_text_for_similarity(text: str) -> str:
    """
    清理文本，供相似度计算使用。
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def assign_job_id(df: pd.DataFrame, job_id_col: str = "job_id") -> pd.DataFrame:
    """
    若不存在 job_id，则补充稳定可追踪的 job_id。
    """
    result = df.copy()
    if job_id_col not in result.columns:
        result[job_id_col] = [f"job_{i}" for i in range(len(result))]
    else:
        result[job_id_col] = result[job_id_col].fillna("").astype(str)
        empty_mask = result[job_id_col].str.strip().eq("")
        if empty_mask.any():
            result.loc[empty_mask, job_id_col] = [f"job_{i}" for i in result.index[empty_mask]]
    return result


def _normalize_key_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    对作为去重 key 的列做轻量字符串规整。
    """
    result = df.copy()
    for col in cols:
        if col in result.columns:
            result[col] = result[col].fillna("").astype(str).str.strip()
    return result


def _validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> tuple[bool, list[str]]:
    if not isinstance(df, pd.DataFrame):
        return False, required_cols
    missing = [col for col in required_cols if col not in df.columns]
    return len(missing) == 0, missing


def choose_keep_record(
    group_df: pd.DataFrame,
    *,
    detail_col: str = "岗位详情",
    scrape_time_col: str = "抓取时间",
    keep_strategy: KeepStrategy = "latest_then_longest",
) -> int:
    """
    在重复组内选择保留记录的 index。

    策略：
    - longest_detail: 保留岗位详情最长的
    - latest_scrape_time: 保留抓取时间最新的
    - latest_then_longest: 优先抓取时间最新，再按详情长度兜底
    """
    if group_df.empty:
        raise ValueError("group_df 不能为空")

    temp = group_df.copy()

    if detail_col not in temp.columns:
        temp[detail_col] = ""
    detail_lengths = temp[detail_col].fillna("").astype(str).str.len()

    if keep_strategy == "longest_detail":
        return detail_lengths.idxmax()

    if scrape_time_col in temp.columns:
        temp["_scrape_dt"] = pd.to_datetime(temp[scrape_time_col], errors="coerce")
    else:
        temp["_scrape_dt"] = pd.NaT

    if keep_strategy == "latest_scrape_time":
        if temp["_scrape_dt"].notna().any():
            return temp["_scrape_dt"].idxmax()
        return detail_lengths.idxmax()

    # default: latest_then_longest
    if temp["_scrape_dt"].notna().any():
        latest_time = temp["_scrape_dt"].max()
        latest_rows = temp[temp["_scrape_dt"] == latest_time]
        if len(latest_rows) == 1:
            return latest_rows.index[0]
        latest_lengths = latest_rows[detail_col].fillna("").astype(str).str.len()
        return latest_lengths.idxmax()

    return detail_lengths.idxmax()


def _init_dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["duplicate_group_id"] = None
    result["is_duplicate"] = False
    result["duplicate_keep"] = True
    result["duplicate_reason"] = ""
    result["max_similarity_in_group"] = 0.0
    result["dedup_method"] = ""
    return result


def deduplicate_exact_jobs(
    df: pd.DataFrame,
    *,
    subset: list[str] | None = None,
    job_id_col: str = "job_id",
    keep_strategy: KeepStrategy = "latest_then_longest",
    detail_col: str = "岗位详情",
    scrape_time_col: str = "抓取时间",
) -> pd.DataFrame:
    """
    基于若干字段完全相同进行精确去重。

    默认优先使用：
    - 企业名称_norm
    - 职位名称_norm
    - 所在地区
    """
    if subset is None:
        subset = ["企业名称_norm", "职位名称_norm", "所在地区"]

    ok, missing = _validate_required_columns(df, subset)
    if not ok:
        raise ValueError(f"精确去重缺少必要字段: {missing}")

    result = assign_job_id(df, job_id_col=job_id_col)
    result = _normalize_key_columns(result, subset)
    result = _init_dedup_columns(result)

    group_counter = 1
    for _, group_df in result.groupby(subset, dropna=False):
        if len(group_df) <= 1:
            continue

        keep_idx = choose_keep_record(
            group_df,
            detail_col=detail_col,
            scrape_time_col=scrape_time_col,
            keep_strategy=keep_strategy,
        )
        group_id = f"EXACT_{group_counter:04d}"
        group_counter += 1

        for idx in group_df.index:
            result.at[idx, "duplicate_group_id"] = group_id
            result.at[idx, "duplicate_reason"] = f"精确去重: {' + '.join(subset)} 完全一致"
            result.at[idx, "dedup_method"] = "exact"
            if idx == keep_idx:
                result.at[idx, "duplicate_keep"] = True
                result.at[idx, "is_duplicate"] = False
            else:
                result.at[idx, "duplicate_keep"] = False
                result.at[idx, "is_duplicate"] = True

    return result


def _connected_components_from_similarity(sim_matrix, threshold: float) -> list[list[int]]:
    """
    根据相似度矩阵构建连通分量。
    """
    n = sim_matrix.shape[0]
    visited = [False] * n
    components = []

    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                graph[i].append(j)
                graph[j].append(i)

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True

        while stack:
            node = stack.pop()
            comp.append(node)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        components.append(comp)

    return components


def deduplicate_similar_jobs(
    df: pd.DataFrame,
    *,
    company_col: str = "企业名称_norm",
    title_col: str = "职位名称_norm",
    detail_col: str = "岗位详情",
    threshold: float = 0.90,
    job_id_col: str = "job_id",
    keep_strategy: KeepStrategy = "latest_then_longest",
    scrape_time_col: str = "抓取时间",
) -> pd.DataFrame:
    """
    对“同公司 + 同职位标准名”内部做岗位详情相似度去重。

    输出新增字段：
    - duplicate_group_id
    - is_duplicate
    - duplicate_keep
    - duplicate_reason
    - max_similarity_in_group
    - dedup_method
    """
    required_cols = [company_col, title_col, detail_col]
    ok, missing = _validate_required_columns(df, required_cols)
    if not ok:
        raise ValueError(f"模糊去重缺少必要字段: {missing}")

    result = assign_job_id(df, job_id_col=job_id_col)
    result = _normalize_key_columns(result, [company_col, title_col])
    result = _init_dedup_columns(result)

    group_counter = 1
    group_cols = [company_col, title_col]

    for _, sub in result.groupby(group_cols, dropna=False):
        if len(sub) <= 1:
            continue

        texts = sub[detail_col].fillna("").astype(str).apply(clean_text_for_similarity).tolist()
        if all(len(t) == 0 for t in texts):
            continue

        try:
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
            X = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(X)
        except Exception:
            continue

        components = _connected_components_from_similarity(sim_matrix, threshold=threshold)
        indices = list(sub.index)

        for comp in components:
            if len(comp) <= 1:
                continue

            real_indices = [indices[i] for i in comp]
            dup_group = result.loc[real_indices]

            keep_idx = choose_keep_record(
                dup_group,
                detail_col=detail_col,
                scrape_time_col=scrape_time_col,
                keep_strategy=keep_strategy,
            )

            max_sim = 0.0
            if len(comp) >= 2:
                for i in comp:
                    for j in comp:
                        if i < j:
                            max_sim = max(max_sim, float(sim_matrix[i, j]))

            group_id = f"SIM_{group_counter:04d}"
            group_counter += 1

            for idx in real_indices:
                result.at[idx, "duplicate_group_id"] = group_id
                result.at[idx, "max_similarity_in_group"] = max_sim
                result.at[idx, "duplicate_reason"] = f"同公司+同职位，岗位详情相似度 >= {threshold}"
                result.at[idx, "dedup_method"] = "similarity"
                if idx == keep_idx:
                    result.at[idx, "duplicate_keep"] = True
                    result.at[idx, "is_duplicate"] = False
                else:
                    result.at[idx, "duplicate_keep"] = False
                    result.at[idx, "is_duplicate"] = True

    return result


def run_deduplication(
    df: pd.DataFrame,
    *,
    mode: DedupMode = "standard",
    exact_subset: list[str] | None = None,
    threshold: float | None = None,
    job_id_col: str = "job_id",
) -> pd.DataFrame:
    """
    统一去重入口。

    mode:
    - conservative:
        仅精确去重
    - standard:
        先精确去重，再做同公司+同职位下的详情相似度去重（默认 threshold=0.90）
    - aggressive:
        先精确去重，再做更激进的相似度去重（默认 threshold=0.85）
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_deduplication expects a pandas.DataFrame")

    if mode not in {"conservative", "standard", "aggressive"}:
        raise ValueError(f"不支持的去重模式: {mode}")

    exact_df = deduplicate_exact_jobs(
        df,
        subset=exact_subset or ["企业名称_norm", "职位名称_norm", "所在地区"],
        job_id_col=job_id_col,
    )
    exact_kept = get_deduped_df(exact_df)

    if mode == "conservative":
        return exact_df

    if threshold is None:
        threshold = 0.90 if mode == "standard" else 0.85

    sim_df = deduplicate_similar_jobs(
        exact_kept,
        threshold=threshold,
        job_id_col=job_id_col,
    )
    return sim_df


def get_deduped_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取去重后保留的记录。
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if "duplicate_keep" not in df.columns:
        return df.copy()

    return df[df["duplicate_keep"] == True].copy()


def get_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取被判定为重复的记录。
    """
    if not isinstance(df, pd.DataFrame) or "is_duplicate" not in df.columns:
        return pd.DataFrame()

    return df[df["is_duplicate"] == True].copy()


def duplicate_summary(df: pd.DataFrame) -> dict:
    """
    返回去重摘要信息。
    """
    if not isinstance(df, pd.DataFrame):
        return {
            "原始岗位数": 0,
            "疑似重复组数": 0,
            "疑似重复记录数": 0,
            "去重后岗位数": 0,
        }

    total = len(df)

    if "duplicate_group_id" not in df.columns:
        return {
            "原始岗位数": total,
            "疑似重复组数": 0,
            "疑似重复记录数": 0,
            "去重后岗位数": total,
        }

    dup_groups = int(df["duplicate_group_id"].dropna().nunique())

    if "is_duplicate" in df.columns:
        dup_records = int(df["is_duplicate"].fillna(False).astype(bool).sum())
    else:
        dup_records = 0

    if "duplicate_keep" in df.columns:
        keep_records = int(df["duplicate_keep"].fillna(True).astype(bool).sum())
    else:
        keep_records = total

    return {
        "原始岗位数": total,
        "疑似重复组数": dup_groups,
        "疑似重复记录数": dup_records,
        "去重后岗位数": keep_records,
    }
