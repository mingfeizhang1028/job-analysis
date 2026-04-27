from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import pandas as pd


# 纯 ASCII token，用于英文/数字关键词边界保护
_ASCII_TOKEN_RE = re.compile(r"^[A-Za-z0-9_+#.\-]+$")
_ASCII_BOUNDARY_TEMPLATE = r"(?<![A-Za-z0-9_]){token}(?![A-Za-z0-9_])"


def to_safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value).strip()


def _is_ascii_token(text: str) -> bool:
    return bool(_ASCII_TOKEN_RE.fullmatch(text or ""))


@lru_cache(maxsize=4096)
def _build_ascii_pattern(token_lower: str) -> re.Pattern:
    pattern = _ASCII_BOUNDARY_TEMPLATE.format(token=re.escape(token_lower))
    return re.compile(pattern)


def _count_keyword_hits(text: str, keyword: str) -> int:
    """
    统计 keyword 在 text 中的命中次数。

    规则：
    - 中文或混合中文：普通 substring count
    - 纯 ASCII token：使用 token 边界保护，避免 AI 命中 paid
    """
    source = to_safe_str(text).lower()
    kw = to_safe_str(keyword).lower()

    if not source or not kw:
        return 0

    if _is_ascii_token(kw):
        return len(_build_ascii_pattern(kw).findall(source))

    return source.count(kw)


def _normalize_trait_dict(trait_dict: dict) -> dict[str, dict[str, list[str]]]:
    """
    规范化 trait_dict 结构为：
    {
      "一级标签": {
        "二级标签": ["关键词1", "关键词2"]
      }
    }

    非法结构会被尽量跳过，而不是报错。
    """
    if not isinstance(trait_dict, dict):
        return {}

    normalized: dict[str, dict[str, list[str]]] = {}

    for lvl1, lvl2_dict in trait_dict.items():
        lvl1_str = to_safe_str(lvl1)
        if not lvl1_str or not isinstance(lvl2_dict, dict):
            continue

        normalized[lvl1_str] = {}

        for lvl2, keywords in lvl2_dict.items():
            lvl2_str = to_safe_str(lvl2)
            if not lvl2_str:
                continue

            if isinstance(keywords, str):
                keywords = [keywords]
            elif not isinstance(keywords, (list, tuple, set)):
                continue

            cleaned_keywords = []
            seen = set()

            for kw in keywords:
                kw_str = to_safe_str(kw)
                if not kw_str:
                    continue
                kw_lower = kw_str.lower()
                if kw_lower not in seen:
                    cleaned_keywords.append(kw_str)
                    seen.add(kw_lower)

            if cleaned_keywords:
                normalized[lvl1_str][lvl2_str] = cleaned_keywords

        if not normalized[lvl1_str]:
            normalized.pop(lvl1_str, None)

    return normalized


def extract_traits_from_text(text: str, trait_dict: dict) -> list[dict[str, Any]]:
    """
    从文本中抽取 trait 命中结果。

    返回格式：
    [
      {
        "一级标签": "...",
        "二级标签": "...",
        "命中次数": 3
      }
    ]
    """
    source = to_safe_str(text)
    if not source:
        return []

    normalized_trait_dict = _normalize_trait_dict(trait_dict)
    if not normalized_trait_dict:
        return []

    matched: list[dict[str, Any]] = []

    for lvl1, lvl2_dict in normalized_trait_dict.items():
        for lvl2, keywords in lvl2_dict.items():
            hit_count = 0
            for kw in keywords:
                hit_count += _count_keyword_hits(source, kw)

            if hit_count > 0:
                matched.append(
                    {
                        "一级标签": lvl1,
                        "二级标签": lvl2,
                        "命中次数": int(hit_count),
                    }
                )

    return matched


def build_trait_table(
    df: pd.DataFrame,
    trait_dict: dict,
    *,
    text_col: str = "岗位详情_clean",
    job_id_col: str = "job_id",
) -> pd.DataFrame:
    """
    从岗位 DataFrame 构建 trait 明细表。

    输出字段默认包括：
    - job_id
    - 职位名称
    - 企业名称
    - 所在地区
    - 经验要求
    - 学历要求
    - 一级标签
    - 二级标签
    - 命中次数
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(
            columns=[
                "job_id", "职位名称", "企业名称", "所在地区", "经验要求", "学历要求",
                "一级标签", "二级标签", "命中次数",
            ]
        )

    normalized_trait_dict = _normalize_trait_dict(trait_dict)
    if not normalized_trait_dict:
        return pd.DataFrame(
            columns=[
                "job_id", "职位名称", "企业名称", "所在地区", "经验要求", "学历要求",
                "一级标签", "二级标签", "命中次数",
            ]
        )

    rows = []

    for idx, row in df.iterrows():
        job_id = row.get(job_id_col, idx)
        text = row.get(text_col, "")

        traits = extract_traits_from_text(text, normalized_trait_dict)
        if not traits:
            continue

        for item in traits:
            rows.append(
                {
                    "job_id": job_id,
                    "职位名称": row.get("职位名称", ""),
                    "企业名称": row.get("企业名称", ""),
                    "所在地区": row.get("所在地区", ""),
                    "经验要求": row.get("经验要求", ""),
                    "学历要求": row.get("学历要求", ""),
                    "一级标签": item["一级标签"],
                    "二级标签": item["二级标签"],
                    "命中次数": item["命中次数"],
                }
            )

    return pd.DataFrame(rows)


def summarize_traits(trait_df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总一级标签统计。

    输出字段：
    - 一级标签
    - 覆盖岗位数
    - 总命中次数
    """
    if not isinstance(trait_df, pd.DataFrame) or trait_df.empty:
        return pd.DataFrame(columns=["一级标签", "覆盖岗位数", "总命中次数"])

    required_cols = {"一级标签", "job_id", "命中次数"}
    if not required_cols.issubset(trait_df.columns):
        return pd.DataFrame(columns=["一级标签", "覆盖岗位数", "总命中次数"])

    summary = (
        trait_df.groupby("一级标签")
        .agg(
            覆盖岗位数=("job_id", "nunique"),
            总命中次数=("命中次数", "sum"),
        )
        .reset_index()
    )

    return summary.sort_values(
        ["覆盖岗位数", "总命中次数", "一级标签"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def trait_heatmap_by_region(trait_df: pd.DataFrame) -> pd.DataFrame:
    """
    按地区生成 trait 一级标签热力透视表。
    """
    if not isinstance(trait_df, pd.DataFrame) or trait_df.empty:
        return pd.DataFrame()

    required_cols = {"所在地区", "一级标签", "job_id"}
    if not required_cols.issubset(trait_df.columns):
        return pd.DataFrame()

    pivot = trait_df.pivot_table(
        index="所在地区",
        columns="一级标签",
        values="job_id",
        aggfunc="nunique",
        fill_value=0,
    )

    return pivot.reset_index()


def trait_heatmap_by_company(trait_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    按企业生成 trait 一级标签热力透视表。
    默认仅保留岗位数最多的 top_n 家企业。
    """
    if not isinstance(trait_df, pd.DataFrame) or trait_df.empty:
        return pd.DataFrame()

    required_cols = {"企业名称", "一级标签", "job_id"}
    if not required_cols.issubset(trait_df.columns):
        return pd.DataFrame()

    top_n = max(int(top_n), 1)

    top_companies = (
        trait_df["企业名称"]
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )

    temp = trait_df[trait_df["企业名称"].isin(top_companies)].copy()
    if temp.empty:
        return pd.DataFrame()

    pivot = temp.pivot_table(
        index="企业名称",
        columns="一级标签",
        values="job_id",
        aggfunc="nunique",
        fill_value=0,
    )

    return pivot.reset_index()
