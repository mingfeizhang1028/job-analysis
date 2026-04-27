"""
[MODULE_SPEC]
module_id: modules.network_analysis
module_path: modules/network_analysis.py
module_name: 多维关系网络构建与图分析模块
module_type: network_analysis
layer: 分析计算层

responsibility:
  - 基于招聘岗位数据构建多维关系网络。
  - 支持标签共现网络、岗位相似网络、公司-标签二部图。
  - 提供网络摘要、核心节点、强关联边、社区发现、节点/边反查岗位等图分析能力。
  - 本模块负责“算图”和“分析图”，不负责最终图形渲染。

notes:
  - 图渲染应由 modules.network_viz.py 或 app.py 负责。
  - 标签提取与标准化应由 preprocessing.py / tag_extraction.py 负责。
[/MODULE_SPEC]
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import log1p
from typing import Literal
import json
import re

import networkx as nx
import pandas as pd


NetworkType = Literal["标签共现网络", "岗位相似网络", "公司能力画像网络", "岗位-标签洞察网络"]
DimensionType = Literal["技术关系", "素质关系", "业务职责关系", "行业场景关系", "综合标签关系"]


TAG_DIMENSION_MAP: dict[str, str] = {
    "技术关系": "网络_技术标签",
    "素质关系": "网络_素质标签",
    "业务职责关系": "网络_业务职责标签",
    "行业场景关系": "网络_行业场景标签",
    "综合标签关系": "网络_综合标签",
}

TAG_DIMENSION_FALLBACK_MAP: dict[str, str] = {
    "技术关系": "硬技能标签",
    "素质关系": "软素质标签",
    "业务职责关系": "业务职责标签",
    "行业场景关系": "行业场景标签",
    "综合标签关系": "全部标签",
}

DIMENSION_COLOR_MAP: dict[str, str] = {
    "技术关系": "#4C78A8",
    "素质关系": "#F58518",
    "业务职责关系": "#54A24B",
    "行业场景关系": "#B279A2",
    "综合标签关系": "#E45756",
    "公司能力画像": "#72B7B2",
}

DEFAULT_DIMENSION = "综合标签关系"
DEFAULT_TAG_COL = "全部标签"

TAG_COLUMN_KEYWORDS = [
    "标签",
    "技能",
    "能力",
    "职责",
    "场景",
    "行业",
    "要求",
    "加分",
    "工具栈",
    "工作内容",
    "岗位类型",
    "LLM",
    "skill",
    "tag",
    "responsibility",
    "requirement",
]

MANUAL_NETWORK_STOP_TAGS = {"LLM", "大模型"}
CROSS_DIMENSION_STOP_TAGS = {"原型设计"}

TAG_TEXT_SOURCE_CANDIDATES = [
    "职位名称_norm",
    "职位名称_raw",
    "岗位详情",
    "岗位职责",
    "职位描述",
    "职位诱惑",
    "任职要求",
]


def _is_meaningful_tag_column_name(col: str) -> bool:
    text = str(col)
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in TAG_COLUMN_KEYWORDS)


def _sample_non_empty_value(series: pd.Series) -> str:
    for value in series.tolist():
        tags = _safe_tags(value)
        if tags:
            return "、".join(tags[:8])
        if value is not None:
            raw = str(value).strip()
            if raw and raw.lower() not in {"nan", "none", "[]", "{}"}:
                return raw[:120]
    return ""


def inspect_candidate_tag_columns(df: pd.DataFrame, include_text_sources: bool = True) -> pd.DataFrame:
    """扫描当前数据中所有疑似标签/技能/职责字段，辅助确认网络构图来源。"""
    columns = ["列名", "非空行数", "可解析标签行数", "非空覆盖率", "唯一标签数", "标签总数", "样例值", "建议"]
    if not _is_valid_df(df):
        return pd.DataFrame(columns=columns)

    total_rows = max(len(df), 1)
    rows = []
    for col in df.columns:
        is_candidate = _is_meaningful_tag_column_name(col) or (include_text_sources and col in TAG_TEXT_SOURCE_CANDIDATES)
        if not is_candidate:
            continue

        series = df[col]
        raw_non_empty = 0
        parsed_non_empty = 0
        unique_tags: set[str] = set()
        total_tags = 0
        for value in series.tolist():
            raw = "" if value is None else str(value).strip()
            if raw and raw.lower() not in {"nan", "none", "[]", "{}"}:
                raw_non_empty += 1
            tags = _safe_tags(value)
            if tags:
                parsed_non_empty += 1
                total_tags += len(tags)
                unique_tags.update(tags)

        if parsed_non_empty > 0:
            suggestion = "可直接作为网络标签列候选"
        elif raw_non_empty > 0 and col in TAG_TEXT_SOURCE_CANDIDATES:
            suggestion = "文本源字段：可用于重新运行词典标签抽取"
        elif raw_non_empty > 0:
            suggestion = "有内容但未解析出标签，可能是长文本/结构化字符串，需检查格式"
        else:
            suggestion = "当前筛选范围为空"

        rows.append({
            "列名": col,
            "非空行数": raw_non_empty,
            "可解析标签行数": parsed_non_empty,
            "非空覆盖率": raw_non_empty / total_rows,
            "唯一标签数": len(unique_tags),
            "标签总数": total_tags,
            "样例值": _sample_non_empty_value(series),
            "建议": suggestion,
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["可解析标签行数", "唯一标签数", "非空行数"],
        ascending=False,
    ).reset_index(drop=True)


def find_best_available_tag_col(
    df: pd.DataFrame,
    dimension: str = DEFAULT_DIMENSION,
) -> tuple[str | None, dict[str, object]]:
    """
    根据当前数据实际可用情况，为指定维度选择最合适的标签列。

    返回:
    - chosen_col: 选中的列名；如果完全没有可用标签列则返回 None
    - diagnostics: 诊断信息，供页面提示使用
    """
    diagnostics: dict[str, object] = {
        "dimension": dimension,
        "preferred_candidates": [],
        "all_candidates": [],
        "column_stats": [],
        "chosen_col": None,
    }

    if not _is_valid_df(df):
        return None, diagnostics

    preferred = []
    preferred_col = TAG_DIMENSION_MAP.get(dimension)
    fallback_col = TAG_DIMENSION_FALLBACK_MAP.get(dimension)
    if preferred_col:
        preferred.append(preferred_col)
    if fallback_col and fallback_col not in preferred:
        preferred.append(fallback_col)

    common_backup = [
        "最终全部标签",
        "全部标签",
        "最终硬技能标签",
        "硬技能标签",
        "最终业务职责标签",
        "业务职责标签",
        "最终软素质标签",
        "软素质标签",
        "最终行业场景标签",
        "行业场景标签",
    ]

    candidates: list[str] = []
    for col in preferred + common_backup:
        if col in df.columns and col not in candidates:
            candidates.append(col)

    diagnostics["preferred_candidates"] = preferred
    diagnostics["all_candidates"] = candidates

    best_col = None
    best_score = (-1, -1)
    total_rows = len(df)

    for col in candidates:
        non_empty_rows = 0
        unique_tags: set[str] = set()
        total_tags = 0
        for value in df[col].tolist():
            tags = _safe_tags(value)
            if tags:
                non_empty_rows += 1
                total_tags += len(tags)
                unique_tags.update(tags)
        coverage = non_empty_rows / total_rows if total_rows else 0
        stat = {
            "column": col,
            "non_empty_rows": non_empty_rows,
            "coverage": coverage,
            "unique_tag_count": len(unique_tags),
            "total_tag_count": total_tags,
        }
        diagnostics["column_stats"].append(stat)
        score = (non_empty_rows, len(unique_tags))
        if score > best_score:
            best_score = score
            best_col = col

    diagnostics["chosen_col"] = best_col
    return best_col, diagnostics


def get_supported_dimensions() -> list[str]:
    return list(TAG_DIMENSION_MAP.keys())


def analyze_dimension_tag_overlap(
    df: pd.DataFrame,
    dimension_columns: dict[str, str] | None = None,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """分析不同维度标签的区分度：Top 标签、独有标签、跨维度交集标签。"""
    result = {
        "top_tags": pd.DataFrame(columns=["维度", "标签", "出现次数", "覆盖率"]),
        "unique_tags": pd.DataFrame(columns=["维度", "标签", "出现次数", "覆盖率"]),
        "overlap_tags": pd.DataFrame(columns=["标签", "出现维度数", "维度列表", "总出现次数"]),
        "tag_dimension_map": pd.DataFrame(columns=["标签", "出现维度数", "维度列表", "总出现次数"]),
    }
    if not _is_valid_df(df):
        return result

    dim_cols = dimension_columns or {
        "技术关系": "网络_技术标签",
        "素质关系": "网络_素质标签",
        "业务职责关系": "网络_业务职责标签",
        "行业场景关系": "网络_行业场景标签",
    }

    total_rows = max(len(df), 1)
    dim_counters: dict[str, Counter] = {}
    dim_coverages: dict[str, Counter] = {}
    tag_dims: defaultdict[str, set[str]] = defaultdict(set)
    tag_total_counter: Counter = Counter()

    for dim_name, col in dim_cols.items():
        if col not in df.columns:
            continue
        counter = Counter()
        coverage_counter = Counter()
        for value in df[col].tolist():
            tags = _safe_tags(value)
            if not tags:
                continue
            unique_tags = set(tags)
            counter.update(tags)
            coverage_counter.update(unique_tags)
            for tag in unique_tags:
                tag_dims[tag].add(dim_name)
                tag_total_counter[tag] += 1
        dim_counters[dim_name] = counter
        dim_coverages[dim_name] = coverage_counter

    top_rows = []
    unique_rows = []
    for dim_name, counter in dim_counters.items():
        coverage_counter = dim_coverages.get(dim_name, Counter())
        for tag, cnt in counter.most_common(top_n):
            top_rows.append({
                "维度": dim_name,
                "标签": tag,
                "出现次数": int(cnt),
                "覆盖率": coverage_counter.get(tag, 0) / total_rows,
            })
        unique_candidates = [
            (tag, cnt) for tag, cnt in counter.most_common()
            if len(tag_dims.get(tag, set())) == 1
        ]
        for tag, cnt in unique_candidates[:top_n]:
            unique_rows.append({
                "维度": dim_name,
                "标签": tag,
                "出现次数": int(cnt),
                "覆盖率": coverage_counter.get(tag, 0) / total_rows,
            })

    mapping_rows = []
    overlap_rows = []
    for tag, dims in tag_dims.items():
        row = {
            "标签": tag,
            "出现维度数": len(dims),
            "维度列表": "、".join(sorted(dims)),
            "总出现次数": int(tag_total_counter.get(tag, 0)),
        }
        mapping_rows.append(row)
        if len(dims) >= 2:
            overlap_rows.append(row)

    result["top_tags"] = pd.DataFrame(top_rows)
    result["unique_tags"] = pd.DataFrame(unique_rows)
    result["overlap_tags"] = pd.DataFrame(overlap_rows).sort_values(
        ["出现维度数", "总出现次数", "标签"],
        ascending=[False, False, True],
    ).reset_index(drop=True) if overlap_rows else result["overlap_tags"]
    result["tag_dimension_map"] = pd.DataFrame(mapping_rows).sort_values(
        ["出现维度数", "总出现次数", "标签"],
        ascending=[False, False, True],
    ).reset_index(drop=True) if mapping_rows else result["tag_dimension_map"]
    return result


def get_supported_network_types() -> list[str]:
    return ["标签共现网络", "岗位相似网络", "公司能力画像网络", "岗位-标签洞察网络"]


def _is_valid_df(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def _resolve_tag_col(dimension: str, *, strict: bool = False, df: pd.DataFrame | None = None) -> str:
    normalized = str(dimension).strip() if dimension is not None else DEFAULT_DIMENSION
    if normalized not in TAG_DIMENSION_MAP:
        if strict:
            raise ValueError(f"未知关系维度: {dimension}，支持: {get_supported_dimensions()}")
        normalized = DEFAULT_DIMENSION

    if df is not None:
        chosen_col, _ = find_best_available_tag_col(df, normalized)
        if chosen_col:
            return chosen_col

    preferred = TAG_DIMENSION_MAP.get(normalized, DEFAULT_TAG_COL)
    if df is not None and preferred not in df.columns:
        return TAG_DIMENSION_FALLBACK_MAP.get(normalized, preferred)
    return preferred


def _safe_tags(tags) -> list[str]:
    """
    将标签字段安全转换为 list[str]。

    支持：
    - list / tuple / set / numpy.ndarray / pd.Series
    - dict 嵌套结构
    - JSON 数组 / Python list 字符串
    - numpy / pandas 字符串化数组：['Docker' 'Python']
    - 常见字符串分隔：, ， 、 ; ； | / 换行
    """
    invalid_values = {"nan", "none", "[]", "{}", ""}

    def _clean_token(value) -> str:
        text = str(value).strip().strip("[](){}")
        text = text.strip(" -•·：:'\"")
        return text

    def _dedupe(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values:
            cleaned = _clean_token(item)
            if not cleaned:
                continue
            if cleaned.lower() in invalid_values:
                continue
            if cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
        return out

    if tags is None:
        return []

    try:
        if pd.isna(tags):
            return []
    except Exception:
        pass

    if isinstance(tags, dict):
        values = []
        for value in tags.values():
            values.extend(_safe_tags(value))
        return _dedupe(values)

    if isinstance(tags, pd.Series):
        values = []
        for item in tags.tolist():
            values.extend(_safe_tags(item))
        return _dedupe(values)

    ndarray_cls = getattr(pd, "array", None)
    if hasattr(tags, "tolist") and not isinstance(tags, (str, bytes, list, tuple, set, dict, pd.Series)):
        try:
            as_list = tags.tolist()
            return _safe_tags(as_list)
        except Exception:
            pass

    if isinstance(tags, (list, tuple, set)):
        values = []
        for item in tags:
            values.extend(_safe_tags(item) if isinstance(item, (list, tuple, set, dict, pd.Series)) or hasattr(item, "tolist") else [str(item).strip()])
        return _dedupe(values)

    if isinstance(tags, str):
        raw_text = tags.strip()
        if not raw_text or raw_text.lower() in invalid_values:
            return []

        if raw_text.startswith("[") and raw_text.endswith("]") and ("'" in raw_text or '"' in raw_text):
            quoted_tokens = re.findall(r"['\"]([^'\"]+)['\"]", raw_text)
            if quoted_tokens:
                return _dedupe(quoted_tokens)

        if (raw_text.startswith("[") and raw_text.endswith("]")) or (raw_text.startswith("{") and raw_text.endswith("}")):
            for loader in (json.loads,):
                try:
                    parsed = loader(raw_text)
                    parsed_tags = _safe_tags(parsed)
                    if parsed_tags:
                        return parsed_tags
                except Exception:
                    pass
            try:
                import ast
                parsed = ast.literal_eval(raw_text)
                parsed_tags = _safe_tags(parsed)
                if parsed_tags:
                    return parsed_tags
            except Exception:
                pass

        text = re.sub(r"^[\[\(\{]+|[\]\)\}]+$", "", raw_text)

        quoted_token_matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", raw_text)
        flattened_quoted = [a or b for a, b in quoted_token_matches if (a or b)]
        if flattened_quoted:
            return _dedupe(flattened_quoted)

        compact_text = text.replace("'", "").replace('"', "")

        if " " in compact_text and not any(sep in compact_text for sep in [",", "，", "、", ";", "；", "|", "\n", "\t"]):
            tokens = [x for x in re.split(r"\s{2,}|\s(?=[A-ZA-Za-z\u4e00-\u9fff])", compact_text) if x.strip()]
            parsed_tokens = _dedupe(tokens)
            if len(parsed_tokens) >= 2:
                return parsed_tokens

        normalized = compact_text
        for sep in ["，", ",", "、", ";", "；", "|", "\n", "\t"]:
            normalized = normalized.replace(sep, ",")
        if len(normalized) < 160:
            normalized = normalized.replace("/", ",")
        values = [x for x in normalized.split(",") if x.strip()]
        return _dedupe(values)

    return []


def normalize_tag_columns_inplace(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    stop_tags: set[str] | None = None,
) -> pd.DataFrame:
    """将常见标签列统一标准化为 list[str]，并可按 stop_tags 做统一过滤。"""
    if not _is_valid_df(df):
        return df

    target_columns = columns or [
        "全部标签",
        "最终全部标签",
        "硬技能标签",
        "最终硬技能标签",
        "软素质标签",
        "最终软素质标签",
        "行业场景标签",
        "最终行业场景标签",
        "业务职责标签",
        "最终业务职责标签",
    ]

    stop_tag_set = {str(x).strip() for x in (stop_tags or set()) if str(x).strip()}
    existing = [col for col in target_columns if col in df.columns]
    for col in existing:
        df[col] = df[col].apply(lambda value: [tag for tag in _safe_tags(value) if tag not in stop_tag_set])
    return df


def get_preferred_tag_columns() -> list[str]:
    return [
        "最终硬技能标签",
        "硬技能标签",
        "最终软素质标签",
        "软素质标签",
        "最终行业场景标签",
        "行业场景标签",
        "最终业务职责标签",
        "业务职责标签",
        "最终全部标签",
        "全部标签",
    ]


def infer_high_coverage_tags(
    df: pd.DataFrame,
    columns: list[str],
    *,
    coverage_threshold: float = 0.99,
) -> pd.DataFrame:
    if not _is_valid_df(df):
        return pd.DataFrame(columns=["标签", "覆盖岗位数", "覆盖率", "来源列"])

    total_rows = max(len(df), 1)
    tag_counter: Counter = Counter()
    tag_sources: defaultdict[str, set[str]] = defaultdict(set)
    valid_columns = [col for col in columns if col in df.columns]
    for _, row in df.iterrows():
        row_tags: set[str] = set()
        row_sources: defaultdict[str, set[str]] = defaultdict(set)
        for col in valid_columns:
            for tag in set(_safe_tags(row.get(col))):
                row_tags.add(tag)
                row_sources[tag].add(col)
        for tag in row_tags:
            tag_counter[tag] += 1
            tag_sources[tag].update(row_sources.get(tag, set()))

    rows = []
    for tag, cnt in tag_counter.items():
        coverage = cnt / total_rows if total_rows else 0
        if coverage >= coverage_threshold:
            rows.append({
                "标签": tag,
                "覆盖岗位数": cnt,
                "覆盖率": coverage,
                "来源列": "、".join(sorted(tag_sources.get(tag, set()))),
            })
    return pd.DataFrame(rows).sort_values(["覆盖率", "覆盖岗位数", "标签"], ascending=[False, False, True]).reset_index(drop=True) if rows else pd.DataFrame(columns=["标签", "覆盖岗位数", "覆盖率", "来源列"])



def get_tag_column_coverage(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if not _is_valid_df(df):
        return pd.DataFrame(columns=["列名", "非空行数", "覆盖率", "唯一标签数", "标签总数"])

    rows = []
    total_rows = max(len(df), 1)
    for col in (columns or get_preferred_tag_columns()):
        if col not in df.columns:
            rows.append({"列名": col, "非空行数": 0, "覆盖率": 0.0, "唯一标签数": 0, "标签总数": 0})
            continue
        non_empty_rows = 0
        unique_tags: set[str] = set()
        total_tags = 0
        for value in df[col].tolist():
            tags_list = _safe_tags(value)
            if tags_list:
                non_empty_rows += 1
                total_tags += len(tags_list)
                unique_tags.update(tags_list)
        rows.append({
            "列名": col,
            "非空行数": non_empty_rows,
            "覆盖率": non_empty_rows / total_rows,
            "唯一标签数": len(unique_tags),
            "标签总数": total_tags,
        })
    return pd.DataFrame(rows)


def merge_dimension_tag_columns(
    df: pd.DataFrame,
    source_columns: list[str],
    target_column: str,
    *,
    stop_tags: set[str] | None = None,
    max_tag_coverage: float | None = None,
) -> pd.DataFrame:
    if not _is_valid_df(df):
        return df

    valid_sources = [col for col in source_columns if col in df.columns]
    if not valid_sources:
        if target_column not in df.columns:
            df[target_column] = [[] for _ in range(len(df))]
        return df

    stop_tags = {str(x).strip() for x in (stop_tags or set()) if str(x).strip()}
    tag_drop_set: set[str] = set(stop_tags)
    if max_tag_coverage is not None and len(df) > 0:
        total_rows = len(df)
        tag_counter = Counter()
        for _, row in df.iterrows():
            merged_tags = []
            for col in valid_sources:
                merged_tags.extend(_safe_tags(row.get(col)))
            for tag in set(_dedupe(merged_tags)):
                tag_counter[tag] += 1
        for tag, cnt in tag_counter.items():
            coverage = cnt / total_rows if total_rows else 0
            if coverage >= float(max_tag_coverage):
                tag_drop_set.add(tag)

    def _merge_row(row: pd.Series) -> list[str]:
        merged: list[str] = []
        for col in valid_sources:
            merged.extend(_safe_tags(row.get(col)))
        result = _dedupe(merged)
        if not tag_drop_set:
            return result
        return [tag for tag in result if tag not in tag_drop_set]

    def _dedupe(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values:
            cleaned = _clean_token(item)
            if not cleaned or cleaned.lower() in invalid_values:
                continue
            if cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
        return out

    def _clean_token(value) -> str:
        text = str(value).strip().strip("[](){}")
        text = text.strip(" -•·：:'\"")
        return text

    invalid_values = {"nan", "none", "[]", "{}", ""}
    df[target_column] = df.apply(_merge_row, axis=1)
    return df


def _safe_company(value) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else "未知公司"


def _pick_first_non_empty(row: pd.Series, candidates: list[str]) -> str:
    for col in candidates:
        if col in row.index:
            value = row.get(col)
            text = "" if value is None else str(value).strip()
            if text and text.lower() not in {"nan", "none", "未知", "未知地区", "未知行业", "未知公司"}:
                return text
    return ""


def _safe_city(row: pd.Series) -> str:
    city = _pick_first_non_empty(row, ["工作城市", "所在地区", "城市_norm", "城市", "地区"])
    return city or "未知地区"


def _safe_industry(row: pd.Series) -> str:
    industry_sources = ["LLM所属行业", "所属行业", "行业"]
    for col in industry_sources:
        if col in row.index:
            value = row.get(col)
            text = "" if value is None else str(value).strip()
            if text and text.lower() not in {"nan", "none", "未知", "未知行业"}:
                return text

    scene_sources = ["最终行业场景标签", "行业场景标签"]
    scene_tags: list[str] = []
    for col in scene_sources:
        if col in row.index:
            tags = _safe_tags(row.get(col))
            if tags:
                scene_tags.extend(tags)
    if scene_tags:
        uniq = []
        seen = set()
        for tag in scene_tags:
            if tag not in seen:
                uniq.append(tag)
                seen.add(tag)
        return "、".join(uniq[:3])

    return "未知行业"


def _safe_role_direction(row: pd.Series) -> str:
    direction = _pick_first_non_empty(row, ["职位方向", "LLM岗位类型", "职位类别"])
    return direction or ""


def _safe_job_id(row: pd.Series, fallback) -> str:
    job_id = row.get("job_id", fallback)
    return str(job_id)


def _empty_graph() -> nx.Graph:
    return nx.Graph()


def _apply_node_importance_scores(G: nx.Graph) -> nx.Graph:
    if G is None or G.number_of_nodes() == 0:
        return G

    weighted_degree_map = dict(G.degree(weight="weight"))
    degree_map = dict(G.degree())
    try:
        centrality_map = nx.degree_centrality(G)
    except Exception:
        centrality_map = {node: 0.0 for node in G.nodes}

    for node, attrs in G.nodes(data=True):
        count = float(attrs.get("coverage_jobs", attrs.get("count", 0)) or 0)
        weighted_degree = float(weighted_degree_map.get(node, 0) or 0)
        centrality = float(centrality_map.get(node, 0) or 0)
        size_score = (
            log1p(max(count, 0)) * 0.48
            + log1p(max(weighted_degree, 0)) * 0.37
            + centrality * 4.5 * 0.15
        )
        attrs["degree"] = int(degree_map.get(node, 0) or 0)
        attrs["weighted_degree"] = round(weighted_degree, 4)
        attrs["centrality"] = round(centrality, 4)
        attrs["size_score"] = round(size_score, 4)
    return G


def build_tag_cooccurrence_network(
    df: pd.DataFrame,
    tag_col: str,
    top_n_tags: int = 50,
    min_cooccur: int = 2,
    dimension: str = "标签关系",
) -> nx.Graph:
    """
    标签共现网络：
    - 节点：标签
    - 边：两个标签在同一个岗位中共同出现
    - 边权：共现岗位数
    """
    G = _empty_graph()

    if not _is_valid_df(df) or tag_col not in df.columns:
        return G

    top_n_tags = max(int(top_n_tags), 1)
    min_cooccur = max(int(min_cooccur), 1)

    tag_counter = Counter()
    pair_counter = Counter()
    tag_job_counter = defaultdict(set)

    for _, row in df.iterrows():
        job_id = _safe_job_id(row, row.name)
        tags = sorted(set(_safe_tags(row.get(tag_col, []))))
        if len(tags) == 0:
            continue

        tag_counter.update(tags)

        for tag in tags:
            tag_job_counter[tag].add(job_id)

        for a, b in combinations(tags, 2):
            pair_counter[(a, b)] += 1

    if not tag_counter:
        return G

    top_tags = {tag for tag, _ in tag_counter.most_common(top_n_tags)}
    color = DIMENSION_COLOR_MAP.get(dimension, "#4C78A8")
    total_jobs = max(len(df), 1)

    for tag in top_tags:
        count = int(tag_counter[tag])
        coverage_jobs = len(tag_job_counter[tag])
        coverage = coverage_jobs / total_jobs

        G.add_node(
            tag,
            label=tag,
            size=10 + min(count * 1.5, 30),
            count=count,
            coverage=coverage,
            coverage_jobs=coverage_jobs,
            title=(
                f"标签：{tag}<br>"
                f"出现次数：{count}<br>"
                f"覆盖岗位数：{coverage_jobs}<br>"
                f"覆盖率：{coverage:.2%}"
            ),
            color=color,
            group=dimension,
            node_type="tag",
        )

    for (a, b), weight in pair_counter.items():
        if a in top_tags and b in top_tags and weight >= min_cooccur:
            G.add_edge(
                a,
                b,
                weight=int(weight),
                relation="标签共现",
                edge_type="cooccurrence",
                title=f"{a} 与 {b}<br>共现岗位数：{weight}",
            )

    return _apply_node_importance_scores(G)


def build_job_similarity_network_by_tags(
    df: pd.DataFrame,
    tag_col: str,
    top_n_jobs: int = 80,
    threshold: float = 0.25,
    dimension: str = "岗位相似",
    sort_by: str | None = None,
) -> nx.Graph:
    """
    岗位相似网络：
    - 节点：岗位
    - 边：标签集合 Jaccard 相似度
    - 边权：Jaccard 相似度

    注意：
    - 当前实现为两两比较，复杂度约 O(n^2)
    - top_n_jobs 应适度控制
    """
    G = _empty_graph()

    if not _is_valid_df(df) or tag_col not in df.columns:
        return G

    top_n_jobs = max(int(top_n_jobs), 1)
    threshold = float(threshold)

    data = df.copy()
    if sort_by and sort_by in data.columns:
        data = data.sort_values(sort_by, ascending=False)
    data = data.head(top_n_jobs).copy()

    for idx, row in data.iterrows():
        job_id = _safe_job_id(row, idx)
        title = str(row.get("职位名称_norm", "") or row.get("职位名称_raw", "") or "").strip()
        company = _safe_company(row.get("企业名称_norm", row.get("企业名称_raw", "未知公司")))
        city = _safe_city(row)
        industry = _safe_industry(row)
        role_direction = _safe_role_direction(row)
        tags = _safe_tags(row.get(tag_col, []))

        parts = [part for part in [city] if part and part != "未知地区"]
        if industry and industry != "未知行业":
            parts.append(industry)
        parts.append(company)
        if title and title != "AI产品经理":
            parts.append(title)
        elif role_direction and role_direction not in {industry, title}:
            parts.append(f"方向:{role_direction}")
        display_label = " - ".join(parts) if parts else (title if title else job_id)

        G.add_node(
            job_id,
            label=display_label[:42] if display_label else job_id,
            size=12 + min(len(tags) * 2, 20),
            title=(
                f"城市：{city or '未知'}<br>"
                f"行业/场景：{industry or '未知'}<br>"
                f"岗位方向：{role_direction or '未知'}<br>"
                f"公司：{company}<br>"
                f"职位：{title or '未知岗位'}<br>"
                f"标签：{'、'.join(tags[:20])}"
            ),
            color=DIMENSION_COLOR_MAP.get(dimension, "#4C78A8"),
            group=company,
            node_type="job",
        )

    rows = list(data.iterrows())
    for i in range(len(rows)):
        idx_i, row_i = rows[i]
        tags_i = set(_safe_tags(row_i.get(tag_col, [])))
        if not tags_i:
            continue

        G.nodes[_safe_job_id(row_i, idx_i)]["count"] = len(tags_i)
        G.nodes[_safe_job_id(row_i, idx_i)]["coverage_jobs"] = 1

        for j in range(i + 1, len(rows)):
            idx_j, row_j = rows[j]
            tags_j = set(_safe_tags(row_j.get(tag_col, [])))
            if not tags_j:
                continue

            union = tags_i | tags_j
            if not union:
                continue

            inter = tags_i & tags_j
            score = len(inter) / len(union)

            if score >= threshold:
                n1 = _safe_job_id(row_i, idx_i)
                n2 = _safe_job_id(row_j, idx_j)
                G.add_edge(
                    n1,
                    n2,
                    weight=round(score, 3),
                    relation=f"{dimension}相似",
                    edge_type="job_similarity",
                    shared_tags="、".join(sorted(inter)),
                    title=(
                        f"相似度：{score:.2f}<br>"
                        f"共同标签：{'、'.join(sorted(inter))}"
                    ),
                )

    return G


def build_company_tag_bipartite_network(
    df: pd.DataFrame,
    tag_col: str,
    top_n_companies: int = 20,
    top_n_tags: int = 40,
    min_weight: int = 2,
    dimension: str = "公司能力画像",
) -> nx.Graph:
    """
    公司-标签二部图：
    - 节点：公司、标签
    - 边：公司岗位中出现某标签的岗位数
    """
    G = _empty_graph()

    if not _is_valid_df(df) or tag_col not in df.columns or "企业名称_norm" not in df.columns:
        return G

    top_n_companies = max(int(top_n_companies), 1)
    top_n_tags = max(int(top_n_tags), 1)
    min_weight = max(int(min_weight), 1)

    company_series = df["企业名称_norm"].fillna("").astype(str).str.strip().replace("", "未知公司")
    top_companies = company_series.value_counts().head(top_n_companies).index.tolist()

    sub = df[company_series.isin(top_companies)].copy()

    tag_counter = Counter()
    company_tag_counter = Counter()

    for _, row in sub.iterrows():
        company = _safe_company(row.get("企业名称_norm", "未知公司"))
        tags = set(_safe_tags(row.get(tag_col, [])))

        tag_counter.update(tags)
        for tag in tags:
            company_tag_counter[(company, tag)] += 1

    if not tag_counter:
        return G

    top_tags = {tag for tag, _ in tag_counter.most_common(top_n_tags)}

    for company in top_companies:
        G.add_node(
            f"company::{company}",
            label=company,
            size=26,
            color="#E45756",
            title=f"公司：{company}",
            group="公司",
            node_type="company",
        )

    for tag in top_tags:
        G.add_node(
            f"tag::{tag}",
            label=tag,
            size=12 + min(tag_counter[tag], 28),
            color=DIMENSION_COLOR_MAP.get(dimension, "#72B7B2"),
            title=f"标签：{tag}<br>出现次数：{tag_counter[tag]}",
            group="标签",
            node_type="tag",
            count=int(tag_counter[tag]),
        )

    for (company, tag), weight in company_tag_counter.items():
        if tag in top_tags and weight >= min_weight:
            G.add_edge(
                f"company::{company}",
                f"tag::{tag}",
                weight=int(weight),
                relation="公司-标签",
                edge_type="bipartite_company_tag",
                title=f"{company} 中有 {weight} 个岗位要求 {tag}",
            )

    return _apply_node_importance_scores(G)


def build_role_tag_insight_network(
    df: pd.DataFrame,
    tag_col: str,
    role_col: str = "职位方向",
    top_n_roles: int = 12,
    top_n_tags: int = 40,
    min_weight: int = 2,
) -> nx.Graph:
    G = _empty_graph()
    if not _is_valid_df(df) or tag_col not in df.columns or role_col not in df.columns:
        return G

    role_counter = Counter()
    tag_counter = Counter()
    pair_counter = Counter()
    role_job_counter = defaultdict(set)
    tag_job_counter = defaultdict(set)

    for _, row in df.iterrows():
        role = str(row.get(role_col, "")).strip() or "未知方向"
        tags = sorted(set(_safe_tags(row.get(tag_col, []))))
        if not tags:
            continue
        job_id = _safe_job_id(row, row.name)
        role_counter[role] += 1
        role_job_counter[role].add(job_id)
        for tag in tags:
            tag_counter[tag] += 1
            tag_job_counter[tag].add(job_id)
            pair_counter[(role, tag)] += 1

    top_roles = {name for name, _ in role_counter.most_common(max(int(top_n_roles), 1))}
    top_tags = {name for name, _ in tag_counter.most_common(max(int(top_n_tags), 1))}
    total_jobs = max(len(df), 1)

    for role in top_roles:
        count = int(role_counter[role])
        coverage_jobs = len(role_job_counter[role])
        coverage = coverage_jobs / total_jobs
        G.add_node(
            f"role::{role}",
            label=role,
            size=18 + min(count * 1.2, 26),
            count=count,
            coverage=coverage,
            coverage_jobs=coverage_jobs,
            title=(
                f"岗位方向：{role}<br>"
                f"岗位数：{count}<br>"
                f"覆盖岗位数：{coverage_jobs}<br>"
                f"覆盖率：{coverage:.2%}"
            ),
            color="#6C8CFF",
            group="岗位方向",
            node_type="role",
        )

    for tag in top_tags:
        count = int(tag_counter[tag])
        coverage_jobs = len(tag_job_counter[tag])
        coverage = coverage_jobs / total_jobs
        G.add_node(
            f"tag::{tag}",
            label=tag,
            size=10 + min(count * 1.1, 22),
            count=count,
            coverage=coverage,
            coverage_jobs=coverage_jobs,
            title=(
                f"标签：{tag}<br>"
                f"出现次数：{count}<br>"
                f"覆盖岗位数：{coverage_jobs}<br>"
                f"覆盖率：{coverage:.2%}"
            ),
            color="#F6C453",
            group="标签",
            node_type="tag",
        )

    for (role, tag), weight in pair_counter.items():
        role_id = f"role::{role}"
        tag_id = f"tag::{tag}"
        if role in top_roles and tag in top_tags and weight >= max(int(min_weight), 1):
            G.add_edge(
                role_id,
                tag_id,
                weight=int(weight),
                relation="岗位-标签关联",
                edge_type="role_tag",
                title=f"{role} 与 {tag}<br>关联岗位数：{weight}",
            )
    return _apply_node_importance_scores(G)


def build_network(
    df: pd.DataFrame,
    dimension: str = "技术关系",
    network_type: NetworkType = "标签共现网络",
    top_n: int = 50,
    min_edge_weight: float = 2,
    similarity_threshold: float = 0.25,
    *,
    strict: bool = False,
) -> nx.Graph:

    """
    统一网络构建入口。
    """
    if network_type not in get_supported_network_types():
        if strict:
            raise ValueError(f"未知网络类型: {network_type}，支持: {get_supported_network_types()}")
        return _empty_graph()

    tag_col = _resolve_tag_col(dimension, strict=strict, df=df)

    if network_type == "标签共现网络":
        return build_tag_cooccurrence_network(
            df,
            tag_col=tag_col,
            top_n_tags=top_n,
            min_cooccur=int(min_edge_weight),
            dimension=dimension,
        )

    if network_type == "岗位相似网络":
        return build_job_similarity_network_by_tags(
            df,
            tag_col=tag_col,
            top_n_jobs=top_n,
            threshold=similarity_threshold,
            dimension=dimension,
        )

    if network_type == "公司能力画像网络":
        return build_company_tag_bipartite_network(
            df,
            tag_col=tag_col,
            top_n_companies=min(20, int(top_n)),
            top_n_tags=top_n,
            min_weight=int(min_edge_weight),
            dimension="公司能力画像",
        )

    if network_type == "岗位-标签洞察网络":
        role_col = "职位方向" if "职位方向" in df.columns else ("LLM岗位类型" if "LLM岗位类型" in df.columns else "职位类别")
        return build_role_tag_insight_network(
            df,
            tag_col=tag_col,
            role_col=role_col,
            top_n_roles=min(15, max(6, int(top_n // 4))),
            top_n_tags=top_n,
            min_weight=int(min_edge_weight),
        )

    return _empty_graph()


def get_network_by_dimension_v2(
    df: pd.DataFrame,
    dimension: str = "技术关系",
    network_type: str = "标签共现网络",
    top_n: int = 50,
    min_edge_weight: float = 2,
    similarity_threshold: float = 0.25,
) -> nx.Graph:
    """
    兼容旧接口。
    """
    return build_network(
        df=df,
        dimension=dimension,
        network_type=network_type,
        top_n=top_n,
        min_edge_weight=min_edge_weight,
        similarity_threshold=similarity_threshold,
        strict=False,
    )


def get_network_summary(G: nx.Graph) -> dict:
    """
    计算网络基础摘要。
    """
    if G is None or G.number_of_nodes() == 0:
        return {
            "节点数": 0,
            "边数": 0,
            "平均度": 0,
            "连通分量数": 0,
            "网络密度": 0,
        }

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / node_count if node_count else 0
    components = nx.number_connected_components(G) if node_count > 0 else 0
    density = nx.density(G) if node_count > 1 else 0

    return {
        "节点数": node_count,
        "边数": edge_count,
        "平均度": round(avg_degree, 2),
        "连通分量数": int(components),
        "网络密度": round(density, 4),
    }


def get_top_nodes(G: nx.Graph, top_n: int = 10) -> pd.DataFrame:
    """
    计算核心节点排行，默认使用加权度。
    """
    if G is None or G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["节点ID", "节点", "节点类型", "加权度", "连接数", "出现次数", "覆盖岗位数", "覆盖率"])

    degree_dict = dict(G.degree(weight="weight"))
    rows = []

    for node, score in sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        attrs = G.nodes[node]
        rows.append({
            "节点ID": node,
            "节点": attrs.get("label", node),
            "节点类型": attrs.get("node_type", ""),
            "加权度": round(score, 3),
            "连接数": int(G.degree(node)),
            "出现次数": attrs.get("count", None),
            "覆盖岗位数": attrs.get("coverage_jobs", None),
            "覆盖率": attrs.get("coverage", None),
        })

    return pd.DataFrame(rows)


def get_all_edges(G: nx.Graph) -> pd.DataFrame:
    """
    输出全部边明细。
    """
    if G is None or G.number_of_edges() == 0:
        return pd.DataFrame(columns=["source_id", "target_id", "节点A", "节点B", "权重", "关系", "边类型", "说明"])

    rows = []
    for u, v, attrs in G.edges(data=True):
        rows.append({
            "source_id": u,
            "target_id": v,
            "节点A": G.nodes[u].get("label", u),
            "节点B": G.nodes[v].get("label", v),
            "权重": attrs.get("weight", 1),
            "关系": attrs.get("relation", ""),
            "边类型": attrs.get("edge_type", ""),
            "说明": attrs.get("title", ""),
        })

    return pd.DataFrame(rows).sort_values("权重", ascending=False).reset_index(drop=True)


def get_top_edges(G: nx.Graph, top_n: int = 10) -> pd.DataFrame:
    """
    输出 strongest edges。
    """
    df = get_all_edges(G)
    if df.empty:
        return df
    return df.head(top_n).copy()


def detect_communities(G: nx.Graph) -> pd.DataFrame:
    """
    社区发现：
    - 优先使用 Louvain
    - 不可用时退化为连通分量
    """
    if G is None or G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["社区ID", "节点ID", "节点", "节点类型", "度"])

    if G.number_of_edges() == 0:
        rows = []
        for i, node in enumerate(G.nodes()):
            rows.append({
                "社区ID": i,
                "节点ID": node,
                "节点": G.nodes[node].get("label", node),
                "节点类型": G.nodes[node].get("node_type", ""),
                "度": int(G.degree(node)),
            })
        return pd.DataFrame(rows).sort_values(["社区ID", "度"], ascending=[True, False])

    try:
        import community as community_louvain

        partition = community_louvain.best_partition(G, weight="weight")
        rows = []
        for node in G.nodes():
            rows.append({
                "社区ID": partition.get(node, -1),
                "节点ID": node,
                "节点": G.nodes[node].get("label", node),
                "节点类型": G.nodes[node].get("node_type", ""),
                "度": int(G.degree(node)),
            })
        return pd.DataFrame(rows).sort_values(["社区ID", "度"], ascending=[True, False]).reset_index(drop=True)

    except Exception:
        rows = []
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                rows.append({
                    "社区ID": i,
                    "节点ID": node,
                    "节点": G.nodes[node].get("label", node),
                    "节点类型": G.nodes[node].get("node_type", ""),
                    "度": int(G.degree(node)),
                })
        return pd.DataFrame(rows).sort_values(["社区ID", "度"], ascending=[True, False]).reset_index(drop=True)


def get_jobs_by_node_label(df: pd.DataFrame, dimension: str, node_label: str, role_col: str | None = None) -> pd.DataFrame:
    """
    根据标签节点名、岗位方向节点名或岗位节点名反查岗位。
    """
    if not _is_valid_df(df):
        return pd.DataFrame()

    node_label = str(node_label).strip()
    if not node_label:
        return pd.DataFrame()

    role_col = role_col or ("职位方向" if "职位方向" in df.columns else ("LLM岗位类型" if "LLM岗位类型" in df.columns else ("职位类别" if "职位类别" in df.columns else None)))
    if role_col and role_col in df.columns:
        role_series = df[role_col].fillna("").astype(str).str.strip()
        role_mask = role_series == node_label
        if role_mask.any():
            return df[role_mask].copy()

    job_title_cols = [col for col in ["职位名称_norm", "职位名称_raw"] if col in df.columns]
    if job_title_cols:
        job_mask = pd.Series(False, index=df.index)
        for col in job_title_cols:
            title_series = df[col].fillna("").astype(str).str.strip()
            short_series = title_series.apply(lambda x: x[:18] if x else x)
            job_mask = job_mask | (title_series == node_label) | (short_series == node_label)

        if not job_mask.any():
            composite_parts = [part.strip() for part in re.split(r"\s*-\s*", node_label) if part.strip()]
            if composite_parts:
                company_cols = [c for c in ["企业名称_norm", "企业名称_raw"] if c in df.columns]
                city_cols = [c for c in ["工作城市", "所在地区", "城市", "城市_norm", "地区"] if c in df.columns]
                industry_cols = [c for c in ["LLM所属行业", "所属行业", "行业", "最终行业场景标签", "行业场景标签"] if c in df.columns]
                role_direction_cols = [c for c in ["职位方向", "LLM岗位类型", "职位类别"] if c in df.columns]
                composite_mask = pd.Series(True, index=df.index)
                matched_any = False
                for part in composite_parts:
                    part_mask = pd.Series(False, index=df.index)
                    for col in job_title_cols + company_cols + city_cols + industry_cols + role_direction_cols:
                        series = df[col].fillna("").astype(str).str.strip()
                        part_mask = part_mask | (series == part)
                    if part_mask.any():
                        composite_mask = composite_mask & part_mask
                        matched_any = True
                if matched_any and composite_mask.any():
                    return df[composite_mask].copy()

        if job_mask.any():
            return df[job_mask].copy()

    tag_col = _resolve_tag_col(dimension, strict=False, df=df)
    if tag_col not in df.columns:
        return pd.DataFrame()

    return df[df[tag_col].apply(lambda tags: node_label in _safe_tags(tags))].copy()


def get_jobs_by_edge_pair(df: pd.DataFrame, dimension: str, node_a: str, node_b: str, role_col: str | None = None) -> pd.DataFrame:
    """
    根据标签对或岗位方向边反查岗位。
    """
    if not _is_valid_df(df):
        return pd.DataFrame()

    node_a = str(node_a).strip()
    node_b = str(node_b).strip()
    if not node_a or not node_b:
        return pd.DataFrame()

    role_col = role_col or ("职位方向" if "职位方向" in df.columns else ("LLM岗位类型" if "LLM岗位类型" in df.columns else ("职位类别" if "职位类别" in df.columns else None)))
    if role_col and role_col in df.columns:
        role_series = df[role_col].fillna("").astype(str).str.strip()
        role_mask = (role_series == node_a) | (role_series == node_b)
        if role_mask.any():
            return df[role_mask].copy()

    tag_col = _resolve_tag_col(dimension, strict=False, df=df)
    if tag_col not in df.columns:
        return pd.DataFrame()

    return df[
        df[tag_col].apply(lambda tags: node_a in _safe_tags(tags) and node_b in _safe_tags(tags))
    ].copy()


def get_jobs_by_company_node(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    """
    根据公司节点反查岗位。
    """
    if not _is_valid_df(df) or "企业名称_norm" not in df.columns:
        return pd.DataFrame()

    company_name = str(company_name).strip()
    if not company_name:
        return pd.DataFrame()

    company_series = df["企业名称_norm"].fillna("").astype(str).str.strip().replace("", "未知公司")
    return df[company_series == company_name].copy()


def get_jobs_by_company_tag_edge(df: pd.DataFrame, company_name: str, tag_col: str, tag_label: str) -> pd.DataFrame:
    """
    根据公司-标签边反查岗位。
    """
    if not _is_valid_df(df):
        return pd.DataFrame()

    if "企业名称_norm" not in df.columns or tag_col not in df.columns:
        return pd.DataFrame()

    company_name = str(company_name).strip()
    tag_label = str(tag_label).strip()

    if not company_name or not tag_label:
        return pd.DataFrame()

    company_series = df["企业名称_norm"].fillna("").astype(str).str.strip().replace("", "未知公司")
    return df[
        (company_series == company_name)
        & df[tag_col].apply(lambda tags: tag_label in _safe_tags(tags))
    ].copy()
