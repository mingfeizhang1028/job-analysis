from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_TAG_DICT_PATH = "data/tag_dict.json"

# 为下游页面提供稳定默认列
DEFAULT_TAG_TYPES = ["硬技能", "软素质", "业务职责", "行业场景"]

# 纯 ASCII token 检测
# 例如：AI / SQL / RAG / Python / C++ / .NET / BI
_ASCII_TOKEN_RE = re.compile(r"^[A-Za-z0-9_+#.\-]+$")

# 英文 token 边界保护
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


@lru_cache(maxsize=32)
def load_tag_dict(path: str | Path = DEFAULT_TAG_DICT_PATH) -> dict[str, Any]:
    """
    加载标签词典。

    如果文件不存在、JSON 解析失败或顶层不是 dict，则返回空 dict。
    """
    path = Path(path)

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, TypeError):
        return {}

    return data if isinstance(data, dict) else {}


def _iter_leaf_words(value: Any) -> Iterable[str]:
    """
    从任意嵌套结构中提取字符串标签词。

    支持：
    - str
    - list[str]
    - dict[str, list[str]]
    - 更深层 dict/list/tuple/set 混合结构
    """
    if isinstance(value, str):
        word = value.strip()
        if word:
            yield word
        return

    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_leaf_words(child)
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_leaf_words(item)
        return


@lru_cache(maxsize=32)
def _flatten_tag_dict_cached(tag_dict_json: str) -> dict[str, list[str]]:
    """
    供 flatten_tag_dict 内部使用的缓存函数。
    """
    try:
        tag_dict = json.loads(tag_dict_json)
    except json.JSONDecodeError:
        return {}

    if not isinstance(tag_dict, dict):
        return {}

    result: dict[str, list[str]] = {}

    for tag_type, groups in tag_dict.items():
        tag_type_str = to_safe_str(tag_type)
        if not tag_type_str:
            continue

        words = {
            to_safe_str(word)
            for word in _iter_leaf_words(groups)
            if to_safe_str(word)
        }

        result[tag_type_str] = sorted(words, key=lambda x: (-len(x), x.lower()))

    return result


def flatten_tag_dict(tag_dict: dict[str, Any]) -> dict[str, list[str]]:
    """
    将嵌套标签词典压平成：
    {
      "硬技能": ["Python", "SQL", "LLM"],
      "软素质": ["沟通能力"]
    }

    排序策略：
    - 长词优先
    - 再按字典序稳定输出
    """
    if not isinstance(tag_dict, dict):
        return {}

    try:
        normalized = json.dumps(tag_dict, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return {}

    return _flatten_tag_dict_cached(normalized)


def _is_ascii_token(text: str) -> bool:
    return bool(_ASCII_TOKEN_RE.fullmatch(text or ""))


@lru_cache(maxsize=2048)
def _build_ascii_pattern(token_lower: str) -> re.Pattern:
    pattern = _ASCII_BOUNDARY_TEMPLATE.format(token=re.escape(token_lower))
    return re.compile(pattern)


def _contains_tag(text: str, tag: str, case_sensitive: bool = False) -> bool:
    """
    判断文本是否命中标签。

    规则：
    - 中文或混合中文标签：普通 substring 匹配
    - 纯 ASCII 标签：使用 token 边界保护，降低误命中
    - 默认大小写不敏感
    """
    if not text or not tag:
        return False

    source = text if case_sensitive else text.lower()
    target = tag if case_sensitive else tag.lower()

    if _is_ascii_token(tag):
        return _build_ascii_pattern(target).search(source) is not None

    return target in source


def normalize_tag_list(tags: Any) -> list[str]:
    """
    将任意标签值规范为 list[str]，去空、去重、保序。
    """
    if not isinstance(tags, list):
        return []

    normalized: list[str] = []
    seen = set()

    for tag in tags:
        tag_str = to_safe_str(tag)
        if not tag_str or tag_str in seen:
            continue
        normalized.append(tag_str)
        seen.add(tag_str)

    return normalized


def _normalize_synonym_map(synonym_map: dict[str, str] | None) -> dict[str, str]:
    """
    规范化 synonym_map，统一 strip。
    """
    if not isinstance(synonym_map, dict):
        return {}

    normalized: dict[str, str] = {}
    for k, v in synonym_map.items():
        key = to_safe_str(k)
        val = to_safe_str(v)
        if key and val:
            normalized[key] = val

    return normalized


def extract_tags_from_text(
    text: str,
    tag_words: list[str],
    *,
    synonym_map: dict[str, str] | None = None,
    case_sensitive: bool = False,
) -> list[str]:
    """
    从文本中抽取命中的标签。

    参数：
    - text: 待抽取文本
    - tag_words: 候选标签词
    - synonym_map: 可选同义词映射，例如 {"LLM": "大语言模型", "大模型": "大语言模型"}
    - case_sensitive: 是否大小写敏感，默认 False

    返回：
    - 去重后的标签列表
    """
    source = to_safe_str(text)
    if not source or not isinstance(tag_words, list) or not tag_words:
        return []

    synonym_map = _normalize_synonym_map(synonym_map)

    hits: list[str] = []
    seen = set()

    for tag in tag_words:
        tag_str = to_safe_str(tag)
        if not tag_str:
            continue

        if _contains_tag(source, tag_str, case_sensitive=case_sensitive):
            normalized_tag = synonym_map.get(tag_str, tag_str)
            normalized_tag = to_safe_str(normalized_tag)

            if normalized_tag and normalized_tag not in seen:
                hits.append(normalized_tag)
                seen.add(normalized_tag)

    return sorted(hits, key=lambda x: (-len(x), x.lower()))


def _build_tag_text(row: pd.Series, detail_col: str) -> str:
    """
    拼接用于标签抽取的文本。

    当前拼接顺序：
    - 职位名称_norm
    - 职位名称_raw
    - 岗位详情
    """
    parts = [
        row.get("职位名称_norm", ""),
        row.get("职位名称_raw", ""),
        row.get(detail_col, ""),
    ]

    clean_parts = []
    for part in parts:
        part_str = to_safe_str(part)
        if part_str:
            clean_parts.append(part_str)

    return " ".join(clean_parts)


def _ensure_default_tag_columns(
    df: pd.DataFrame,
    tag_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    确保默认标签列存在，且每列值稳定为 list[str]。
    """
    tag_types = tag_types or DEFAULT_TAG_TYPES

    for tag_type in tag_types:
        col = f"{tag_type}标签"
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        else:
            df[col] = df[col].apply(normalize_tag_list)

    if "全部标签" not in df.columns:
        df["全部标签"] = [[] for _ in range(len(df))]
    else:
        df["全部标签"] = df["全部标签"].apply(normalize_tag_list)

    return df


def _merge_all_tags(row: pd.Series) -> list[str]:
    merged = []
    seen = set()

    for tags in row:
        if not isinstance(tags, list):
            continue
        for tag in tags:
            tag_str = to_safe_str(tag)
            if not tag_str or tag_str in seen:
                continue
            merged.append(tag_str)
            seen.add(tag_str)

    return sorted(merged, key=lambda x: (-len(x), x.lower()))


def apply_tag_extraction(
    df: pd.DataFrame,
    tag_dict_path: str | Path = DEFAULT_TAG_DICT_PATH,
    detail_col: str = "岗位详情",
    *,
    synonym_map: dict[str, str] | None = None,
    default_tag_types: list[str] | None = None,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    对 DataFrame 应用标签抽取。

    输入：
    - df: 岗位数据 DataFrame
    - tag_dict_path: 标签词典路径
    - detail_col: 岗位详情字段名
    - synonym_map: 可选同义词映射
    - default_tag_types: 即使词典缺失也要确保存在的标签类别
    - case_sensitive: 是否大小写敏感

    输出：
    - 返回新增标签列后的 DataFrame，不修改原 df
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("apply_tag_extraction expects a pandas.DataFrame")

    result = df.copy()

    if detail_col not in result.columns:
        result[detail_col] = ""

    ensured_tag_types = default_tag_types or DEFAULT_TAG_TYPES
    result = _ensure_default_tag_columns(result, ensured_tag_types)

    tag_dict = load_tag_dict(tag_dict_path)
    flat = flatten_tag_dict(tag_dict)

    # 词典为空时，也保证默认标签列和全部标签存在
    if not flat:
        result["全部标签"] = [[] for _ in range(len(result))]
        return result

    tag_text = result.apply(lambda row: _build_tag_text(row, detail_col), axis=1)

    generated_tag_cols: list[str] = []

    for tag_type, words in flat.items():
        col = f"{tag_type}标签"
        generated_tag_cols.append(col)

        result[col] = tag_text.apply(
            lambda text: extract_tags_from_text(
                text,
                words,
                synonym_map=synonym_map,
                case_sensitive=case_sensitive,
            )
        )

    all_tag_cols = sorted(
        set([f"{t}标签" for t in ensured_tag_types] + generated_tag_cols)
    )

    for col in all_tag_cols:
        if col not in result.columns:
            result[col] = [[] for _ in range(len(result))]
        else:
            result[col] = result[col].apply(normalize_tag_list)

    result["全部标签"] = result[all_tag_cols].apply(_merge_all_tags, axis=1)

    return result


def explode_tag_column(df: pd.DataFrame, tag_col: str) -> pd.DataFrame:
    """
    将 list 标签列展开为长表。
    如果 tag_col 不存在，返回空 DataFrame。
    """
    if not isinstance(df, pd.DataFrame) or tag_col not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    temp[tag_col] = temp[tag_col].apply(normalize_tag_list)

    temp = temp.explode(tag_col)
    temp = temp[temp[tag_col].notna() & (temp[tag_col] != "")]

    return temp.reset_index(drop=True)


def calc_tag_stats(
    df: pd.DataFrame,
    tag_col: str,
    group_col: str | None = None,
    job_id_col: str = "job_id",
) -> pd.DataFrame:
    """
    统计标签词频、覆盖岗位数和覆盖率。

    输出字段：
    - 标签
    - 词频
    - 覆盖岗位数
    - 总岗位数
    - 覆盖率

    如果 group_col 不为空，则按 group_col + 标签统计。
    """
    if not isinstance(df, pd.DataFrame) or tag_col not in df.columns:
        return pd.DataFrame()

    temp = df.copy()

    if job_id_col not in temp.columns:
        temp[job_id_col] = temp.index.astype(str)

    temp[tag_col] = temp[tag_col].apply(normalize_tag_list)

    total_jobs = temp[job_id_col].nunique()
    exploded = explode_tag_column(temp, tag_col)

    if exploded.empty:
        base_cols = ["标签", "词频", "覆盖岗位数", "总岗位数", "覆盖率"]
        if group_col is not None:
            base_cols = [group_col] + base_cols
        return pd.DataFrame(columns=base_cols)

    if group_col is None or group_col not in temp.columns:
        stat = (
            exploded.groupby(tag_col)
            .agg(
                词频=(tag_col, "count"),
                覆盖岗位数=(job_id_col, "nunique"),
            )
            .reset_index()
            .rename(columns={tag_col: "标签"})
        )
        stat["总岗位数"] = int(total_jobs)
        stat["覆盖率"] = stat["覆盖岗位数"] / max(int(total_jobs), 1)

        return stat.sort_values(
            ["覆盖岗位数", "词频", "标签"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    group_total = (
        temp.groupby(group_col)[job_id_col]
        .nunique()
        .reset_index()
        .rename(columns={job_id_col: "总岗位数"})
    )

    stat = (
        exploded.groupby([group_col, tag_col])
        .agg(
            词频=(tag_col, "count"),
            覆盖岗位数=(job_id_col, "nunique"),
        )
        .reset_index()
        .rename(columns={tag_col: "标签"})
    )

    stat = stat.merge(group_total, on=group_col, how="left")
    stat["总岗位数"] = stat["总岗位数"].fillna(0).astype(int)
    stat["覆盖率"] = stat["覆盖岗位数"] / stat["总岗位数"].replace(0, 1)

    return stat.sort_values(
        [group_col, "覆盖率", "覆盖岗位数", "词频", "标签"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)


def filter_jobs_by_tag(df: pd.DataFrame, tag_col: str, keyword: str) -> pd.DataFrame:
    """
    按标签列筛选包含 keyword 的岗位。

    注意：
    - 这里是精确标签匹配，不是模糊文本匹配。
    """
    if not isinstance(df, pd.DataFrame) or tag_col not in df.columns:
        return pd.DataFrame()

    keyword = to_safe_str(keyword)
    if not keyword:
        return pd.DataFrame()

    def contains_tag(tags: Any) -> bool:
        return keyword in normalize_tag_list(tags)

    return df[df[tag_col].apply(contains_tag)].copy()


def keyword_context(text: str, keyword: str, window: int = 40) -> str:
    """
    提取关键词在文本中的上下文片段。

    默认大小写不敏感。
    """
    source = to_safe_str(text)
    kw = to_safe_str(keyword)

    if not source or not kw:
        return ""

    match = re.search(re.escape(kw), source, flags=re.IGNORECASE)
    if not match:
        return ""

    start = max(0, match.start() - max(window, 0))
    end = min(len(source), match.end() + max(window, 0))

    return source[start:end]
