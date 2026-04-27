"""
[MODULE_SPEC]
module_id: modules.keyword_analysis
module_path: modules/keyword_analysis.py
module_name: 标签关键词分析模块
module_type: keyword_analysis
layer: 分析服务层

responsibility:
  - 基于 tag_extraction.py 产出的标签列进行关键词统计与岗位筛选。
  - 将前端选择的标签类型映射到具体标签列。
  - 为 app.py 提供“关键词统计表”“按关键词查看岗位”“命中片段”等分析结果。
  - 不负责标签提取规则本身，不负责页面渲染。

notes:
  - 标签生成逻辑在 modules/tag_extraction.py。
  - 本模块主要是标签结果的查询、统计和轻量包装。
[/MODULE_SPEC]
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from modules.tag_extraction import calc_tag_stats, filter_jobs_by_tag, keyword_context


TagType = Literal["硬技能", "软素质", "业务职责", "行业场景", "全部"]


TAG_TYPE_TO_COL: dict[str, str] = {
    "硬技能": "硬技能标签",
    "软素质": "软素质标签",
    "业务职责": "业务职责标签",
    "行业场景": "行业场景标签",
    "全部": "全部标签",
}

DEFAULT_TAG_TYPE: TagType = "全部"
DEFAULT_CONTEXT_WINDOW = 50


def get_available_tag_types() -> list[str]:
    """
    返回当前支持的标签类型列表。
    可用于前端 selectbox/radio 的选项。
    """
    return list(TAG_TYPE_TO_COL.keys())


def get_tag_col_by_type(
    tag_type: str | None,
    *,
    default: str = "全部标签",
    strict: bool = False,
) -> str:
    """
    将中文标签类型映射为 DataFrame 中的标签列名。

    参数：
    - tag_type: 例如 硬技能/软素质/业务职责/行业场景/全部
    - default: 非严格模式下的默认标签列
    - strict: 若为 True，未知 tag_type 将抛出 ValueError
    """
    normalized_type = str(tag_type).strip() if tag_type is not None else DEFAULT_TAG_TYPE

    if normalized_type in TAG_TYPE_TO_COL:
        return TAG_TYPE_TO_COL[normalized_type]

    if strict:
        raise ValueError(
            f"未知标签类型: {tag_type}，可选值: {get_available_tag_types()}"
        )

    return default


def _is_valid_dataframe(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def _validate_tag_col(
    df: pd.DataFrame,
    tag_col: str,
    *,
    strict: bool = False,
) -> bool:
    """
    检查标签列是否存在。

    strict=True 时缺列抛异常；
    strict=False 时返回 False。
    """
    if not isinstance(df, pd.DataFrame):
        if strict:
            raise TypeError("df 必须是 pandas.DataFrame")
        return False

    if tag_col not in df.columns:
        if strict:
            raise ValueError(f"缺少标签列: {tag_col}")
        return False

    return True


def get_keyword_stats_by_mode(
    df: pd.DataFrame,
    tag_type: str = "硬技能",
    group_col: str | None = None,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """
    根据标签类型获取关键词统计表。

    参数：
    - df: 已完成标签提取的数据
    - tag_type: 标签类型，如 硬技能/软素质/业务职责/行业场景/全部
    - group_col: 可选分组列，例如 所在地区、企业名称_norm、学历要求
    - strict: 是否严格校验 tag_type 和标签列存在性

    返回：
    - pandas.DataFrame
    """
    if not _is_valid_dataframe(df):
        return pd.DataFrame()

    tag_col = get_tag_col_by_type(tag_type, strict=strict)

    if not _validate_tag_col(df, tag_col, strict=strict):
        return pd.DataFrame()

    if group_col is not None and group_col not in df.columns:
        if strict:
            raise ValueError(f"分组列不存在: {group_col}")
        group_col = None

    return calc_tag_stats(df, tag_col=tag_col, group_col=group_col)


def get_jobs_by_keyword(
    df: pd.DataFrame,
    tag_type: str,
    keyword: str,
    detail_col: str = "岗位详情",
    *,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    strict: bool = False,
) -> pd.DataFrame:
    """
    根据标签类型和关键词筛选命中的岗位，并可附加“命中片段”。

    参数：
    - df: 已完成标签提取的数据
    - tag_type: 标签类型
    - keyword: 要筛选的关键词
    - detail_col: 用于截取命中上下文的文本列
    - context_window: 命中片段窗口大小
    - strict: 是否严格校验标签列和详情列

    返回：
    - 命中岗位 DataFrame
    """
    if not _is_valid_dataframe(df):
        return pd.DataFrame()

    keyword = str(keyword).strip() if keyword is not None else ""
    if not keyword:
        return pd.DataFrame()

    tag_col = get_tag_col_by_type(tag_type, strict=strict)

    if not _validate_tag_col(df, tag_col, strict=strict):
        return pd.DataFrame()

    result = filter_jobs_by_tag(df, tag_col=tag_col, keyword=keyword)

    if result is None or result.empty:
        return pd.DataFrame(columns=df.columns)

    result = result.copy()

    if detail_col in result.columns:
        result["命中片段"] = result[detail_col].apply(
            lambda x: keyword_context(x, keyword, window=context_window)
        )
    else:
        if strict:
            raise ValueError(f"缺少用于生成命中片段的详情列: {detail_col}")
        result["命中片段"] = ""

    return result


def get_keyword_overview(
    df: pd.DataFrame,
    *,
    group_col: str | None = None,
    strict: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    返回所有标签类型的关键词统计结果。

    适用于 app.py 中一次性展示多个标签面板。
    """
    overview: dict[str, pd.DataFrame] = {}

    for tag_type in get_available_tag_types():
        overview[tag_type] = get_keyword_stats_by_mode(
            df,
            tag_type=tag_type,
            group_col=group_col,
            strict=strict,
        )

    return overview
