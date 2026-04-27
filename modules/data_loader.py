"""
[MODULE_SPEC]
module_id: modules.data_loader
module_path: modules/data_loader.py
module_name: 招聘数据读取模块
module_type: data_loader
layer: 数据输入层

responsibility:
  - 负责从本地文件路径或上传文件对象中读取原始招聘数据。
  - 支持 CSV、Excel、JSON 等常见结构化文件格式。
  - 输出 pandas.DataFrame，供 preprocessing.py 做进一步标准化。
  - 仅做轻量读取、基础字段校验和兼容性清洗，不承载复杂业务清洗逻辑。

notes:
  - 若需复杂字段标准化、薪资解析、标签提取、去重策略优化，应转到 preprocessing.py 或其他分析模块。
[/MODULE_SPEC]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from config import REQUIRED_COLUMNS, validate_required_columns


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


def _is_file_like(obj: Any) -> bool:
    """
    判断对象是否像上传文件或文件流。
    """
    return hasattr(obj, "read")


def _get_extension(source: Any, file_type: str | None = None) -> str:
    """
    获取文件扩展名。

    优先级：
    1. 显式传入 file_type
    2. 从 source.name 推断（上传文件对象常见）
    3. 从路径字符串推断
    """
    if file_type:
        ext = str(file_type).strip().lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return ext

    if hasattr(source, "name"):
        name = str(getattr(source, "name", "")).strip()
        if name:
            return Path(name).suffix.lower()

    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower()

    return ""


def load_csv(source: str | Path | BinaryIO, **kwargs) -> pd.DataFrame:
    """
    读取 CSV 文件。

    默认尝试多种常见中文编码。
    可通过 kwargs 覆盖 pandas.read_csv 参数。
    """
    encodings = kwargs.pop("encodings", ["utf-8", "utf-8-sig", "gbk", "gb18030"])

    last_error = None
    for encoding in encodings:
        try:
            if _is_file_like(source) and hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding, **kwargs)
        except Exception as e:
            last_error = e

    raise ValueError(f"CSV 读取失败，已尝试编码 {encodings}。最后错误: {last_error}")


def load_excel(
    source: str | Path | BinaryIO,
    sheet_name: str | int | None = 0,
    **kwargs,
) -> pd.DataFrame:
    """
    读取 Excel 文件。
    """
    if _is_file_like(source) and hasattr(source, "seek"):
        source.seek(0)

    return pd.read_excel(source, sheet_name=sheet_name, **kwargs)


def load_json(source: str | Path | BinaryIO, **kwargs) -> pd.DataFrame:
    """
    读取 JSON 文件。

    注意：
    - 默认使用 pandas.read_json
    - 若 JSON 为 records 列表，通常可直接读取
    """
    if _is_file_like(source) and hasattr(source, "seek"):
        source.seek(0)

    return pd.read_json(source, **kwargs)


def load_data(
    source: str | Path | BinaryIO,
    file_type: str | None = None,
    *,
    sheet_name: str | int | None = 0,
    strict_empty_check: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    统一数据读取入口。

    参数：
    - source: 本地路径、Path 对象或上传文件对象
    - file_type: 可显式指定格式，如 csv/xlsx/json
    - sheet_name: Excel sheet 选择
    - strict_empty_check: 是否在读取后对空 DataFrame 抛异常
    - kwargs: 透传给 pandas 读取函数

    返回：
    - pandas.DataFrame
    """
    if source is None:
        raise ValueError("读取数据失败：source 不能为空")

    ext = _get_extension(source, file_type=file_type)

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"暂不支持的数据格式: {ext or '未知'}，支持格式: {sorted(SUPPORTED_EXTENSIONS)}")

    if ext == ".csv":
        df = load_csv(source, **kwargs)
    elif ext in {".xlsx", ".xls"}:
        df = load_excel(source, sheet_name=sheet_name, **kwargs)
    elif ext == ".json":
        df = load_json(source, **kwargs)
    else:
        # 理论上不会走到这里，作为保护分支保留
        raise ValueError(f"未识别的数据格式: {ext}")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("读取失败：返回结果不是 pandas.DataFrame")

    if strict_empty_check and df.empty:
        raise ValueError("读取失败：数据为空")

    return df


def check_empty(df: pd.DataFrame) -> bool:
    """
    判断 DataFrame 是否为空或无有效行。
    """
    return not isinstance(df, pd.DataFrame) or df.empty


def validate_columns(
    df: pd.DataFrame,
    *,
    raise_error: bool = True,
) -> tuple[bool, list[str]]:
    """
    校验 DataFrame 是否包含配置中的必填字段。

    参数：
    - raise_error: 为 True 时缺失字段直接抛异常；否则返回 (False, missing)

    返回：
    - (ok, missing)
    """
    ok, missing = validate_required_columns(df)

    if not ok and raise_error:
        raise ValueError(f"缺少必要字段: {missing}")

    return ok, missing


def basic_clean(
    df: pd.DataFrame,
    *,
    fill_missing_required: bool = True,
    deduplicate: bool = True,
    create_job_id: bool = True,
    parse_time: bool = True,
) -> pd.DataFrame:
    """
    对读取后的原始数据进行轻量兼容性清理。

    本函数只做“尽量不破坏原始数据”的基础处理：
    - 可选补齐缺失必填列为空字符串
    - 对常用文本字段填空
    - 轻量去重
    - 构建 job_id
    - 尝试解析抓取时间

    注意：
    - 更复杂的字段标准化、职位/企业名称清洗、薪资解析等应放在 preprocessing.py
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("basic_clean expects a pandas.DataFrame")

    result = df.copy()

    # 1) 补齐必填字段，避免后续页面或流程报错
    if fill_missing_required:
        for col in REQUIRED_COLUMNS:
            if col not in result.columns:
                result[col] = ""

    # 2) 文本列填空并转字符串，时间列先跳过
    for col in result.columns:
        if col == "抓取时间":
            continue
        if result[col].dtype == "object" or col in REQUIRED_COLUMNS:
            result[col] = result[col].fillna("").astype(str)

    # 3) 轻量去重：优先详情链接，其次页面URL
    if deduplicate:
        if "详情链接" in result.columns and result["详情链接"].astype(str).str.strip().ne("").any():
            result = result.drop_duplicates(subset=["详情链接"], keep="first")
        elif "页面URL" in result.columns and result["页面URL"].astype(str).str.strip().ne("").any():
            result = result.drop_duplicates(subset=["页面URL"], keep="first")

    # 4) 重建索引与 job_id
    result = result.reset_index(drop=True)
    if create_job_id:
        result["job_id"] = result.index.astype(str)

    # 5) 抓取时间解析
    if parse_time and "抓取时间" in result.columns:
        result["抓取时间"] = pd.to_datetime(result["抓取时间"], errors="coerce")

    return result


def load_and_prepare(
    source: str | Path | BinaryIO,
    file_type: str | None = None,
    *,
    sheet_name: str | int | None = 0,
    strict_empty_check: bool = True,
    validate_required: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    便捷入口：读取 + 轻量清理。

    适用于 app.py 中快速接入数据流程。

    参数：
    - validate_required:
        True  -> 严格校验 REQUIRED_COLUMNS，缺失时报错
        False -> 不强制，后续可由 preprocessing.py 或页面层继续处理
    """
    df = load_data(
        source,
        file_type=file_type,
        sheet_name=sheet_name,
        strict_empty_check=strict_empty_check,
        **kwargs,
    )

    df = basic_clean(df)

    if validate_required:
        validate_columns(df, raise_error=True)

    return df
