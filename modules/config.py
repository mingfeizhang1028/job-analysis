# -*- coding: utf-8 -*-
"""
项目全局配置模块。

职责：
- 定义项目基础目录、数据目录、输出目录
- 定义常用资源文件路径
- 定义默认输入文件路径
- 定义数据处理所需的必填字段
- 提供轻量路径/字段校验辅助函数

说明：
- 本模块只负责“配置”和“轻量校验辅助”，不承载具体业务逻辑。
- 若需要跨环境运行，优先通过环境变量覆盖默认路径。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


# ========== 基础目录 ==========
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


# ========== 默认资源路径 ==========
STOPWORDS_PATH = DATA_DIR / "stopwords.txt"
USER_DICT_PATH = DATA_DIR / "user_dict.txt"
TRAIT_DICT_PATH = DATA_DIR / "trait_dict.json"
DEFAULT_EXCEL_FILENAME = "current_selected_job_detail.xlsx"


# ========== 默认输入文件 ==========
# 优先读取环境变量 JOB_DATA_EXCEL_PATH
# 若未设置，则回退到项目 data 目录下的默认文件名
DEFAULT_EXCEL_PATH = Path(
    os.getenv("JOB_DATA_EXCEL_PATH", str(DATA_DIR / DEFAULT_EXCEL_FILENAME))
)


# ========== 数据契约：核心必填字段 ==========
REQUIRED_COLUMNS: tuple[str, ...] = (
    "抓取时间",
    "页面URL",
    "职位名称",
    "企业名称",
    "所在地区",
    "薪资解析",
    "经验要求",
    "学历要求",
    "工作地址",
    "岗位详情",
    "详情链接",
)


def ensure_directories() -> None:
    """
    确保项目运行所需目录存在。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_default_excel_path() -> Path:
    """
    获取默认 Excel 路径。

    优先级：
    1. 环境变量 JOB_DATA_EXCEL_PATH
    2. 项目 data/current_selected_job_detail.xlsx
    """
    return DEFAULT_EXCEL_PATH


def path_exists(path: str | Path) -> bool:
    """
    判断路径是否存在。
    """
    try:
        return Path(path).exists()
    except (TypeError, OSError):
        return False


def get_missing_required_columns(columns: Iterable[str]) -> list[str]:
    """
    根据列名序列，返回缺失的必填字段列表。
    """
    column_set = set(columns)
    return [col for col in REQUIRED_COLUMNS if col not in column_set]


def validate_required_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    校验 DataFrame 是否包含所有必填字段。

    返回：
    - (True, [])：全部存在
    - (False, [缺失字段...])：存在缺失项
    """
    if not isinstance(df, pd.DataFrame):
        return False, list(REQUIRED_COLUMNS)

    missing = get_missing_required_columns(df.columns)
    return len(missing) == 0, missing


# 模块加载时可确保目录存在，避免输出时报错
ensure_directories()
