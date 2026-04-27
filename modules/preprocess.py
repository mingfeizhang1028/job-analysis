"""
modules/preprocessing.py

职责：
- 提供轻量文本预处理能力
- 提供地区归一化能力
- 提供分词前后处理能力
- 提供停用词、用户词典、trait 字典加载能力

注意：
- 本模块当前不负责 DataFrame 级整体清洗
- 不负责职位名/公司名标准化主逻辑
- 不负责薪资、经验、学历解析
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import jieba


LOW_VALUE_WORDS = {
    "产品", "技术", "业务", "需求", "设计", "落地", "用户", "数据", "场景", "应用",
    "优化", "项目", "工具", "智能", "推动", "行业", "来自", "沟通", "迭代", "分析",
    "持续", "理解", "运营", "体验", "流程", "开发", "功能", "岗位", "职位", "任职",
    "要求", "相关", "负责", "参与", "进行", "完成", "协助", "配合", "能够", "具备",
    "熟悉", "了解", "优先", "良好", "较强", "一定", "以上", "以下", "包括", "以及",
    "公司", "团队", "工作"
}

CITY_ALIASES = {
    "北京市": "北京",
    "北京": "北京",
    "上海市": "上海",
    "上海": "上海",
    "深圳市": "深圳",
    "深圳": "深圳",
    "广州市": "广州",
    "广州": "广州",
    "杭州市": "杭州",
    "杭州": "杭州",
    "成都市": "成都",
    "成都": "成都",
    "南京市": "南京",
    "南京": "南京",
    "苏州市": "苏州",
    "苏州": "苏州",
    "武汉市": "武汉",
    "武汉": "武汉",
    "西安市": "西安",
    "西安": "西安",
    "天津市": "天津",
    "天津": "天津",
    "重庆市": "重庆",
    "重庆": "重庆",
    "长沙市": "长沙",
    "长沙": "长沙",
    "郑州市": "郑州",
    "郑州": "郑州",
    "合肥市": "合肥",
    "合肥": "合肥",
    "青岛市": "青岛",
    "青岛": "青岛",
    "厦门市": "厦门",
    "厦门": "厦门",
}


RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"http[s]?://\S+")
RE_BLANK = re.compile(r"[\r\n\t]+")
RE_MULTI_SPACE = re.compile(r"\s+")
RE_KEEP_TEXT = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9+#.]+")
RE_REGION_SPLIT = re.compile(r"[-/|,，、\s]+")


def to_safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if value != value:  # NaN
            return default
    except Exception:
        pass
    return str(value).strip()


@lru_cache(maxsize=64)
def load_word_set(file_path: str | Path) -> set[str]:
    """
    从文本文件中加载词集合，每行一个词。
    - 文件不存在时返回空集合
    - 自动去空白、转小写
    """
    path = Path(file_path)
    if not path.exists():
        return set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}
    except OSError:
        return set()


def load_stopwords(stopwords_path: str | Path) -> set[str]:
    """
    加载停用词。
    """
    return load_word_set(stopwords_path)


def load_user_dict(user_dict_path: str | Path) -> bool:
    """
    加载 jieba 用户词典。
    成功返回 True，失败返回 False。
    """
    path = Path(user_dict_path)
    if not path.exists():
        return False

    try:
        jieba.load_userdict(str(path))
        return True
    except Exception:
        return False


@lru_cache(maxsize=32)
def load_trait_dict(trait_dict_path: str | Path) -> dict:
    """
    加载 trait 字典 JSON。
    文件不存在或解析失败时返回空 dict。
    """
    path = Path(trait_dict_path)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def normalize_text(text: Any) -> str:
    """
    通用文本清洗：
    - 转小写
    - 去 HTML
    - 去 URL
    - 去换行和制表
    - 只保留中英文、数字、+#.
    - 压缩多余空格
    """
    s = to_safe_str(text)
    if not s:
        return ""

    s = s.lower()
    s = RE_HTML.sub(" ", s)
    s = RE_URL.sub(" ", s)
    s = RE_BLANK.sub(" ", s)
    s = RE_KEEP_TEXT.sub(" ", s)
    s = RE_MULTI_SPACE.sub(" ", s).strip()
    return s


def normalize_region(region: Any) -> str:
    """
    地区归一：
    - 北京市 -> 北京
    - 北京-朝阳区 -> 北京
    - 上海 / 上海市 / 上海-浦东 -> 上海
    - 多城市文本中优先返回第一个命中的已知城市
    """
    s = to_safe_str(region)
    if not s:
        return ""

    s = s.replace("·", "-").replace("_", "-").strip()

    if s in CITY_ALIASES:
        return CITY_ALIASES[s]

    parts = [p for p in RE_REGION_SPLIT.split(s) if p]
    for part in parts:
        if part in CITY_ALIASES:
            return CITY_ALIASES[part]

    for city_alias, city_norm in CITY_ALIASES.items():
        if city_alias in s:
            return city_norm

    return s


def tokenize_text(
    text: Any,
    stopwords: set[str] | None = None,
    min_len: int = 2,
    remove_low_value: bool = True,
    auto_normalize: bool = True,
) -> list[str]:
    """
    对文本进行分词和过滤。

    参数：
    - stopwords: 停用词集合
    - min_len: 最小词长度
    - remove_low_value: 是否过滤低信息词
    - auto_normalize: 是否先进行 normalize_text

    返回：
    - 清洗后的 token 列表
    """
    if stopwords is None:
        stopwords = set()

    s = normalize_text(text) if auto_normalize else to_safe_str(text)
    if not s:
        return []

    try:
        words = jieba.lcut(s)
    except Exception:
        return []

    cleaned: list[str] = []
    for w in words:
        token = to_safe_str(w).lower()
        if not token:
            continue
        if len(token) < min_len:
            continue
        if token in stopwords:
            continue
        if remove_low_value and token in LOW_VALUE_WORDS:
            continue
        cleaned.append(token)

    return cleaned
