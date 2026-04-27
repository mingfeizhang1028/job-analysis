# modules/normalization.py

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


UNKNOWN_COMPANY = "未知公司"
UNKNOWN_TITLE = "未知职位"
UNKNOWN_CATEGORY = "未知类别"
DEFAULT_DIRECTION = "通用方向"


COMPANY_SUFFIXES = [
    "集团股份有限公司",
    "信息技术有限公司",
    "网络科技有限公司",
    "科技有限公司",
    "股份有限公司",
    "有限责任公司",
    "技术有限公司",
    "集团有限公司",
    "有限公司",
    "集团",
    "公司",
]

COMMON_AREA_PREFIXES = [
    "北京", "上海", "深圳", "广州", "杭州", "南京", "成都", "武汉", "西安",
    "苏州", "天津", "重庆", "厦门", "长沙", "郑州", "合肥", "青岛",
]

EN_ABBR_MAP = {
    "ai": "AI",
    "llm": "LLM",
    "nlp": "NLP",
    "sql": "SQL",
    "api": "API",
    "rag": "RAG",
    "aigc": "AIGC",
    "bi": "BI",
    "tob": "ToB",
    "toc": "ToC",
}

# 常见招聘噪声词。只做轻量清理，避免过度改写岗位含义。
JOB_NOISE_PATTERNS = [
    r"急聘",
    r"诚聘",
    r"高薪",
    r"双休",
    r"五险一金",
    r"六险一金",
    r"周末双休",
    r"包吃包住",
    r"提供住宿",
    r"可实习",
    r"接受应届生",
    r"应届生可投",
    r"校招",
    r"社招",
]

# 职级词默认不删除，只用于可选的岗位规则判断。
JOB_LEVEL_WORDS = [
    "实习", "初级", "中级", "高级", "资深", "专家", "主管", "经理", "负责人", "总监",
]

RE_CN_PARENS = re.compile(r"[（）]")
RE_PAREN_CONTENT = re.compile(r"（.*?）|\(.*?\)|【.*?】|\[.*?\]")
RE_MULTI_SPACE = re.compile(r"\s+")
RE_COMPANY_SEPARATORS = re.compile(r"[｜|/\\]")
RE_JOB_SEPARATORS_TO_SPACE = re.compile(r"[｜|/\\]")
RE_JOB_REMOVE_SYMBOLS = re.compile(r"[【】\[\]（）(),，。:：;；!！?？\"“”'‘’]")
RE_JOB_DASH_SYMBOLS = re.compile(r"[-—_]+")
RE_CN_CHAR = r"\u4e00-\u9fa5"

# 英文缩写匹配：
# - 支持 AI产品经理、ai 产品经理、产品经理AI
# - 避免替换英文单词内部片段，例如 detail 里的 ai
EN_ABBR_PATTERN = re.compile(
    r"(?<![A-Za-z])(" + "|".join(map(re.escape, sorted(EN_ABBR_MAP, key=len, reverse=True))) + r")(?![A-Za-z])",
    flags=re.IGNORECASE,
)


def is_missing(value: Any) -> bool:
    """统一判断 None / NaN。避免 pd.isna(list) 返回数组导致异常。"""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=32)
def load_json_cached(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def load_json(path: str | Path) -> dict:
    """
    读取 JSON 配置文件。

    设计原则：
    - 文件不存在时返回空 dict。
    - JSON 损坏时不让主流程崩溃，返回空 dict。
    - 使用缓存，避免 apply_normalization 多次调用时重复读文件。
    """
    return load_json_cached(str(Path(path)))


def to_safe_str(value: Any, default: str = "") -> str:
    if is_missing(value):
        return default
    return str(value).strip()


def normalize_spaces(text: str) -> str:
    return RE_MULTI_SPACE.sub(" ", text).strip()


def remove_all_spaces(text: str) -> str:
    return RE_MULTI_SPACE.sub("", text).strip()


def normalize_brackets(text: str) -> str:
    return text.replace("（", "(").replace("）", ")")


def build_alias_reverse_map(alias_dict: dict) -> dict:
    """
    将：
    {
      "快手": ["北京快手科技有限公司", "快手科技"]
    }

    转成：
    {
      "北京快手科技有限公司": "快手",
      "快手科技": "快手",
      "快手": "快手"
    }

    兼容：
    {
      "快手": "北京快手科技有限公司"
    }

    注意：
    - 若不同标准名配置了同一 alias，后出现的会覆盖先出现的。
    - 建议后续在配置校验阶段显式提示冲突。
    """
    if not isinstance(alias_dict, dict):
        return {}

    reverse: dict[str, str] = {}

    for norm_name, aliases in alias_dict.items():
        norm = to_safe_str(norm_name)
        if not norm:
            continue

        reverse[norm] = norm

        if isinstance(aliases, str):
            aliases = [aliases]
        elif not isinstance(aliases, list):
            continue

        for alias in aliases:
            alias_str = to_safe_str(alias)
            if alias_str:
                reverse[alias_str] = norm

                # 同时加入去空格版本，提升命中率。
                alias_no_space = remove_all_spaces(alias_str)
                if alias_no_space:
                    reverse[alias_no_space] = norm

    return reverse


def normalize_english_abbr(text: str) -> str:
    """
    对常见英文缩写做大小写统一。

    相比原实现：
    - 避免对英文单词内部片段误替换。
    - 支持中文相邻场景：ai产品经理 -> AI产品经理。
    """
    if not isinstance(text, str):
        return ""

    def repl(match: re.Match) -> str:
        key = match.group(1).lower()
        return EN_ABBR_MAP.get(key, match.group(1))

    return EN_ABBR_PATTERN.sub(repl, text)


def normalize_company_name(name: Any, alias_map: dict | None = None) -> str:
    """
    企业名称标准化。

    规则优先级：
    1. 空值保护
    2. 原始值精确别名映射
    3. 基础清洗：去括号内容、去空白、统一分隔符
    4. 清洗后别名映射
    5. 去公司后缀
    6. 轻度去地区前缀
    7. 再次别名映射
    8. 兜底返回
    """
    raw = to_safe_str(name)
    if not raw:
        return UNKNOWN_COMPANY

    alias_map = alias_map or {}

    if raw in alias_map:
        return alias_map[raw]

    s = raw

    # 去括号内容：适合企业主体名清洗，如 “北京某某科技有限公司（上海分公司）”
    s = RE_PAREN_CONTENT.sub("", s)

    # 去除常见分隔符后的补充描述，仅保留主体前半部分。
    # 例如：某某科技有限公司｜招聘 -> 某某科技有限公司
    s = RE_COMPANY_SEPARATORS.split(s)[0]

    s = remove_all_spaces(s)

    if s in alias_map:
        return alias_map[s]

    # 去常见公司后缀，先长后短。
    for suffix in COMPANY_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)]
            break

    # 轻度去地区前缀：
    # 仅当去掉地区后仍保留至少 3 个字符时处理，避免“上海电气”等品牌被错误改成“电气”。
    for prefix in COMMON_AREA_PREFIXES:
        if s.startswith(prefix) and len(s) >= len(prefix) + 3:
            candidate = s[len(prefix):]
            if candidate:
                s = candidate
            break

    if s in alias_map:
        return alias_map[s]

    return s or raw or UNKNOWN_COMPANY


def clean_job_title_base(title: str) -> str:
    """
    职位名称基础清洗：
    - 统一括号
    - 统一英文缩写大小写
    - 分隔符转空格
    - 删除常见噪声词
    - 删除标点符号
    - 去除多余空白
    """
    s = normalize_brackets(title)
    s = normalize_spaces(s)
    s = normalize_english_abbr(s)

    # 常见分隔符转空格，避免“AI/产品经理”直接拼成不可读文本。
    s = RE_JOB_SEPARATORS_TO_SPACE.sub(" ", s)
    s = RE_JOB_DASH_SYMBOLS.sub(" ", s)

    for pattern in JOB_NOISE_PATTERNS:
        s = re.sub(pattern, "", s, flags=re.IGNORECASE)

    s = RE_JOB_REMOVE_SYMBOLS.sub("", s)
    s = normalize_spaces(s)

    # 去掉中文与英文缩写之间的多余空格：
    # AI 产品经理 -> AI产品经理；产品经理 AI -> 产品经理AI
    abbr_group = "|".join(map(re.escape, EN_ABBR_MAP.values()))
    s = re.sub(rf"\b({abbr_group})\b\s+([{RE_CN_CHAR}])", r"\1\2", s, flags=re.IGNORECASE)
    s = re.sub(rf"([{RE_CN_CHAR}])\s+\b({abbr_group})\b", r"\1\2", s, flags=re.IGNORECASE)

    return remove_all_spaces(s)


def normalize_job_title(title: Any, title_rules: dict | None = None) -> str:
    """
    职位名称标准化。

    title_rules 推荐结构：
    {
      "normalization": {
        "AI产品经理": "AI产品经理",
        "人工智能产品经理": "AI产品经理"
      },
      "category_rules": {
        "产品": ["产品经理", "产品运营"]
      },
      "direction_rules": {
        "大模型": ["LLM", "大模型"]
      }
    }
    """
    raw = to_safe_str(title)
    if not raw:
        return UNKNOWN_TITLE

    rules = (title_rules or {}).get("normalization", {})
    rules = rules if isinstance(rules, dict) else {}

    # 原始值先做一次精确映射。
    if raw in rules:
        return rules[raw]

    s_clean = clean_job_title_base(raw)

    if not s_clean:
        return UNKNOWN_TITLE

    # 清洗后精确映射。
    if s_clean in rules:
        return rules[s_clean]

    # 轻量规则型归一：尽量保守，避免误合并。
    # 优先识别更具体的大模型产品经理，再识别泛 AI 产品经理。
    if "产品经理" in s_clean and ("大模型" in s_clean or "LLM" in s_clean):
        return "大模型产品经理"

    if "产品经理" in s_clean and ("AI" in s_clean or "人工智能" in s_clean or "AIGC" in s_clean):
        return "AI产品经理"

    if ("数据分析" in s_clean or "BI" in s_clean) and ("工程师" not in s_clean):
        # 保守处理：不把数据开发/数据工程合并到数据分析。
        if "产品" in s_clean:
            return "数据产品经理"
        if "师" in s_clean or "分析" in s_clean:
            return "数据分析师"

    if "算法工程师" in s_clean and ("大模型" in s_clean or "LLM" in s_clean or "NLP" in s_clean):
        return "大模型算法工程师"

    return s_clean or raw or UNKNOWN_TITLE


def _match_keywords(text: str, rules: dict, default: str) -> str:
    if not isinstance(rules, dict) or not text:
        return default

    text_lower = text.lower()

    for label, keywords in rules.items():
        if isinstance(keywords, str):
            keywords = [keywords]
        if not isinstance(keywords, list):
            continue

        for kw in keywords:
            kw_str = to_safe_str(kw)
            if kw_str and kw_str.lower() in text_lower:
                return str(label)

    return default


def classify_job_category(title_norm: Any, title_rules: dict | None = None) -> str:
    title = to_safe_str(title_norm)
    if not title or title == UNKNOWN_TITLE:
        return UNKNOWN_CATEGORY

    rules = (title_rules or {}).get("category_rules", {})
    return _match_keywords(title, rules, "其他")


def classify_job_direction(
    title_norm: Any,
    detail_text: Any = "",
    title_rules: dict | None = None,
) -> str:
    title = to_safe_str(title_norm)
    detail = to_safe_str(detail_text)
    text = f"{title} {detail}".strip()

    rules = (title_rules or {}).get("direction_rules", {})
    return _match_keywords(text, rules, DEFAULT_DIRECTION)


def apply_normalization(
    df: pd.DataFrame,
    company_alias_path: str = "data/company_alias.json",
    job_title_rules_path: str = "data/job_title_rules.json",
    company_col: str = "企业名称",
    title_col: str = "职位名称",
    detail_col: str = "岗位详情",
) -> pd.DataFrame:
    """
    对招聘岗位 DataFrame 应用企业名称和职位名称标准化。

    输出字段：
    - 企业名称_raw
    - 企业名称_norm
    - 职位名称_raw
    - 职位名称_norm
    - 职位类别
    - 职位方向

    设计原则：
    - 不修改原始输入 DataFrame。
    - 缺失列时生成安全兜底字段，避免下游页面直接报错。
    - raw 字段保留原值，norm 字段用于聚合分析。
    """
    if df is None:
        df = pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("apply_normalization expects a pandas.DataFrame")

    df = df.copy()

    alias_dict = load_json(company_alias_path)
    alias_map = build_alias_reverse_map(alias_dict)

    title_rules = load_json(job_title_rules_path)
    if not isinstance(title_rules, dict):
        title_rules = {}

    if company_col in df.columns:
        df["企业名称_raw"] = df[company_col]
        df["企业名称_norm"] = df[company_col].map(lambda x: normalize_company_name(x, alias_map))
    else:
        df["企业名称_raw"] = UNKNOWN_COMPANY
        df["企业名称_norm"] = UNKNOWN_COMPANY

    if title_col in df.columns:
        df["职位名称_raw"] = df[title_col]
        df["职位名称_norm"] = df[title_col].map(lambda x: normalize_job_title(x, title_rules))
    else:
        df["职位名称_raw"] = UNKNOWN_TITLE
        df["职位名称_norm"] = UNKNOWN_TITLE

    if detail_col not in df.columns:
        df[detail_col] = ""

    df["职位类别"] = df["职位名称_norm"].map(lambda x: classify_job_category(x, title_rules))

    # apply axis=1 在这里可接受，因为方向规则通常需要拼接 detail。
    df["职位方向"] = df.apply(
        lambda r: classify_job_direction(
            r.get("职位名称_norm", ""),
            r.get(detail_col, ""),
            title_rules,
        ),
        axis=1,
    )

    return df
