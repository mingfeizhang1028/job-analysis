from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:50:59 2026

@author: Rosem
"""

# -*- coding: utf-8 -*-
"""
CFPS 变量协调半自动系统
版本：Python 3.10+
作者：ChatGPT（按用户研究框架定制）
用途：
1．读取已有元数据 Excel：cfps_metadata.xlsx
2．构建 variable_cards enriched
3．生成候选变量对 candidate_pairs
4．为候选对构造 LLM prompt
5．调用 OpenAI 兼容 API（预留 provider / model / api base / api key）
6．解析结构化 JSON 结果
7．导出人工审核模板 harmonization_review_template

依赖：
pip install pandas numpy openpyxl rapidfuzz requests tqdm

说明：
1．默认使用 OpenAI 兼容 Chat Completions API。
2．你只需要修改“0．配置区”。
3．所有随机过程固定 seed。
4．为了可复现，程序会保存：
   - 候选对
   - prompts
   - 原始 responses
   - 解析结果
   - 审核模板

建议：
1．先将 RUN_LLM = False，检查 candidate_pairs 是否合理；
2．确认无误后，再将 RUN_LLM = True；
3．首次建议 MAX_CANDIDATES_FOR_LLM 控制在 100～300。
"""


import os
import re
import json
import time
import math
import random
import hashlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz
from tqdm import tqdm


# =========================================================
# 0．配置区：你只需要改这里
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 元数据文件
METADATA_XLSX = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\cfps_metadata.xlsx")

# 输出目录
OUTPUT_DIR = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\harmonization_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取哪些 sheet
SHEET_VARIABLE_DICTIONARY = "variable_dictionary"
SHEET_VALUE_LABELS = "value_labels"
SHEET_VAR_YEAR_SUMMARY = "var_year_summary"

# 是否运行 LLM
RUN_LLM = False

# API 配置：OpenAI 兼容接口
MODEL_PROVIDER = "openai_compatible"
API_BASE = "https://api2.zufe.top"
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL_NAME = "gpt-5.4"

# 也可填成其它兼容服务：
# API_BASE = "https://api.deepseek.com/v1"
# MODEL_NAME = "deepseek-chat"

# 请求参数
LLM_TIMEOUT = 120
LLM_MAX_RETRIES = 3
LLM_RETRY_SLEEP = 5
LLM_TEMPERATURE = 0.0

# 候选对控制
MAX_CANDIDATES_TOTAL = 5000
MAX_CANDIDATES_FOR_LLM = 300
MIN_FINAL_SCORE = 45

# 是否只比较相邻年份或同年份
PRIORITIZE_NEARBY_YEARS = True
YEAR_GAP_PREFERRED = 4

# 主题关键词，用于召回与聚类
TOPIC_KEYWORDS = {
    "income": ["income", "wage", "earn", "salary", "收入", "工资", "报酬", "务工"],
    "internet_access": ["internet", "online", "net", "上网", "互联网", "手机上网", "电脑上网"],
    "learning": ["learn", "study", "train", "教育", "学习", "培训", "网课"],
    "shopping": ["shop", "buy", "purchase", "购物", "网购", "消费"],
    "social": ["social", "wechat", "chat", "share", "社交", "微信", "朋友圈"],
    "entertainment": ["game", "video", "music", "entertainment", "娱乐", "游戏", "短视频"],
    "work": ["work", "job", "employ", "occupation", "工作", "就业", "职业"],
    "education": ["edu", "school", "degree", "education", "学历", "教育程度", "受教育年限"],
    "hukou": ["hukou", "户口", "农业户口", "非农"],
    "health": ["health", "healthy", "ill", "病", "健康", "身体"],
    "marriage": ["marry", "marriage", "婚姻", "结婚", "配偶"],
    "ethnicity": ["minzu", "ethnic", "民族", "汉族"],
    "party": ["party", "党员", "党"],
    "region": ["prov", "province", "city", "region", "地区", "省", "市"],
}

# 文本字段长度控制
MAX_VALUE_LABEL_TEXT_LEN = 1200
MAX_CONTEXT_TEXT_LEN = 2400
MAX_PROMPT_FIELD_LEN = 1800

# 是否跳过已存在 response 的记录
SKIP_IF_ALREADY_DONE = True

# 是否导出低分候选
EXPORT_ALL_CANDIDATES = True


# =========================================================
# 1．基础工具函数
# =========================================================

def normalize_text(x: Any) -> str:
    if pd.isna(x) or x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def short_text(x: Any, max_len: int = 300) -> str:
    s = "" if x is None or pd.isna(x) else str(x)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def safe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def list_to_str(lst: List[Any], sep: str = " | ") -> str:
    vals = [str(i) for i in lst if i is not None and str(i).strip() != ""]
    return sep.join(vals)


def tokenize_simple(text: str) -> List[str]:
    s = normalize_text(text)
    if not s:
        return []
    tokens = s.split()
    return [t for t in tokens if t]


def infer_topics(text: str) -> List[str]:
    s = normalize_text(text)
    hits = []
    for topic, kws in TOPIC_KEYWORDS.items():
        for kw in kws:
            if normalize_text(kw) in s:
                hits.append(topic)
                break
    return sorted(set(hits))


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def build_value_label_text(value_label_df: pd.DataFrame) -> str:
    if value_label_df is None or value_label_df.empty:
        return ""

    tmp = value_label_df.copy()

    def parse_code_for_sort(x):
        if pd.isna(x):
            return (2, "")
        s = str(x).strip()

        # 优先按数值排序
        try:
            return (0, float(s))
        except Exception:
            return (1, s.lower())

    tmp["_code_sort_key"] = tmp["code"].apply(parse_code_for_sort)
    tmp = tmp.sort_values(by="_code_sort_key", na_position="last")

    rows = []
    for _, r in tmp.iterrows():
        code = r.get("code", "")
        lab = r.get("label", "")
        rows.append(f"{code}={lab}")

    txt = "；".join(rows)
    if len(txt) > MAX_VALUE_LABEL_TEXT_LEN:
        txt = txt[:MAX_VALUE_LABEL_TEXT_LEN] + "..."
    return txt

def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_json_from_response(text: str) -> Optional[dict]:
    if text is None:
        return None
    text = text.strip()

    # 直接 parse
    obj = safe_json_loads(text)
    if obj is not None:
        return obj

    # ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        obj = safe_json_loads(m.group(1))
        if obj is not None:
            return obj

    # 最外层 { ... }
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        obj = safe_json_loads(m.group(1))
        if obj is not None:
            return obj

    return None


# =========================================================
# 2．读取元数据
# =========================================================

def load_metadata(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"未找到元数据文件：{xlsx_path}")

    variable_dictionary = pd.read_excel(xlsx_path, sheet_name=SHEET_VARIABLE_DICTIONARY)
    value_labels = pd.read_excel(xlsx_path, sheet_name=SHEET_VALUE_LABELS)
    var_year_summary = pd.read_excel(xlsx_path, sheet_name=SHEET_VAR_YEAR_SUMMARY)

    return variable_dictionary, value_labels, var_year_summary


# =========================================================
# 3．构建 variable_cards enriched
# =========================================================

def aggregate_value_labels(value_labels: pd.DataFrame) -> pd.DataFrame:
    if value_labels.empty:
        return pd.DataFrame(columns=["file_path", "value_label_name", "value_labels_text"])

    grouped = (
        value_labels
        .groupby(["file_path", "value_label_name"], dropna=False)
        .apply(build_value_label_text)
        .reset_index(name="value_labels_text")
    )
    return grouped


def build_variable_cards_from_metadata(
    variable_dictionary: pd.DataFrame,
    value_labels: pd.DataFrame
) -> pd.DataFrame:

    vdict = variable_dictionary.copy()
    vdict["var_name_norm"] = vdict["var_name"].apply(normalize_text)
    vdict["var_label_norm"] = vdict["var_label"].apply(normalize_text)
    vdict["file_name_norm"] = vdict["file_name"].apply(normalize_text)

    value_label_text_df = aggregate_value_labels(value_labels)

    cards = vdict.merge(
        value_label_text_df,
        on=["file_path", "value_label_name"],
        how="left"
    )

    cards["year"] = cards["year"].apply(safe_int)
    cards["unique_n"] = cards["unique_n"].apply(safe_int)
    cards["min"] = cards["min"].apply(safe_float)
    cards["max"] = cards["max"].apply(safe_float)

    cards["measure"] = cards["measure"].fillna("")
    cards["source_stage"] = cards["source_stage"].fillna("")
    cards["dataset_class"] = cards["dataset_class"].fillna("")
    cards["survey_role"] = cards["survey_role"].fillna("")
    cards["path_group"] = cards["path_group"].fillna("")
    cards["special_missing_candidates"] = cards["special_missing_candidates"].fillna("")

    cards["topic_hits"] = cards.apply(
        lambda r: ",".join(infer_topics(" ".join([
            str(r.get("var_name", "")),
            str(r.get("var_label", "")),
            str(r.get("value_labels_text", "")),
            str(r.get("file_name", "")),
            str(r.get("source_note", "")),
        ]))),
        axis=1
    )

    def make_context(r):
        parts = [
            f"变量名：{r.get('var_name', '')}",
            f"变量标签：{r.get('var_label', '')}",
            f"年份：{r.get('year', '')}",
            f"文件：{r.get('file_name', '')}",
            f"数据阶段：{r.get('source_stage', '')}",
            f"数据类别：{r.get('dataset_class', '')}",
            f"问卷角色：{r.get('survey_role', '')}",
            f"测量类型：{r.get('measure', '')}",
            f"唯一值数：{r.get('unique_n', '')}",
            f"最小值：{r.get('min', '')}",
            f"最大值：{r.get('max', '')}",
            f"特殊缺失候选：{r.get('special_missing_candidates', '')}",
            f"值标签：{r.get('value_labels_text', '')}",
            f"主题标签：{r.get('topic_hits', '')}",
        ]
        txt = "；".join([p for p in parts if p.strip("；") != ""])
        if len(txt) > MAX_CONTEXT_TEXT_LEN:
            txt = txt[:MAX_CONTEXT_TEXT_LEN] + "..."
        return txt

    cards["context_text"] = cards.apply(make_context, axis=1)

    cards["card_id"] = cards.apply(
        lambda r: hash_text(
            "||".join([
                str(r.get("file_path", "")),
                str(r.get("var_name", "")),
                str(r.get("year", "")),
                str(r.get("var_label", "")),
            ])
        ),
        axis=1
    )

    cards["is_binary_like"] = cards["unique_n"].apply(
        lambda x: 1 if x is not None and pd.notna(x) and 1 <= x <= 3 else 0
    )
    cards["is_ordered_like"] = cards["unique_n"].apply(
        lambda x: 1 if x is not None and pd.notna(x) and 3 <= x <= 10 else 0
    )

    return cards


# =========================================================
# 4．候选变量对生成
# =========================================================

def score_name_similarity(a_name: str, b_name: str) -> float:
    if not a_name or not b_name:
        return 0.0
    return max(
        fuzz.ratio(a_name, b_name),
        fuzz.partial_ratio(a_name, b_name),
        fuzz.token_sort_ratio(a_name, b_name)
    ) / 100.0


def score_label_similarity(a_label: str, b_label: str) -> float:
    if not a_label or not b_label:
        return 0.0
    return max(
        fuzz.ratio(a_label, b_label),
        fuzz.partial_ratio(a_label, b_label),
        fuzz.token_sort_ratio(a_label, b_label)
    ) / 100.0


def score_value_label_similarity(a_vl: str, b_vl: str) -> float:
    if not a_vl or not b_vl:
        return 0.0
    return max(
        fuzz.ratio(a_vl, b_vl),
        fuzz.partial_ratio(a_vl, b_vl),
        fuzz.token_sort_ratio(a_vl, b_vl)
    ) / 100.0


def score_topic_similarity(a_topics: str, b_topics: str) -> float:
    at = [i for i in str(a_topics).split(",") if i]
    bt = [i for i in str(b_topics).split(",") if i]
    return jaccard_similarity(at, bt)


def score_year_proximity(a_year: Optional[int], b_year: Optional[int]) -> float:
    if a_year is None or b_year is None:
        return 0.0
    gap = abs(a_year - b_year)
    if gap == 0:
        return 1.0
    if gap == YEAR_GAP_PREFERRED:
        return 0.9
    if gap <= 2:
        return 0.8
    if gap <= 4:
        return 0.7
    if gap <= 6:
        return 0.5
    return 0.2


def score_role_match(a_role: str, b_role: str) -> float:
    if not a_role or not b_role:
        return 0.0
    return 1.0 if a_role == b_role else 0.3


def score_dataset_match(a_ds: str, b_ds: str) -> float:
    if not a_ds or not b_ds:
        return 0.0
    if a_ds == b_ds:
        return 1.0
    if "raw" in a_ds and "raw" in b_ds:
        return 0.8
    if "processed" in a_ds and "processed" in b_ds:
        return 0.8
    return 0.3


def score_measure_match(a_m: str, b_m: str) -> float:
    if not a_m or not b_m:
        return 0.0
    return 1.0 if a_m == b_m else 0.2


def score_type_shape(a_unique, b_unique, a_min, a_max, b_min, b_max) -> float:
    score = 0.0
    try:
        if a_unique is not None and b_unique is not None:
            diff = abs(int(a_unique) - int(b_unique))
            if diff == 0:
                score += 0.5
            elif diff <= 2:
                score += 0.35
            elif diff <= 5:
                score += 0.2

        if all(v is not None for v in [a_min, a_max, b_min, b_max]):
            ra = float(a_max) - float(a_min)
            rb = float(b_max) - float(b_min)
            if ra == rb:
                score += 0.5
            elif abs(ra - rb) <= 2:
                score += 0.3
            elif abs(ra - rb) <= 5:
                score += 0.15
    except Exception:
        pass
    return min(score, 1.0)


def build_pair_reason(row: dict) -> str:
    reasons = []
    if row["name_sim"] >= 0.75:
        reasons.append("变量名高度相似")
    if row["label_sim"] >= 0.75:
        reasons.append("变量标签高度相似")
    if row["value_label_sim"] >= 0.75:
        reasons.append("值标签结构高度相似")
    if row["topic_sim"] >= 0.5:
        reasons.append("主题关键词重合")
    if row["role_match"] >= 0.9:
        reasons.append("问卷角色一致")
    if row["dataset_match"] >= 0.8:
        reasons.append("数据类别接近")
    if row["year_prox"] >= 0.7:
        reasons.append("年份接近")
    if row["shape_sim"] >= 0.5:
        reasons.append("取值结构相近")

    return "；".join(reasons)


def score_candidate_pair(a: pd.Series, b: pd.Series) -> dict:
    a_name = normalize_text(a.get("var_name", ""))
    b_name = normalize_text(b.get("var_name", ""))
    a_label = normalize_text(a.get("var_label", ""))
    b_label = normalize_text(b.get("var_label", ""))
    a_vl = normalize_text(a.get("value_labels_text", ""))
    b_vl = normalize_text(b.get("value_labels_text", ""))
    a_topics = a.get("topic_hits", "")
    b_topics = b.get("topic_hits", "")

    name_sim = score_name_similarity(a_name, b_name)
    label_sim = score_label_similarity(a_label, b_label)
    value_label_sim = score_value_label_similarity(a_vl, b_vl)
    topic_sim = score_topic_similarity(a_topics, b_topics)
    year_prox = score_year_proximity(safe_int(a.get("year")), safe_int(b.get("year")))
    role_match = score_role_match(str(a.get("survey_role", "")), str(b.get("survey_role", "")))
    dataset_match = score_dataset_match(str(a.get("dataset_class", "")), str(b.get("dataset_class", "")))
    measure_match = score_measure_match(str(a.get("measure", "")), str(b.get("measure", "")))
    shape_sim = score_type_shape(
        a.get("unique_n"), b.get("unique_n"),
        a.get("min"), a.get("max"),
        b.get("min"), b.get("max")
    )

    # 加权总分
    final_score = (
        0.24 * name_sim +
        0.24 * label_sim +
        0.12 * value_label_sim +
        0.14 * topic_sim +
        0.08 * year_prox +
        0.06 * role_match +
        0.05 * dataset_match +
        0.03 * measure_match +
        0.04 * shape_sim
    ) * 100

    return {
        "name_sim": round(name_sim, 4),
        "label_sim": round(label_sim, 4),
        "value_label_sim": round(value_label_sim, 4),
        "topic_sim": round(topic_sim, 4),
        "year_prox": round(year_prox, 4),
        "role_match": round(role_match, 4),
        "dataset_match": round(dataset_match, 4),
        "measure_match": round(measure_match, 4),
        "shape_sim": round(shape_sim, 4),
        "final_score": round(final_score, 2),
    }


def should_compare(a: pd.Series, b: pd.Series) -> bool:
    # 不与自己比
    if a["card_id"] == b["card_id"]:
        return False

    # 宏观变量和 CFPS 个体变量通常不直接比较
    a_stage = str(a.get("source_stage", ""))
    b_stage = str(b.get("source_stage", ""))
    if a_stage != b_stage:
        # 若一个是 macro，另一个不是，通常不比较
        if "macro" in {a_stage, b_stage}:
            return False

    # 如果都没有标签、名字也完全不相近，则跳过
    a_name = normalize_text(a.get("var_name", ""))
    b_name = normalize_text(b.get("var_name", ""))
    a_label = normalize_text(a.get("var_label", ""))
    b_label = normalize_text(b.get("var_label", ""))

    rough_name = score_name_similarity(a_name, b_name)
    rough_label = score_label_similarity(a_label, b_label)

    # 主题命中
    a_topics = set([i for i in str(a.get("topic_hits", "")).split(",") if i])
    b_topics = set([i for i in str(b.get("topic_hits", "")).split(",") if i])
    has_topic_overlap = len(a_topics & b_topics) > 0

    # 同名跨年必须保留
    if a_name and b_name and a_name == b_name:
        return True

    # 标签一致也保留
    if a_label and b_label and a_label == b_label:
        return True

    # 名字或标签粗相似
    if rough_name >= 0.45 or rough_label >= 0.50:
        return True

    # 主题重合且年份接近
    a_year = safe_int(a.get("year"))
    b_year = safe_int(b.get("year"))
    if has_topic_overlap:
        if a_year is None or b_year is None:
            return True
        if abs(a_year - b_year) <= 6:
            return True

    return False


def generate_candidate_pairs(cards: pd.DataFrame) -> pd.DataFrame:
    records = []
    n = len(cards)

    cards = cards.reset_index(drop=True)

    for i in tqdm(range(n), desc="生成候选变量对"):
        a = cards.iloc[i]

        for j in range(i + 1, n):
            b = cards.iloc[j]

            if not should_compare(a, b):
                continue

            pair_score = score_candidate_pair(a, b)

            if pair_score["final_score"] < MIN_FINAL_SCORE:
                if not EXPORT_ALL_CANDIDATES:
                    continue

            pair_id = hash_text(a["card_id"] + "||" + b["card_id"])

            rec = {
                "pair_id": pair_id,

                "a_card_id": a["card_id"],
                "a_var_name": a.get("var_name"),
                "a_var_label": a.get("var_label"),
                "a_file_name": a.get("file_name"),
                "a_file_path": a.get("file_path"),
                "a_year": a.get("year"),
                "a_source_stage": a.get("source_stage"),
                "a_dataset_class": a.get("dataset_class"),
                "a_survey_role": a.get("survey_role"),
                "a_measure": a.get("measure"),
                "a_unique_n": a.get("unique_n"),
                "a_min": a.get("min"),
                "a_max": a.get("max"),
                "a_value_labels_text": short_text(a.get("value_labels_text", ""), 800),
                "a_context_text": short_text(a.get("context_text", ""), 1200),
                "a_topic_hits": a.get("topic_hits"),

                "b_card_id": b["card_id"],
                "b_var_name": b.get("var_name"),
                "b_var_label": b.get("var_label"),
                "b_file_name": b.get("file_name"),
                "b_file_path": b.get("file_path"),
                "b_year": b.get("year"),
                "b_source_stage": b.get("source_stage"),
                "b_dataset_class": b.get("dataset_class"),
                "b_survey_role": b.get("survey_role"),
                "b_measure": b.get("measure"),
                "b_unique_n": b.get("unique_n"),
                "b_min": b.get("min"),
                "b_max": b.get("max"),
                "b_value_labels_text": short_text(b.get("value_labels_text", ""), 800),
                "b_context_text": short_text(b.get("context_text", ""), 1200),
                "b_topic_hits": b.get("topic_hits"),
            }
            rec.update(pair_score)
            rec["pair_reason"] = build_pair_reason(rec)

            records.append(rec)

    cand = pd.DataFrame(records)

    if cand.empty:
        return cand

    # 排序
    cand = cand.sort_values(
        by=["final_score", "label_sim", "name_sim", "value_label_sim"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    # 去重策略：同名同标签同年同文件的重复配对一般不会出现；pair_id 已唯一
    cand = cand.drop_duplicates(subset=["pair_id"])

    # 若设置总量上限
    if MAX_CANDIDATES_TOTAL is not None and len(cand) > MAX_CANDIDATES_TOTAL:
        cand = cand.head(MAX_CANDIDATES_TOTAL).copy()

    return cand


# =========================================================
# 5．LLM Prompt 设计
# =========================================================

SYSTEM_PROMPT = """
你是经济学与社会调查方法领域的高级研究助理，任务是帮助识别 CFPS 变量之间的关系，以支持跨年变量协调（harmonization）。
请严格遵守以下原则：
1．你不能武断认定两个变量“完全相同”，除非证据充分。
2．如果两个变量只是概念相关，但题意、量纲、对象、时间窗、编码方式不同，则不得判定为同一变量。
3．若变量可通过重编码、取并集、取最大值、构造上位概念等方式进入同一分析维度，可以判定为 harmonizable 或 constructible。
4．如果信息不足，请输出 uncertain。
5．请只输出 JSON，不要输出额外解释文字，不要加 markdown。
6．relation_type 只能取以下之一：
   same_variable
   harmonizable
   constructible
   related_but_distinct
   not_comparable
   uncertain

输出 JSON 格式必须包含以下字段：
{
  "relation_type": "...",
  "confidence": 0.0,
  "reason": "...",
  "suggested_unified_name": "...",
  "suggested_recoding": "...",
  "comparability_risk": "low | medium | high",
  "needs_human_review": true,
  "key_evidence": ["...", "...", "..."]
}
""".strip()


def truncate_for_prompt(s: Any, max_len: int = MAX_PROMPT_FIELD_LEN) -> str:
    s = "" if s is None or pd.isna(s) else str(s)
    return s if len(s) <= max_len else s[:max_len] + "..."


def build_llm_prompt(pair_row: pd.Series) -> str:
    content = {
        "task": "判断以下两个变量是否可视为同一变量、可协调变量、可作为上位概念构造项，或不可比较。",
        "variable_a": {
            "var_name": truncate_for_prompt(pair_row.get("a_var_name")),
            "var_label": truncate_for_prompt(pair_row.get("a_var_label")),
            "file_name": truncate_for_prompt(pair_row.get("a_file_name")),
            "year": pair_row.get("a_year"),
            "source_stage": truncate_for_prompt(pair_row.get("a_source_stage")),
            "dataset_class": truncate_for_prompt(pair_row.get("a_dataset_class")),
            "survey_role": truncate_for_prompt(pair_row.get("a_survey_role")),
            "measure": truncate_for_prompt(pair_row.get("a_measure")),
            "unique_n": pair_row.get("a_unique_n"),
            "min": pair_row.get("a_min"),
            "max": pair_row.get("a_max"),
            "value_labels_text": truncate_for_prompt(pair_row.get("a_value_labels_text")),
            "context_text": truncate_for_prompt(pair_row.get("a_context_text")),
            "topic_hits": truncate_for_prompt(pair_row.get("a_topic_hits")),
        },
        "variable_b": {
            "var_name": truncate_for_prompt(pair_row.get("b_var_name")),
            "var_label": truncate_for_prompt(pair_row.get("b_var_label")),
            "file_name": truncate_for_prompt(pair_row.get("b_file_name")),
            "year": pair_row.get("b_year"),
            "source_stage": truncate_for_prompt(pair_row.get("b_source_stage")),
            "dataset_class": truncate_for_prompt(pair_row.get("b_dataset_class")),
            "survey_role": truncate_for_prompt(pair_row.get("b_survey_role")),
            "measure": truncate_for_prompt(pair_row.get("b_measure")),
            "unique_n": pair_row.get("b_unique_n"),
            "min": pair_row.get("b_min"),
            "max": pair_row.get("b_max"),
            "value_labels_text": truncate_for_prompt(pair_row.get("b_value_labels_text")),
            "context_text": truncate_for_prompt(pair_row.get("b_context_text")),
            "topic_hits": truncate_for_prompt(pair_row.get("b_topic_hits")),
        },
        "rule_based_scores": {
            "name_sim": pair_row.get("name_sim"),
            "label_sim": pair_row.get("label_sim"),
            "value_label_sim": pair_row.get("value_label_sim"),
            "topic_sim": pair_row.get("topic_sim"),
            "year_prox": pair_row.get("year_prox"),
            "role_match": pair_row.get("role_match"),
            "dataset_match": pair_row.get("dataset_match"),
            "measure_match": pair_row.get("measure_match"),
            "shape_sim": pair_row.get("shape_sim"),
            "final_score": pair_row.get("final_score"),
            "pair_reason": truncate_for_prompt(pair_row.get("pair_reason")),
        },
        "decision_rules": {
            "same_variable": "题意、对象、量纲、编码方式基本一致，仅可能是跨年改名、前缀变化、命名习惯不同。",
            "harmonizable": "核心含义相近，可通过明确重编码规则统一为一个分析变量。",
            "constructible": "不能直接视作同一原始变量，但可作为同一上位概念的组成项来构造新变量。",
            "related_but_distinct": "概念相关，但应保留为不同变量。",
            "not_comparable": "对象、量纲、题意或时间窗不同，不应合并。",
            "uncertain": "信息不足，必须人工核查问卷原文。"
        }
    }
    return json.dumps(content, ensure_ascii=False)


# =========================================================
# 6．API 调用
# =========================================================

def call_openai_compatible_api(
    system_prompt: str,
    user_prompt: str,
    api_base: str,
    api_key: str,
    model_name: str,
    temperature: float = 0.0,
    timeout: int = 120
) -> dict:

    if not api_key:
        raise ValueError("API_KEY 为空，请在配置区填写，或设置环境变量 LLM_API_KEY。")

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def parse_chat_completion_content(resp_json: dict) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp_json, ensure_ascii=False)


def call_llm_api(user_prompt: str) -> Tuple[str, dict]:
    if MODEL_PROVIDER != "openai_compatible":
        raise NotImplementedError(f"暂未实现 provider：{MODEL_PROVIDER}")

    last_err = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp_json = call_openai_compatible_api(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                api_base=API_BASE,
                api_key=API_KEY,
                model_name=MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                timeout=LLM_TIMEOUT
            )
            content = parse_chat_completion_content(resp_json)
            return content, resp_json
        except Exception as e:
            last_err = e
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_SLEEP)
            else:
                raise last_err


# =========================================================
# 7．运行 LLM 并保存中间结果
# =========================================================

def load_done_pair_ids(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    try:
        df = pd.read_csv(results_path)
        if "pair_id" in df.columns:
            return set(df["pair_id"].astype(str).tolist())
        return set()
    except Exception:
        return set()


def normalize_llm_result(obj: dict) -> dict:
    if obj is None:
        return {
            "relation_type": "uncertain",
            "confidence": None,
            "reason": "",
            "suggested_unified_name": "",
            "suggested_recoding": "",
            "comparability_risk": "high",
            "needs_human_review": True,
            "key_evidence": "",
        }

    relation_type = obj.get("relation_type", "uncertain")
    if relation_type not in {
        "same_variable", "harmonizable", "constructible",
        "related_but_distinct", "not_comparable", "uncertain"
    }:
        relation_type = "uncertain"

    risk = obj.get("comparability_risk", "high")
    if risk not in {"low", "medium", "high"}:
        risk = "high"

    key_evidence = obj.get("key_evidence", [])
    if isinstance(key_evidence, list):
        key_evidence = " | ".join([str(i) for i in key_evidence])
    else:
        key_evidence = str(key_evidence)

    confidence = obj.get("confidence", None)
    try:
        if confidence is not None:
            confidence = float(confidence)
    except Exception:
        confidence = None

    return {
        "relation_type": relation_type,
        "confidence": confidence,
        "reason": str(obj.get("reason", "")),
        "suggested_unified_name": str(obj.get("suggested_unified_name", "")),
        "suggested_recoding": str(obj.get("suggested_recoding", "")),
        "comparability_risk": risk,
        "needs_human_review": bool(obj.get("needs_human_review", True)),
        "key_evidence": key_evidence,
    }


def run_llm_on_candidates(candidates: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    prompts_jsonl = out_dir / "llm_prompts.jsonl"
    raw_responses_jsonl = out_dir / "llm_raw_responses.jsonl"
    parsed_results_csv = out_dir / "llm_harmonization_results.csv"

    done_pair_ids = load_done_pair_ids(parsed_results_csv) if SKIP_IF_ALREADY_DONE else set()

    # 仅送前 N 条高分候选
    to_run = candidates.copy().head(MAX_CANDIDATES_FOR_LLM).copy()

    results = []
    if parsed_results_csv.exists():
        try:
            old = pd.read_csv(parsed_results_csv)
            results.extend(old.to_dict("records"))
        except Exception:
            pass

    with open(prompts_jsonl, "a", encoding="utf-8") as f_prompt, \
         open(raw_responses_jsonl, "a", encoding="utf-8") as f_resp:

        for _, row in tqdm(to_run.iterrows(), total=len(to_run), desc="LLM 判别"):
            pair_id = str(row["pair_id"])

            if pair_id in done_pair_ids:
                continue

            prompt = build_llm_prompt(row)

            f_prompt.write(json.dumps({
                "pair_id": pair_id,
                "prompt": prompt
            }, ensure_ascii=False) + "\n")
            f_prompt.flush()

            raw_text = ""
            raw_json = None
            parse_ok = False
            parse_obj = None
            err_msg = ""

            try:
                raw_text, raw_json = call_llm_api(prompt)
                parse_obj = extract_json_from_response(raw_text)
                parse_ok = parse_obj is not None
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"

            f_resp.write(json.dumps({
                "pair_id": pair_id,
                "raw_text": raw_text,
                "raw_json": raw_json,
                "parse_ok": parse_ok,
                "error": err_msg,
            }, ensure_ascii=False) + "\n")
            f_resp.flush()

            norm = normalize_llm_result(parse_obj)

            result_row = {
                "pair_id": pair_id,
                "a_var_name": row.get("a_var_name"),
                "a_var_label": row.get("a_var_label"),
                "a_file_name": row.get("a_file_name"),
                "a_year": row.get("a_year"),
                "b_var_name": row.get("b_var_name"),
                "b_var_label": row.get("b_var_label"),
                "b_file_name": row.get("b_file_name"),
                "b_year": row.get("b_year"),
                "final_score": row.get("final_score"),
                "pair_reason": row.get("pair_reason"),
                "parse_ok": parse_ok,
                "api_error": err_msg,
                "raw_text": raw_text,
            }
            result_row.update(norm)
            results.append(result_row)

            pd.DataFrame(results).to_csv(parsed_results_csv, index=False, encoding="utf-8-sig")

            # 稍微 sleep，避免触发限流
            time.sleep(0.5)

    df_results = pd.DataFrame(results)
    return df_results


# =========================================================
# 8．人工审核模板
# =========================================================

def build_review_template(
    candidates: pd.DataFrame,
    llm_results: Optional[pd.DataFrame] = None
) -> pd.DataFrame:

    base_cols = [
        "pair_id",
        "a_var_name", "a_var_label", "a_file_name", "a_year",
        "b_var_name", "b_var_label", "b_file_name", "b_year",
        "name_sim", "label_sim", "value_label_sim", "topic_sim",
        "year_prox", "role_match", "dataset_match", "measure_match",
        "shape_sim", "final_score", "pair_reason",
    ]

    review = candidates[base_cols].copy()

    if llm_results is not None and not llm_results.empty:
        keep_cols = [
            "pair_id",
            "relation_type", "confidence", "reason",
            "suggested_unified_name", "suggested_recoding",
            "comparability_risk", "needs_human_review",
            "key_evidence", "parse_ok", "api_error"
        ]
        tmp = llm_results[keep_cols].drop_duplicates(subset=["pair_id"])
        review = review.merge(tmp, on="pair_id", how="left")
    else:
        review["relation_type"] = ""
        review["confidence"] = ""
        review["reason"] = ""
        review["suggested_unified_name"] = ""
        review["suggested_recoding"] = ""
        review["comparability_risk"] = ""
        review["needs_human_review"] = ""
        review["key_evidence"] = ""
        review["parse_ok"] = ""
        review["api_error"] = ""

    # 人工审核列
    review["human_decision"] = ""
    review["final_harmonized_var"] = ""
    review["approved_recode_rule"] = ""
    review["approved_relation_type"] = ""
    review["review_notes"] = ""
    review["reviewer"] = ""
    review["review_date"] = ""

    return review


def build_grouped_variable_suggestions(llm_results: pd.DataFrame) -> pd.DataFrame:
    if llm_results is None or llm_results.empty:
        return pd.DataFrame(columns=[
            "suggested_unified_name", "pair_count", "member_examples", "risk_levels"
        ])

    df = llm_results.copy()
    df["suggested_unified_name"] = df["suggested_unified_name"].fillna("").astype(str)

    df = df[df["suggested_unified_name"].str.strip() != ""].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "suggested_unified_name", "pair_count", "member_examples", "risk_levels"
        ])

    rows = []
    for name, g in df.groupby("suggested_unified_name", dropna=False):
        members = set()
        for _, r in g.iterrows():
            members.add(f"{r.get('a_var_name')}({r.get('a_year')})")
            members.add(f"{r.get('b_var_name')}({r.get('b_year')})")
        rows.append({
            "suggested_unified_name": name,
            "pair_count": len(g),
            "member_examples": " | ".join(sorted(members)[:50]),
            "risk_levels": ",".join(sorted(set(g["comparability_risk"].fillna("").astype(str).tolist()))),
        })
    out = pd.DataFrame(rows).sort_values(by=["pair_count", "suggested_unified_name"], ascending=[False, True])
    return out


# =========================================================
# 9．导出
# =========================================================

def export_excel_with_sheets(path: Path, sheets: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            if df is None:
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


# =========================================================
# 10．主程序
# =========================================================

def main():
    print("=" * 80)
    print("CFPS 变量协调半自动系统启动")
    print("=" * 80)

    print("第 1 步：读取元数据")
    variable_dictionary, value_labels, var_year_summary = load_metadata(METADATA_XLSX)
    print(f"variable_dictionary：{variable_dictionary.shape}")
    print(f"value_labels：{value_labels.shape}")
    print(f"var_year_summary：{var_year_summary.shape}")

    print("-" * 80)
    print("第 2 步：构建 enriched variable_cards")
    variable_cards = build_variable_cards_from_metadata(variable_dictionary, value_labels)
    variable_cards_csv = OUTPUT_DIR / "variable_cards_enriched.csv"
    variable_cards_xlsx = OUTPUT_DIR / "variable_cards_enriched.xlsx"
    variable_cards.to_csv(variable_cards_csv, index=False, encoding="utf-8-sig")
    export_excel_with_sheets(variable_cards_xlsx, {"variable_cards_enriched": variable_cards})
    print(f"[完成] 已导出：{variable_cards_csv}")
    print(f"[完成] 已导出：{variable_cards_xlsx}")

    print("-" * 80)
    print("第 3 步：生成候选变量对")
    candidates = generate_candidate_pairs(variable_cards)
    if candidates.empty:
        print("[提示] 未生成任何候选变量对，请降低 MIN_FINAL_SCORE 或放宽 should_compare 规则。")
        return

    candidate_csv = OUTPUT_DIR / "candidate_pairs.csv"
    candidate_xlsx = OUTPUT_DIR / "candidate_pairs.xlsx"
    candidates.to_csv(candidate_csv, index=False, encoding="utf-8-sig")
    export_excel_with_sheets(candidate_xlsx, {"candidate_pairs": candidates})
    print(f"[完成] 候选对数量：{len(candidates)}")
    print(f"[完成] 已导出：{candidate_csv}")
    print(f"[完成] 已导出：{candidate_xlsx}")

    llm_results = pd.DataFrame()

    if RUN_LLM:
        print("-" * 80)
        print("第 4 步：调用 LLM 判别候选变量对")
        llm_results = run_llm_on_candidates(candidates, OUTPUT_DIR)
        llm_results_xlsx = OUTPUT_DIR / "llm_harmonization_results.xlsx"
        export_excel_with_sheets(llm_results_xlsx, {"llm_harmonization_results": llm_results})
        print(f"[完成] LLM 结果条数：{len(llm_results)}")
        print(f"[完成] 已导出：{llm_results_xlsx}")
    else:
        parsed_results_csv = OUTPUT_DIR / "llm_harmonization_results.csv"
        if parsed_results_csv.exists():
            llm_results = pd.read_csv(parsed_results_csv)
            print(f"[提示] RUN_LLM=False，但已读取历史 LLM 结果：{parsed_results_csv}")
        else:
            print("[提示] RUN_LLM=False，跳过 LLM 调用。")

    print("-" * 80)
    print("第 5 步：生成人工审核模板")
    review_template = build_review_template(candidates, llm_results)
    grouped_suggestions = build_grouped_variable_suggestions(llm_results)

    review_csv = OUTPUT_DIR / "harmonization_review_template.csv"
    review_xlsx = OUTPUT_DIR / "harmonization_review_template.xlsx"

    review_template.to_csv(review_csv, index=False, encoding="utf-8-sig")
    export_excel_with_sheets(review_xlsx, {
        "review_template": review_template,
        "grouped_suggestions": grouped_suggestions,
    })

    print(f"[完成] 已导出：{review_csv}")
    print(f"[完成] 已导出：{review_xlsx}")

    print("-" * 80)
    print("第 6 步：导出总汇总工作簿")
    all_in_one_xlsx = OUTPUT_DIR / "cfps_variable_harmonization_workbook.xlsx"
    export_excel_with_sheets(all_in_one_xlsx, {
        "variable_cards_enriched": variable_cards,
        "candidate_pairs": candidates,
        "llm_results": llm_results,
        "review_template": review_template,
        "grouped_suggestions": grouped_suggestions,
    })
    print(f"[完成] 已导出：{all_in_one_xlsx}")

    print("=" * 80)
    print("全部完成")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()