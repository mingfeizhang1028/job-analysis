# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:18:17 2026

@author: Rosem
"""

from __future__ import annotations

# -*- coding: utf-8 -*-
"""
CFPS 变量协调重构版
版本：Python 3.10+
作者：ChatGPT（按用户要求重构）
用途：
1．读取 cfps_metadata.xlsx
2．构建变量卡片
3．严格识别“同一题不同选项/不同展开槽位”的 strict_question_group
4．在 strict_question_group 层面构造宽口径 broad_concept_cluster
5．自动标记类内问题，供后续 LLM / 人工审核
6．导出结构化结果

依赖：
pip install pandas numpy openpyxl xlsxwriter rapidfuzz scikit-learn
"""

import re
import json
import math
import hashlib
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 0．配置区
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ===== 输入 =====
METADATA_XLSX = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\cfps_metadata.xlsx")
SHEET_VARIABLE_DICTIONARY = "variable_dictionary"
SHEET_VALUE_LABELS = "value_labels"

# ===== 输出目录 =====
OUTPUT_DIR = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\harmonization_redesigned_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 范围限制 =====
ONLY_COMPARE_DIR_TAG = "CFPS_RAW"
REQUIRE_SAME_DATASET_CLASS = True
REQUIRE_SAME_SURVEY_ROLE = True

EXCLUDE_DATASET_CLASSES = {
    # 可按需排除
    # "cfps_id_link",
    # "cfps_family_link",
}

# ===== 严格题组识别参数 =====
STRICT_LABEL_SIM_MIN_FOR_PATTERN_GROUP = 35
STRICT_LABEL_SIM_MIN_FOR_FALLBACK = 92
STRICT_NAME_SIM_MIN_FOR_FALLBACK = 95

# ===== broad cluster 参数 =====
BROAD_CLUSTER_MIN_FUZZY_SCORE = 68
BROAD_CLUSTER_MIN_COSINE_SCORE = 0.35
BROAD_CLUSTER_MIN_COMBINED_SCORE = 72
BROAD_CLUSTER_TOPK_PER_LEFT = 5

# ===== TF-IDF 参数 =====
TFIDF_ANALYZER = "char_wb"
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_MAX_FEATURES = 25000

# ===== 文本截断 =====
MAX_VALUE_LABEL_TEXT_LEN = 1500
MAX_CONTEXT_TEXT_LEN = 2000

# ===== 输出文件 =====
OUT_VARIABLE_CARDS = OUTPUT_DIR / "variable_cards_enriched.csv"
OUT_STRICT_GROUP_MAP = OUTPUT_DIR / "strict_question_group_map.csv"
OUT_STRICT_GROUP_SUMMARY = OUTPUT_DIR / "strict_question_group_summary.csv"
OUT_BROAD_CLUSTER_MAP = OUTPUT_DIR / "broad_concept_cluster_map.csv"
OUT_BROAD_CLUSTER_SUMMARY = OUTPUT_DIR / "broad_concept_cluster_summary.csv"
OUT_LLM_REVIEW_CANDIDATES = OUTPUT_DIR / "llm_review_candidates.csv"
OUT_WORKBOOK = OUTPUT_DIR / "cfps_harmonization_redesigned_workbook.xlsx"


# =========================================================
# 1．基础工具函数
# =========================================================

def safe_str(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip()


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def normalize_text(x: Any) -> str:
    s = safe_str(x).lower()
    s = s.replace("_", " ")
    s = re.sub(r"[，。；：、“”‘’（）()【】\[\]\-_/,:;!?]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_key_text(x: Any) -> str:
    s = normalize_text(x)
    s = s.replace(" ", "")
    return s


def short_text(x: Any, max_len: int = 300) -> str:
    s = safe_str(x)
    return s if len(s) <= max_len else s[:max_len] + "..."


def hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def join_unique(values: List[str], sep: str = " | ", top_n: int = 20) -> str:
    out = []
    seen = set()
    for v in values:
        sv = safe_str(v)
        if sv and sv not in seen:
            out.append(sv)
            seen.add(sv)
        if len(out) >= top_n:
            break
    return sep.join(out)


def mode_nonempty(series: pd.Series) -> str:
    vals = [safe_str(x) for x in series if safe_str(x)]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def first_not_empty(series: pd.Series) -> str:
    for x in series:
        sx = safe_str(x)
        if sx:
            return sx
    return ""


def export_excel_with_sheets(path: Path, sheets: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            if df is None:
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


# =========================================================
# 2．读取元数据
# =========================================================

def load_metadata(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"未找到元数据文件：{xlsx_path}")

    variable_dictionary = pd.read_excel(xlsx_path, sheet_name=SHEET_VARIABLE_DICTIONARY)
    value_labels = pd.read_excel(xlsx_path, sheet_name=SHEET_VALUE_LABELS)

    return variable_dictionary, value_labels


def build_value_label_text(value_label_df: pd.DataFrame) -> str:
    if value_label_df is None or value_label_df.empty:
        return ""

    tmp = value_label_df.copy()

    def parse_code_for_sort(x):
        if pd.isna(x):
            return (2, "")
        s = str(x).strip()
        try:
            return (0, float(s))
        except Exception:
            return (1, s.lower())

    tmp["_sort_key"] = tmp["code"].apply(parse_code_for_sort)
    tmp = tmp.sort_values(by="_sort_key", na_position="last")

    rows = []
    for _, r in tmp.iterrows():
        rows.append(f"{safe_str(r.get('code'))}={safe_str(r.get('label'))}")

    txt = "；".join(rows)
    if len(txt) > MAX_VALUE_LABEL_TEXT_LEN:
        txt = txt[:MAX_VALUE_LABEL_TEXT_LEN] + "..."
    return txt


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


# =========================================================
# 3．变量卡片构建
# =========================================================

def infer_structure_type(var_name: str, var_label: str, value_labels_text: str) -> str:
    text = " ".join([safe_str(var_name), safe_str(var_label), safe_str(value_labels_text)]).lower()

    multi_option_patterns = [
        r"_s_\d+$",
        r"_a_\d+$",
        r"_b_\d+$",
        r"_c_\d+$",
        r"选择\s*\d+",
        r"选项\s*\d+",
        r"option\s*\d+",
    ]
    roster_patterns = [
        r"姓名\s*\d+",
        r"成员\s*\d+",
        r"家庭成员\s*\d+",
        r"子女\s*\d+",
        r"兄弟姐妹\s*\d+",
        r"member\s*\d+",
        r"person\s*\d+",
        r"(?:^|[_])[a-z]+[_]?\d+$",
        r"(?:^|[_])[a-z]+[_]\d+[_]",
    ]
    derived_patterns = [
        r"isco",
        r"isei",
        r"siops",
        r"职业.*编码",
        r"行业.*编码",
        r"\bcode\b",
        r"编码",
        r"得分",
        r"score",
    ]

    if any(re.search(p, text) for p in multi_option_patterns):
        return "multi_option_like"
    if any(re.search(p, text) for p in roster_patterns):
        return "roster_like"
    if any(re.search(p, text) for p in derived_patterns):
        return "derived_code_score"
    return "single_item"


def extract_varname_base_tokens(var_name: str) -> List[str]:
    s = safe_str(var_name).lower()
    if not s:
        return []
    toks = re.split(r"[_\-\s]+", s)
    return [t for t in toks if t]


def detect_time_unit_tokens(var_name: str, var_label: str) -> List[str]:
    text = " ".join([safe_str(var_name).lower(), safe_str(var_label).lower()])

    units = []
    patterns = {
        "year": [r"\by\b", r"按年", r"年份", r"每年", r"年"],
        "month": [r"\bm\b", r"按月", r"月份", r"每月", r"月"],
        "day": [r"\bd\b", r"按日", r"每日", r"天", r"日"],
        "week": [r"\bw\b", r"按周", r"每周", r"周"],
        "hour": [r"\bh\b", r"小时", r"钟头"],
    }
    for k, ps in patterns.items():
        if any(re.search(p, text) for p in ps):
            units.append(k)
    return sorted(set(units))


def detect_embedded_year_tokens(var_name: str, var_label: str) -> List[str]:
    text = " ".join([safe_str(var_name), safe_str(var_label)])
    years = re.findall(r"(19\d{2}|20\d{2})", text)
    return sorted(set(years))


def parse_strict_slot_pattern(var_name: str) -> Dict[str, Any]:
    """
    严格识别“同一题不同选项/槽位”的变量名模式。
    核心原则：
    1. 末尾必须是独立编号槽位，如 _1
    2. 去掉最后槽位后，前面的完整 base 必须完全一致，才可归为同一 strict group
    3. 不进一步粗暴去掉 base 内的数字或年份
    """

    s = safe_str(var_name).lower().strip()
    result = {
        "matched": False,
        "strict_base_stem": "",
        "slot_index": "",
        "slot_marker": "",
        "strict_group_type": "",
        "strict_match_rule": "",
    }

    # 模式1：xxx_s_1, xxx_a_2, xxx_b_3
    m1 = re.match(r"^(.*)_(s|a|b|c)_(\d+)$", s)
    if m1:
        result.update({
            "matched": True,
            "strict_base_stem": f"{m1.group(1)}_{m1.group(2)}",
            "slot_index": m1.group(3),
            "slot_marker": m1.group(2),
            "strict_group_type": "strict_suffix_slot",
            "strict_match_rule": "full_base_plus_slot_suffix",
        })
        return result

    # 模式2：xxx_1
    m2 = re.match(r"^(.*)_(\d+)$", s)
    if m2:
        base = m2.group(1)
        slot = m2.group(2)
        result.update({
            "matched": True,
            "strict_base_stem": base,
            "slot_index": slot,
            "slot_marker": "",
            "strict_group_type": "strict_trailing_number",
            "strict_match_rule": "full_base_plus_trailing_number",
        })
        return result

    # 模式3：xxx1_yyy, fammem1_age
    # 这类只做保守识别：中间编号槽位，去掉这个槽位后前后主干完全一致
    m3 = re.match(r"^([a-z\u4e00-\u9fff]+)(\d+)([_\-].+)$", s)
    if m3:
        result.update({
            "matched": True,
            "strict_base_stem": f"{m3.group(1)}#{m3.group(3)}",
            "slot_index": m3.group(2),
            "slot_marker": "embedded",
            "strict_group_type": "strict_embedded_member_slot",
            "strict_match_rule": "embedded_index_slot",
        })
        return result

    return result


def build_variable_cards(variable_dictionary: pd.DataFrame, value_labels: pd.DataFrame) -> pd.DataFrame:
    vdict = variable_dictionary.copy()

    required_cols_defaults = {
        "dir_tag": "",
        "source_stage": "",
        "dataset_class": "",
        "survey_role": "",
        "path_group": "",
        "source_note": "",
        "file_name": "",
        "file_path": "",
        "var_name": "",
        "var_label": "",
        "measure": "",
        "special_missing_candidates": "",
        "year": None,
        "unique_n": None,
        "min": None,
        "max": None,
        "value_label_name": "",
    }
    for c, default in required_cols_defaults.items():
        if c not in vdict.columns:
            vdict[c] = default

    value_label_text_df = aggregate_value_labels(value_labels)

    cards = vdict.merge(
        value_label_text_df,
        on=["file_path", "value_label_name"],
        how="left"
    )

    cards["dir_tag"] = cards["dir_tag"].fillna("").astype(str)
    cards["dataset_class"] = cards["dataset_class"].fillna("").astype(str)
    cards["survey_role"] = cards["survey_role"].fillna("").astype(str)
    cards["file_name"] = cards["file_name"].fillna("").astype(str)
    cards["file_path"] = cards["file_path"].fillna("").astype(str)
    cards["var_name"] = cards["var_name"].fillna("").astype(str)
    cards["var_label"] = cards["var_label"].fillna("").astype(str)
    cards["measure"] = cards["measure"].fillna("").astype(str)
    cards["source_stage"] = cards["source_stage"].fillna("").astype(str)
    cards["path_group"] = cards["path_group"].fillna("").astype(str)
    cards["source_note"] = cards["source_note"].fillna("").astype(str)
    cards["special_missing_candidates"] = cards["special_missing_candidates"].fillna("").astype(str)
    cards["value_labels_text"] = cards["value_labels_text"].fillna("").astype(str)

    cards["year"] = cards["year"].apply(safe_int)
    cards["unique_n"] = cards["unique_n"].apply(safe_int)
    cards["min"] = cards["min"].apply(safe_float)
    cards["max"] = cards["max"].apply(safe_float)

    cards = cards[cards["dir_tag"].astype(str) == ONLY_COMPARE_DIR_TAG].copy()
    if EXCLUDE_DATASET_CLASSES:
        cards = cards[~cards["dataset_class"].astype(str).isin(EXCLUDE_DATASET_CLASSES)].copy()

    cards["var_name_norm"] = cards["var_name"].apply(normalize_text)
    cards["var_label_norm"] = cards["var_label"].apply(normalize_text)
    cards["var_name_key"] = cards["var_name"].apply(normalize_key_text)
    cards["var_label_key"] = cards["var_label"].apply(normalize_key_text)

    cards["structure_type"] = cards.apply(
        lambda r: infer_structure_type(r["var_name"], r["var_label"], r["value_labels_text"]),
        axis=1
    )

    strict_info_df = cards["var_name"].apply(parse_strict_slot_pattern).apply(pd.Series)
    cards = pd.concat([cards, strict_info_df], axis=1)

    cards["time_unit_tokens"] = cards.apply(
        lambda r: ",".join(detect_time_unit_tokens(r["var_name"], r["var_label"])),
        axis=1
    )
    cards["embedded_year_tokens"] = cards.apply(
        lambda r: ",".join(detect_embedded_year_tokens(r["var_name"], r["var_label"])),
        axis=1
    )

    def make_context(r):
        parts = [
            f"变量名：{r.get('var_name', '')}",
            f"变量标签：{r.get('var_label', '')}",
            f"年份：{r.get('year', '')}",
            f"文件：{r.get('file_name', '')}",
            f"数据类别：{r.get('dataset_class', '')}",
            f"问卷角色：{r.get('survey_role', '')}",
            f"结构类型：{r.get('structure_type', '')}",
            f"strict_base：{r.get('strict_base_stem', '')}",
            f"slot_index：{r.get('slot_index', '')}",
            f"值标签：{r.get('value_labels_text', '')}",
        ]
        return short_text("；".join(parts), MAX_CONTEXT_TEXT_LEN)

    cards["context_text"] = cards.apply(make_context, axis=1)

    cards["card_id"] = cards.apply(
        lambda r: hash_text(
            "||".join([
                safe_str(r.get("file_path")),
                safe_str(r.get("var_name")),
                safe_str(r.get("year")),
                safe_str(r.get("var_label")),
            ])
        ),
        axis=1
    )

    cards = cards.reset_index(drop=True)
    return cards


# =========================================================
# 4．并查集
# =========================================================

class UnionFind:
    def __init__(self, items: List[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# =========================================================
# 5．严格题组构建
# =========================================================

def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return max(
        fuzz.ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_sort_ratio(a, b)
    )


def should_merge_strict(a: pd.Series, b: pd.Series) -> Tuple[bool, str]:
    """
    严格题组归并规则：
    1. 同年、同 dataset_class、同 survey_role
    2. 优先依赖 strict slot pattern，且 strict_base_stem 必须完全一致
    3. 不因为“都有 _a_1”就自动合并，必须是完整 base 一致
    4. fallback 规则极保守，仅用于明显重复项
    """
    if safe_int(a.get("year")) != safe_int(b.get("year")):
        return False, ""

    if REQUIRE_SAME_DATASET_CLASS:
        if safe_str(a.get("dataset_class")) != safe_str(b.get("dataset_class")):
            return False, ""

    if REQUIRE_SAME_SURVEY_ROLE:
        if safe_str(a.get("survey_role")) != safe_str(b.get("survey_role")):
            return False, ""

    # 规则1：严格 slot/base 完全一致
    if bool(a.get("matched")) and bool(b.get("matched")):
        if safe_str(a.get("strict_group_type")) == safe_str(b.get("strict_group_type")):
            if safe_str(a.get("strict_base_stem")) and safe_str(a.get("strict_base_stem")) == safe_str(b.get("strict_base_stem")):
                label_sim = fuzzy_score(normalize_text(a.get("var_label")), normalize_text(b.get("var_label")))
                if label_sim >= STRICT_LABEL_SIM_MIN_FOR_PATTERN_GROUP or safe_str(a.get("var_label")) == "" or safe_str(b.get("var_label")) == "":
                    return True, "strict_same_full_base"

    # 规则2：极保守 fallback
    # 仅当变量名完全一致或标签完全一致时
    name_key_a = safe_str(a.get("var_name_key"))
    name_key_b = safe_str(b.get("var_name_key"))
    label_key_a = safe_str(a.get("var_label_key"))
    label_key_b = safe_str(b.get("var_label_key"))

    if name_key_a and name_key_a == name_key_b:
        return True, "fallback_same_var_name"

    if label_key_a and label_key_a == label_key_b:
        return True, "fallback_same_var_label"

    # 规则3：近乎完全一致的重复项
    name_sim = fuzzy_score(name_key_a, name_key_b)
    label_sim = fuzzy_score(label_key_a, label_key_b)
    if name_sim >= STRICT_NAME_SIM_MIN_FOR_FALLBACK and label_sim >= STRICT_LABEL_SIM_MIN_FOR_FALLBACK:
        return True, "fallback_near_duplicate"

    return False, ""


def build_strict_question_groups(cards: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []

    grouped = cards.groupby(["year", "dataset_class", "survey_role"], dropna=False)

    for group_key, g in grouped:
        g = g.copy().reset_index(drop=True)
        if g.empty:
            continue

        ids = g["card_id"].tolist()
        uf = UnionFind(ids)

        merge_rule_map = {}

        n = len(g)
        for i in range(n):
            a = g.iloc[i]
            for j in range(i + 1, n):
                b = g.iloc[j]
                ok, rule = should_merge_strict(a, b)
                if ok:
                    uf.union(a["card_id"], b["card_id"])
                    merge_rule_map[(a["card_id"], b["card_id"])] = rule

        g["strict_root"] = g["card_id"].apply(uf.find)
        root_order = {root: i + 1 for i, root in enumerate(pd.unique(g["strict_root"]))}
        year = safe_int(g.iloc[0]["year"])
        g["strict_group_id"] = g["strict_root"].map(lambda x: f"SG{year}_{root_order[x]:05d}")

        records.append(g)

    strict_map = pd.concat(records, ignore_index=True).reset_index(drop=True)

    # group summary
    summary_rows = []
    for gid, sub in strict_map.groupby("strict_group_id", dropna=False):
        temp = sub.copy()
        temp["label_len"] = temp["var_label"].fillna("").astype(str).map(len)
        temp = temp.sort_values(by=["label_len", "var_name"], ascending=[False, True])
        anchor = temp.iloc[0]

        strict_group_type = mode_nonempty(temp["strict_group_type"])
        strict_base_stem = mode_nonempty(temp["strict_base_stem"])
        dataset_class = mode_nonempty(temp["dataset_class"])
        survey_role = mode_nonempty(temp["survey_role"])
        year = safe_int(temp["year"].iloc[0])

        group_label = first_not_empty(temp["var_label"])
        if not group_label:
            group_label = first_not_empty(temp["var_name"])

        slot_indices = sorted(set([safe_str(x) for x in temp["slot_index"] if safe_str(x)]))
        slot_index_range = ",".join(slot_indices)

        # 严格题组内部风险标记
        label_norms = set([normalize_key_text(x) for x in temp["var_label"] if safe_str(x)])
        strict_label_inconsistency = 1 if len(label_norms) > 1 and len(temp) > 1 else 0

        summary_rows.append({
            "strict_group_id": gid,
            "year": year,
            "dataset_class": dataset_class,
            "survey_role": survey_role,
            "strict_group_type": strict_group_type if strict_group_type else "singleton_or_fallback",
            "strict_base_stem": strict_base_stem,
            "group_label": group_label,
            "anchor_var": safe_str(anchor.get("var_name")),
            "anchor_label": safe_str(anchor.get("var_label")),
            "member_count": len(temp),
            "slot_indices": slot_index_range,
            "member_var_names": join_unique(temp["var_name"].astype(str).tolist(), top_n=100),
            "member_labels": join_unique(temp["var_label"].astype(str).tolist(), top_n=100),
            "value_label_summary": join_unique(temp["value_labels_text"].astype(str).tolist(), top_n=10),
            "file_names": join_unique(temp["file_name"].astype(str).tolist(), top_n=20),
            "structure_type_mode": mode_nonempty(temp["structure_type"]),
            "strict_label_inconsistency": strict_label_inconsistency,
            "broad_text": normalize_text(" ".join([
                group_label,
                safe_str(anchor.get("var_name")),
                strict_base_stem,
                dataset_class,
                survey_role,
                join_unique(temp["value_labels_text"].astype(str).tolist(), top_n=5),
            ])),
            "embedded_year_tokens_group": join_unique(temp["embedded_year_tokens"].astype(str).tolist(), top_n=20),
            "time_unit_tokens_group": join_unique(temp["time_unit_tokens"].astype(str).tolist(), top_n=20),
        })

    strict_summary = pd.DataFrame(summary_rows).sort_values(
        by=["year", "dataset_class", "survey_role", "strict_group_id"]
    ).reset_index(drop=True)

    strict_map = strict_map.sort_values(
        by=["year", "strict_group_id", "var_name"]
    ).reset_index(drop=True)

    return strict_map, strict_summary


# =========================================================
# 6．broad cluster 构建
# =========================================================

def build_text_vector_similarity(texts: List[str]) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros((0, 0))
    if len(texts) == 1:
        return np.array([[1.0]])

    vectorizer = TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=1
    )
    X = vectorizer.fit_transform(texts)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 1.0)
    return sim


def pairwise_broad_score(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    label_a = normalize_text(a.get("group_label"))
    label_b = normalize_text(b.get("group_label"))
    anchor_a = normalize_text(a.get("anchor_var"))
    anchor_b = normalize_text(b.get("anchor_var"))
    text_a = normalize_text(a.get("broad_text"))
    text_b = normalize_text(b.get("broad_text"))
    base_a = normalize_text(a.get("strict_base_stem"))
    base_b = normalize_text(b.get("strict_base_stem"))

    label_sim = fuzzy_score(label_a, label_b)
    anchor_sim = fuzzy_score(anchor_a, anchor_b)
    text_sim = fuzzy_score(text_a, text_b)
    base_sim = fuzzy_score(base_a, base_b)

    structure_match = 100.0 if safe_str(a.get("structure_type_mode")) == safe_str(b.get("structure_type_mode")) else 50.0
    dataset_match = 100.0 if safe_str(a.get("dataset_class")) == safe_str(b.get("dataset_class")) else 50.0
    role_match = 100.0 if safe_str(a.get("survey_role")) == safe_str(b.get("survey_role")) else 50.0

    combined = (
        0.28 * label_sim +
        0.15 * anchor_sim +
        0.25 * text_sim +
        0.12 * base_sim +
        0.08 * structure_match +
        0.07 * dataset_match +
        0.05 * role_match
    )

    return {
        "label_sim": round(label_sim, 2),
        "anchor_sim": round(anchor_sim, 2),
        "text_sim": round(text_sim, 2),
        "base_sim": round(base_sim, 2),
        "structure_match": round(structure_match, 2),
        "dataset_match": round(dataset_match, 2),
        "role_match": round(role_match, 2),
        "combined_score": round(combined, 2),
    }


def build_broad_concept_clusters(strict_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if strict_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_edges = []

    grouped = strict_summary.groupby(["dataset_class", "survey_role"], dropna=False)

    for group_key, g in grouped:
        g = g.copy().reset_index(drop=True)
        if len(g) == 0:
            continue
        if len(g) == 1:
            continue

        texts = g["broad_text"].fillna("").astype(str).tolist()
        sim_mat = build_text_vector_similarity(texts)

        candidate_records = defaultdict(list)

        for i in range(len(g)):
            a = g.iloc[i]
            for j in range(len(g)):
                if i == j:
                    continue
                b = g.iloc[j]

                # 宽口径可以跨年，必须不同年
                if safe_int(a.get("year")) == safe_int(b.get("year")):
                    continue

                score_dict = pairwise_broad_score(a, b)
                cosine_score = float(sim_mat[i, j])

                # 宽一些，但要防止过宽
                pass_rule = (
                    (score_dict["combined_score"] >= BROAD_CLUSTER_MIN_COMBINED_SCORE) or
                    (score_dict["label_sim"] >= 82 and cosine_score >= 0.25) or
                    (score_dict["base_sim"] >= 88 and score_dict["text_sim"] >= 70)
                )

                if not pass_rule:
                    continue

                rec = {
                    "left_strict_group_id": safe_str(a.get("strict_group_id")),
                    "left_year": safe_int(a.get("year")),
                    "left_group_label": safe_str(a.get("group_label")),
                    "right_strict_group_id": safe_str(b.get("strict_group_id")),
                    "right_year": safe_int(b.get("year")),
                    "right_group_label": safe_str(b.get("group_label")),
                    "dataset_class": safe_str(a.get("dataset_class")),
                    "survey_role": safe_str(a.get("survey_role")),
                    "cosine_score": round(cosine_score, 6),
                }
                rec.update(score_dict)
                candidate_records[rec["left_strict_group_id"]].append(rec)

        # 每个左侧保留 top-k
        for left_gid, recs in candidate_records.items():
            recs_sorted = sorted(
                recs,
                key=lambda x: (x["combined_score"], x["label_sim"], x["cosine_score"]),
                reverse=True
            )[:BROAD_CLUSTER_TOPK_PER_LEFT]
            all_edges.extend(recs_sorted)

    edge_df = pd.DataFrame(all_edges).drop_duplicates(
        subset=["left_strict_group_id", "right_strict_group_id"]
    )

    # 并查集归并 broad cluster
    nodes = strict_summary["strict_group_id"].astype(str).tolist()
    uf = UnionFind(nodes)

    if not edge_df.empty:
        for _, r in edge_df.iterrows():
            uf.union(safe_str(r["left_strict_group_id"]), safe_str(r["right_strict_group_id"]))

    out_map = strict_summary.copy()
    out_map["broad_root"] = out_map["strict_group_id"].astype(str).apply(uf.find)
    root_order = {root: i + 1 for i, root in enumerate(pd.unique(out_map["broad_root"]))}
    out_map["broad_cluster_id"] = out_map["broad_root"].map(lambda x: f"BC{root_order[x]:05d}")

    # 汇总 broad cluster
    summary_rows = []
    for bcid, sub in out_map.groupby("broad_cluster_id", dropna=False):
        years = sorted(set([safe_int(y) for y in sub["year"] if safe_int(y) is not None]))
        year_count = len(years)

        candidate_concept_label = mode_nonempty(sub["group_label"])
        main_dataset_class = mode_nonempty(sub["dataset_class"])
        main_survey_role = mode_nonempty(sub["survey_role"])
        main_structure_type = mode_nonempty(sub["structure_type_mode"])

        total_member_groups = len(sub)
        total_member_variables = pd.to_numeric(sub["member_count"], errors="coerce").fillna(0).sum()

        # ========== 问题标记 ==========
        issue_flags = []
        issue_examples = []

        # 1. mixed structure
        if sub["structure_type_mode"].nunique(dropna=True) > 1:
            issue_flags.append("mixed_structure_type")
            issue_examples.append("类内混有不同结构类型")

        # 2. mixed time unit
        time_unit_tokens_all = set()
        for x in sub["time_unit_tokens_group"].astype(str).tolist():
            for t in [i.strip() for i in x.split("|")]:
                if t:
                    for z in [j.strip() for j in t.split(",")]:
                        if z:
                            time_unit_tokens_all.add(z)
        if len(time_unit_tokens_all) > 1:
            issue_flags.append("mixed_time_unit")
            issue_examples.append(f"时间单位混杂：{','.join(sorted(time_unit_tokens_all))}")

        # 3. mixed embedded years
        embedded_year_tokens_all = set()
        for x in sub["embedded_year_tokens_group"].astype(str).tolist():
            for t in [i.strip() for i in x.split("|")]:
                if t:
                    for z in [j.strip() for j in t.split(",")]:
                        if z:
                            embedded_year_tokens_all.add(z)
        if len(embedded_year_tokens_all) > 1:
            issue_flags.append("mixed_target_year")
            issue_examples.append(f"变量名/标签中出现多个目标年份：{','.join(sorted(embedded_year_tokens_all))}")

        # 4. mixed value label
        value_label_patterns = set([normalize_key_text(x)[:80] for x in sub["value_label_summary"].astype(str).tolist() if safe_str(x)])
        if len(value_label_patterns) > 1:
            issue_flags.append("mixed_value_label_pattern")
            issue_examples.append("值标签模式差异较大")

        # 5. same name diff label / same label diff name
        labels_norm = set([normalize_key_text(x) for x in sub["group_label"].astype(str).tolist() if safe_str(x)])
        names_norm = set([normalize_key_text(x) for x in sub["anchor_var"].astype(str).tolist() if safe_str(x)])
        if len(names_norm) <= 2 and len(labels_norm) >= 3:
            issue_flags.append("same_name_diff_label")
            issue_examples.append("变量名较接近，但标签差异较大")
        if len(labels_norm) <= 2 and len(names_norm) >= 3:
            issue_flags.append("same_label_diff_name")
            issue_examples.append("标签较接近，但变量主干差异较大")

        # 6. cross year large gap
        if year_count >= 3 and (max(years) - min(years) >= 8):
            issue_flags.append("cross_year_gap_large")
            issue_examples.append("年份跨度较大，建议核对题意变化")

        # 7. strict group warning inside cluster
        if pd.to_numeric(sub["strict_label_inconsistency"], errors="coerce").fillna(0).sum() > 0:
            issue_flags.append("strict_group_internal_inconsistency")
            issue_examples.append("部分严格题组内部标签并不完全一致")

        if not issue_flags:
            issue_flags.append("no_major_issue_detected")
            issue_examples.append("未发现明显结构性风险，但仍建议抽样审核")

        year_distribution = (
            sub.groupby("year")["strict_group_id"]
            .count()
            .to_dict()
        )

        summary_rows.append({
            "broad_cluster_id": bcid,
            "candidate_concept_label": candidate_concept_label,
            "year_count": year_count,
            "years_covered": ",".join([str(y) for y in years]),
            "first_year": min(years) if years else None,
            "last_year": max(years) if years else None,
            "member_group_count": total_member_groups,
            "member_variable_count": total_member_variables,
            "main_dataset_class": main_dataset_class,
            "main_survey_role": main_survey_role,
            "main_structure_type": main_structure_type,
            "cluster_issue_flags": " | ".join(issue_flags),
            "issue_examples": " | ".join(issue_examples),
            "member_group_labels": join_unique(sub["group_label"].astype(str).tolist(), top_n=100),
            "member_anchor_vars": join_unique(sub["anchor_var"].astype(str).tolist(), top_n=100),
            "member_strict_group_ids": join_unique(sub["strict_group_id"].astype(str).tolist(), top_n=200),
            "year_distribution": json.dumps(year_distribution, ensure_ascii=False),
        })

    broad_summary = pd.DataFrame(summary_rows).sort_values(
        by=["year_count", "member_variable_count", "broad_cluster_id"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return out_map, broad_summary


# =========================================================
# 7．LLM / 人工审核候选表
# =========================================================

def build_llm_review_candidates(
    broad_map: pd.DataFrame,
    broad_summary: pd.DataFrame
) -> pd.DataFrame:
    if broad_summary.empty:
        return pd.DataFrame()

    rows = []
    for _, r in broad_summary.iterrows():
        bcid = safe_str(r["broad_cluster_id"])
        sub = broad_map[broad_map["broad_cluster_id"].astype(str) == bcid].copy()

        review_priority = "medium"
        issue_flags = safe_str(r.get("cluster_issue_flags"))

        if (
            "mixed_target_year" in issue_flags or
            "mixed_time_unit" in issue_flags or
            "mixed_structure_type" in issue_flags
        ):
            review_priority = "high"
        elif "no_major_issue_detected" in issue_flags and safe_int(r.get("year_count")) is not None and safe_int(r.get("year_count")) >= 3:
            review_priority = "low"

        rows.append({
            "broad_cluster_id": bcid,
            "review_priority": review_priority,
            "candidate_concept_label": safe_str(r.get("candidate_concept_label")),
            "cluster_issue_flags": issue_flags,
            "issue_examples": safe_str(r.get("issue_examples")),
            "years_covered": safe_str(r.get("years_covered")),
            "year_count": safe_int(r.get("year_count")),
            "member_group_count": safe_int(r.get("member_group_count")),
            "member_variable_count": safe_int(r.get("member_variable_count")),
            "main_dataset_class": safe_str(r.get("main_dataset_class")),
            "main_survey_role": safe_str(r.get("main_survey_role")),
            "main_structure_type": safe_str(r.get("main_structure_type")),
            "member_group_labels": safe_str(r.get("member_group_labels")),
            "member_anchor_vars": safe_str(r.get("member_anchor_vars")),
            "member_strict_group_ids": safe_str(r.get("member_strict_group_ids")),
            "suggested_review_focus": (
                "请核查类内是否混入不同时间单位、不同目标年份、不同结构题型；"
                "确认哪些 strict_group 可视为同一概念，哪些应拆分。"
            ),
            "human_decision": "",
            "approved_concept_name": "",
            "approved_subcluster_scheme": "",
            "review_notes": "",
        })

    review_df = pd.DataFrame(rows).sort_values(
        by=["review_priority", "year_count", "member_variable_count"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    priority_order = {"high": 0, "medium": 1, "low": 2}
    review_df["_priority_order"] = review_df["review_priority"].map(priority_order)
    review_df = review_df.sort_values(
        by=["_priority_order", "year_count", "member_variable_count"],
        ascending=[True, False, False]
    ).drop(columns=["_priority_order"]).reset_index(drop=True)

    return review_df


# =========================================================
# 8．主程序
# =========================================================

def main():
    print("=" * 80)
    print("CFPS 变量协调重构版启动")
    print("=" * 80)

    print("第 1 步：读取元数据")
    variable_dictionary, value_labels = load_metadata(METADATA_XLSX)
    print(f"variable_dictionary.shape = {variable_dictionary.shape}")
    print(f"value_labels.shape = {value_labels.shape}")

    print("-" * 80)
    print("第 2 步：构建变量卡片")
    cards = build_variable_cards(variable_dictionary, value_labels)
    cards.to_csv(OUT_VARIABLE_CARDS, index=False, encoding="utf-8-sig")
    print(f"[完成] 变量卡片数：{len(cards)}")
    print(f"[完成] 导出：{OUT_VARIABLE_CARDS}")

    print("-" * 80)
    print("第 3 步：构建严格题组 strict_question_group")
    strict_map, strict_summary = build_strict_question_groups(cards)
    strict_map.to_csv(OUT_STRICT_GROUP_MAP, index=False, encoding="utf-8-sig")
    strict_summary.to_csv(OUT_STRICT_GROUP_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"[完成] strict_group 映射数：{len(strict_map)}")
    print(f"[完成] strict_group 数：{len(strict_summary)}")
    print(f"[完成] 导出：{OUT_STRICT_GROUP_MAP}")
    print(f"[完成] 导出：{OUT_STRICT_GROUP_SUMMARY}")

    print("-" * 80)
    print("第 4 步：构建宽口径 broad_concept_cluster")
    broad_map, broad_summary = build_broad_concept_clusters(strict_summary)
    broad_map.to_csv(OUT_BROAD_CLUSTER_MAP, index=False, encoding="utf-8-sig")
    broad_summary.to_csv(OUT_BROAD_CLUSTER_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"[完成] broad_cluster 映射数：{len(broad_map)}")
    print(f"[完成] broad_cluster 数：{broad_summary['broad_cluster_id'].nunique() if not broad_summary.empty else 0}")
    print(f"[完成] 导出：{OUT_BROAD_CLUSTER_MAP}")
    print(f"[完成] 导出：{OUT_BROAD_CLUSTER_SUMMARY}")

    print("-" * 80)
    print("第 5 步：生成 LLM / 人工审核候选表")
    review_df = build_llm_review_candidates(broad_map, broad_summary)
    review_df.to_csv(OUT_LLM_REVIEW_CANDIDATES, index=False, encoding="utf-8-sig")
    print(f"[完成] 审核候选类数：{len(review_df)}")
    print(f"[完成] 导出：{OUT_LLM_REVIEW_CANDIDATES}")

    print("-" * 80)
    print("第 6 步：导出总工作簿")
    export_excel_with_sheets(OUT_WORKBOOK, {
        "variable_cards": cards,
        "strict_group_map": strict_map,
        "strict_group_summary": strict_summary,
        "broad_cluster_map": broad_map,
        "broad_cluster_summary": broad_summary,
        "llm_review_candidates": review_df,
    })
    print(f"[完成] 导出：{OUT_WORKBOOK}")

    print("=" * 80)
    print("全部完成")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()