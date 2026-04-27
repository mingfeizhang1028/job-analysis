# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:29:20 2026

@author: Rosem
"""

from __future__ import annotations

# -*- coding: utf-8 -*-
"""
CFPS 跨年变量候选匹配 + 并行 LLM 判别 + 推荐文件输出
版本：Python 3.10+
用途：
1．读取 strict_question_group_summary.csv
2．为每个 strict group 生成跨年候选池
3．并行调用 LLM 进行标准化判别
4．合并 pre-score 与 LLM judgement
5．输出推荐文件与人工审核文件

依赖：
pip install pandas numpy rapidfuzz scikit-learn requests openpyxl xlsxwriter

说明：
1．默认以 strict_question_group_summary.csv 为输入
2．并行数已按用户要求设置为 MAX_WORKERS = 8
3．使用缓存 llm_cache.jsonl，支持断点续跑
"""

import os
import re
import json
import time
import math
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 0．全局配置
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ===== 输入文件 =====
INPUT_STRICT_GROUP_SUMMARY = Path(
    r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\harmonization_redesigned_output\strict_question_group_summary.csv"
)

# ===== 输出目录 =====
OUTPUT_DIR = Path(
    r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\cross_year_match_output"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 模型配置 =====
MODEL_PROVIDER = "OpenAi_Responce"
API_BASE = "https://api.siliconflow.cn"
API_KEY = os.getenv("LLM_API_KEY", "sk-nxvyrqxdmlclzuymuaehxteplvhsyvhvbkhucdfwffonjuzr")
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

# ===== 并行配置 =====
MAX_WORKERS = 8
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_SLEEP_BASE = 2
USE_LLM_CACHE = True

# ===== 候选池生成参数 =====
REQUIRE_SAME_DATASET_CLASS = True
REQUIRE_SAME_SURVEY_ROLE = True
FORBID_SAME_YEAR_MATCH = True

TOP_K_PER_TARGET = 8
SEND_TO_LLM_TOP_K = 5

LABEL_SIM_MIN = 45
PRE_SCORE_MIN = 58
COSINE_MIN = 0.18

# ===== 最终推荐参数 =====
LLM_LEVEL_TO_SCORE = {
    "high": 95,
    "medium": 75,
    "low": 50,
    "reject": 10,
}
FINAL_SCORE_WEIGHT_PRE = 0.70
FINAL_SCORE_WEIGHT_LLM = 0.30
FINAL_HIGH_THRESHOLD = 85
FINAL_MEDIUM_THRESHOLD = 72

# ===== TF-IDF 参数 =====
TFIDF_ANALYZER = "char_wb"
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_MAX_FEATURES = 30000

# ===== 缓存与输出文件 =====
OUT_CANDIDATE_POOL = OUTPUT_DIR / "candidate_pool_pairs.csv"
OUT_LLM_JUDGEMENT = OUTPUT_DIR / "llm_pair_judgement.csv"
OUT_FINAL_RECOMMEND = OUTPUT_DIR / "cross_year_match_recommendation.csv"
OUT_MANUAL_REVIEW = OUTPUT_DIR / "manual_review_sheet.csv"
OUT_TARGET_SUMMARY = OUTPUT_DIR / "target_level_summary.csv"
OUT_LLM_CACHE = OUTPUT_DIR / "llm_cache.jsonl"
OUT_WORKBOOK = OUTPUT_DIR / "cross_year_match_workbook.xlsx"


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
    return normalize_text(x).replace(" ", "")


def join_unique(values: List[str], sep: str = " | ", top_n: int = 50) -> str:
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


def short_text(x: Any, max_len: int = 500) -> str:
    s = safe_str(x)
    return s if len(s) <= max_len else s[:max_len] + "..."


def hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return max(
        fuzz.ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_sort_ratio(a, b)
    )


def export_excel_with_sheets(path: Path, sheets: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            if df is None:
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2．读取 strict group summary
# =========================================================

def load_strict_group_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到输入文件：{path}")

    df = pd.read_csv(path)
    needed_cols = [
        "strict_group_id", "year", "dataset_class", "survey_role",
        "strict_group_type", "strict_base_stem", "group_label",
        "anchor_var", "anchor_label", "member_count", "slot_indices",
        "member_var_names", "member_labels", "value_label_summary",
        "file_names", "structure_type_mode", "strict_label_inconsistency",
        "broad_text", "embedded_year_tokens_group", "time_unit_tokens_group",
    ]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = ""

    df["year"] = df["year"].apply(safe_int)
    df["member_count"] = pd.to_numeric(df["member_count"], errors="coerce").fillna(0).astype(int)

    df["group_label_norm"] = df["group_label"].apply(normalize_text)
    df["anchor_var_norm"] = df["anchor_var"].apply(normalize_text)
    df["strict_base_stem_norm"] = df["strict_base_stem"].apply(normalize_text)
    df["value_label_summary_norm"] = df["value_label_summary"].apply(normalize_text)
    df["broad_text_norm"] = df["broad_text"].apply(normalize_text)

    df["target_uid"] = df["strict_group_id"].astype(str)

    return df


# =========================================================
# 3．文本向量与预评分
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


def parse_token_set(text: str) -> set:
    if not safe_str(text):
        return set()
    out = set()
    for part in re.split(r"[|,，；;\s]+", safe_str(text)):
        p = part.strip()
        if p:
            out.add(p)
    return out


def pre_score_pair(a: pd.Series, b: pd.Series, cosine_score: float) -> Dict[str, Any]:
    label_sim = fuzzy_score(a["group_label_norm"], b["group_label_norm"])
    anchor_sim = fuzzy_score(a["anchor_var_norm"], b["anchor_var_norm"])
    base_sim = fuzzy_score(a["strict_base_stem_norm"], b["strict_base_stem_norm"])
    value_label_sim = fuzzy_score(a["value_label_summary_norm"], b["value_label_summary_norm"])
    broad_text_sim = fuzzy_score(a["broad_text_norm"], b["broad_text_norm"])

    structure_match = 100.0 if safe_str(a["structure_type_mode"]) == safe_str(b["structure_type_mode"]) else 50.0
    dataset_match = 100.0 if safe_str(a["dataset_class"]) == safe_str(b["dataset_class"]) else 0.0
    role_match = 100.0 if safe_str(a["survey_role"]) == safe_str(b["survey_role"]) else 0.0

    # 风险惩罚项
    penalty = 0.0

    # 时间单位冲突
    tu_a = parse_token_set(a["time_unit_tokens_group"])
    tu_b = parse_token_set(b["time_unit_tokens_group"])
    if tu_a and tu_b and tu_a != tu_b:
        penalty += 8.0

    # 嵌入年份冲突
    ey_a = parse_token_set(a["embedded_year_tokens_group"])
    ey_b = parse_token_set(b["embedded_year_tokens_group"])
    if ey_a and ey_b and ey_a != ey_b:
        penalty += 12.0

    pre_score = (
        0.28 * label_sim +
        0.14 * anchor_sim +
        0.12 * base_sim +
        0.12 * value_label_sim +
        0.12 * broad_text_sim +
        0.10 * cosine_score * 100 +
        0.07 * structure_match +
        0.03 * dataset_match +
        0.02 * role_match
        - penalty
    )

    return {
        "label_similarity": round(label_sim, 2),
        "anchor_similarity": round(anchor_sim, 2),
        "base_similarity": round(base_sim, 2),
        "value_label_similarity": round(value_label_sim, 2),
        "broad_text_similarity": round(broad_text_sim, 2),
        "cosine_similarity": round(cosine_score, 6),
        "structure_match": round(structure_match, 2),
        "dataset_match": round(dataset_match, 2),
        "role_match": round(role_match, 2),
        "penalty": round(penalty, 2),
        "pre_score": round(pre_score, 2),
    }


# =========================================================
# 4．生成跨年候选池
# =========================================================

def build_candidate_pool(strict_df: pd.DataFrame) -> pd.DataFrame:
    df = strict_df.copy().reset_index(drop=True)

    # 分桶：优先在 dataset_class + survey_role 内比较
    bucket_cols = []
    if REQUIRE_SAME_DATASET_CLASS:
        bucket_cols.append("dataset_class")
    if REQUIRE_SAME_SURVEY_ROLE:
        bucket_cols.append("survey_role")

    if not bucket_cols:
        df["_all_bucket"] = "ALL"
        bucket_cols = ["_all_bucket"]

    candidate_rows = []

    for bucket_key, g in df.groupby(bucket_cols, dropna=False):
        g = g.copy().reset_index(drop=True)
        if len(g) <= 1:
            continue

        texts = g["broad_text_norm"].fillna("").astype(str).tolist()
        sim_mat = build_text_vector_similarity(texts)

        for i in range(len(g)):
            a = g.iloc[i]
            recs = []

            for j in range(len(g)):
                if i == j:
                    continue
                b = g.iloc[j]

                if FORBID_SAME_YEAR_MATCH and safe_int(a["year"]) == safe_int(b["year"]):
                    continue

                cosine_score = float(sim_mat[i, j])
                score_dict = pre_score_pair(a, b, cosine_score)

                pass_rule = (
                    score_dict["pre_score"] >= PRE_SCORE_MIN and
                    (
                        score_dict["label_similarity"] >= LABEL_SIM_MIN or
                        score_dict["base_similarity"] >= 60 or
                        cosine_score >= COSINE_MIN
                    )
                )

                if not pass_rule:
                    continue

                row = {
                    "target_id": safe_str(a["strict_group_id"]),
                    "target_year": safe_int(a["year"]),
                    "target_group_label": safe_str(a["group_label"]),
                    "target_anchor_var": safe_str(a["anchor_var"]),
                    "target_anchor_label": safe_str(a["anchor_label"]),
                    "target_dataset_class": safe_str(a["dataset_class"]),
                    "target_survey_role": safe_str(a["survey_role"]),
                    "target_structure_type": safe_str(a["structure_type_mode"]),
                    "target_member_count": safe_int(a["member_count"]),
                    "target_member_var_names": safe_str(a["member_var_names"]),
                    "target_member_labels": safe_str(a["member_labels"]),
                    "target_value_label_summary": short_text(a["value_label_summary"], 1000),
                    "target_time_unit_tokens": safe_str(a["time_unit_tokens_group"]),
                    "target_embedded_year_tokens": safe_str(a["embedded_year_tokens_group"]),

                    "candidate_id": safe_str(b["strict_group_id"]),
                    "candidate_year": safe_int(b["year"]),
                    "candidate_group_label": safe_str(b["group_label"]),
                    "candidate_anchor_var": safe_str(b["anchor_var"]),
                    "candidate_anchor_label": safe_str(b["anchor_label"]),
                    "candidate_dataset_class": safe_str(b["dataset_class"]),
                    "candidate_survey_role": safe_str(b["survey_role"]),
                    "candidate_structure_type": safe_str(b["structure_type_mode"]),
                    "candidate_member_count": safe_int(b["member_count"]),
                    "candidate_member_var_names": safe_str(b["member_var_names"]),
                    "candidate_member_labels": safe_str(b["member_labels"]),
                    "candidate_value_label_summary": short_text(b["value_label_summary"], 1000),
                    "candidate_time_unit_tokens": safe_str(b["time_unit_tokens_group"]),
                    "candidate_embedded_year_tokens": safe_str(b["embedded_year_tokens_group"]),
                }
                row.update(score_dict)
                recs.append(row)

            if recs:
                recs_df = pd.DataFrame(recs).sort_values(
                    by=["pre_score", "label_similarity", "cosine_similarity"],
                    ascending=[False, False, False]
                ).head(TOP_K_PER_TARGET)

                recs_df["rank_within_target"] = range(1, len(recs_df) + 1)
                candidate_rows.append(recs_df)

    if candidate_rows:
        candidate_pool = pd.concat(candidate_rows, ignore_index=True)
    else:
        candidate_pool = pd.DataFrame()

    if not candidate_pool.empty:
        candidate_pool["pair_id"] = candidate_pool.apply(
            lambda r: hash_text(f"{r['target_id']}||{r['candidate_id']}"),
            axis=1
        )

    return candidate_pool


# =========================================================
# 5．LLM 缓存
# =========================================================

def load_llm_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    cache = {}
    if not cache_path.exists():
        return cache

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pair_id = safe_str(obj.get("pair_id"))
                if pair_id:
                    cache[pair_id] = obj
            except Exception:
                continue
    return cache


def append_cache_record(cache_path: Path, record: Dict[str, Any]):
    ensure_parent_dir(cache_path)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================================================
# 6．LLM Prompt 与请求
# =========================================================

def build_llm_prompt(row: pd.Series) -> str:
    prompt = f"""
你是一名严谨的社会调查变量协调研究助理。请判断下面这两个不同年份的 CFPS strict question group 是否可视为“跨年对应变量/题组”。

请严格按以下标准判断：
1．如果两者本质上是同一概念，只存在轻微命名、措辞或年份变化，判为 high。
2．如果两者大概率是同一概念，但可能存在时间单位、统计口径、参考对象、值标签结构、题干范围等差异，判为 medium。
3．如果两者只是同模块相近问题，不能稳定视为同一概念，判为 low。
4．如果两者明显不是同一概念，判为 reject。

重点核查：
- 是否是同一测量概念，而不只是同一模块；
- 是否存在时间单位差异，如年/月/日；
- 是否存在嵌入年份差异，如 2012 vs 2013；
- 是否值标签结构不同；
- 是否变量结构不同；
- 是否只是同题不同选项，或实际上是不同问题。

请仅输出 JSON，不要输出任何额外文字。格式如下：
{{
  "match_level": "high/medium/low/reject",
  "confidence": 0,
  "reason": "不超过120字",
  "risk_flags": ["flag1", "flag2"],
  "suggested_action": "accept/review/reject"
}}

下面是待比较的两个题组：

【目标题组】
strict_group_id: {safe_str(row["target_id"])}
year: {safe_str(row["target_year"])}
dataset_class: {safe_str(row["target_dataset_class"])}
survey_role: {safe_str(row["target_survey_role"])}
structure_type: {safe_str(row["target_structure_type"])}
group_label: {safe_str(row["target_group_label"])}
anchor_var: {safe_str(row["target_anchor_var"])}
anchor_label: {safe_str(row["target_anchor_label"])}
member_count: {safe_str(row["target_member_count"])}
member_var_names: {short_text(row["target_member_var_names"], 800)}
member_labels: {short_text(row["target_member_labels"], 800)}
value_label_summary: {short_text(row["target_value_label_summary"], 800)}
time_unit_tokens: {safe_str(row["target_time_unit_tokens"])}
embedded_year_tokens: {safe_str(row["target_embedded_year_tokens"])}

【候选题组】
strict_group_id: {safe_str(row["candidate_id"])}
year: {safe_str(row["candidate_year"])}
dataset_class: {safe_str(row["candidate_dataset_class"])}
survey_role: {safe_str(row["candidate_survey_role"])}
structure_type: {safe_str(row["candidate_structure_type"])}
group_label: {safe_str(row["candidate_group_label"])}
anchor_var: {safe_str(row["candidate_anchor_var"])}
anchor_label: {safe_str(row["candidate_anchor_label"])}
member_count: {safe_str(row["candidate_member_count"])}
member_var_names: {short_text(row["candidate_member_var_names"], 800)}
member_labels: {short_text(row["candidate_member_labels"], 800)}
value_label_summary: {short_text(row["candidate_value_label_summary"], 800)}
time_unit_tokens: {safe_str(row["candidate_time_unit_tokens"])}
embedded_year_tokens: {safe_str(row["candidate_embedded_year_tokens"])}

【规则预评分信息】
pre_score: {safe_str(row["pre_score"])}
label_similarity: {safe_str(row["label_similarity"])}
anchor_similarity: {safe_str(row["anchor_similarity"])}
base_similarity: {safe_str(row["base_similarity"])}
value_label_similarity: {safe_str(row["value_label_similarity"])}
broad_text_similarity: {safe_str(row["broad_text_similarity"])}
cosine_similarity: {safe_str(row["cosine_similarity"])}
penalty: {safe_str(row["penalty"])}
"""
    return prompt.strip()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = safe_str(text).strip()
    if not text:
        raise ValueError("LLM 返回为空")

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidate = m.group(0)
        return json.loads(candidate)

    raise ValueError("无法从 LLM 返回中解析 JSON")


def call_openai_compatible_api(prompt: str) -> Dict[str, Any]:
    url = API_BASE.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严格遵守输出格式的社会科学变量匹配助手。只输出 JSON。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    parsed = extract_json_from_text(content)
    return {
        "raw_response": content,
        "parsed": parsed,
        "api_response": data,
    }


def llm_worker(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    pair_id = safe_str(row_dict["pair_id"])
    row = pd.Series(row_dict)

    prompt = build_llm_prompt(row)
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = call_openai_compatible_api(prompt)
            parsed = result["parsed"]

            match_level = safe_str(parsed.get("match_level")).lower()
            if match_level not in {"high", "medium", "low", "reject"}:
                match_level = "reject"

            confidence = safe_int(parsed.get("confidence"))
            if confidence is None:
                confidence = 0
            confidence = max(0, min(100, confidence))

            risk_flags = parsed.get("risk_flags", [])
            if not isinstance(risk_flags, list):
                risk_flags = [safe_str(risk_flags)] if safe_str(risk_flags) else []

            out = {
                "pair_id": pair_id,
                "target_id": safe_str(row_dict["target_id"]),
                "candidate_id": safe_str(row_dict["candidate_id"]),
                "llm_match_level": match_level,
                "llm_confidence": confidence,
                "llm_reason": short_text(parsed.get("reason", ""), 200),
                "llm_risk_flags": " | ".join([safe_str(x) for x in risk_flags if safe_str(x)]),
                "llm_suggested_action": safe_str(parsed.get("suggested_action", "")),
                "llm_raw_response": short_text(result.get("raw_response", ""), 2000),
                "llm_status": "success",
                "llm_attempts": attempt,
                "llm_error": "",
                "model_provider": MODEL_PROVIDER,
                "model_name": MODEL_NAME,
            }
            return out

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_BASE ** attempt)
            continue

    return {
        "pair_id": pair_id,
        "target_id": safe_str(row_dict["target_id"]),
        "candidate_id": safe_str(row_dict["candidate_id"]),
        "llm_match_level": "reject",
        "llm_confidence": 0,
        "llm_reason": "",
        "llm_risk_flags": "",
        "llm_suggested_action": "reject",
        "llm_raw_response": "",
        "llm_status": "failed",
        "llm_attempts": MAX_RETRIES,
        "llm_error": short_text(last_error, 500),
        "model_provider": MODEL_PROVIDER,
        "model_name": MODEL_NAME,
    }


# =========================================================
# 7．并行运行 LLM 判别
# =========================================================

def run_llm_judgement(candidate_pool: pd.DataFrame) -> pd.DataFrame:
    if candidate_pool.empty:
        return pd.DataFrame()

    send_df = candidate_pool.copy()
    send_df = send_df.sort_values(
        by=["target_id", "rank_within_target", "pre_score"],
        ascending=[True, True, False]
    )
    send_df = send_df.groupby("target_id", as_index=False, group_keys=False).head(SEND_TO_LLM_TOP_K).copy()

    cache = load_llm_cache(OUT_LLM_CACHE) if USE_LLM_CACHE else {}
    results = []

    todo_rows = []
    for _, row in send_df.iterrows():
        pair_id = safe_str(row["pair_id"])
        if USE_LLM_CACHE and pair_id in cache:
            cached = cache[pair_id]
            results.append(cached)
        else:
            todo_rows.append(row.to_dict())

    print(f"LLM 候选对总数：{len(send_df)}")
    print(f"缓存命中数：{len(results)}")
    print(f"待调用数：{len(todo_rows)}")
    print(f"MAX_WORKERS = {MAX_WORKERS}")

    if todo_rows:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_pair = {executor.submit(llm_worker, row_dict): row_dict["pair_id"] for row_dict in todo_rows}

            for idx, future in enumerate(as_completed(future_to_pair), start=1):
                res = future.result()
                results.append(res)

                if USE_LLM_CACHE:
                    append_cache_record(OUT_LLM_CACHE, res)

                if idx % 20 == 0 or idx == len(todo_rows):
                    print(f"LLM 已完成：{idx}/{len(todo_rows)}")

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.drop_duplicates(subset=["pair_id"], keep="last").reset_index(drop=True)

    return result_df


# =========================================================
# 8．最终推荐结果
# =========================================================

def build_final_recommendation(candidate_pool: pd.DataFrame, llm_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidate_pool.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merged = candidate_pool.merge(
        llm_df[
            [
                "pair_id", "llm_match_level", "llm_confidence", "llm_reason",
                "llm_risk_flags", "llm_suggested_action", "llm_status",
                "llm_attempts", "llm_error"
            ]
        ] if not llm_df.empty else pd.DataFrame(columns=["pair_id"]),
        on="pair_id",
        how="left"
    )

    merged["llm_match_level"] = merged["llm_match_level"].fillna("reject")
    merged["llm_confidence"] = pd.to_numeric(merged["llm_confidence"], errors="coerce").fillna(0)
    merged["llm_numeric_score"] = merged["llm_match_level"].map(LLM_LEVEL_TO_SCORE).fillna(10)

    merged["final_recommendation_score"] = (
        FINAL_SCORE_WEIGHT_PRE * pd.to_numeric(merged["pre_score"], errors="coerce").fillna(0) +
        FINAL_SCORE_WEIGHT_LLM * pd.to_numeric(merged["llm_numeric_score"], errors="coerce").fillna(0)
    ).round(2)

    def classify_recommend(row):
        score = safe_float(row["final_recommendation_score"]) or 0
        level = safe_str(row["llm_match_level"])
        if score >= FINAL_HIGH_THRESHOLD and level == "high":
            return "strong_recommend"
        if score >= FINAL_MEDIUM_THRESHOLD and level in {"high", "medium"}:
            return "recommend_with_review"
        if level in {"medium", "low"}:
            return "weak_candidate"
        return "not_recommended"

    merged["recommendation_level"] = merged.apply(classify_recommend, axis=1)

    def manual_review_needed(row):
        if safe_str(row["recommendation_level"]) == "strong_recommend":
            if "time_unit" in safe_str(row["llm_risk_flags"]) or "year" in safe_str(row["llm_risk_flags"]):
                return 1
            return 0
        if safe_str(row["recommendation_level"]) == "recommend_with_review":
            return 1
        if safe_str(row["recommendation_level"]) == "weak_candidate":
            return 1
        return 0

    merged["manual_review_needed"] = merged.apply(manual_review_needed, axis=1)

    merged = merged.sort_values(
        by=["target_id", "final_recommendation_score", "pre_score", "llm_confidence"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)

    merged["recommendation_rank"] = merged.groupby("target_id").cumcount() + 1

    recommend_df = merged.copy()

    # target-level summary
    summary_rows = []
    for target_id, sub in recommend_df.groupby("target_id", dropna=False):
        top1 = sub.iloc[0]
        n_candidates = len(sub)
        strong_n = int((sub["recommendation_level"] == "strong_recommend").sum())
        medium_n = int((sub["recommendation_level"] == "recommend_with_review").sum())
        weak_n = int((sub["recommendation_level"] == "weak_candidate").sum())

        summary_rows.append({
            "target_id": safe_str(target_id),
            "target_year": safe_int(top1["target_year"]),
            "target_group_label": safe_str(top1["target_group_label"]),
            "target_anchor_var": safe_str(top1["target_anchor_var"]),
            "candidate_count": n_candidates,
            "strong_recommend_count": strong_n,
            "review_recommend_count": medium_n,
            "weak_candidate_count": weak_n,
            "best_candidate_id": safe_str(top1["candidate_id"]),
            "best_candidate_year": safe_int(top1["candidate_year"]),
            "best_candidate_group_label": safe_str(top1["candidate_group_label"]),
            "best_pre_score": safe_float(top1["pre_score"]),
            "best_llm_match_level": safe_str(top1["llm_match_level"]),
            "best_final_score": safe_float(top1["final_recommendation_score"]),
            "best_recommendation_level": safe_str(top1["recommendation_level"]),
            "manual_review_needed_for_best": safe_int(top1["manual_review_needed"]),
        })

    target_summary = pd.DataFrame(summary_rows).sort_values(
        by=["target_year", "target_id"]
    ).reset_index(drop=True)

    # manual review sheet：每个 target 保留前3
    manual_review = recommend_df.groupby("target_id", as_index=False, group_keys=False).head(3).copy()
    manual_review["human_final_decision"] = ""
    manual_review["human_selected_rank"] = ""
    manual_review["human_notes"] = ""

    return recommend_df, target_summary, manual_review


# =========================================================
# 9．主程序
# =========================================================

def main():
    print("=" * 90)
    print("CFPS 跨年候选匹配 + 并行 LLM 判别系统启动")
    print("=" * 90)

    print("第 1 步：读取 strict_question_group_summary.csv")
    strict_df = load_strict_group_summary(INPUT_STRICT_GROUP_SUMMARY)
    print(f"strict group 数：{len(strict_df)}")

    print("-" * 90)
    print("第 2 步：生成跨年候选池")
    candidate_pool = build_candidate_pool(strict_df)
    candidate_pool.to_csv(OUT_CANDIDATE_POOL, index=False, encoding="utf-8-sig")
    print(f"候选对数：{len(candidate_pool)}")
    print(f"已导出：{OUT_CANDIDATE_POOL}")

    print("-" * 90)
    print("第 3 步：并行运行 LLM 判别")
    llm_df = run_llm_judgement(candidate_pool)
    llm_df.to_csv(OUT_LLM_JUDGEMENT, index=False, encoding="utf-8-sig")
    print(f"LLM 判别结果数：{len(llm_df)}")
    print(f"已导出：{OUT_LLM_JUDGEMENT}")

    print("-" * 90)
    print("第 4 步：生成最终推荐结果")
    recommend_df, target_summary, manual_review = build_final_recommendation(candidate_pool, llm_df)

    recommend_df.to_csv(OUT_FINAL_RECOMMEND, index=False, encoding="utf-8-sig")
    target_summary.to_csv(OUT_TARGET_SUMMARY, index=False, encoding="utf-8-sig")
    manual_review.to_csv(OUT_MANUAL_REVIEW, index=False, encoding="utf-8-sig")

    print(f"最终推荐记录数：{len(recommend_df)}")
    print(f"target summary 数：{len(target_summary)}")
    print(f"manual review 记录数：{len(manual_review)}")
    print(f"已导出：{OUT_FINAL_RECOMMEND}")
    print(f"已导出：{OUT_TARGET_SUMMARY}")
    print(f"已导出：{OUT_MANUAL_REVIEW}")

    print("-" * 90)
    print("第 5 步：导出工作簿")
    export_excel_with_sheets(OUT_WORKBOOK, {
        "candidate_pool_pairs": candidate_pool,
        "llm_pair_judgement": llm_df,
        "cross_year_recommend": recommend_df,
        "target_level_summary": target_summary,
        "manual_review_sheet": manual_review,
    })
    print(f"已导出：{OUT_WORKBOOK}")

    print("=" * 90)
    print("全部完成")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()