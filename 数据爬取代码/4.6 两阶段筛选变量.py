# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:00:22 2026

@author: Rosem
"""

from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Created on 2026-04-06

@author: ChatGPT
"""

# -*- coding: utf-8 -*-
"""
CFPS 变量协调：基于现有 broad / strict 汇总表的双阶段 LLM 审核系统
版本：Python 3.10+
用途：
1．直接读取现有 broad_concept_cluster_summary.csv 与 strict_question_group_summary.csv
2．阶段一：对 broad cluster 做 LLM 诊断
3．阶段二：对 strict group 相对于所属 broad cluster 做 LLM 判别（strict -> cluster）
4．全流程支持断点续跑
5．阶段内并行调用 API，加快运行速度
6．尽量依据表格结构选择必要字段，减少污染
7．导出：
   - broad / strict 精简卡片
   - prompts
   - raw responses
   - broad_llm_results
   - strict_llm_results
   - review workbook

依赖：
pip install pandas numpy openpyxl xlsxwriter requests tqdm

说明：
1．本脚本不重跑 strict / broad 构建流程。
2．请仅修改“0．配置区”。
3．默认使用 OpenAI 兼容 Chat Completions API。
4．所有随机过程固定 seed。
"""


import os
import re
import ast
import json
import time
import random
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# =========================================================
# 0．配置区：你只需要改这里
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 输入文件
BROAD_CSV = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\harmonization_redesigned_output\broad_concept_cluster_summary.csv")
STRICT_CSV = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\harmonization_redesigned_output\strict_question_group_summary.csv")

# 输出目录
OUTPUT_DIR = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\harmonization_redesigned_output\llm_two_stage_review")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 模型配置 =====
MODEL_PROVIDER = "openai_compatible"
API_BASE = "https://api.siliconflow.cn"
API_KEY = os.getenv("LLM_API_KEY", "sk-nxvyrqxdmlclzuymuaehxteplvhsyvhvbkhucdfwffonjuzr")
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

# 请求参数
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 1
LLM_RETRY_SLEEP = 5
LLM_TEMPERATURE = 0.0

# 并行参数
BROAD_MAX_WORKERS = 2
STRICT_MAX_WORKERS = 12
REQUEST_SLEEP_SECONDS = 0.2

# 运行控制
RUN_BROAD_STAGE = True
RUN_STRICT_STAGE = True
SKIP_IF_ALREADY_DONE = True

# 数量控制
MAX_BROAD_FOR_LLM = 10          # None 表示全部
MAX_STRICT_FOR_LLM = None         # None 表示全部

# strict 阶段是否仅保留 broad 阶段可用簇
STRICT_ONLY_FROM_USABLE_BROAD = False
USABLE_BROAD_STATUSES = {"coherent", "partially_coherent", "noisy"}

# 文本裁剪控制：尽量减少污染
MAX_BROAD_MEMBER_LABELS = 24
MAX_BROAD_MEMBER_ANCHORS = 24
MAX_BROAD_ISSUE_EXAMPLES = 8

MAX_STRICT_MEMBER_VARS = 18
MAX_STRICT_MEMBER_LABELS = 18

MAX_PROMPT_FIELD_LEN = 2200
MAX_RAW_TEXT_EXPORT_LEN = 30000


# =========================================================
# 1．基础工具函数
# =========================================================

def normalize_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def short_text(x: Any, max_len: int = 500) -> str:
    s = normalize_text(x)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def safe_int(x: Any) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_json_from_response(text: str) -> Optional[dict]:
    if text is None:
        return None
    text = text.strip()

    obj = safe_json_loads(text)
    if obj is not None:
        return obj

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        obj = safe_json_loads(m.group(1))
        if obj is not None:
            return obj

    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        obj = safe_json_loads(m.group(1))
        if obj is not None:
            return obj

    return None


def hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def truncate_for_prompt(x: Any, max_len: int = MAX_PROMPT_FIELD_LEN) -> str:
    s = normalize_text(x)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def split_pipe_text(x: Any) -> List[str]:
    s = normalize_text(x)
    if not s:
        return []
    return [i.strip() for i in s.split("|") if i.strip()]


def split_issue_flags(x: Any) -> List[str]:
    s = normalize_text(x)
    if not s:
        return []
    return [i.strip() for i in s.split("|") if i.strip()]


def parse_member_ids(x: Any) -> List[str]:
    s = normalize_text(x)
    if not s:
        return []
    return [i.strip() for i in s.split("|") if i.strip()]


def parse_year_distribution(x: Any) -> Dict[str, int]:
    s = normalize_text(x)
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return {str(k): int(v) for k, v in obj.items()}
    except Exception:
        try:
            obj = ast.literal_eval(s)
            return {str(k): int(v) for k, v in obj.items()}
        except Exception:
            return {}


def take_first_n(items: List[str], n: int) -> List[str]:
    return items[:n] if len(items) > n else items


def to_joined_text(items: List[str], sep: str = " | ") -> str:
    return sep.join([normalize_text(i) for i in items if normalize_text(i)])


def export_excel_with_sheets(path: Path, sheets: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            if df is None:
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


# =========================================================
# 2．读取与预处理
# =========================================================

def load_input_files(broad_csv: Path, strict_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not broad_csv.exists():
        raise FileNotFoundError(f"未找到 broad 文件：{broad_csv}")
    if not strict_csv.exists():
        raise FileNotFoundError(f"未找到 strict 文件：{strict_csv}")

    broad = pd.read_csv(broad_csv)
    strict = pd.read_csv(strict_csv)
    return broad, strict


def prepare_broad_cards(broad: pd.DataFrame) -> pd.DataFrame:
    df = broad.copy()

    required_cols = [
        "broad_cluster_id", "candidate_concept_label", "year_count", "years_covered",
        "first_year", "last_year", "member_group_count", "member_variable_count",
        "main_dataset_class", "main_survey_role", "main_structure_type",
        "cluster_issue_flags", "issue_examples", "member_group_labels",
        "member_anchor_vars", "member_strict_group_ids", "year_distribution"
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    df["broad_cluster_id"] = df["broad_cluster_id"].astype(str)
    df["candidate_concept_label"] = df["candidate_concept_label"].fillna("").astype(str)

    rows = []
    for _, r in df.iterrows():
        group_labels = split_pipe_text(r.get("member_group_labels", ""))
        anchor_vars = split_pipe_text(r.get("member_anchor_vars", ""))
        issue_examples = split_pipe_text(r.get("issue_examples", ""))
        strict_ids = parse_member_ids(r.get("member_strict_group_ids", ""))
        issue_flags = split_issue_flags(r.get("cluster_issue_flags", ""))
        year_dist = parse_year_distribution(r.get("year_distribution", ""))

        selected_group_labels = take_first_n(group_labels, MAX_BROAD_MEMBER_LABELS)
        selected_anchor_vars = take_first_n(anchor_vars, MAX_BROAD_MEMBER_ANCHORS)
        selected_issue_examples = take_first_n(issue_examples, MAX_BROAD_ISSUE_EXAMPLES)

        prompt_summary = {
            "broad_cluster_id": r.get("broad_cluster_id"),
            "candidate_concept_label": r.get("candidate_concept_label"),
            "year_count": safe_int(r.get("year_count")),
            "first_year": safe_int(r.get("first_year")),
            "last_year": safe_int(r.get("last_year")),
            "member_group_count": safe_int(r.get("member_group_count")),
            "member_variable_count": safe_int(r.get("member_variable_count")),
            "main_dataset_class": normalize_text(r.get("main_dataset_class")),
            "main_survey_role": normalize_text(r.get("main_survey_role")),
            "main_structure_type": normalize_text(r.get("main_structure_type")),
            "cluster_issue_flags": issue_flags,
            "issue_examples_selected": selected_issue_examples,
            "member_group_labels_selected": selected_group_labels,
            "member_anchor_vars_selected": selected_anchor_vars,
            "year_distribution": year_dist,
            "strict_group_id_count": len(strict_ids),
        }

        rows.append({
            "broad_cluster_id": r.get("broad_cluster_id"),
            "candidate_concept_label": r.get("candidate_concept_label"),
            "year_count": safe_int(r.get("year_count")),
            "years_covered": normalize_text(r.get("years_covered")),
            "first_year": safe_int(r.get("first_year")),
            "last_year": safe_int(r.get("last_year")),
            "member_group_count": safe_int(r.get("member_group_count")),
            "member_variable_count": safe_int(r.get("member_variable_count")),
            "main_dataset_class": normalize_text(r.get("main_dataset_class")),
            "main_survey_role": normalize_text(r.get("main_survey_role")),
            "main_structure_type": normalize_text(r.get("main_structure_type")),
            "cluster_issue_flags": to_joined_text(issue_flags),
            "issue_examples_selected": to_joined_text(selected_issue_examples),
            "member_group_labels_selected": to_joined_text(selected_group_labels),
            "member_anchor_vars_selected": to_joined_text(selected_anchor_vars),
            "member_strict_group_ids": to_joined_text(strict_ids),
            "year_distribution": json.dumps(year_dist, ensure_ascii=False),
            "strict_group_id_count": len(strict_ids),
            "broad_prompt_payload": json.dumps(prompt_summary, ensure_ascii=False),
        })

    out = pd.DataFrame(rows)
    return out


def prepare_strict_cards(strict: pd.DataFrame) -> pd.DataFrame:
    df = strict.copy()

    required_cols = [
        "strict_group_id", "year", "dataset_class", "survey_role", "strict_group_type",
        "strict_base_stem", "group_label", "anchor_var", "anchor_label", "member_count",
        "slot_indices", "member_var_names", "member_labels", "value_label_summary",
        "file_names", "structure_type_mode", "strict_label_inconsistency",
        "broad_text", "embedded_year_tokens_group", "time_unit_tokens_group"
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    df["strict_group_id"] = df["strict_group_id"].astype(str)

    rows = []
    for _, r in df.iterrows():
        member_vars = split_pipe_text(r.get("member_var_names", ""))
        member_labels = split_pipe_text(r.get("member_labels", ""))
        file_names = split_pipe_text(r.get("file_names", ""))

        selected_member_vars = take_first_n(member_vars, MAX_STRICT_MEMBER_VARS)
        selected_member_labels = take_first_n(member_labels, MAX_STRICT_MEMBER_LABELS)

        strict_payload = {
            "strict_group_id": r.get("strict_group_id"),
            "year": safe_int(r.get("year")),
            "dataset_class": normalize_text(r.get("dataset_class")),
            "survey_role": normalize_text(r.get("survey_role")),
            "strict_group_type": normalize_text(r.get("strict_group_type")),
            "strict_base_stem": normalize_text(r.get("strict_base_stem")),
            "group_label": normalize_text(r.get("group_label")),
            "anchor_var": normalize_text(r.get("anchor_var")),
            "anchor_label": normalize_text(r.get("anchor_label")),
            "member_count": safe_int(r.get("member_count")),
            "slot_indices": normalize_text(r.get("slot_indices")),
            "member_var_names_selected": selected_member_vars,
            "member_labels_selected": selected_member_labels,
            "value_label_summary": short_text(r.get("value_label_summary", ""), 1000),
            "file_names_selected": take_first_n(file_names, 10),
            "structure_type_mode": normalize_text(r.get("structure_type_mode")),
            "strict_label_inconsistency": safe_int(r.get("strict_label_inconsistency")),
            "embedded_year_tokens_group": normalize_text(r.get("embedded_year_tokens_group")),
            "time_unit_tokens_group": normalize_text(r.get("time_unit_tokens_group")),
        }

        rows.append({
            "strict_group_id": r.get("strict_group_id"),
            "year": safe_int(r.get("year")),
            "dataset_class": normalize_text(r.get("dataset_class")),
            "survey_role": normalize_text(r.get("survey_role")),
            "strict_group_type": normalize_text(r.get("strict_group_type")),
            "strict_base_stem": normalize_text(r.get("strict_base_stem")),
            "group_label": normalize_text(r.get("group_label")),
            "anchor_var": normalize_text(r.get("anchor_var")),
            "anchor_label": normalize_text(r.get("anchor_label")),
            "member_count": safe_int(r.get("member_count")),
            "slot_indices": normalize_text(r.get("slot_indices")),
            "member_var_names_selected": to_joined_text(selected_member_vars),
            "member_labels_selected": to_joined_text(selected_member_labels),
            "value_label_summary_selected": short_text(r.get("value_label_summary", ""), 1000),
            "file_names_selected": to_joined_text(take_first_n(file_names, 10)),
            "structure_type_mode": normalize_text(r.get("structure_type_mode")),
            "strict_label_inconsistency": safe_int(r.get("strict_label_inconsistency")),
            "embedded_year_tokens_group": normalize_text(r.get("embedded_year_tokens_group")),
            "time_unit_tokens_group": normalize_text(r.get("time_unit_tokens_group")),
            "strict_prompt_payload": json.dumps(strict_payload, ensure_ascii=False),
        })

    out = pd.DataFrame(rows)
    return out


# =========================================================
# 3．Prompt 设计
# =========================================================

BROAD_SYSTEM_PROMPT = """
你是经济学与社会调查方法领域的高级研究助理，任务是诊断 CFPS 的 broad concept cluster 是否适合作为跨年变量协调的概念簇。
请严格遵守以下规则：
1．broad cluster 只是候选概念簇，不保证内部变量真的同质。
2．如果簇内明显混入不同主题、不同结构、不同时间单位或不同目标年份的变量，应明确指出污染，不可武断认定为可直接协调。
3．你需要判断该簇更适合：
   - 作为较一致的概念簇
   - 作为部分一致但需要人工拆分的概念簇
   - 作为噪声很高、仅可用于导航的聚类
   - 完全不适合用于协调
4．请只输出 JSON，不要输出额外解释文字，不要加 markdown。
5．输出中 cluster_status 只能取：
   coherent
   partially_coherent
   noisy
   unusable

main_issue_type 中每个元素只能取：
   mixed_topic
   mixed_structure
   mixed_time_unit
   mixed_target_year
   mixed_value_label_pattern
   anchor_drift
   weak_concept_label
   oversized_cluster
   other

输出 JSON 必须包含以下字段：
{
  "cluster_status": "...",
  "confidence": 0.0,
  "reason": "...",
  "core_concept": "...",
  "main_issue_type": ["..."],
  "should_split": true,
  "split_suggestions": ["...", "..."],
  "usable_for_direct_harmonization": false,
  "usable_for_navigation_only": true,
  "needs_human_review": true,
  "key_evidence": ["...", "...", "..."]
}
""".strip()


STRICT_SYSTEM_PROMPT = """
你是经济学与社会调查方法领域的高级研究助理，任务是判断某个 strict question group 相对于其所属 broad concept cluster 的关系。
请严格遵守以下规则：
1．broad cluster 只是候选概念簇，strict group 不一定真正属于该概念核心。
2．如果 strict group 与 broad 概念高度一致，可判为 core_member。
3．如果 strict group 与 broad 概念相关，但只是近邻题意、下位概念、上位概念或衍生结构，可判为 related_member。
4．如果 strict group 与 broad 概念仅弱相关、可能是污染带入，可判为 weak_member。
5．如果 strict group 明显不属于该 broad 概念，应判为 exclude。
6．信息不足时判为 uncertain。
7．请只输出 JSON，不要输出额外解释文字，不要加 markdown。

relation_to_cluster 只能取：
   core_member
   related_member
   weak_member
   exclude
   uncertain

comparability_risk 只能取：
   low
   medium
   high

输出 JSON 必须包含以下字段：
{
  "relation_to_cluster": "...",
  "confidence": 0.0,
  "reason": "...",
  "suggested_subconcept": "...",
  "comparability_risk": "low",
  "needs_human_review": true,
  "key_evidence": ["...", "...", "..."]
}
""".strip()


def build_broad_prompt(row: pd.Series) -> str:
    payload = {
        "task": "诊断该 broad cluster 是否是可用于跨年变量协调的有效概念簇。",
        "cluster_card": {
            "broad_cluster_id": row.get("broad_cluster_id"),
            "candidate_concept_label": truncate_for_prompt(row.get("candidate_concept_label")),
            "year_count": row.get("year_count"),
            "first_year": row.get("first_year"),
            "last_year": row.get("last_year"),
            "member_group_count": row.get("member_group_count"),
            "member_variable_count": row.get("member_variable_count"),
            "main_dataset_class": truncate_for_prompt(row.get("main_dataset_class")),
            "main_survey_role": truncate_for_prompt(row.get("main_survey_role")),
            "main_structure_type": truncate_for_prompt(row.get("main_structure_type")),
            "cluster_issue_flags": truncate_for_prompt(row.get("cluster_issue_flags")),
            "issue_examples_selected": truncate_for_prompt(row.get("issue_examples_selected"), 1800),
            "member_group_labels_selected": truncate_for_prompt(row.get("member_group_labels_selected"), 1800),
            "member_anchor_vars_selected": truncate_for_prompt(row.get("member_anchor_vars_selected"), 1500),
            "year_distribution": truncate_for_prompt(row.get("year_distribution")),
            "strict_group_id_count": row.get("strict_group_id_count"),
        },
        "evaluation_focus": [
            "该簇是否围绕一个可辨识的核心概念",
            "是否存在明显跨主题污染",
            "是否存在结构类型污染",
            "是否存在时间单位或目标年份污染",
            "该簇是否能直接用于变量协调，还是只能作为导航线索"
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


def build_strict_prompt(broad_row: pd.Series, strict_row: pd.Series) -> str:
    payload = {
        "task": "判断该 strict group 相对于 broad cluster 的归属关系。",
        "broad_cluster_card": {
            "broad_cluster_id": broad_row.get("broad_cluster_id"),
            "candidate_concept_label": truncate_for_prompt(broad_row.get("candidate_concept_label")),
            "main_dataset_class": truncate_for_prompt(broad_row.get("main_dataset_class")),
            "main_survey_role": truncate_for_prompt(broad_row.get("main_survey_role")),
            "main_structure_type": truncate_for_prompt(broad_row.get("main_structure_type")),
            "cluster_issue_flags": truncate_for_prompt(broad_row.get("cluster_issue_flags")),
            "member_group_labels_selected": truncate_for_prompt(broad_row.get("member_group_labels_selected"), 1600),
            "member_anchor_vars_selected": truncate_for_prompt(broad_row.get("member_anchor_vars_selected"), 1400),
            "year_distribution": truncate_for_prompt(broad_row.get("year_distribution")),
        },
        "strict_group_card": {
            "strict_group_id": strict_row.get("strict_group_id"),
            "year": strict_row.get("year"),
            "dataset_class": truncate_for_prompt(strict_row.get("dataset_class")),
            "survey_role": truncate_for_prompt(strict_row.get("survey_role")),
            "strict_group_type": truncate_for_prompt(strict_row.get("strict_group_type")),
            "strict_base_stem": truncate_for_prompt(strict_row.get("strict_base_stem")),
            "group_label": truncate_for_prompt(strict_row.get("group_label")),
            "anchor_var": truncate_for_prompt(strict_row.get("anchor_var")),
            "anchor_label": truncate_for_prompt(strict_row.get("anchor_label")),
            "member_count": strict_row.get("member_count"),
            "slot_indices": truncate_for_prompt(strict_row.get("slot_indices")),
            "member_var_names_selected": truncate_for_prompt(strict_row.get("member_var_names_selected"), 1600),
            "member_labels_selected": truncate_for_prompt(strict_row.get("member_labels_selected"), 1800),
            "value_label_summary_selected": truncate_for_prompt(strict_row.get("value_label_summary_selected"), 1200),
            "file_names_selected": truncate_for_prompt(strict_row.get("file_names_selected")),
            "structure_type_mode": truncate_for_prompt(strict_row.get("structure_type_mode")),
            "strict_label_inconsistency": strict_row.get("strict_label_inconsistency"),
            "embedded_year_tokens_group": truncate_for_prompt(strict_row.get("embedded_year_tokens_group")),
            "time_unit_tokens_group": truncate_for_prompt(strict_row.get("time_unit_tokens_group")),
        },
        "decision_focus": [
            "该 strict group 是否属于 broad 概念核心成员",
            "是否只是相邻题意或衍生题组",
            "是否明显属于误聚类污染",
            "是否需要单独拆出子概念"
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


# =========================================================
# 4．API 调用
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

    url = api_base.rstrip("/") + "/v1/chat/completions"
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


def call_llm_api(system_prompt: str, user_prompt: str) -> Tuple[str, dict]:
    last_err = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp_json = call_openai_compatible_api(
                system_prompt=system_prompt,
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
# 5．结果标准化
# =========================================================

def normalize_broad_llm_result(obj: Optional[dict]) -> dict:
    default = {
        "cluster_status": "unusable",
        "confidence": None,
        "reason": "",
        "core_concept": "",
        "main_issue_type": "",
        "should_split": True,
        "split_suggestions": "",
        "usable_for_direct_harmonization": False,
        "usable_for_navigation_only": True,
        "needs_human_review": True,
        "key_evidence": "",
    }
    if obj is None:
        return default

    cluster_status = obj.get("cluster_status", "unusable")
    if cluster_status not in {"coherent", "partially_coherent", "noisy", "unusable"}:
        cluster_status = "unusable"

    confidence = obj.get("confidence", None)
    try:
        confidence = None if confidence is None else float(confidence)
    except Exception:
        confidence = None

    main_issue_type = obj.get("main_issue_type", [])
    if isinstance(main_issue_type, list):
        valid = {
            "mixed_topic", "mixed_structure", "mixed_time_unit", "mixed_target_year",
            "mixed_value_label_pattern", "anchor_drift", "weak_concept_label",
            "oversized_cluster", "other"
        }
        main_issue_type = [i for i in main_issue_type if i in valid]
        main_issue_type = " | ".join(main_issue_type)
    else:
        main_issue_type = str(main_issue_type)

    split_suggestions = obj.get("split_suggestions", [])
    if isinstance(split_suggestions, list):
        split_suggestions = " | ".join([str(i) for i in split_suggestions])
    else:
        split_suggestions = str(split_suggestions)

    key_evidence = obj.get("key_evidence", [])
    if isinstance(key_evidence, list):
        key_evidence = " | ".join([str(i) for i in key_evidence])
    else:
        key_evidence = str(key_evidence)

    return {
        "cluster_status": cluster_status,
        "confidence": confidence,
        "reason": str(obj.get("reason", "")),
        "core_concept": str(obj.get("core_concept", "")),
        "main_issue_type": main_issue_type,
        "should_split": bool(obj.get("should_split", True)),
        "split_suggestions": split_suggestions,
        "usable_for_direct_harmonization": bool(obj.get("usable_for_direct_harmonization", False)),
        "usable_for_navigation_only": bool(obj.get("usable_for_navigation_only", True)),
        "needs_human_review": bool(obj.get("needs_human_review", True)),
        "key_evidence": key_evidence,
    }


def normalize_strict_llm_result(obj: Optional[dict]) -> dict:
    default = {
        "relation_to_cluster": "uncertain",
        "confidence": None,
        "reason": "",
        "suggested_subconcept": "",
        "comparability_risk": "high",
        "needs_human_review": True,
        "key_evidence": "",
    }
    if obj is None:
        return default

    relation = obj.get("relation_to_cluster", "uncertain")
    if relation not in {"core_member", "related_member", "weak_member", "exclude", "uncertain"}:
        relation = "uncertain"

    risk = obj.get("comparability_risk", "high")
    if risk not in {"low", "medium", "high"}:
        risk = "high"

    confidence = obj.get("confidence", None)
    try:
        confidence = None if confidence is None else float(confidence)
    except Exception:
        confidence = None

    key_evidence = obj.get("key_evidence", [])
    if isinstance(key_evidence, list):
        key_evidence = " | ".join([str(i) for i in key_evidence])
    else:
        key_evidence = str(key_evidence)

    return {
        "relation_to_cluster": relation,
        "confidence": confidence,
        "reason": str(obj.get("reason", "")),
        "suggested_subconcept": str(obj.get("suggested_subconcept", "")),
        "comparability_risk": risk,
        "needs_human_review": bool(obj.get("needs_human_review", True)),
        "key_evidence": key_evidence,
    }


# =========================================================
# 6．断点续跑与并行写入
# =========================================================

WRITE_LOCK = threading.Lock()


def load_done_ids(csv_path: Path, id_col: str) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        if id_col in df.columns:
            return set(df[id_col].astype(str).tolist())
        return set()
    except Exception:
        return set()


def append_jsonl(path: Path, record: dict):
    with WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_result_csv(path: Path, row: dict):
    with WRITE_LOCK:
        df = pd.DataFrame([row])
        header = not path.exists()
        df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")


# =========================================================
# 7．Broad 阶段
# =========================================================

def run_single_broad_task(row: pd.Series, prompts_path: Path, raw_resp_path: Path) -> dict:
    broad_cluster_id = str(row["broad_cluster_id"])
    prompt = build_broad_prompt(row)

    append_jsonl(prompts_path, {
        "broad_cluster_id": broad_cluster_id,
        "prompt": prompt
    })

    raw_text = ""
    raw_json = None
    parse_ok = False
    parse_obj = None
    err_msg = ""

    try:
        raw_text, raw_json = call_llm_api(BROAD_SYSTEM_PROMPT, prompt)
        if len(raw_text) > MAX_RAW_TEXT_EXPORT_LEN:
            raw_text = raw_text[:MAX_RAW_TEXT_EXPORT_LEN] + "..."
        parse_obj = extract_json_from_response(raw_text)
        parse_ok = parse_obj is not None
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"

    append_jsonl(raw_resp_path, {
        "broad_cluster_id": broad_cluster_id,
        "raw_text": raw_text,
        "raw_json": raw_json,
        "parse_ok": parse_ok,
        "error": err_msg,
    })

    norm = normalize_broad_llm_result(parse_obj)

    out = {
        "broad_cluster_id": broad_cluster_id,
        "candidate_concept_label": row.get("candidate_concept_label"),
        "year_count": row.get("year_count"),
        "first_year": row.get("first_year"),
        "last_year": row.get("last_year"),
        "member_group_count": row.get("member_group_count"),
        "member_variable_count": row.get("member_variable_count"),
        "main_dataset_class": row.get("main_dataset_class"),
        "main_survey_role": row.get("main_survey_role"),
        "main_structure_type": row.get("main_structure_type"),
        "cluster_issue_flags": row.get("cluster_issue_flags"),
        "issue_examples_selected": row.get("issue_examples_selected"),
        "member_group_labels_selected": row.get("member_group_labels_selected"),
        "member_anchor_vars_selected": row.get("member_anchor_vars_selected"),
        "year_distribution": row.get("year_distribution"),
        "parse_ok": parse_ok,
        "api_error": err_msg,
        "raw_text": raw_text,
    }
    out.update(norm)

    time.sleep(REQUEST_SLEEP_SECONDS)
    return out


def run_broad_stage(broad_cards: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    prompts_path = out_dir / "broad_cluster_llm_prompts.jsonl"
    raw_resp_path = out_dir / "broad_cluster_llm_raw_responses.jsonl"
    results_csv = out_dir / "broad_cluster_llm_results.csv"

    done_ids = load_done_ids(results_csv, "broad_cluster_id") if SKIP_IF_ALREADY_DONE else set()

    work = broad_cards.copy()
    if MAX_BROAD_FOR_LLM is not None:
        work = work.head(MAX_BROAD_FOR_LLM).copy()

    tasks = []
    for _, row in work.iterrows():
        bid = str(row["broad_cluster_id"])
        if bid in done_ids:
            continue
        tasks.append(row)

    if not tasks:
        if results_csv.exists():
            return pd.read_csv(results_csv)
        return pd.DataFrame()

    results = []
    with ThreadPoolExecutor(max_workers=BROAD_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_broad_task, row, prompts_path, raw_resp_path): str(row["broad_cluster_id"])
            for row in tasks
        }

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Broad 阶段 LLM 诊断"):
            row_out = future.result()
            append_result_csv(results_csv, row_out)
            results.append(row_out)

    if results_csv.exists():
        return pd.read_csv(results_csv)
    return pd.DataFrame(results)


# =========================================================
# 8．Strict 阶段任务生成
# =========================================================

def build_strict_tasks(
    broad_cards: pd.DataFrame,
    strict_cards: pd.DataFrame,
    broad_results: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    broad_use = broad_cards.copy()

    if STRICT_ONLY_FROM_USABLE_BROAD and broad_results is not None and not broad_results.empty:
        ok = broad_results[broad_results["cluster_status"].astype(str).isin(USABLE_BROAD_STATUSES)].copy()
        broad_use = broad_use[broad_use["broad_cluster_id"].astype(str).isin(ok["broad_cluster_id"].astype(str))].copy()

    broad_lookup = broad_use.set_index("broad_cluster_id", drop=False).to_dict("index")
    strict_lookup = strict_cards.set_index("strict_group_id", drop=False).to_dict("index")

    records = []
    for _, brow in broad_use.iterrows():
        bid = str(brow["broad_cluster_id"])
        strict_ids = parse_member_ids(brow.get("member_strict_group_ids", ""))
        for sid in strict_ids:
            if sid not in strict_lookup:
                continue
            srow = strict_lookup[sid]

            task_id = hash_text(bid + "||" + sid)
            records.append({
                "task_id": task_id,
                "broad_cluster_id": bid,
                "strict_group_id": sid,

                "candidate_concept_label": brow.get("candidate_concept_label"),
                "broad_main_dataset_class": brow.get("main_dataset_class"),
                "broad_main_survey_role": brow.get("main_survey_role"),
                "broad_main_structure_type": brow.get("main_structure_type"),
                "broad_cluster_issue_flags": brow.get("cluster_issue_flags"),
                "broad_member_group_labels_selected": brow.get("member_group_labels_selected"),
                "broad_member_anchor_vars_selected": brow.get("member_anchor_vars_selected"),
                "broad_year_distribution": brow.get("year_distribution"),

                "strict_year": srow.get("year"),
                "strict_dataset_class": srow.get("dataset_class"),
                "strict_survey_role": srow.get("survey_role"),
                "strict_group_type": srow.get("strict_group_type"),
                "strict_base_stem": srow.get("strict_base_stem"),
                "strict_group_label": srow.get("group_label"),
                "strict_anchor_var": srow.get("anchor_var"),
                "strict_anchor_label": srow.get("anchor_label"),
                "strict_member_count": srow.get("member_count"),
                "strict_structure_type_mode": srow.get("structure_type_mode"),
                "strict_label_inconsistency": srow.get("strict_label_inconsistency"),
                "strict_member_var_names_selected": srow.get("member_var_names_selected"),
                "strict_member_labels_selected": srow.get("member_labels_selected"),
                "strict_value_label_summary_selected": srow.get("value_label_summary_selected"),
                "strict_embedded_year_tokens_group": srow.get("embedded_year_tokens_group"),
                "strict_time_unit_tokens_group": srow.get("time_unit_tokens_group"),
            })

    tasks_df = pd.DataFrame(records)
    if not tasks_df.empty and MAX_STRICT_FOR_LLM is not None:
        tasks_df = tasks_df.head(MAX_STRICT_FOR_LLM).copy()
    return tasks_df


# =========================================================
# 9．Strict 阶段
# =========================================================

def run_single_strict_task(task_row: pd.Series, broad_lookup: Dict[str, dict], strict_lookup: Dict[str, dict],
                           prompts_path: Path, raw_resp_path: Path) -> dict:
    task_id = str(task_row["task_id"])
    broad_cluster_id = str(task_row["broad_cluster_id"])
    strict_group_id = str(task_row["strict_group_id"])

    broad_row = pd.Series(broad_lookup[broad_cluster_id])
    strict_row = pd.Series(strict_lookup[strict_group_id])

    prompt = build_strict_prompt(broad_row, strict_row)

    append_jsonl(prompts_path, {
        "task_id": task_id,
        "broad_cluster_id": broad_cluster_id,
        "strict_group_id": strict_group_id,
        "prompt": prompt
    })

    raw_text = ""
    raw_json = None
    parse_ok = False
    parse_obj = None
    err_msg = ""

    try:
        raw_text, raw_json = call_llm_api(STRICT_SYSTEM_PROMPT, prompt)
        if len(raw_text) > MAX_RAW_TEXT_EXPORT_LEN:
            raw_text = raw_text[:MAX_RAW_TEXT_EXPORT_LEN] + "..."
        parse_obj = extract_json_from_response(raw_text)
        parse_ok = parse_obj is not None
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"

    append_jsonl(raw_resp_path, {
        "task_id": task_id,
        "broad_cluster_id": broad_cluster_id,
        "strict_group_id": strict_group_id,
        "raw_text": raw_text,
        "raw_json": raw_json,
        "parse_ok": parse_ok,
        "error": err_msg,
    })

    norm = normalize_strict_llm_result(parse_obj)

    out = {
        "task_id": task_id,
        "broad_cluster_id": broad_cluster_id,
        "strict_group_id": strict_group_id,
        "candidate_concept_label": task_row.get("candidate_concept_label"),
        "broad_main_dataset_class": task_row.get("broad_main_dataset_class"),
        "broad_main_survey_role": task_row.get("broad_main_survey_role"),
        "broad_main_structure_type": task_row.get("broad_main_structure_type"),
        "broad_cluster_issue_flags": task_row.get("broad_cluster_issue_flags"),
        "strict_year": task_row.get("strict_year"),
        "strict_dataset_class": task_row.get("strict_dataset_class"),
        "strict_survey_role": task_row.get("strict_survey_role"),
        "strict_group_type": task_row.get("strict_group_type"),
        "strict_base_stem": task_row.get("strict_base_stem"),
        "strict_group_label": task_row.get("strict_group_label"),
        "strict_anchor_var": task_row.get("strict_anchor_var"),
        "strict_anchor_label": task_row.get("strict_anchor_label"),
        "strict_member_count": task_row.get("strict_member_count"),
        "strict_structure_type_mode": task_row.get("strict_structure_type_mode"),
        "strict_label_inconsistency": task_row.get("strict_label_inconsistency"),
        "parse_ok": parse_ok,
        "api_error": err_msg,
        "raw_text": raw_text,
    }
    out.update(norm)

    time.sleep(REQUEST_SLEEP_SECONDS)
    return out


def run_strict_stage(
    strict_tasks: pd.DataFrame,
    broad_cards: pd.DataFrame,
    strict_cards: pd.DataFrame,
    out_dir: Path
) -> pd.DataFrame:
    prompts_path = out_dir / "strict_cluster_llm_prompts.jsonl"
    raw_resp_path = out_dir / "strict_cluster_llm_raw_responses.jsonl"
    results_csv = out_dir / "strict_cluster_llm_results.csv"

    done_ids = load_done_ids(results_csv, "task_id") if SKIP_IF_ALREADY_DONE else set()

    broad_lookup = broad_cards.set_index("broad_cluster_id", drop=False).to_dict("index")
    strict_lookup = strict_cards.set_index("strict_group_id", drop=False).to_dict("index")

    work = strict_tasks.copy()
    tasks = []
    for _, row in work.iterrows():
        tid = str(row["task_id"])
        if tid in done_ids:
            continue
        tasks.append(row)

    if not tasks:
        if results_csv.exists():
            return pd.read_csv(results_csv)
        return pd.DataFrame()

    results = []
    with ThreadPoolExecutor(max_workers=STRICT_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_strict_task, row, broad_lookup, strict_lookup, prompts_path, raw_resp_path): str(row["task_id"])
            for row in tasks
        }

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Strict→Cluster 阶段 LLM 判别"):
            row_out = future.result()
            append_result_csv(results_csv, row_out)
            results.append(row_out)

    if results_csv.exists():
        return pd.read_csv(results_csv)
    return pd.DataFrame(results)


# =========================================================
# 10．人工审核模板
# =========================================================

def build_broad_review_template(broad_cards: pd.DataFrame, broad_results: pd.DataFrame) -> pd.DataFrame:
    base = broad_cards.copy()

    if broad_results is not None and not broad_results.empty:
        keep_cols = [
            "broad_cluster_id",
            "cluster_status", "confidence", "reason", "core_concept",
            "main_issue_type", "should_split", "split_suggestions",
            "usable_for_direct_harmonization", "usable_for_navigation_only",
            "needs_human_review", "key_evidence", "parse_ok", "api_error"
        ]
        tmp = broad_results[keep_cols].drop_duplicates(subset=["broad_cluster_id"])
        base = base.merge(tmp, on="broad_cluster_id", how="left")

    base["human_decision"] = ""
    base["approved_cluster_status"] = ""
    base["approved_core_concept"] = ""
    base["approved_split_plan"] = ""
    base["review_notes"] = ""
    base["reviewer"] = ""
    base["review_date"] = ""
    return base


def build_strict_review_template(strict_tasks: pd.DataFrame, strict_results: pd.DataFrame) -> pd.DataFrame:
    base = strict_tasks.copy()

    if strict_results is not None and not strict_results.empty:
        keep_cols = [
            "task_id", "relation_to_cluster", "confidence", "reason",
            "suggested_subconcept", "comparability_risk",
            "needs_human_review", "key_evidence", "parse_ok", "api_error"
        ]
        tmp = strict_results[keep_cols].drop_duplicates(subset=["task_id"])
        base = base.merge(tmp, on="task_id", how="left")

    base["human_decision"] = ""
    base["approved_relation_to_cluster"] = ""
    base["approved_subconcept"] = ""
    base["review_notes"] = ""
    base["reviewer"] = ""
    base["review_date"] = ""
    return base


# =========================================================
# 11．主程序
# =========================================================

def main():
    print("=" * 90)
    print("CFPS broad / strict 双阶段 LLM 审核系统启动")
    print("=" * 90)

    print("第 1 步：读取输入文件")
    broad_raw, strict_raw = load_input_files(BROAD_CSV, STRICT_CSV)
    print(f"broad_raw：{broad_raw.shape}")
    print(f"strict_raw：{strict_raw.shape}")

    print("-" * 90)
    print("第 2 步：构造精简卡片，尽量减少污染字段")
    broad_cards = prepare_broad_cards(broad_raw)
    strict_cards = prepare_strict_cards(strict_raw)

    broad_cards_csv = OUTPUT_DIR / "broad_cluster_cards.csv"
    strict_cards_csv = OUTPUT_DIR / "strict_group_cards.csv"
    broad_cards.to_csv(broad_cards_csv, index=False, encoding="utf-8-sig")
    strict_cards.to_csv(strict_cards_csv, index=False, encoding="utf-8-sig")
    print(f"[完成] broad 卡片：{broad_cards_csv}")
    print(f"[完成] strict 卡片：{strict_cards_csv}")

    broad_results = pd.DataFrame()
    strict_tasks = pd.DataFrame()
    strict_results = pd.DataFrame()

    if RUN_BROAD_STAGE:
        print("-" * 90)
        print("第 3 步：Broad 阶段 LLM 诊断")
        broad_results = run_broad_stage(broad_cards, OUTPUT_DIR)
        broad_results_xlsx = OUTPUT_DIR / "broad_cluster_llm_results.xlsx"
        export_excel_with_sheets(broad_results_xlsx, {"broad_cluster_llm_results": broad_results})
        print(f"[完成] broad LLM 结果：{broad_results.shape}")
        print(f"[完成] 导出：{broad_results_xlsx}")
    else:
        broad_csv = OUTPUT_DIR / "broad_cluster_llm_results.csv"
        if broad_csv.exists():
            broad_results = pd.read_csv(broad_csv)
            print(f"[提示] 跳过 broad 运行，读取历史结果：{broad_csv}")

    if RUN_STRICT_STAGE:
        print("-" * 90)
        print("第 4 步：生成 Strict→Cluster 任务")
        strict_tasks = build_strict_tasks(broad_cards, strict_cards, broad_results)
        strict_tasks_csv = OUTPUT_DIR / "strict_cluster_tasks.csv"
        strict_tasks.to_csv(strict_tasks_csv, index=False, encoding="utf-8-sig")
        print(f"[完成] strict 任务数量：{len(strict_tasks)}")
        print(f"[完成] 导出：{strict_tasks_csv}")

        print("-" * 90)
        print("第 5 步：Strict→Cluster 阶段 LLM 判别")
        strict_results = run_strict_stage(strict_tasks, broad_cards, strict_cards, OUTPUT_DIR)
        strict_results_xlsx = OUTPUT_DIR / "strict_cluster_llm_results.xlsx"
        export_excel_with_sheets(strict_results_xlsx, {"strict_cluster_llm_results": strict_results})
        print(f"[完成] strict LLM 结果：{strict_results.shape}")
        print(f"[完成] 导出：{strict_results_xlsx}")
    else:
        strict_csv = OUTPUT_DIR / "strict_cluster_llm_results.csv"
        if strict_csv.exists():
            strict_results = pd.read_csv(strict_csv)
            print(f"[提示] 跳过 strict 运行，读取历史结果：{strict_csv}")

    print("-" * 90)
    print("第 6 步：构建人工审核模板")
    broad_review = build_broad_review_template(broad_cards, broad_results)
    if strict_tasks.empty:
        strict_tasks_path = OUTPUT_DIR / "strict_cluster_tasks.csv"
        if strict_tasks_path.exists():
            strict_tasks = pd.read_csv(strict_tasks_path)
    strict_review = build_strict_review_template(strict_tasks, strict_results) if not strict_tasks.empty else pd.DataFrame()

    review_xlsx = OUTPUT_DIR / "cfps_llm_two_stage_review_workbook.xlsx"
    export_excel_with_sheets(review_xlsx, {
        "broad_cards": broad_cards,
        "strict_cards": strict_cards,
        "broad_llm_results": broad_results,
        "strict_tasks": strict_tasks,
        "strict_llm_results": strict_results,
        "broad_review_template": broad_review,
        "strict_review_template": strict_review,
    })
    print(f"[完成] 导出总工作簿：{review_xlsx}")

    print("=" * 90)
    print("全部完成")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()