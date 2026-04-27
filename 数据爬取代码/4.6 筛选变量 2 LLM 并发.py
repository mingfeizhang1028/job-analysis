
from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:18:39 2026

@author: Rosem
"""

# -*- coding: utf-8 -*-
"""
聚类后簇内 LLM 校正版（并发版）
输入：
    cluster_harmonization_output/variable_cluster_map.csv
    cluster_harmonization_output/concept_clusters_summary.csv

输出：
    llm_cluster_correction_output/
        1. llm_review_requests.jsonl
        2. llm_review_responses.jsonl
        3. llm_review_failures.csv
        4. variable_cluster_map_llm_corrected.csv
        5. concept_clusters_summary_llm_corrected.csv
        6. llm_cluster_review.xlsx

Python 3.13
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd
import requests


# =========================
# 0. 全局配置
# =========================
SEED = 20250406
random.seed(SEED)
np.random.seed(SEED)

# ===== 模型配置 =====
MODEL_PROVIDER = "OpenAi_Responce"
API_BASE = "https://api.siliconflow.cn"
API_KEY = os.getenv("LLM_API_KEY", "sk-nxvyrqxdmlclzuymuaehxteplvhsyvhvbkhucdfwffonjuzr")
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

# ===== 输入输出路径 =====
INPUT_DIR = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\cluster_harmonization_output")
INPUT_VAR_MAP = INPUT_DIR / "variable_cluster_map.csv"
INPUT_CLUSTER_SUMMARY = INPUT_DIR / "concept_clusters_summary.csv"

OUTPUT_DIR = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\llm_cluster_correction_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REQUESTS = OUTPUT_DIR / "llm_review_requests.jsonl"
OUTPUT_RESPONSES = OUTPUT_DIR / "llm_review_responses.jsonl"
OUTPUT_FAILURES = OUTPUT_DIR / "llm_review_failures.csv"
OUTPUT_VAR_MAP_CORRECTED = OUTPUT_DIR / "variable_cluster_map_llm_corrected.csv"
OUTPUT_CLUSTER_SUMMARY_CORRECTED = OUTPUT_DIR / "concept_clusters_summary_llm_corrected.csv"
OUTPUT_REVIEW_XLSX = OUTPUT_DIR / "llm_cluster_review.xlsx"

# ===== 审核规则 =====
LOW_CONF_THRESHOLD = 0.60
LARGE_CLUSTER_SIZE = 8

# ===== LLM 调用参数 =====
REQUEST_TIMEOUT = 180
MAX_RETRIES = 3
RETRY_SLEEP = 5
TEMPERATURE = 0.2

# ===== 并发参数 =====
MAX_WORKERS = 4
JITTER_SLEEP_MAX = 0.8

# ===== 控制 prompt 大小 =====
MAX_MEMBERS_PER_CLUSTER_IN_PROMPT = 60
TRUNCATE_TEXT_LENGTH = 240

# ===== 是否只处理待审簇，其余簇保持原样 =====
ONLY_REVIEW_SUSPICIOUS_CLUSTERS = True


# =========================
# 1. 工具函数
# =========================
def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_text(x) -> str:
    s = safe_str(x).lower()
    s = re.sub(r"\s+", " ", s)
    return s


def truncate_text(s: str, max_len: int = 240) -> str:
    s = safe_str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len] + " ..."


def ensure_file_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"未找到文件：{path}")


def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_json_from_response(text: str) -> Optional[dict]:
    """
    尝试从 LLM 输出中提取 JSON。
    支持：
    1. 纯 JSON
    2. ```json ... ```
    3. 前后带解释文本
    """
    text = safe_str(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    fence_pattern = r"```json\s*(\{.*?\})\s*```"
    m = re.search(fence_pattern, text, flags=re.S)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    brace_pattern = r"(\{.*\})"
    m2 = re.search(brace_pattern, text, flags=re.S)
    if m2:
        candidate = m2.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


# =========================
# 2. 数据读取与筛选待审簇
# =========================
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_file_exists(INPUT_VAR_MAP)
    ensure_file_exists(INPUT_CLUSTER_SUMMARY)

    var_map = pd.read_csv(INPUT_VAR_MAP, encoding="utf-8-sig")
    cluster_summary = pd.read_csv(INPUT_CLUSTER_SUMMARY, encoding="utf-8-sig")

    return var_map, cluster_summary


def detect_clusters_for_review(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    cs = cluster_summary.copy()

    if "avg_cluster_confidence" not in cs.columns:
        cs["avg_cluster_confidence"] = np.nan
    if "cluster_size" not in cs.columns:
        raise ValueError("concept_clusters_summary.csv 缺少 cluster_size 字段。")

    cs["need_review_low_conf"] = cs["avg_cluster_confidence"].fillna(0) < LOW_CONF_THRESHOLD
    cs["need_review_large"] = cs["cluster_size"].fillna(0) >= LARGE_CLUSTER_SIZE
    cs["need_review"] = cs["need_review_low_conf"] | cs["need_review_large"]

    return cs


# =========================
# 3. Prompt 构造
# =========================
def build_cluster_member_records(cdf: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for _, r in cdf.iterrows():
        rows.append({
            "var_name": safe_str(r.get("var_name", "")),
            "var_label": truncate_text(safe_str(r.get("var_label", "")), TRUNCATE_TEXT_LENGTH),
            "year": safe_str(r.get("year", "")),
            "source_file": safe_str(r.get("source_file", "")),
            "module": safe_str(r.get("module", "")),
            "question_text": truncate_text(safe_str(r.get("question_text", "")), TRUNCATE_TEXT_LENGTH),
            "value_labels": truncate_text(safe_str(r.get("value_labels", "")), TRUNCATE_TEXT_LENGTH),
            "cluster_confidence": safe_str(r.get("cluster_confidence", ""))
        })
    return rows


def build_prompt_for_cluster(cluster_row: pd.Series, cdf: pd.DataFrame) -> Dict[str, Any]:
    cdf = cdf.copy()

    if len(cdf) > MAX_MEMBERS_PER_CLUSTER_IN_PROMPT:
        cdf = cdf.sort_values(
            by=["cluster_confidence", "year", "var_name"],
            ascending=[True, True, True]
        ).head(MAX_MEMBERS_PER_CLUSTER_IN_PROMPT).copy()

    cluster_id = safe_str(cluster_row.get("cluster_id", ""))
    concept_label = safe_str(cluster_row.get("concept_label", ""))
    anchor_var = safe_str(cluster_row.get("anchor_var", ""))
    anchor_year = safe_str(cluster_row.get("anchor_year", ""))
    avg_conf = safe_str(cluster_row.get("avg_cluster_confidence", ""))
    cluster_size = safe_str(cluster_row.get("cluster_size", ""))
    pre_group = safe_str(cluster_row.get("pre_group", ""))

    members = build_cluster_member_records(cdf)

    system_prompt = (
        "你是一名严格的问卷变量协调专家。"
        "你的任务是判断一个变量簇是否混合了多个概念，并在必要时拆分子簇。"
        "你必须只输出严格 JSON，不要输出任何额外解释。"
    )

    user_payload = {
        "task": "review_cluster_and_optionally_split",
        "instructions": [
            "判断该簇中的变量是否属于同一概念。",
            "如果属于同一概念，则 keep_cluster=true，并给出 revised_concept_label 和 recommended_anchor_var。",
            "如果不属于同一概念，则 keep_cluster=false，并拆分为若干 subclusters。",
            "subclusters 中每个成员必须来自输入 members，不能虚构变量。",
            "每个成员必须且只能出现一次。",
            "subcluster_label 应简洁、概念化。",
            "recommended_anchor_var 必须是该子簇成员之一。",
            "如果信息不足，也要尽量基于变量名、变量标签、题干和年份做保守判断。",
            "不要新建空子簇。",
            "输出必须是严格 JSON。"
        ],
        "cluster_meta": {
            "cluster_id": cluster_id,
            "concept_label": concept_label,
            "anchor_var": anchor_var,
            "anchor_year": anchor_year,
            "avg_cluster_confidence": avg_conf,
            "cluster_size": cluster_size,
            "pre_group": pre_group
        },
        "members": members,
        "required_json_schema": {
            "cluster_id": "string",
            "keep_cluster": "boolean",
            "reason": "string",
            "revised_concept_label": "string",
            "recommended_anchor_var": "string",
            "subclusters": [
                {
                    "subcluster_label": "string",
                    "recommended_anchor_var": "string",
                    "members": ["var_name_1", "var_name_2"]
                }
            ]
        }
    }

    return {
        "cluster_id": cluster_id,
        "system_prompt": system_prompt,
        "user_prompt": json.dumps(user_payload, ensure_ascii=False, indent=2)
    }


# =========================
# 4. LLM 调用
# =========================
def call_openai_compatible(system_prompt: str, user_prompt: str) -> str:
    if not API_KEY:
        raise ValueError("环境变量 LLM_API_KEY 为空，无法调用 LLM。")

    url = API_BASE.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    retryable_status_codes = {429, 500, 502, 503, 504}
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(random.uniform(0, JITTER_SLEEP_MAX))

            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content

            if resp.status_code in retryable_status_codes:
                err_msg = f"HTTP {resp.status_code}: {resp.text[:1000]}"
                last_err = RuntimeError(err_msg)
                if attempt < MAX_RETRIES:
                    sleep_s = RETRY_SLEEP * (2 ** (attempt - 1))
                    time.sleep(sleep_s)
                    continue
                raise last_err

            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")

        except requests.exceptions.Timeout as e:
            last_err = e
            if attempt < MAX_RETRIES:
                sleep_s = RETRY_SLEEP * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue
            raise last_err

        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < MAX_RETRIES:
                sleep_s = RETRY_SLEEP * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue
            raise last_err

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                sleep_s = RETRY_SLEEP * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue
            raise last_err

    raise last_err


def process_one_cluster_review(req: Dict[str, Any], var_map: pd.DataFrame) -> Dict[str, Any]:
    """
    单簇处理函数，供并发调用。
    返回统一结果结构，避免在线程中直接写文件。
    """
    cid = req["cluster_id"]
    cdf = var_map[var_map["cluster_id"].astype(str) == str(cid)].copy()
    member_var_names = cdf["var_name"].astype(str).tolist()

    try:
        raw_text = call_openai_compatible(req["system_prompt"], req["user_prompt"])
        parsed = extract_json_from_response(raw_text)

        if parsed is None:
            raise ValueError("无法从 LLM 输出中解析 JSON")

        ok, msg = validate_llm_result(parsed, member_var_names)
        if not ok:
            raise ValueError(f"LLM 输出校验失败：{msg}")

        return {
            "cluster_id": cid,
            "success": True,
            "raw_response": raw_text,
            "parsed_response": parsed,
            "error_message": ""
        }

    except Exception as e:
        return {
            "cluster_id": cid,
            "success": False,
            "raw_response": "",
            "parsed_response": None,
            "error_message": safe_str(str(e))
        }


# =========================
# 5. 响应解析与合法性校验
# =========================
def validate_llm_result(result: Dict[str, Any], member_var_names: List[str]) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "result 不是 dict"

    if "cluster_id" not in result:
        return False, "缺少 cluster_id"
    if "keep_cluster" not in result:
        return False, "缺少 keep_cluster"

    keep_cluster = result["keep_cluster"]

    if keep_cluster is True:
        if "revised_concept_label" not in result:
            return False, "keep_cluster=true 但缺少 revised_concept_label"
        if "recommended_anchor_var" not in result:
            return False, "keep_cluster=true 但缺少 recommended_anchor_var"
        if result["recommended_anchor_var"] not in member_var_names:
            return False, "recommended_anchor_var 不在原簇成员中"
        return True, "ok"

    if keep_cluster is False:
        subs = result.get("subclusters", None)
        if not isinstance(subs, list) or len(subs) == 0:
            return False, "keep_cluster=false 但 subclusters 为空"

        collected = []
        for i, sc in enumerate(subs):
            if "subcluster_label" not in sc:
                return False, f"subclusters[{i}] 缺少 subcluster_label"
            if "recommended_anchor_var" not in sc:
                return False, f"subclusters[{i}] 缺少 recommended_anchor_var"
            if "members" not in sc or not isinstance(sc["members"], list) or len(sc["members"]) == 0:
                return False, f"subclusters[{i}] members 非法"

            if sc["recommended_anchor_var"] not in sc["members"]:
                return False, f"subclusters[{i}] 的 recommended_anchor_var 不在本子簇成员中"

            for m in sc["members"]:
                if m not in member_var_names:
                    return False, f"子簇成员 {m} 不在原始成员中"
            collected.extend(sc["members"])

        if sorted(collected) != sorted(member_var_names):
            return False, "拆分后成员覆盖不完整或有重复"

        return True, "ok"

    return False, "keep_cluster 既不是 true 也不是 false"


# =========================
# 6. 将 LLM 结果回写到数据
# =========================
def apply_llm_correction_to_cluster(
    cdf: pd.DataFrame,
    cluster_row: pd.Series,
    llm_result: Dict[str, Any],
    new_cluster_counter_start: int
) -> tuple[pd.DataFrame, List[Dict[str, Any]], int]:
    """
    返回：
        corrected_df,
        summary_records,
        next_counter
    """
    cdf = cdf.copy()
    original_cluster_id = safe_str(cluster_row.get("cluster_id", ""))

    if llm_result["keep_cluster"] is True:
        revised_label = safe_str(llm_result.get("revised_concept_label", "")) or safe_str(cluster_row.get("concept_label", ""))
        revised_anchor = safe_str(llm_result.get("recommended_anchor_var", "")) or safe_str(cluster_row.get("anchor_var", ""))

        cdf["cluster_id_corrected"] = original_cluster_id
        cdf["concept_label_corrected"] = revised_label
        cdf["anchor_var_corrected"] = revised_anchor
        cdf["llm_action"] = "keep"
        cdf["llm_reason"] = safe_str(llm_result.get("reason", ""))

        anchor_year = ""
        anchor_rows = cdf[cdf["var_name"].astype(str) == revised_anchor]
        if len(anchor_rows) > 0:
            anchor_year = safe_str(anchor_rows.iloc[0].get("year", ""))

        summary_records = [{
            "cluster_id_corrected": original_cluster_id,
            "source_cluster_id": original_cluster_id,
            "concept_label_corrected": revised_label,
            "anchor_var_corrected": revised_anchor,
            "anchor_year_corrected": anchor_year,
            "cluster_size_corrected": len(cdf),
            "avg_cluster_confidence_corrected": round(float(pd.to_numeric(cdf["cluster_confidence"], errors="coerce").fillna(0).mean()), 6),
            "llm_action": "keep",
            "llm_reason": safe_str(llm_result.get("reason", "")),
            "years_covered": ", ".join(sorted(set(cdf["year"].astype(str).tolist()))),
            "member_vars": " | ".join(sorted([f"{safe_str(r['var_name'])}[{safe_str(r['year'])}]" for _, r in cdf.iterrows()]))
        }]

        return cdf, summary_records, new_cluster_counter_start

    corrected_parts = []
    summary_records = []
    next_counter = new_cluster_counter_start

    for sc in llm_result["subclusters"]:
        next_counter += 1
        sub_id = f"LC{next_counter:06d}"

        members = sc["members"]
        sub_label = safe_str(sc.get("subcluster_label", ""))
        sub_anchor = safe_str(sc.get("recommended_anchor_var", ""))

        subdf = cdf[cdf["var_name"].astype(str).isin(members)].copy()
        subdf["cluster_id_corrected"] = sub_id
        subdf["concept_label_corrected"] = sub_label
        subdf["anchor_var_corrected"] = sub_anchor
        subdf["llm_action"] = "split"
        subdf["llm_reason"] = safe_str(llm_result.get("reason", ""))

        corrected_parts.append(subdf)

        anchor_year = ""
        anchor_rows = subdf[subdf["var_name"].astype(str) == sub_anchor]
        if len(anchor_rows) > 0:
            anchor_year = safe_str(anchor_rows.iloc[0].get("year", ""))

        summary_records.append({
            "cluster_id_corrected": sub_id,
            "source_cluster_id": original_cluster_id,
            "concept_label_corrected": sub_label,
            "anchor_var_corrected": sub_anchor,
            "anchor_year_corrected": anchor_year,
            "cluster_size_corrected": len(subdf),
            "avg_cluster_confidence_corrected": round(float(pd.to_numeric(subdf["cluster_confidence"], errors="coerce").fillna(0).mean()), 6),
            "llm_action": "split",
            "llm_reason": safe_str(llm_result.get("reason", "")),
            "years_covered": ", ".join(sorted(set(subdf["year"].astype(str).tolist()))),
            "member_vars": " | ".join(sorted([f"{safe_str(r['var_name'])}[{safe_str(r['year'])}]" for _, r in subdf.iterrows()]))
        })

    corrected_df = pd.concat(corrected_parts, ignore_index=True)
    return corrected_df, summary_records, next_counter


# =========================
# 7. 主流程
# =========================
def main():
    var_map, cluster_summary = load_inputs()

    review_clusters = detect_clusters_for_review(cluster_summary)

    suspicious = review_clusters[review_clusters["need_review"]].copy()
    suspicious_cluster_ids = suspicious["cluster_id"].astype(str).tolist()

    print("=" * 80)
    print(f"总簇数：{len(cluster_summary):,}")
    print(f"待审簇数：{len(suspicious):,}")

    request_records = []
    for _, crow in suspicious.iterrows():
        cid = safe_str(crow["cluster_id"])
        cdf = var_map[var_map["cluster_id"].astype(str) == cid].copy()
        prompt_obj = build_prompt_for_cluster(crow, cdf)

        request_records.append({
            "cluster_id": cid,
            "system_prompt": prompt_obj["system_prompt"],
            "user_prompt": prompt_obj["user_prompt"]
        })

    write_jsonl(OUTPUT_REQUESTS, request_records)
    print(f"已写出请求文件：{OUTPUT_REQUESTS}")

    # 并发调用 LLM
    if OUTPUT_RESPONSES.exists():
        OUTPUT_RESPONSES.unlink()

    response_records = []
    failures = []
    print_lock = Lock()

    total_n = len(request_records)
    completed_n = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_req = {
            executor.submit(process_one_cluster_review, req, var_map): req
            for req in request_records
        }

        for future in as_completed(future_to_req):
            req = future_to_req[future]
            cid = req["cluster_id"]
            completed_n += 1

            try:
                result = future.result()

                with print_lock:
                    print("-" * 80)
                    print(f"[{completed_n}/{total_n}] 已完成簇：{cid}，success={result['success']}")

                if result["success"]:
                    record = {
                        "cluster_id": cid,
                        "success": True,
                        "raw_response": result["raw_response"],
                        "parsed_response": result["parsed_response"]
                    }
                    append_jsonl(OUTPUT_RESPONSES, record)
                    response_records.append(record)

                else:
                    err = {
                        "cluster_id": cid,
                        "success": False,
                        "error_message": result["error_message"]
                    }
                    append_jsonl(OUTPUT_RESPONSES, err)
                    failures.append(err)

                    with print_lock:
                        print(f"[失败] {cid}: {result['error_message']}")

            except Exception as e:
                err = {
                    "cluster_id": cid,
                    "success": False,
                    "error_message": safe_str(str(e))
                }
                append_jsonl(OUTPUT_RESPONSES, err)
                failures.append(err)

                with print_lock:
                    print("-" * 80)
                    print(f"[{completed_n}/{total_n}] 已完成簇：{cid}，success=False")
                    print(f"[失败] {cid}: {e}")

    corrected_parts_all = []
    corrected_summary_records = []

    existing_corrected_counter = 0

    response_ok_map = {}
    for r in response_records:
        response_ok_map[r["cluster_id"]] = r["parsed_response"]

    all_cluster_ids = cluster_summary["cluster_id"].astype(str).tolist()

    for cid in all_cluster_ids:
        cdf = var_map[var_map["cluster_id"].astype(str) == cid].copy()
        crow = cluster_summary[cluster_summary["cluster_id"].astype(str) == cid].iloc[0]

        if cid in response_ok_map:
            llm_result = response_ok_map[cid]
            corrected_df, summary_records, existing_corrected_counter = apply_llm_correction_to_cluster(
                cdf=cdf,
                cluster_row=crow,
                llm_result=llm_result,
                new_cluster_counter_start=existing_corrected_counter
            )
            corrected_parts_all.append(corrected_df)
            corrected_summary_records.extend(summary_records)

        else:
            cdf = cdf.copy()
            cdf["cluster_id_corrected"] = cid
            cdf["concept_label_corrected"] = safe_str(crow.get("concept_label", ""))
            cdf["anchor_var_corrected"] = safe_str(crow.get("anchor_var", ""))
            cdf["llm_action"] = "unchanged"
            cdf["llm_reason"] = "not_reviewed_or_failed"

            corrected_parts_all.append(cdf)

            corrected_summary_records.append({
                "cluster_id_corrected": cid,
                "source_cluster_id": cid,
                "concept_label_corrected": safe_str(crow.get("concept_label", "")),
                "anchor_var_corrected": safe_str(crow.get("anchor_var", "")),
                "anchor_year_corrected": safe_str(crow.get("anchor_year", "")),
                "cluster_size_corrected": len(cdf),
                "avg_cluster_confidence_corrected": safe_str(crow.get("avg_cluster_confidence", "")),
                "llm_action": "unchanged",
                "llm_reason": "not_reviewed_or_failed",
                "years_covered": ", ".join(sorted(set(cdf["year"].astype(str).tolist()))),
                "member_vars": " | ".join(sorted([f"{safe_str(r['var_name'])}[{safe_str(r['year'])}]" for _, r in cdf.iterrows()]))
            })

    var_map_corrected = pd.concat(corrected_parts_all, ignore_index=True)
    cluster_summary_corrected = pd.DataFrame(corrected_summary_records)

    var_map_corrected = var_map_corrected.sort_values(
        by=["cluster_id_corrected", "year", "var_name"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    cluster_summary_corrected = cluster_summary_corrected.sort_values(
        by=["cluster_id_corrected", "cluster_size_corrected"],
        ascending=[True, False]
    ).reset_index(drop=True)

    var_cols = [
        "cluster_id", "cluster_id_corrected",
        "concept_label", "concept_label_corrected",
        "anchor_var", "anchor_var_corrected",
        "cluster_size", "cluster_confidence",
        "llm_action", "llm_reason",
        "pre_group", "var_name", "var_label", "year",
        "source_file", "module", "question_text", "value_labels", "var_type"
    ]
    var_cols = [c for c in var_cols if c in var_map_corrected.columns]
    var_map_corrected[var_cols].to_csv(OUTPUT_VAR_MAP_CORRECTED, index=False, encoding="utf-8-sig")

    summary_cols = [
        "cluster_id_corrected", "source_cluster_id",
        "concept_label_corrected",
        "anchor_var_corrected", "anchor_year_corrected",
        "cluster_size_corrected", "avg_cluster_confidence_corrected",
        "llm_action", "llm_reason", "years_covered", "member_vars"
    ]
    summary_cols = [c for c in summary_cols if c in cluster_summary_corrected.columns]
    cluster_summary_corrected[summary_cols].to_csv(OUTPUT_CLUSTER_SUMMARY_CORRECTED, index=False, encoding="utf-8-sig")

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(OUTPUT_FAILURES, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_REVIEW_XLSX, engine="openpyxl") as writer:
        suspicious.to_excel(writer, sheet_name="suspicious_clusters", index=False)
        cluster_summary_corrected[summary_cols].to_excel(writer, sheet_name="corrected_cluster_summary", index=False)
        var_map_corrected[var_cols].to_excel(writer, sheet_name="corrected_variable_map", index=False)
        failures_df.to_excel(writer, sheet_name="failures", index=False)

    print("=" * 80)
    print("[完成] 输出文件：")
    print(f"1. {OUTPUT_REQUESTS}")
    print(f"2. {OUTPUT_RESPONSES}")
    print(f"3. {OUTPUT_FAILURES}")
    print(f"4. {OUTPUT_VAR_MAP_CORRECTED}")
    print(f"5. {OUTPUT_CLUSTER_SUMMARY_CORRECTED}")
    print(f"6. {OUTPUT_REVIEW_XLSX}")
    print("=" * 80)
    print(f"待审簇数：{len(suspicious):,}")
    print(f"成功审查簇数：{len(response_records):,}")
    print(f"失败簇数：{len(failures):,}")


if __name__ == "__main__":
    main()