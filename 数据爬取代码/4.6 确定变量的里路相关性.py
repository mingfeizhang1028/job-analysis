# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:02:25 2026

@author: Rosem
"""

from __future__ import annotations

# -*- coding: utf-8 -*-
"""
CFPS strict question group 理论相关性筛选系统
主题：数字参与、收入水平与收入不平等
版本：Python 3.10+

功能：
1．读取 strict_question_group_summary.csv
2．基于理论词典做规则粗筛
3．并行调用 LLM 做理论相关性判别
4．输出推荐保留、人工复核与排除建议

依赖：
pip install pandas numpy requests openpyxl xlsxwriter
"""

import os
import re
import json
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests


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
    r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\topic_relevance_output"
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

# ===== 规则筛选阈值 =====
RULE_SEND_TO_LLM_MIN_SCORE = 5
RULE_PRIORITY_KEEP_MIN_SCORE = 18
RULE_KEEP_MIN_SCORE = 12

# ===== 输出文件 =====
OUT_RULE_SCREEN = OUTPUT_DIR / "topic_rule_screen.csv"
OUT_LLM_JUDGEMENT = OUTPUT_DIR / "topic_llm_judgement.csv"
OUT_RECOMMEND = OUTPUT_DIR / "topic_relevance_recommendation.csv"
OUT_PRIORITY_KEEP = OUTPUT_DIR / "topic_priority_keep.csv"
OUT_MANUAL_REVIEW = OUTPUT_DIR / "topic_manual_review_sheet.csv"
OUT_CACHE = OUTPUT_DIR / "topic_llm_cache.jsonl"
OUT_WORKBOOK = OUTPUT_DIR / "topic_relevance_workbook.xlsx"


# =========================================================
# 1．基础工具
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


def short_text(x: Any, max_len: int = 600) -> str:
    s = safe_str(x)
    return s if len(s) <= max_len else s[:max_len] + "..."


def normalize_text(x: Any) -> str:
    s = safe_str(x).lower()
    s = s.replace("_", " ")
    s = re.sub(r"[，。；：、“”‘’（）()【】\[\]\-_/,:;!?]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def export_excel_with_sheets(path: Path, sheets: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            if df is None:
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def load_jsonl_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
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
                key = safe_str(obj.get("strict_group_id"))
                if key:
                    cache[key] = obj
            except Exception:
                continue
    return cache


def append_jsonl_cache(cache_path: Path, record: Dict[str, Any]):
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        "anchor_var", "anchor_label", "member_count",
        "member_var_names", "member_labels", "value_label_summary",
        "broad_text", "structure_type_mode"
    ]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = ""

    df["year"] = df["year"].apply(safe_int)
    df["member_count"] = pd.to_numeric(df["member_count"], errors="coerce").fillna(0).astype(int)

    # 综合文本
    df["topic_text"] = (
        df["group_label"].fillna("").astype(str) + "；" +
        df["anchor_label"].fillna("").astype(str) + "；" +
        df["member_labels"].fillna("").astype(str) + "；" +
        df["value_label_summary"].fillna("").astype(str) + "；" +
        df["broad_text"].fillna("").astype(str)
    )

    df["topic_text_norm"] = df["topic_text"].apply(normalize_text)

    return df


# =========================================================
# 3．理论词典
# =========================================================

TOPIC_DICTIONARY = {
    "digital_access": {
        "keywords": [
            "上网", "互联网", "网络", "宽带", "wifi", "无线网络", "电脑", "计算机",
            "手机上网", "智能手机", "平板", "移动设备", "电子设备", "网络接入",
            "是否使用互联网", "是否上网", "是否有电脑", "是否有手机", "是否有智能手机",
            "internet", "online", "web", "computer", "smartphone", "mobile phone"
        ],
        "weight": 10
    },
    "digital_use": {
        "keywords": [
            "网购", "网上购物", "电商", "电子商务", "线上购物", "移动支付", "数字支付",
            "网络支付", "支付宝", "微信支付", "数字金融", "互联网金融", "上网频率",
            "上网时长", "社交媒体", "微信", "微博", "短视频", "网络娱乐", "网络游戏",
            "在线学习", "网络学习", "在线教育", "数字技能", "信息获取", "网上银行",
            "网上办公", "远程办公", "线上办公", "平台工作", "接单平台", "数字平台",
            "online shopping", "e commerce", "social media", "digital finance", "internet finance"
        ],
        "weight": 10
    },
    "income_level": {
        "keywords": [
            "收入", "工资", "工资性收入", "劳动收入", "个人收入", "总收入", "年收入",
            "月收入", "家庭收入", "纯收入", "经营收入", "财产收入", "转移收入",
            "工资水平", "薪酬", "报酬", "earnings", "income", "wage", "salary"
        ],
        "weight": 10
    },
    "income_inequality": {
        "keywords": [
            "不平等", "收入差距", "收入分配", "相对剥夺", "贫困", "低收入", "高收入",
            "收入位置", "收入排名", "分位数", "机会不平等", "社会流动", "收入流动",
            "收入阶层", "阶层地位", "财富差距", "剥夺", "不公平", "inequality", "deprivation"
        ],
        "weight": 9
    },
    "control": {
        "keywords": [
            "年龄", "性别", "教育", "受教育程度", "学历", "婚姻", "健康", "户口",
            "城乡", "地区", "省份", "民族", "宗教", "工作状态", "就业状态", "失业",
            "行业", "职业", "工作单位", "家庭规模", "子女数", "家庭人口", "居住地",
            "党员", "身体状况", "hukou", "education", "age", "gender", "marriage", "health"
        ],
        "weight": 5
    },
    "mechanism": {
        "keywords": [
            "技能", "能力", "培训", "职业培训", "就业质量", "求职", "找工作", "创业",
            "融资", "信贷", "借贷", "金融可得性", "社会资本", "人力资本", "信息渠道",
            "信息获取", "职业转换", "非农就业", "劳动供给", "工作时间", "副业",
            "技能提升", "培训经历", "创业行为", "创业活动", "技能证书", "资本约束"
        ],
        "weight": 7
    },
    "heterogeneity": {
        "keywords": [
            "教育分层", "学历层次", "城乡差异", "地区差异", "性别差异", "年龄组",
            "代际", "代沟", "家庭背景", "父母教育", "父母职业", "先赋条件",
            "社会出身", "阶层来源", "出生背景", "初始条件", "区域异质性"
        ],
        "weight": 6
    }
}

NEGATIVE_KEYWORDS = [
    "身高", "体重", "血压", "脉搏", "体温", "饮食口味", "烹饪", "宗教仪式",
    "宠物", "星座", "头发颜色", "鞋码", "牙齿", "视力", "听力测试"
]


# =========================================================
# 4．规则理论筛选
# =========================================================

def count_keyword_hits(text: str, keywords: List[str]) -> Tuple[int, List[str]]:
    hits = []
    for kw in keywords:
        if kw.lower() in text:
            hits.append(kw)
    return len(hits), hits


def rule_topic_screen(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        text = safe_str(r["topic_text_norm"])

        role_scores = {}
        role_hits = {}

        for role, spec in TOPIC_DICTIONARY.items():
            n_hits, hits = count_keyword_hits(text, [k.lower() for k in spec["keywords"]])
            role_scores[role] = n_hits * spec["weight"]
            role_hits[role] = hits

        negative_hits_n, negative_hits = count_keyword_hits(text, [k.lower() for k in NEGATIVE_KEYWORDS])
        negative_penalty = negative_hits_n * 8

        # 规则总分
        topic_rule_score = sum(role_scores.values()) - negative_penalty

        # 主角色猜测
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        top_role = sorted_roles[0][0] if sorted_roles else "other"
        second_role = sorted_roles[1][0] if len(sorted_roles) > 1 else "other"

        matched_keywords_all = []
        for role in TOPIC_DICTIONARY.keys():
            matched_keywords_all.extend([f"{role}:{x}" for x in role_hits[role]])

        if topic_rule_score >= RULE_PRIORITY_KEEP_MIN_SCORE:
            rule_keep_guess = "priority_keep"
        elif topic_rule_score >= RULE_KEEP_MIN_SCORE:
            rule_keep_guess = "keep"
        elif topic_rule_score >= RULE_SEND_TO_LLM_MIN_SCORE:
            rule_keep_guess = "review"
        else:
            rule_keep_guess = "drop"

        rows.append({
            "strict_group_id": safe_str(r["strict_group_id"]),
            "year": safe_int(r["year"]),
            "dataset_class": safe_str(r["dataset_class"]),
            "survey_role": safe_str(r["survey_role"]),
            "strict_group_type": safe_str(r["strict_group_type"]),
            "group_label": safe_str(r["group_label"]),
            "anchor_var": safe_str(r["anchor_var"]),
            "anchor_label": safe_str(r["anchor_label"]),
            "member_count": safe_int(r["member_count"]),
            "structure_type_mode": safe_str(r["structure_type_mode"]),
            "topic_text": short_text(r["topic_text"], 2000),

            "score_digital_access": role_scores["digital_access"],
            "score_digital_use": role_scores["digital_use"],
            "score_income_level": role_scores["income_level"],
            "score_income_inequality": role_scores["income_inequality"],
            "score_control": role_scores["control"],
            "score_mechanism": role_scores["mechanism"],
            "score_heterogeneity": role_scores["heterogeneity"],

            "negative_penalty": negative_penalty,
            "negative_hits": " | ".join(negative_hits),
            "topic_rule_score": topic_rule_score,
            "rule_topic_role_guess": top_role,
            "rule_topic_role_second": second_role,
            "rule_keep_guess": rule_keep_guess,
            "matched_keywords": " | ".join(matched_keywords_all[:100]),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["topic_rule_score", "year", "strict_group_id"],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    return out


# =========================================================
# 5．LLM Prompt 与接口
# =========================================================

def build_llm_prompt(row: pd.Series) -> str:
    prompt = f"""
你是一名经济学论文变量筛选助手。论文主题是：

“数字参与、收入水平与收入不平等”。

研究主线是：
1．数字参与是否提高收入水平；
2．数字参与是否影响收入不平等；
3．同时关注控制变量、机制变量、异质性变量。

请你根据下面这个 CFPS strict question group 的信息，判断它与该论文主题的理论相关程度。

请严格按以下标准输出：
- core：与主题高度直接相关，通常可作为核心解释变量或核心结果变量。
- important：与主题明确相关，可作为关键控制变量、重要机制变量或重要异质性变量。
- potential：有一定相关性，但用途不稳定或需人工判断。
- irrelevant：与论文主题关系很弱，通常不建议保留。

同时，请判断它在论文中的功能角色 topic_role，只能从下面选择一个：
- digital_access
- digital_use
- income_level
- income_inequality
- control
- mechanism
- heterogeneity
- other

最后，请给出保留建议 keep_for_study，只能从下面选择一个：
- yes
- review
- no

请只输出 JSON，不要输出任何额外解释。格式如下：
{{
  "topic_relevance_level": "core/important/potential/irrelevant",
  "topic_role": "digital_access/digital_use/income_level/income_inequality/control/mechanism/heterogeneity/other",
  "keep_for_study": "yes/review/no",
  "reason": "不超过120字",
  "risk_flags": ["flag1", "flag2"]
}}

下面是变量题组信息：

strict_group_id: {safe_str(row["strict_group_id"])}
year: {safe_str(row["year"])}
dataset_class: {safe_str(row["dataset_class"])}
survey_role: {safe_str(row["survey_role"])}
strict_group_type: {safe_str(row["strict_group_type"])}
group_label: {safe_str(row["group_label"])}
anchor_var: {safe_str(row["anchor_var"])}
anchor_label: {safe_str(row["anchor_label"])}
member_count: {safe_str(row["member_count"])}
structure_type_mode: {safe_str(row["structure_type_mode"])}

topic_text: {short_text(row["topic_text"], 1800)}

规则粗筛信息：
topic_rule_score: {safe_str(row["topic_rule_score"])}
rule_topic_role_guess: {safe_str(row["rule_topic_role_guess"])}
rule_topic_role_second: {safe_str(row["rule_topic_role_second"])}
rule_keep_guess: {safe_str(row["rule_keep_guess"])}
matched_keywords: {short_text(row["matched_keywords"], 1000)}
negative_hits: {safe_str(row["negative_hits"])}
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
        return json.loads(m.group(0))

    raise ValueError("无法解析 JSON")


def call_openai_compatible_api(prompt: str) -> Dict[str, Any]:
    url = API_BASE.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严格遵守格式的经济学论文变量筛选助手。你只能输出 JSON。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
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


def llm_topic_worker(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    strict_group_id = safe_str(row_dict["strict_group_id"])
    row = pd.Series(row_dict)

    prompt = build_llm_prompt(row)
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = call_openai_compatible_api(prompt)
            parsed = result["parsed"]

            topic_relevance_level = safe_str(parsed.get("topic_relevance_level")).lower()
            if topic_relevance_level not in {"core", "important", "potential", "irrelevant"}:
                topic_relevance_level = "irrelevant"

            topic_role = safe_str(parsed.get("topic_role")).lower()
            if topic_role not in {
                "digital_access", "digital_use", "income_level", "income_inequality",
                "control", "mechanism", "heterogeneity", "other"
            }:
                topic_role = "other"

            keep_for_study = safe_str(parsed.get("keep_for_study")).lower()
            if keep_for_study not in {"yes", "review", "no"}:
                keep_for_study = "review"

            risk_flags = parsed.get("risk_flags", [])
            if not isinstance(risk_flags, list):
                risk_flags = [safe_str(risk_flags)] if safe_str(risk_flags) else []

            return {
                "strict_group_id": strict_group_id,
                "llm_topic_relevance_level": topic_relevance_level,
                "llm_topic_role": topic_role,
                "llm_keep_for_study": keep_for_study,
                "llm_reason": short_text(parsed.get("reason", ""), 200),
                "llm_risk_flags": " | ".join([safe_str(x) for x in risk_flags if safe_str(x)]),
                "llm_status": "success",
                "llm_attempts": attempt,
                "llm_error": "",
                "llm_raw_response": short_text(result.get("raw_response", ""), 1500),
                "model_provider": MODEL_PROVIDER,
                "model_name": MODEL_NAME,
            }

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_BASE ** attempt)
            continue

    return {
        "strict_group_id": strict_group_id,
        "llm_topic_relevance_level": "irrelevant",
        "llm_topic_role": "other",
        "llm_keep_for_study": "no",
        "llm_reason": "",
        "llm_risk_flags": "",
        "llm_status": "failed",
        "llm_attempts": MAX_RETRIES,
        "llm_error": short_text(last_error, 500),
        "llm_raw_response": "",
        "model_provider": MODEL_PROVIDER,
        "model_name": MODEL_NAME,
    }


# =========================================================
# 6．并行运行 LLM
# =========================================================

def run_llm_topic_screen(rule_df: pd.DataFrame) -> pd.DataFrame:
    if rule_df.empty:
        return pd.DataFrame()

    # 只把有一定规则相关性的送给 LLM
    send_df = rule_df[rule_df["topic_rule_score"] >= RULE_SEND_TO_LLM_MIN_SCORE].copy()

    cache = load_jsonl_cache(OUT_CACHE) if USE_LLM_CACHE else {}
    results = []
    todo_rows = []

    for _, row in send_df.iterrows():
        gid = safe_str(row["strict_group_id"])
        if USE_LLM_CACHE and gid in cache:
            results.append(cache[gid])
        else:
            todo_rows.append(row.to_dict())

    print(f"理论筛选：规则入围数 = {len(send_df)}")
    print(f"缓存命中数 = {len(results)}")
    print(f"待调用数 = {len(todo_rows)}")
    print(f"MAX_WORKERS = {MAX_WORKERS}")

    if todo_rows:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_gid = {
                executor.submit(llm_topic_worker, row_dict): row_dict["strict_group_id"]
                for row_dict in todo_rows
            }

            for idx, future in enumerate(as_completed(future_to_gid), start=1):
                res = future.result()
                results.append(res)

                if USE_LLM_CACHE:
                    append_jsonl_cache(OUT_CACHE, res)

                if idx % 20 == 0 or idx == len(todo_rows):
                    print(f"LLM 理论判别已完成：{idx}/{len(todo_rows)}")

    llm_df = pd.DataFrame(results)
    if not llm_df.empty:
        llm_df = llm_df.drop_duplicates(subset=["strict_group_id"], keep="last").reset_index(drop=True)

    return llm_df


# =========================================================
# 7．合并与最终推荐
# =========================================================

def build_final_topic_recommendation(rule_df: pd.DataFrame, llm_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = rule_df.merge(
        llm_df[
            [
                "strict_group_id", "llm_topic_relevance_level", "llm_topic_role",
                "llm_keep_for_study", "llm_reason", "llm_risk_flags",
                "llm_status", "llm_attempts", "llm_error"
            ]
        ] if not llm_df.empty else pd.DataFrame(columns=["strict_group_id"]),
        on="strict_group_id",
        how="left"
    )

    merged["llm_topic_relevance_level"] = merged["llm_topic_relevance_level"].fillna("irrelevant")
    merged["llm_topic_role"] = merged["llm_topic_role"].fillna(merged["rule_topic_role_guess"])
    merged["llm_keep_for_study"] = merged["llm_keep_for_study"].fillna("no")

    relevance_score_map = {
        "core": 100,
        "important": 80,
        "potential": 55,
        "irrelevant": 10
    }
    merged["llm_relevance_score"] = merged["llm_topic_relevance_level"].map(relevance_score_map).fillna(10)

    # 规则分缩放
    merged["topic_rule_score_scaled"] = merged["topic_rule_score"].clip(lower=0, upper=100)

    merged["topic_final_score"] = (
        0.45 * merged["topic_rule_score_scaled"] +
        0.55 * merged["llm_relevance_score"]
    ).round(2)

    def final_decision(row):
        llm_level = safe_str(row["llm_topic_relevance_level"])
        keep = safe_str(row["llm_keep_for_study"])
        score = float(row["topic_final_score"])

        if llm_level == "core" and keep == "yes" and score >= 60:
            return "priority_keep"
        if llm_level in {"core", "important"} and keep in {"yes", "review"} and score >= 45:
            return "keep"
        if llm_level == "potential" or keep == "review":
            return "review"
        return "drop"

    merged["topic_final_decision"] = merged.apply(final_decision, axis=1)

    def manual_review_needed(row):
        if safe_str(row["topic_final_decision"]) == "review":
            return 1
        if safe_str(row["topic_final_decision"]) == "keep" and safe_str(row["llm_topic_relevance_level"]) == "important":
            return 1
        return 0

    merged["manual_review_needed"] = merged.apply(manual_review_needed, axis=1)

    merged = merged.sort_values(
        by=["topic_final_decision", "topic_final_score", "topic_rule_score"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    priority_keep = merged[merged["topic_final_decision"].isin(["priority_keep", "keep"])].copy()

    manual_review = merged[merged["manual_review_needed"] == 1].copy()
    manual_review["human_final_decision"] = ""
    manual_review["human_role_revision"] = ""
    manual_review["human_notes"] = ""

    return merged, priority_keep, manual_review


# =========================================================
# 8．主程序
# =========================================================

def main():
    print("=" * 90)
    print("CFPS 变量理论相关性筛选系统启动")
    print("=" * 90)

    print("第 1 步：读取 strict question group summary")
    strict_df = load_strict_group_summary(INPUT_STRICT_GROUP_SUMMARY)
    print(f"strict group 数：{len(strict_df)}")

    print("-" * 90)
    print("第 2 步：规则理论粗筛")
    rule_df = rule_topic_screen(strict_df)
    rule_df.to_csv(OUT_RULE_SCREEN, index=False, encoding="utf-8-sig")
    print(f"规则筛选完成，记录数：{len(rule_df)}")
    print(f"已导出：{OUT_RULE_SCREEN}")

    print("-" * 90)
    print("第 3 步：并行运行 LLM 理论判别")
    llm_df = run_llm_topic_screen(rule_df)
    llm_df.to_csv(OUT_LLM_JUDGEMENT, index=False, encoding="utf-8-sig")
    print(f"LLM 理论判别完成，记录数：{len(llm_df)}")
    print(f"已导出：{OUT_LLM_JUDGEMENT}")

    print("-" * 90)
    print("第 4 步：合并并生成最终建议")
    recommend_df, priority_keep_df, manual_review_df = build_final_topic_recommendation(rule_df, llm_df)

    recommend_df.to_csv(OUT_RECOMMEND, index=False, encoding="utf-8-sig")
    priority_keep_df.to_csv(OUT_PRIORITY_KEEP, index=False, encoding="utf-8-sig")
    manual_review_df.to_csv(OUT_MANUAL_REVIEW, index=False, encoding="utf-8-sig")

    print(f"最终建议记录数：{len(recommend_df)}")
    print(f"优先保留数：{len(priority_keep_df)}")
    print(f"人工复核数：{len(manual_review_df)}")
    print(f"已导出：{OUT_RECOMMEND}")
    print(f"已导出：{OUT_PRIORITY_KEEP}")
    print(f"已导出：{OUT_MANUAL_REVIEW}")

    print("-" * 90)
    print("第 5 步：导出 Excel 工作簿")
    export_excel_with_sheets(OUT_WORKBOOK, {
        "topic_rule_screen": rule_df,
        "topic_llm_judgement": llm_df,
        "topic_recommendation": recommend_df,
        "priority_keep": priority_keep_df,
        "manual_review": manual_review_df,
    })
    print(f"已导出：{OUT_WORKBOOK}")

    print("=" * 90)
    print("全部完成")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()