from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import pandas as pd

from modules.tag_extraction import apply_tag_extraction
from modules.tag_source_resolver import resolve_tag_column


DETAIL_COL_CANDIDATES = ["岗位详情", "职位详情", "职位描述", "JD", "岗位描述"]
TITLE_COL_CANDIDATES = ["职位名称_norm", "职位名称_raw", "职位名称"]
COMPANY_COL_CANDIDATES = ["企业名称_norm", "企业名称_raw", "企业名称"]
ROLE_COL_CANDIDATES = ["LLM岗位类型", "职位方向", "职位类别", "职位名称_norm"]
EXPERIENCE_COL_CANDIDATES = ["经验要求", "LLM经验要求", "LLM资历级别"]
DEGREE_COL_CANDIDATES = ["学历要求", "LLM学历要求"]

REQUIREMENT_SIGNALS = [
    "任职要求", "岗位要求", "职位要求", "必须", "必备", "熟练", "精通", "掌握", "需要具备",
    "要求具备", "能够", "具备", "本科及以上", "经验要求",
]
BONUS_SIGNALS = ["优先", "加分", "更佳", "有经验者优先", "熟悉者优先", "加分项", "nice to have"]
RESPONSIBILITY_SIGNALS = [
    "岗位职责", "工作职责", "工作内容", "负责", "参与", "协助", "推进", "完成", "输出", "落地",
]
FRESH_POSITIVE_SIGNALS = [
    "应届", "校招", "校园招聘", "2026届", "2025届", "经验不限", "不限经验", "无经验",
    "接受无经验", "实习转正", "培养", "导师", "带教", "管培", "助理",
]
FRESH_NEGATIVE_SIGNALS = [
    "3年以上", "三年以上", "5年以上", "五年以上", "独立负责", "专家", "高级", "资深",
    "负责人", "带团队", "管理经验", "成熟项目经验", "行业经验",
]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text.strip("[]")
        return [x.strip(" '\"\t\r\n") for x in re.split(r"[，,；;、\n]+", text) if x.strip(" '\"\t\r\n")]
    return []


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _has_non_empty_tags(df: pd.DataFrame, col: str | None) -> bool:
    if not col or col not in df.columns:
        return False
    return bool(df[col].apply(lambda x: len(_safe_list(x)) > 0).any())


def resolve_available_tag_column(df: pd.DataFrame, tag_type: str, source_mode: str) -> str | None:
    """Prefer the requested tag source, then fall back to dictionary tags if it is empty."""
    target_col = resolve_tag_column(tag_type, source_mode)
    if _has_non_empty_tags(df, target_col):
        return target_col

    if source_mode == "最终标签":
        fallback_col = resolve_tag_column(tag_type, "词典标签")
        if _has_non_empty_tags(df, fallback_col):
            return fallback_col

    return target_col if target_col in df.columns else None


def ensure_tag_source(df: pd.DataFrame, tag_type: str, source_mode: str) -> tuple[pd.DataFrame, str | None]:
    """Return a dataframe and usable tag column, deriving rule tags on the fly if needed."""
    tag_col = resolve_available_tag_column(df, tag_type, source_mode)
    if _has_non_empty_tags(df, tag_col):
        return df, tag_col

    try:
        enriched = apply_tag_extraction(df)
    except Exception:
        return df, tag_col

    tag_col = resolve_available_tag_column(enriched, tag_type, source_mode)
    return enriched, tag_col


def _job_id(row: pd.Series, idx: Any) -> str:
    return _safe_str(row.get("job_id")) or str(idx)


def _contains(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    if re.fullmatch(r"[A-Za-z0-9_+#.\-]+", keyword):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(keyword)}(?![A-Za-z0-9_])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return keyword.lower() in text.lower()


def _context(text: str, keyword: str, window: int = 90) -> str:
    source = _safe_str(text)
    kw = _safe_str(keyword)
    if not source or not kw:
        return ""
    flags = re.IGNORECASE
    if re.fullmatch(r"[A-Za-z0-9_+#.\-]+", kw):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(kw)}(?![A-Za-z0-9_])"
    else:
        pattern = re.escape(kw)
    match = re.search(pattern, source, flags=flags)
    if not match:
        return ""
    start = max(0, match.start() - window)
    end = min(len(source), match.end() + window)
    return source[start:end].replace("\n", " ").strip()


def _has_any(text: str, signals: list[str]) -> bool:
    lowered = _safe_str(text).lower()
    return any(signal.lower() in lowered for signal in signals)


def _tag_signal_type(text: str, tag: str) -> tuple[bool, bool, bool]:
    snippet = _context(text, tag, window=120)
    if not snippet:
        return False, False, False
    return (
        _has_any(snippet, REQUIREMENT_SIGNALS),
        _has_any(snippet, BONUS_SIGNALS),
        _has_any(snippet, RESPONSIBILITY_SIGNALS),
    )


def score_fresh_graduate_friendliness(row: pd.Series, detail_col: str | None = None) -> tuple[int, str]:
    parts = []
    for col in EXPERIENCE_COL_CANDIDATES + DEGREE_COL_CANDIDATES + ["LLM应届友好度", "LLM资历级别"]:
        if col in row.index:
            parts.append(_safe_str(row.get(col)))
    if detail_col and detail_col in row.index:
        parts.append(_safe_str(row.get(detail_col)))
    text = " ".join([p for p in parts if p])

    score = 50
    pos_hits = [s for s in FRESH_POSITIVE_SIGNALS if s.lower() in text.lower()]
    neg_hits = [s for s in FRESH_NEGATIVE_SIGNALS if s.lower() in text.lower()]
    score += min(len(pos_hits) * 12, 42)
    score -= min(len(neg_hits) * 14, 50)

    if re.search(r"[1一]-?[2二]年", text):
        score -= 8
    if re.search(r"[3三]-?[5五]年|[3三]年以上|[5五]年以上", text):
        score -= 28
    if "应届" in text or "经验不限" in text:
        score += 18

    score = max(0, min(100, score))
    if score >= 75:
        level = "高"
    elif score >= 55:
        level = "中"
    else:
        level = "低"
    return int(score), level


def _level_by_score(score: float) -> str:
    if score >= 72:
        return "高"
    if score >= 45:
        return "中"
    return "低"


def _priority_level(row: dict[str, Any]) -> str:
    if row["必备字段命中率"] >= 0.45 and row["覆盖率"] >= 0.20:
        return "核心门槛"
    if row["重要性得分"] >= 55:
        return "重点强化"
    if row["加分字段命中率"] >= 0.35 and row["必备字段命中率"] < 0.35:
        return "加分项"
    return "观察项"


def build_tag_evidence_table(
    df: pd.DataFrame,
    tag_type: str = "全部",
    source_mode: str = "最终标签",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df, tag_col = ensure_tag_source(df, tag_type, source_mode)
    if not tag_col or tag_col not in df.columns:
        return pd.DataFrame()

    detail_col = _pick_col(df, DETAIL_COL_CANDIDATES)
    title_col = _pick_col(df, TITLE_COL_CANDIDATES)
    company_col = _pick_col(df, COMPANY_COL_CANDIDATES)
    role_col = _pick_col(df, ROLE_COL_CANDIDATES)

    total_jobs = max(df["job_id"].nunique() if "job_id" in df.columns else len(df), 1)
    total_companies = max(df[company_col].nunique() if company_col else 1, 1)
    total_roles = max(df[role_col].astype(str).nunique() if role_col else 1, 1)

    tag_jobs: dict[str, set[str]] = defaultdict(set)
    tag_companies: dict[str, set[str]] = defaultdict(set)
    tag_roles: dict[str, set[str]] = defaultdict(set)
    tag_freq = Counter()
    req_hits = Counter()
    bonus_hits = Counter()
    resp_hits = Counter()
    title_hits = Counter()
    fresh_scores: dict[str, list[int]] = defaultdict(list)
    evidence_count = Counter()

    for idx, row in df.iterrows():
        job_id = _job_id(row, idx)
        tags = list(dict.fromkeys(_safe_list(row.get(tag_col))))
        if not tags:
            continue

        detail = _safe_str(row.get(detail_col)) if detail_col else ""
        title = _safe_str(row.get(title_col)) if title_col else ""
        company = _safe_str(row.get(company_col)) if company_col else ""
        role = _safe_str(row.get(role_col)) if role_col else ""
        fresh_score, _ = score_fresh_graduate_friendliness(row, detail_col)

        for tag in tags:
            tag_freq[tag] += 1
            tag_jobs[tag].add(job_id)
            if company:
                tag_companies[tag].add(company)
            if role:
                tag_roles[tag].add(role)
            if title and _contains(title, tag):
                title_hits[tag] += 1
            if detail and _contains(detail, tag):
                evidence_count[tag] += 1
            is_req, is_bonus, is_resp = _tag_signal_type(detail, tag)
            if is_req:
                req_hits[tag] += 1
            if is_bonus:
                bonus_hits[tag] += 1
            if is_resp:
                resp_hits[tag] += 1
            fresh_scores[tag].append(fresh_score)

    rows = []
    for tag, jobs in tag_jobs.items():
        coverage_jobs = len(jobs)
        coverage_rate = coverage_jobs / total_jobs
        company_coverage = len(tag_companies[tag]) / total_companies
        role_coverage = len(tag_roles[tag]) / total_roles
        req_rate = req_hits[tag] / max(coverage_jobs, 1)
        bonus_rate = bonus_hits[tag] / max(coverage_jobs, 1)
        resp_rate = resp_hits[tag] / max(coverage_jobs, 1)
        title_rate = title_hits[tag] / max(coverage_jobs, 1)
        evidence_rate = evidence_count[tag] / max(coverage_jobs, 1)
        fresh_avg = sum(fresh_scores[tag]) / max(len(fresh_scores[tag]), 1)

        importance = (
            coverage_rate * 32
            + company_coverage * 18
            + role_coverage * 12
            + req_rate * 20
            + resp_rate * 10
            + title_rate * 5
            + evidence_rate * 3
        )
        row = {
            "标签": tag,
            "词频": int(tag_freq[tag]),
            "覆盖岗位数": int(coverage_jobs),
            "覆盖率": coverage_rate,
            "覆盖公司数": int(len(tag_companies[tag])),
            "公司覆盖率": company_coverage,
            "岗位方向覆盖数": int(len(tag_roles[tag])),
            "岗位方向覆盖率": role_coverage,
            "必备字段命中率": req_rate,
            "职责字段命中率": resp_rate,
            "加分字段命中率": bonus_rate,
            "标题命中率": title_rate,
            "证据片段覆盖率": evidence_rate,
            "平均应届友好分": round(fresh_avg, 1),
            "重要性得分": round(importance, 1),
            "置信等级": _level_by_score(importance),
        }
        row["求职优先级"] = _priority_level(row)
        row["求职建议"] = make_tag_action_advice(row)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["重要性得分", "覆盖岗位数", "必备字段命中率"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def make_tag_action_advice(row: dict[str, Any] | pd.Series) -> str:
    priority = row.get("求职优先级", "")
    tag = row.get("标签", "")
    if priority == "核心门槛":
        return f"简历中必须体现「{tag}」的真实使用场景，优先写进项目/实习经历，而不只放在技能栏。"
    if priority == "重点强化":
        return f"建议围绕「{tag}」补充方法、工具、业务结果或协作对象，提高和 JD 的对应关系。"
    if priority == "加分项":
        return f"「{tag}」更像差异化加分项，适合放在项目亮点、补充技能或面试表达中。"
    return f"「{tag}」当前证据有限，可作为观察项，先查看命中 JD 后再决定是否投入准备。"


def build_tag_evidence_examples(
    df: pd.DataFrame,
    tag: str,
    tag_type: str = "全部",
    source_mode: str = "最终标签",
    limit: int = 8,
) -> pd.DataFrame:
    df, tag_col = ensure_tag_source(df, tag_type, source_mode)
    if df is None or df.empty or not tag_col or tag_col not in df.columns or not tag:
        return pd.DataFrame()

    detail_col = _pick_col(df, DETAIL_COL_CANDIDATES)
    title_col = _pick_col(df, TITLE_COL_CANDIDATES)
    company_col = _pick_col(df, COMPANY_COL_CANDIDATES)
    role_col = _pick_col(df, ROLE_COL_CANDIDATES)

    rows = []
    for idx, row in df.iterrows():
        tags = _safe_list(row.get(tag_col))
        if tag not in tags:
            continue
        detail = _safe_str(row.get(detail_col)) if detail_col else ""
        snippet = _context(detail, tag, window=120)
        is_req, is_bonus, is_resp = _tag_signal_type(detail, tag)
        fresh_score, fresh_level = score_fresh_graduate_friendliness(row, detail_col)
        rows.append(
            {
                "职位": _safe_str(row.get(title_col)) if title_col else "",
                "公司": _safe_str(row.get(company_col)) if company_col else "",
                "岗位方向": _safe_str(row.get(role_col)) if role_col else "",
                "应届友好分": fresh_score,
                "应届友好度": fresh_level,
                "证据类型": "必备要求" if is_req else "加分项" if is_bonus else "岗位职责" if is_resp else "普通命中",
                "JD证据片段": snippet,
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def build_skill_combination_table(
    df: pd.DataFrame,
    tag_type: str = "全部",
    source_mode: str = "最终标签",
    min_support: int = 2,
    top_n: int = 20,
) -> pd.DataFrame:
    df, tag_col = ensure_tag_source(df, tag_type, source_mode)
    if df is None or df.empty or not tag_col or tag_col not in df.columns:
        return pd.DataFrame()

    total_jobs = max(df["job_id"].nunique() if "job_id" in df.columns else len(df), 1)
    pair_counter = Counter()
    for _, row in df.iterrows():
        tags = sorted(set(_safe_list(row.get(tag_col))))
        for a, b in combinations(tags, 2):
            pair_counter[(a, b)] += 1

    rows = []
    for (a, b), count in pair_counter.most_common():
        if count < min_support:
            continue
        rows.append(
            {
                "能力A": a,
                "能力B": b,
                "共现岗位数": int(count),
                "共现覆盖率": count / total_jobs,
                "简历表达建议": f"如果同时具备「{a}」和「{b}」，建议在同一段项目经历中写出方法链路和结果。"
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def build_fresh_friendly_jobs(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    detail_col = _pick_col(df, DETAIL_COL_CANDIDATES)
    title_col = _pick_col(df, TITLE_COL_CANDIDATES)
    company_col = _pick_col(df, COMPANY_COL_CANDIDATES)
    role_col = _pick_col(df, ROLE_COL_CANDIDATES)

    rows = []
    for _, row in df.iterrows():
        score, level = score_fresh_graduate_friendliness(row, detail_col)
        detail = _safe_str(row.get(detail_col)) if detail_col else ""
        pos = [s for s in FRESH_POSITIVE_SIGNALS if s.lower() in detail.lower()]
        neg = [s for s in FRESH_NEGATIVE_SIGNALS if s.lower() in detail.lower()]
        rows.append(
            {
                "职位": _safe_str(row.get(title_col)) if title_col else "",
                "公司": _safe_str(row.get(company_col)) if company_col else "",
                "岗位方向": _safe_str(row.get(role_col)) if role_col else "",
                "应届友好分": score,
                "应届友好度": level,
                "正向信号": "、".join(pos[:5]),
                "风险信号": "、".join(neg[:5]),
            }
        )
    return pd.DataFrame(rows).sort_values("应届友好分", ascending=False).head(top_n).reset_index(drop=True)


def build_job_direction_opportunity_table(
    df: pd.DataFrame,
    tag_table: pd.DataFrame | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    detail_col = _pick_col(df, DETAIL_COL_CANDIDATES)
    role_col = _pick_col(df, ROLE_COL_CANDIDATES)
    company_col = _pick_col(df, COMPANY_COL_CANDIDATES)
    if not role_col:
        return pd.DataFrame()

    rows = []
    for role, group in df.groupby(role_col):
        if not _safe_str(role):
            continue
        scores = [score_fresh_graduate_friendliness(row, detail_col)[0] for _, row in group.iterrows()]
        job_count = len(group)
        company_count = group[company_col].nunique() if company_col else 0
        fresh_avg = sum(scores) / max(len(scores), 1)
        opportunity_score = min(job_count, 30) * 1.2 + min(company_count, 20) * 1.3 + fresh_avg * 0.55
        rows.append(
            {
                "岗位方向": role,
                "岗位数": int(job_count),
                "覆盖公司数": int(company_count),
                "平均应届友好分": round(fresh_avg, 1),
                "投递优先级得分": round(opportunity_score, 1),
                "建议": "优先投递" if opportunity_score >= 65 else "可重点观察" if opportunity_score >= 45 else "谨慎投入",
            }
        )
    return pd.DataFrame(rows).sort_values("投递优先级得分", ascending=False).head(top_n).reset_index(drop=True)


def build_evidence_overview(tag_table: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    if tag_table is None or tag_table.empty:
        return {
            "岗位样本数": int(len(df) if isinstance(df, pd.DataFrame) else 0),
            "高置信能力数": 0,
            "核心门槛数": 0,
            "重点强化数": 0,
            "加分项数": 0,
        }
    return {
        "岗位样本数": int(df["job_id"].nunique() if "job_id" in df.columns else len(df)),
        "高置信能力数": int((tag_table["置信等级"] == "高").sum()),
        "核心门槛数": int((tag_table["求职优先级"] == "核心门槛").sum()),
        "重点强化数": int((tag_table["求职优先级"] == "重点强化").sum()),
        "加分项数": int((tag_table["求职优先级"] == "加分项").sum()),
    }
