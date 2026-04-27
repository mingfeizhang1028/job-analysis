from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "resume_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


LIST_COLUMNS = [
    "匹配技能",
    "缺失技能",
    "LLM岗位要求",
    "LLM岗位工作内容",
    "LLM加分项",
    "LLM所属行业",
    "LLM岗位类型",
    "LLM必须技能",
    "LLM加分技能",
    "LLM工具栈",
]


def _json_default(value: Any):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    return str(value)


def make_resume_cache_key(
    resume_text: str,
    jobs_fingerprint: str,
    model_signature: str,
    version: str = "v1",
) -> str:
    raw = f"resume_match|{version}|{jobs_fingerprint}|{model_signature}|{resume_text.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_jobs_fingerprint(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"

    cols = [
        c
        for c in ["job_id", "职位名称_norm", "企业名称_norm", "岗位详情", "duplicate_group_id"]
        if c in df.columns
    ]
    sample = df[cols].fillna("").astype(str).head(200).to_dict(orient="records") if cols else []
    raw = json.dumps(
        {
            "rows": int(len(df)),
            "cols": cols,
            "sample": sample,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _get_cache_paths(cache_key: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{cache_key}.json", CACHE_DIR / f"{cache_key}.parquet"


def load_resume_match_cache(cache_key: str) -> dict[str, Any] | None:
    meta_path, match_path = _get_cache_paths(cache_key)
    if not meta_path.exists() or not match_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        match_df = pd.read_parquet(match_path)
        for col in LIST_COLUMNS:
            if col in match_df.columns:
                match_df[col] = match_df[col].apply(_restore_list_value)
        meta["match_df"] = match_df
        return meta
    except Exception:
        return None


def save_resume_match_cache(cache_key: str, payload: dict[str, Any]) -> None:
    meta_path, match_path = _get_cache_paths(cache_key)

    match_df = payload.get("match_df", pd.DataFrame())
    if not isinstance(match_df, pd.DataFrame):
        match_df = pd.DataFrame(match_df)

    match_to_save = match_df.copy()
    for col in LIST_COLUMNS:
        if col in match_to_save.columns:
            match_to_save[col] = match_to_save[col].apply(_prepare_list_value)
    match_to_save.to_parquet(match_path, index=False)

    meta_payload = {k: v for k, v in payload.items() if k != "match_df"}
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def list_resume_cache_files() -> list[str]:
    return sorted([p.name for p in CACHE_DIR.glob("*.json")], reverse=True)


def _prepare_list_value(value: Any):
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return value


def _restore_list_value(value: Any):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            return value
    return value
