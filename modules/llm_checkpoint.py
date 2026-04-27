from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = BASE_DIR / "data" / "llm_runs"
RUN_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any):
    return str(value)


def save_llm_checkpoint(df: pd.DataFrame, metadata: dict[str, Any] | None = None, name: str = "llm_enriched") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (name or "llm_enriched")).strip("_") or "llm_enriched"
    parquet_path = RUN_DIR / f"{safe_name}_{ts}.parquet"
    meta_path = RUN_DIR / f"{safe_name}_{ts}.meta.json"

    df.to_parquet(parquet_path, index=False)
    meta_payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    if metadata:
        meta_payload.update(metadata)
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return parquet_path


def list_llm_checkpoints() -> list[Path]:
    return sorted(RUN_DIR.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_llm_checkpoint(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_llm_checkpoint_meta(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
