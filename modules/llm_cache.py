from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def make_cache_key(
    task: str,
    model: str,
    jd_text: str,
    version: str = "v1",
) -> str:
    raw = f"{task}|{model}|{version}|{jd_text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_json_cache(cache_name: str) -> Dict[str, Any]:
    path = CACHE_DIR / cache_name

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_json_cache(cache_name: str, cache: Dict[str, Any]) -> None:
    path = CACHE_DIR / cache_name
    tmp_path = path.with_suffix(".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)
