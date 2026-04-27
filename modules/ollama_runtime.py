from __future__ import annotations

import subprocess
import time
from typing import Tuple

import requests


DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def _normalize_base_url(url: str) -> str:
    url = str(url or DEFAULT_OLLAMA_BASE).strip()
    if not url:
        return DEFAULT_OLLAMA_BASE
    if "/api/" in url:
        return url.split("/api/")[0].rstrip("/")
    return url.rstrip("/")


def is_ollama_running(base_url: str = DEFAULT_OLLAMA_BASE, timeout: int = 2) -> bool:
    base = _normalize_base_url(base_url)
    try:
        resp = requests.get(f"{base}/api/tags", timeout=timeout)
        return resp.ok
    except Exception:
        return False


def ensure_ollama_running(base_url: str = DEFAULT_OLLAMA_BASE, wait_seconds: int = 15) -> Tuple[bool, str]:
    base = _normalize_base_url(base_url)

    if is_ollama_running(base):
        return True, ""

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
    except Exception as e:
        return False, f"自动启动Ollama失败：{e}"

    for _ in range(max(wait_seconds, 1)):
        if is_ollama_running(base):
            return True, ""
        time.sleep(1)

    return False, "Ollama服务未在预期时间内启动，请手动执行 ollama serve"
