from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import requests

from modules.ollama_runtime import ensure_ollama_running

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def call_ollama_generate(
    prompt: str,
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
    temperature: float = 0.1,
    num_predict: int = 1024,
) -> Tuple[str, str]:
    ok, err = ensure_ollama_running(ollama_url)
    if not ok:
        return "", err

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "") or ""), ""
    except Exception as e:
        return "", str(e)


def call_openai_compatible_generate(
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    api_key: str = DEFAULT_OPENAI_API_KEY,
    timeout: int = 120,
    temperature: float = 0.1,
) -> Tuple[str, str]:
    base_url = str(base_url or "").strip().rstrip("/")
    api_key = str(api_key or "").strip()
    model = str(model or DEFAULT_OPENAI_MODEL).strip()

    if not base_url or not api_key:
        return "", "未配置远程LLM的 base_url 或 api_key"

    endpoint = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "", "远程LLM未返回choices"
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content or ""), ""
    except Exception as e:
        return "", str(e)
