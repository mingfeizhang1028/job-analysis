from __future__ import annotations

from typing import List, Tuple

import requests

from modules.ollama_runtime import ensure_ollama_running

DEFAULT_EMBED_MODEL = "qwen3-embedding:8b"
DEFAULT_OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"


def get_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_EMBED_URL,
) -> Tuple[List[float], str]:
    text = str(text or "").strip()
    if not text:
        return [], "empty text"

    ensure_ollama_running()

    try:
        resp = requests.post(
            ollama_url,
            json={"model": model, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding", [])
        if isinstance(embedding, list) and embedding:
            return embedding, ""
        return [], "embedding empty"
    except Exception as e:
        return [], str(e)
