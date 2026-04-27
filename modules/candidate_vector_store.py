from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

from modules.candidate_kb_loader import KB_BASE_DIR, ensure_kb_dirs
from modules.candidate_embedding import get_embedding, DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_EMBED_URL

VECTOR_STORE_PATH = os.path.join(KB_BASE_DIR, "vector_store.json")


def _load_json(path: str, default: Any):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def load_vector_store() -> List[Dict[str, Any]]:
    ensure_kb_dirs()
    data = _load_json(VECTOR_STORE_PATH, [])
    return data if isinstance(data, list) else []


def save_vector_store(records: List[Dict[str, Any]]) -> None:
    ensure_kb_dirs()
    _save_json(VECTOR_STORE_PATH, records)


def upsert_chunks_to_vector_store(
    chunks: List[Dict[str, Any]],
    model: str = DEFAULT_EMBED_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_EMBED_URL,
) -> Dict[str, Any]:
    store = load_vector_store()
    store = [x for x in store if x.get("chunk_id") not in {c.get("chunk_id") for c in chunks}]
    inserted = 0
    errors = []

    for chunk in chunks:
        embedding, err = get_embedding(chunk.get("chunk_text", ""), model=model, ollama_url=ollama_url)
        if err or not embedding:
            errors.append({"chunk_id": chunk.get("chunk_id"), "error": err or "embedding empty"})
            continue
        store.append({
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "doc_type": chunk.get("doc_type"),
            "chunk_type": chunk.get("chunk_type"),
            "title": chunk.get("title", ""),
            "chunk_text": chunk.get("chunk_text", ""),
            "metadata": chunk.get("metadata", {}),
            "embedding": embedding,
        })
        inserted += 1

    save_vector_store(store)
    return {"success": True, "inserted": inserted, "errors": errors, "total": len(store)}


def semantic_search(
    query_text: str,
    top_k: int = 5,
    model: str = DEFAULT_EMBED_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_EMBED_URL,
) -> List[Dict[str, Any]]:
    query_embedding, err = get_embedding(query_text, model=model, ollama_url=ollama_url)
    if err or not query_embedding:
        return []

    store = load_vector_store()
    scored = []
    for item in store:
        score = cosine_similarity(query_embedding, item.get("embedding", []))
        scored.append({**item, "score": round(score, 4)})
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored[:top_k]
