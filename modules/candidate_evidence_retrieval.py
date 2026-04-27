from __future__ import annotations

from typing import Any, Dict, List

from modules.candidate_vector_store import semantic_search
from modules.jd_query_builder import build_jd_query_from_row


def retrieve_evidence_for_job(
    row,
    top_k: int = 6,
    model: str = "qwen3-embedding:8b",
    ollama_url: str = "http://localhost:11434/api/embeddings",
) -> Dict[str, Any]:
    jd_query = build_jd_query_from_row(row)
    results = semantic_search(
        query_text=jd_query.get("query_text", ""),
        top_k=top_k,
        model=model,
        ollama_url=ollama_url,
    )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        chunk_type = item.get("chunk_type", "other")
        grouped.setdefault(chunk_type, []).append(item)

    return {
        "jd_query": jd_query,
        "results": results,
        "grouped_results": grouped,
    }
