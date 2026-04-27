from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from modules.candidate_kb_loader import PARSED_DIR, ensure_kb_dirs, load_raw_document


def _clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    chunk = para[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    if end >= len(para):
                        break
                    start = max(end - overlap, start + 1)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def infer_chunk_type(doc_type: str, chunk_text: str) -> str:
    text = str(chunk_text or "")
    if doc_type in {"project", "internship", "competition", "research", "resume"}:
        return doc_type
    if any(k in text for k in ["项目", "平台", "系统", "算法"]):
        return "project"
    if any(k in text for k in ["实习", "公司", "部门"]):
        return "internship"
    if any(k in text for k in ["竞赛", "比赛", "获奖"]):
        return "competition"
    if any(k in text for k in ["论文", "研究", "课题"]):
        return "research"
    return "other"


def build_document_chunks(doc_id: str, chunk_size: int = 500, overlap: int = 80) -> Dict[str, Any]:
    ensure_kb_dirs()
    raw_doc = load_raw_document(doc_id)
    if not raw_doc:
        return {"success": False, "error": "未找到原始材料"}

    text = raw_doc.get("text", "")
    doc_type = raw_doc.get("doc_type", "other")
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)

    chunk_records = []
    for idx, chunk_text in enumerate(chunks, start=1):
        chunk_records.append({
            "chunk_id": f"{doc_id}_chunk_{idx:03d}",
            "doc_id": doc_id,
            "doc_type": doc_type,
            "chunk_index": idx,
            "chunk_type": infer_chunk_type(doc_type, chunk_text),
            "chunk_text": chunk_text,
            "metadata": raw_doc.get("metadata", {}),
            "title": raw_doc.get("title", ""),
        })

    parsed_record = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": raw_doc.get("title", ""),
        "chunks": chunk_records,
    }

    path = os.path.join(PARSED_DIR, f"{doc_id}_chunks.json")
    with open(path, "w", encoding="utf-8") as f:
        import json
        json.dump(parsed_record, f, ensure_ascii=False, indent=2)

    return {"success": True, "doc_id": doc_id, "chunk_count": len(chunk_records), "chunks": chunk_records}
