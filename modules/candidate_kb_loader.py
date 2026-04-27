from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from modules.resume_loader import load_resume_text

KB_BASE_DIR = os.path.join("data", "candidate_kb")
RAW_DIR = os.path.join(KB_BASE_DIR, "raw")
PARSED_DIR = os.path.join(KB_BASE_DIR, "parsed")
DOC_INDEX_PATH = os.path.join(KB_BASE_DIR, "document_index.json")

SUPPORTED_DOC_TYPES = [
    "resume",
    "project",
    "internship",
    "competition",
    "research",
    "certificate",
    "other",
]


def ensure_kb_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PARSED_DIR, exist_ok=True)


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


def load_document_index() -> List[Dict[str, Any]]:
    ensure_kb_dirs()
    data = _load_json(DOC_INDEX_PATH, [])
    return data if isinstance(data, list) else []


def save_document_index(index_data: List[Dict[str, Any]]) -> None:
    ensure_kb_dirs()
    _save_json(DOC_INDEX_PATH, index_data)


def build_doc_id(file_name: str, text: str, doc_type: str) -> str:
    digest = hashlib.md5(f"{doc_type}|{file_name}|{text[:1000]}".encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{doc_type}_{digest}"


def ingest_candidate_document(
    text_input: str = "",
    uploaded_file: Any = None,
    doc_type: str = "other",
    title: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    ensure_kb_dirs()
    text = load_resume_text(text_input=text_input, uploaded_file=uploaded_file)
    if not text:
        return {"success": False, "error": "未读取到材料内容"}

    file_name = title.strip() or getattr(uploaded_file, "name", "manual_input") or "manual_input"
    doc_type = doc_type if doc_type in SUPPORTED_DOC_TYPES else "other"
    doc_id = build_doc_id(file_name=file_name, text=text, doc_type=doc_type)

    raw_record = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": title.strip() or file_name,
        "file_name": file_name,
        "text": text,
        "metadata": {
            "tags": tags or [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "upload" if uploaded_file is not None else "text_input",
            "char_count": len(text),
        },
    }

    _save_json(os.path.join(RAW_DIR, f"{doc_id}.json"), raw_record)

    index_data = load_document_index()
    index_data = [x for x in index_data if x.get("doc_id") != doc_id]
    index_data.append({
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": raw_record["title"],
        "file_name": file_name,
        "char_count": len(text),
        "created_at": raw_record["metadata"]["created_at"],
        "tags": raw_record["metadata"]["tags"],
    })
    index_data = sorted(index_data, key=lambda x: x.get("created_at", ""), reverse=True)
    save_document_index(index_data)

    return {"success": True, "doc_id": doc_id, "record": raw_record}


def load_raw_document(doc_id: str) -> Dict[str, Any]:
    ensure_kb_dirs()
    return _load_json(os.path.join(RAW_DIR, f"{doc_id}.json"), {})
