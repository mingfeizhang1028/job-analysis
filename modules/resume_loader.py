from __future__ import annotations

from io import BytesIO
import os
import re
import tempfile
from typing import Any, Tuple, Dict

try:
    import docx2txt  # type: ignore
except Exception:
    docx2txt = None

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None


def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            text = raw.decode(encoding).strip()
            if text:
                return text
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore").strip()


def _collect_docx_table_text(doc: Any) -> list[str]:
    parts: list[str] = []
    try:
        for table in getattr(doc, "tables", []):
            for row in getattr(table, "rows", []):
                row_parts: list[str] = []
                for cell in getattr(row, "cells", []):
                    text = str(getattr(cell, "text", "") or "").strip()
                    if text:
                        row_parts.append(text)
                if row_parts:
                    parts.append(" | ".join(row_parts))
    except Exception:
        pass
    return parts


def _deduplicate_lines(parts: list[str]) -> str:
    seen = set()
    cleaned: list[str] = []
    for item in parts:
        text = str(item or "").strip()
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            cleaned.append(text)
    return "\n".join(cleaned).strip()


def _normalize_resume_like_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip(" |\t") for line in text.split("\n")]
    return _deduplicate_lines(lines)


def _extract_docx_via_zip_xml(raw: bytes) -> str:
    try:
        import zipfile
        import xml.etree.ElementTree as ET

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        parts: list[str] = []
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            targets = [
                name for name in zf.namelist()
                if name.startswith("word/") and name.endswith(".xml") and any(k in name for k in ["document", "header", "footer"])
            ]
            for name in targets:
                xml_bytes = zf.read(name)
                root = ET.fromstring(xml_bytes)
                for para in root.findall(".//w:p", ns):
                    texts = [node.text or "" for node in para.findall(".//w:t", ns)]
                    merged = "".join(texts).strip()
                    if merged:
                        parts.append(merged)
        return _normalize_resume_like_text("\n".join(parts))
    except Exception:
        return ""


def _load_docx_text(raw: bytes) -> str:
    try:
        from docx import Document  # type: ignore

        doc = Document(BytesIO(raw))
        parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        parts.extend(_collect_docx_table_text(doc))
        merged = _normalize_resume_like_text("\n".join(parts))
        if merged and len(merged) >= 40:
            return merged
    except Exception:
        pass

    xml_text = _extract_docx_via_zip_xml(raw)
    if xml_text and len(xml_text) >= 40:
        return xml_text

    if docx2txt is not None:
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(raw)
                temp_path = tmp.name
            text = _normalize_resume_like_text(docx2txt.process(temp_path) or "")
            if text:
                return text
        except Exception:
            pass
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    return ""


def _load_pdf_text(raw: bytes) -> str:
    if pdfplumber is None:
        return ""
    try:
        pages = []
        with pdfplumber.open(BytesIO(raw)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text.strip())
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def load_resume_text(text_input: str = "", uploaded_file: Any = None) -> str:
    """读取简历文本，优先使用文本框，其次读取上传文件。支持 txt/md/docx/pdf。"""
    text, _ = load_resume_text_with_meta(text_input=text_input, uploaded_file=uploaded_file)
    return text


def load_resume_text_with_meta(text_input: str = "", uploaded_file: Any = None) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "source": "",
        "file_name": getattr(uploaded_file, "name", "") if uploaded_file is not None else "",
        "suffix": "",
        "raw_size": 0,
        "extractor": "",
        "error": "",
        "char_count": 0,
    }

    if text_input and str(text_input).strip():
        text = _normalize_resume_like_text(str(text_input))
        meta.update({"source": "text_input", "extractor": "text_input", "char_count": len(text)})
        return text, meta

    if uploaded_file is None:
        meta["error"] = "未提供简历文本或上传文件"
        return "", meta

    try:
        file_name = getattr(uploaded_file, "name", "") or ""
        suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
        meta["source"] = "upload"
        meta["suffix"] = suffix
        try:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
        except Exception:
            pass
        raw = uploaded_file.read()
        meta["raw_size"] = len(raw) if isinstance(raw, (bytes, bytearray)) else 0
        if not isinstance(raw, bytes):
            text = _normalize_resume_like_text(str(raw))
            meta.update({"extractor": "string_cast", "char_count": len(text)})
            return text, meta

        if suffix == "docx":
            text = _load_docx_text(raw)
            meta["extractor"] = "docx_pipeline"
            meta["char_count"] = len(text)
            if text:
                return text, meta
            meta["error"] = "docx 未提取到文本，可能是异常格式、文本框/图片型内容或文件损坏"
            return "", meta

        if suffix == "pdf":
            text = _load_pdf_text(raw)
            meta["extractor"] = "pdfplumber"
            meta["char_count"] = len(text)
            if text:
                return _normalize_resume_like_text(text), meta
            meta["error"] = "pdf 未提取到文本，可能为扫描版 PDF"
            return "", meta

        text = _normalize_resume_like_text(_decode_bytes(raw))
        meta.update({"extractor": "byte_decode", "char_count": len(text)})
        if not text:
            meta["error"] = "文件已读取，但未解析出有效文本"
        return text, meta
    except Exception as e:
        meta["error"] = str(e)
        return "", meta


def supported_resume_types() -> list[str]:
    return ["txt", "md", "docx", "pdf"]
