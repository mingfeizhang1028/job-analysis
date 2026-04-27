from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


DEBUG_PORT = 9222
CAPTURE_DIR = Path("data/boss_jobs")
DEFAULT_CAPTURE_FILE = CAPTURE_DIR / "boss_captured_jobs.csv"
LLM_SETTINGS_FILE = Path("data/llm_settings.json")

PRIVATE_DIGIT_MAP = {
    "\ue03a": "9",
    "\ue031": "0",
    "\ue032": "1",
    "\ue033": "2",
    "\ue034": "3",
    "\ue035": "4",
    "\ue036": "5",
    "\ue037": "6",
    "\ue038": "7",
    "\ue039": "8",
}

CAPTURE_COLUMNS = [
    "抓取时间",
    "页面URL",
    "职位名称",
    "企业名称",
    "所在地区",
    "薪资原始",
    "薪资解析",
    "经验要求",
    "学历要求",
    "工作地址",
    "岗位详情",
    "详情链接",
    "数据来源",
]

CAPTURE_TIME_COL = CAPTURE_COLUMNS[0]
PAGE_URL_COL = CAPTURE_COLUMNS[1]
JOB_NAME_COL = CAPTURE_COLUMNS[2]
COMPANY_COL = CAPTURE_COLUMNS[3]
CITY_COL = CAPTURE_COLUMNS[4]
SALARY_RAW_COL = CAPTURE_COLUMNS[5]
SALARY_PARSED_COL = CAPTURE_COLUMNS[6]
EXPERIENCE_COL = CAPTURE_COLUMNS[7]
EDUCATION_COL = CAPTURE_COLUMNS[8]
ADDRESS_COL = CAPTURE_COLUMNS[9]
JOB_DETAIL_COL = CAPTURE_COLUMNS[10]
DETAIL_URL_COL = CAPTURE_COLUMNS[11]
SOURCE_COL = CAPTURE_COLUMNS[12]


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def clean_job_detail_text(text: str | None, *, keep_linebreaks: bool = False) -> str:
    """Normalize JD text before saving and tag extraction."""
    if not text:
        return ""

    value = str(text)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\xa0", " ").replace("\u3000", " ")
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"[ \t]*\n[ \t]*", "\n", value)
    value = re.sub(r"\n{2,}", "\n", value)

    lines = [re.sub(r" {2,}", " ", line).strip() for line in value.splitlines()]
    lines = [line for line in lines if line]
    if keep_linebreaks:
        return "\n".join(lines).strip()
    return re.sub(r" {2,}", " ", " ".join(lines)).strip()


def enrich_captured_job(row: dict[str, Any]) -> dict[str, Any]:
    """Clean captured text and append rule-based tag columns for downstream analysis."""
    if not isinstance(row, dict):
        return {}

    enriched = dict(row)
    if JOB_DETAIL_COL in enriched:
        enriched[JOB_DETAIL_COL] = clean_job_detail_text(enriched.get(JOB_DETAIL_COL))

    try:
        from modules.tag_extraction import apply_tag_extraction
        from modules.jd_rule_extraction import apply_rule_jd_extraction

        tagged = apply_tag_extraction(pd.DataFrame([enriched]), detail_col=JOB_DETAIL_COL)
        tagged = apply_rule_jd_extraction(tagged, detail_col=JOB_DETAIL_COL)
        if not tagged.empty:
            enriched.update(tagged.iloc[0].to_dict())
    except Exception:
        # Tag extraction is a convenience at capture time; saving the JD should still work.
        pass

    return enriched


def decode_private_digits(text: str | None) -> str:
    if not text:
        return ""
    out = str(text)
    for src, dst in PRIVATE_DIGIT_MAP.items():
        out = out.replace(src, dst)
    return out


def list_debug_tabs(debug_port: int = DEBUG_PORT) -> list[dict[str, Any]]:
    url = f"http://127.0.0.1:{int(debug_port)}/json"
    with urllib.request.urlopen(url, timeout=3) as resp:
        tabs = json.loads(resp.read().decode("utf-8"))
    return [tab for tab in tabs if tab.get("type") == "page"]


def format_tab_label(tab: dict[str, Any]) -> str:
    title = clean_text(tab.get("title")) or "未命名页面"
    url = clean_text(tab.get("url"))
    marker = "BOSS" if "zhipin.com" in url else "网页"
    return f"{marker}｜{title[:48]}｜{url[:80]}"


def _call_cdp(ws_url: str, method: str, params: dict | None = None, msg_id: int = 1) -> dict:
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError("缺少 websocket-client 依赖，请先安装 websocket-client。") from exc

    ws = websocket.create_connection(ws_url, timeout=8)
    try:
        ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            result = json.loads(ws.recv())
            if result.get("id") == msg_id:
                return result
    finally:
        ws.close()


def get_tab_html(tab: dict[str, Any]) -> tuple[str, str, str]:
    ws_url = tab.get("webSocketDebuggerUrl")
    if not ws_url:
        raise RuntimeError("当前标签页没有 webSocketDebuggerUrl，无法读取页面 HTML。")

    _call_cdp(ws_url, "Page.enable", msg_id=1)
    result = _call_cdp(
        ws_url,
        "Runtime.evaluate",
        {
            "expression": "document.documentElement.outerHTML",
            "returnByValue": True,
        },
        msg_id=2,
    )
    html = result.get("result", {}).get("result", {}).get("value", "")
    if not html:
        raise RuntimeError("未能从当前标签页读取 HTML。")
    return clean_text(tab.get("title")), clean_text(tab.get("url")), html


def _first_text(root, selectors: Iterable[str]) -> str:
    for selector in selectors:
        node = root.select_one(selector)
        if node:
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                return text
    return ""


def _absolute_boss_url(href: str | None) -> str:
    href = clean_text(href)
    if not href:
        return ""
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return "https://www.zhipin.com" + href
    return href


def _extract_salary(detail_box, soup: BeautifulSoup) -> tuple[str, str]:
    salary_raw = _first_text(
        detail_box,
        [
            ".job-detail-info .job-salary",
            ".job-banner .salary",
            ".salary",
        ],
    )
    if not salary_raw:
        salary_raw = _first_text(soup, [".job-card-wrap.active .job-salary", ".job-card-wrapper.active .salary"])
    return salary_raw, decode_private_digits(salary_raw)


def _clean_job_desc(detail_box) -> str:
    desc_node = detail_box.select_one("p.desc") or detail_box.select_one(".job-sec-text") or detail_box.select_one(".job-detail-section")
    if not desc_node:
        return ""

    for bad in desc_node.select("style, script"):
        bad.decompose()

    for tag in desc_node.find_all(True):
        style = (tag.get("style") or "").replace(" ", "").lower()
        txt = tag.get_text(strip=True)
        if "display:none" in style or "visibility:hidden" in style or txt.lower() in {"boss", "kanzhun", "boss直聘"}:
            tag.decompose()

    text = desc_node.get_text("\n", strip=True)
    text = decode_private_digits(clean_text(text))
    text = re.sub(r"(BOSS直聘|kanzhun|boss直聘|boss)", "", text, flags=re.I)
    return clean_job_detail_text(text)


def _extract_company(detail_box, soup: BeautifulSoup) -> str:
    boss_info_attr = detail_box.select_one(".boss-info-attr")
    if boss_info_attr:
        text = clean_text(boss_info_attr.get_text(" ", strip=True))
        if text:
            return text.split("·")[0].split("路")[0].strip()
    return _first_text(
        soup,
        [
            ".job-card-wrap.active .boss-name",
            ".job-card-wrapper.active .company-name",
            ".company-info .name",
            ".company-name",
        ],
    )


def _extract_detail_url(detail_box, soup: BeautifulSoup, page_url: str) -> str:
    for selector in ["a.more-job-btn", ".job-card-wrap.active .job-name", ".job-card-wrapper.active a"]:
        node = detail_box.select_one(selector) or soup.select_one(selector)
        if node and node.get("href"):
            return _absolute_boss_url(node.get("href"))
    return page_url if "zhipin.com" in page_url else ""


def extract_current_job_detail(html: str, page_url: str = "") -> dict[str, str]:
    soup = BeautifulSoup(html or "", "html.parser")
    detail_box = soup.select_one(".job-detail-box") or soup.select_one(".job-detail") or soup.select_one(".job-banner")
    if not detail_box:
        raise RuntimeError("没有识别到 BOSS 岗位详情区域。请先在 BOSS 页面选中一个岗位，或打开岗位详情页。")

    job_name = _first_text(
        detail_box,
        [
            ".job-detail-info .job-name",
            ".job-banner .name",
            ".name",
        ],
    )
    salary_raw, salary_decoded = _extract_salary(detail_box, soup)

    tags = [clean_text(node.get_text(" ", strip=True)) for node in detail_box.select(".tag-list li")]
    city = tags[0] if len(tags) >= 1 else ""
    exp = tags[1] if len(tags) >= 2 else ""
    edu = tags[2] if len(tags) >= 3 else ""

    company = _extract_company(detail_box, soup)
    address = _first_text(detail_box, [".job-address-desc", ".location-address", ".job-location"])
    job_desc = _clean_job_desc(detail_box)
    detail_url = _extract_detail_url(detail_box, soup, page_url)

    return enrich_captured_job(
        {
            CAPTURE_TIME_COL: time.strftime("%Y-%m-%d %H:%M:%S"),
            PAGE_URL_COL: page_url,
            JOB_NAME_COL: job_name,
            COMPANY_COL: company,
            CITY_COL: city,
            SALARY_RAW_COL: salary_raw,
            SALARY_PARSED_COL: salary_decoded,
            EXPERIENCE_COL: exp,
            EDUCATION_COL: edu,
            ADDRESS_COL: address,
            JOB_DETAIL_COL: job_desc,
            DETAIL_URL_COL: detail_url,
            SOURCE_COL: "BOSS直聘",
        }
    )


def _row_key(row: pd.Series) -> str:
    link = clean_text(row.get(DETAIL_URL_COL)) or clean_text(row.get(PAGE_URL_COL))
    if link:
        return "url:" + link
    payload = "|".join(
        clean_text(row.get(col))
        for col in [JOB_NAME_COL, COMPANY_COL, CITY_COL, JOB_DETAIL_COL]
    )
    return "hash:" + hashlib.md5(payload.encode("utf-8")).hexdigest()


def build_capture_key(row: dict[str, Any] | pd.Series) -> str:
    if isinstance(row, pd.Series):
        return _row_key(row)
    return _row_key(pd.Series(row or {}))


def deduplicate_job_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CAPTURE_COLUMNS)
    result = df.copy()
    for col in CAPTURE_COLUMNS:
        if col not in result.columns:
            result[col] = ""
    result["_capture_key"] = result.apply(_row_key, axis=1)
    result = result.drop_duplicates("_capture_key", keep="last").drop(columns=["_capture_key"])
    return result.reset_index(drop=True)


def load_captured_jobs(path: Path = DEFAULT_CAPTURE_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CAPTURE_COLUMNS)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    except Exception:
        df = pd.read_csv(path).fillna("")
    return deduplicate_job_rows(df)


def _normalize_cell_value(value: Any) -> str:
    if isinstance(value, list):
        return "，".join(clean_text(item) for item in value if clean_text(item))
    return clean_text(value)


def normalize_captured_jobs_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CAPTURE_COLUMNS)

    result = df.copy().fillna("")
    for col in CAPTURE_COLUMNS:
        if col not in result.columns:
            result[col] = ""

    if JOB_DETAIL_COL in result.columns:
        result[JOB_DETAIL_COL] = result[JOB_DETAIL_COL].apply(clean_job_detail_text)

    try:
        from modules.tag_extraction import apply_tag_extraction
        from modules.jd_rule_extraction import apply_rule_jd_extraction

        result = apply_tag_extraction(result, detail_col=JOB_DETAIL_COL)
        result = apply_rule_jd_extraction(result, detail_col=JOB_DETAIL_COL)
    except Exception:
        pass

    for col in result.columns:
        result[col] = result[col].apply(_normalize_cell_value)

    return deduplicate_job_rows(result)


def write_captured_jobs(df: pd.DataFrame, path: Path = DEFAULT_CAPTURE_FILE) -> pd.DataFrame:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    result = normalize_captured_jobs_df(df)
    result.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return result


def import_captured_jobs(
    imported_df: pd.DataFrame,
    *,
    path: Path = DEFAULT_CAPTURE_FILE,
    mode: str = "merge_update",
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    imported = normalize_captured_jobs_df(imported_df)
    existing = load_captured_jobs(path)
    existing_count = int(len(existing))
    import_count = int(len(imported))

    if mode == "overwrite":
        result = imported
    elif mode == "append_new":
        existing_keys = set(existing.apply(_row_key, axis=1).tolist()) if not existing.empty else set()
        if imported.empty:
            new_rows = imported
        else:
            new_rows = imported[~imported.apply(_row_key, axis=1).isin(existing_keys)].copy()
        result = pd.concat([existing, new_rows], ignore_index=True)
        result = deduplicate_job_rows(result)
    else:
        mode = "merge_update"
        result = pd.concat([existing, imported], ignore_index=True)
        result = deduplicate_job_rows(result)

    result = write_captured_jobs(result, path=path)
    final_count = int(len(result))
    summary = {
        "mode": mode,
        "existing_count": existing_count,
        "import_count": import_count,
        "final_count": final_count,
        "changed_count": max(final_count - existing_count, 0),
        "duplicate_count": max(import_count - max(final_count - existing_count, 0), 0),
    }
    return result, summary


def save_captured_job_with_status(row: dict[str, Any], path: Path = DEFAULT_CAPTURE_FILE) -> tuple[pd.DataFrame, str]:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_captured_jobs(path)
    enriched = enrich_captured_job(row)
    capture_key = build_capture_key(enriched)
    existing_keys = set(existing.apply(_row_key, axis=1).tolist()) if not existing.empty else set()
    status = "updated" if capture_key in existing_keys else "created"
    columns = list(dict.fromkeys([*CAPTURE_COLUMNS, *existing.columns.tolist(), *enriched.keys()]))
    new_row = {col: enriched.get(col, "") for col in columns}
    for col, value in list(new_row.items()):
        new_row[col] = _normalize_cell_value(value)
    for col in columns:
        if col not in existing.columns:
            existing[col] = ""
    existing = existing[columns]
    combined = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    combined = deduplicate_job_rows(combined)
    combined.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return combined, status


def save_captured_job(row: dict[str, Any], path: Path = DEFAULT_CAPTURE_FILE) -> pd.DataFrame:
    combined, _ = save_captured_job_with_status(row, path=path)
    return combined


def _load_jd_struct_llm_config() -> dict[str, Any]:
    defaults = {
        "provider": "local",
        "local_model": "qwen3:8b",
        "local_url": "http://localhost:11434/api/generate",
        "remote_enabled": False,
        "remote_model": "gpt-5.4",
        "remote_base_url": "",
        "remote_api_key": "",
    }
    settings_path = Path(__file__).resolve().parent.parent / LLM_SETTINGS_FILE
    try:
        loaded = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    except Exception:
        loaded = {}
    if not isinstance(loaded, dict):
        loaded = {}
    return {
        "provider": loaded.get("jd_struct_provider", defaults["provider"]),
        "local_model": loaded.get("jd_struct_local_model", defaults["local_model"]),
        "local_url": loaded.get("jd_struct_local_url", defaults["local_url"]),
        "remote_enabled": bool(loaded.get("jd_struct_remote_enabled", defaults["remote_enabled"])),
        "remote_model": loaded.get("jd_struct_remote_model", defaults["remote_model"]),
        "remote_base_url": loaded.get("jd_struct_remote_base_url", defaults["remote_base_url"]),
        "remote_api_key": loaded.get("jd_struct_remote_api_key", defaults["remote_api_key"]),
    }


def apply_jd_structuring_to_captured_row(row: dict[str, Any], llm_config: dict[str, Any] | None = None) -> dict[str, Any]:
    from modules.llm_jd_structuring import apply_llm_jd_structuring

    config = llm_config or _load_jd_struct_llm_config()
    provider = str(config.get("provider") or "local").lower()
    recognized = apply_llm_jd_structuring(
        pd.DataFrame([row]),
        detail_col=JOB_DETAIL_COL,
        model=str(config.get("local_model") or "qwen3:8b"),
        ollama_url=str(config.get("local_url") or "http://localhost:11434/api/generate"),
        limit=1,
        overwrite=True,
        provider=provider,
        remote_enabled=bool(config.get("remote_enabled") or provider == "remote"),
        remote_model=str(config.get("remote_model") or "gpt-5.4"),
        remote_base_url=str(config.get("remote_base_url") or ""),
        remote_api_key=str(config.get("remote_api_key") or ""),
    )
    return recognized.iloc[0].to_dict()


def delete_captured_job(capture_key: str, path: Path = DEFAULT_CAPTURE_FILE) -> pd.DataFrame:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_captured_jobs(path)
    if existing.empty or not capture_key:
        return existing

    result = existing[existing.apply(_row_key, axis=1) != capture_key].copy().reset_index(drop=True)
    result.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return result


def merge_captured_jobs(base_df: pd.DataFrame, captured_df: pd.DataFrame) -> pd.DataFrame:
    if captured_df is None or captured_df.empty:
        return base_df.copy() if isinstance(base_df, pd.DataFrame) else pd.DataFrame()
    if base_df is None or base_df.empty:
        return deduplicate_job_rows(captured_df)

    base = base_df.copy()
    captured = captured_df.copy()
    for col in base.columns:
        if col not in captured.columns:
            captured[col] = ""
    for col in captured.columns:
        if col not in base.columns:
            base[col] = ""
    merged = pd.concat([base, captured[base.columns]], ignore_index=True)
    return deduplicate_job_rows(merged)


def _candidate_capture_tabs(debug_port: int = DEBUG_PORT, preferred_url: str | None = None) -> list[dict[str, Any]]:
    tabs = list_debug_tabs(debug_port)
    if not tabs:
        raise RuntimeError("没有找到可读取的 Chrome 页面标签。")

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_tab(tab: dict[str, Any]) -> None:
        key = clean_text(tab.get("id")) or clean_text(tab.get("webSocketDebuggerUrl")) or clean_text(tab.get("url"))
        if key and key in seen:
            return
        if key:
            seen.add(key)
        ordered.append(tab)

    if preferred_url:
        for tab in tabs:
            if clean_text(tab.get("url")) == clean_text(preferred_url):
                add_tab(tab)
                break

    boss_tabs = [tab for tab in tabs if "zhipin.com" in str(tab.get("url", ""))]
    for tab in boss_tabs:
        add_tab(tab)
    for tab in tabs:
        add_tab(tab)
    return ordered


def pick_capture_tab(debug_port: int = DEBUG_PORT, preferred_url: str | None = None) -> dict[str, Any]:
    return _candidate_capture_tabs(debug_port, preferred_url=preferred_url)[0]


def capture_and_save_current_job(
    *,
    debug_port: int = DEBUG_PORT,
    path: Path = DEFAULT_CAPTURE_FILE,
    preferred_url: str | None = None,
    auto_llm_struct: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    last_error: Exception | None = None
    for tab in _candidate_capture_tabs(debug_port, preferred_url=preferred_url):
        try:
            _, url, html = get_tab_html(tab)
            row = extract_current_job_detail(html, url)
            if auto_llm_struct:
                row = apply_jd_structuring_to_captured_row(row)
            saved, status = save_captured_job_with_status(row, path=path)
            row["_capture_save_status"] = status
            return row, saved
        except Exception as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    raise RuntimeError("没有找到可抓取的 BOSS 岗位页面。")


def parse_hotkey(hotkey: str) -> tuple[set[int], int]:
    """Parse a hotkey string like 'ctrl+shift+z' into (modifier_vk_set, key_vk).

    Modifier VKs: VK_CONTROL, VK_SHIFT, VK_MENU (alt).
    Returns (modifiers, primary_key_vk).
    """
    try:
        import ctypes
        user32 = ctypes.windll.user32
    except (ImportError, AttributeError, OSError):
        raise RuntimeError("此功能仅在 Windows 系统上可用。")

    # Virtual-key codes
    VK_MAP = {
        "ctrl": 0x11,    # VK_CONTROL
        "shift": 0x10,   # VK_SHIFT
        "alt": 0x12,     # VK_MENU
    }

    parts = [p.strip().lower() for p in hotkey.split("+")]
    if len(parts) < 1:
        raise ValueError(f"无效的快捷键：{hotkey}")

    modifiers = set()
    for part in parts[:-1]:
        vk = VK_MAP.get(part)
        if vk is None:
            raise ValueError(f"不支持的修饰键：{part}（仅支持 ctrl / shift / alt）")
        modifiers.add(vk)

    key_part = parts[-1]
    if len(key_part) == 1 and "a" <= key_part <= "z":
        primary_vk = ord(key_part.upper())
    elif key_part in VK_MAP:
        raise ValueError(f"'{key_part}' 不能作为主键，请将其放在组合键末尾之前。")
    else:
        raise ValueError(f"不支持的主键：{key_part}")

    return modifiers, primary_vk


def _wait_hotkey_released(modifiers: set[int], primary_vk: int, poll_interval: float = 0.05) -> None:
    """Block until all keys in the hotkey combo are released."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
    except (ImportError, AttributeError, OSError):
        return

    all_vks = set(modifiers) | {primary_vk}
    while True:
        any_down = False
        for vk in all_vks:
            if user32.GetAsyncKeyState(vk) & 0x8000:
                any_down = True
                break
        if not any_down:
            break
        time.sleep(poll_interval)


def run_hotkey_capture(
    *,
    debug_port: int = DEBUG_PORT,
    path: Path = DEFAULT_CAPTURE_FILE,
    preferred_url: str | None = None,
    capture_hotkey: str = "ctrl+shift+z",
    quit_hotkey: str = "ctrl+shift+x",
    auto_llm_struct: bool = False,
) -> None:
    try:
        import ctypes
        user32 = ctypes.windll.user32
    except (ImportError, AttributeError, OSError):
        raise RuntimeError("此功能仅在 Windows 系统上可用。")

    capture_mods, capture_key = parse_hotkey(capture_hotkey)
    quit_mods, quit_key = parse_hotkey(quit_hotkey)

    running = {"value": True}
    last_capture_time = 0.0
    debounce_seconds = 0.5

    def capture_once() -> None:
        nonlocal last_capture_time
        now = time.time()
        if now - last_capture_time < debounce_seconds:
            return
        last_capture_time = now
        try:
            row, saved = capture_and_save_current_job(
                debug_port=debug_port,
                path=path,
                preferred_url=preferred_url,
                auto_llm_struct=auto_llm_struct,
            )
            action = "更新" if row.get("_capture_save_status") == "updated" else "新增"
            print(
                f"[OK] {action} | 已保存 {len(saved)} 条 | "
                f"{row.get(JOB_NAME_COL, '-') or '-'} | "
                f"{row.get(COMPANY_COL, '-') or '-'} | "
                f"{row.get(DETAIL_URL_COL, '-') or '-'}",
                flush=True,
            )
        except Exception as exc:
            print(f"[ERROR] 抓取失败：{exc}", flush=True)

    def check_combo(modifiers: set[int], primary_vk: int) -> bool:
        for mod_vk in modifiers:
            if not (user32.GetAsyncKeyState(mod_vk) & 0x8000):
                return False
        return bool(user32.GetAsyncKeyState(primary_vk) & 0x8000)

    print("BOSS 岗位快捷键抓取已启动。", flush=True)
    print(f"抓取并保存：{capture_hotkey}", flush=True)
    print(f"退出监听：{quit_hotkey}", flush=True)
    print(f"保存位置：{path}", flush=True)
    if preferred_url:
        print(f"固定读取标签：{preferred_url}", flush=True)
    else:
        print("读取方式：每次按键重新扫描 Chrome 可读页面，优先抓取可解析的 BOSS 岗位页。", flush=True)
    if auto_llm_struct:
        print("自动识别：已开启，抓取后会立即执行 JD 结构化 LLM 识别并保存。", flush=True)
    print("请保持 Chrome 使用 --remote-debugging-port 启动，并在 BOSS 页面选中岗位。", flush=True)

    try:
        while running["value"]:
            time.sleep(0.05)

            if check_combo(quit_mods, quit_key):
                running["value"] = False
                print("收到退出快捷键，监听结束。", flush=True)
                break

            if check_combo(capture_mods, capture_key):
                capture_once()
                _wait_hotkey_released(capture_mods, capture_key)
    except KeyboardInterrupt:
        print("监听已终止。", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BOSS 岗位详情快捷键抓取")
    parser.add_argument("--hotkeys", action="store_true", help="启动全局快捷键监听")
    parser.add_argument("--debug-port", type=int, default=DEBUG_PORT, help="Chrome remote debugging 端口")
    parser.add_argument("--output", type=Path, default=DEFAULT_CAPTURE_FILE, help="保存 CSV 路径")
    parser.add_argument("--preferred-url", default="", help="优先读取的 Chrome 标签 URL")
    parser.add_argument("--capture-hotkey", default="ctrl+shift+z", help="抓取快捷键")
    parser.add_argument("--quit-hotkey", default="ctrl+shift+x", help="退出快捷键")
    parser.add_argument("--auto-llm-struct", action="store_true", help="抓取后自动执行 JD 结构化 LLM 识别")
    args = parser.parse_args()

    if args.hotkeys:
        run_hotkey_capture(
            debug_port=args.debug_port,
            path=args.output,
            preferred_url=args.preferred_url or None,
            capture_hotkey=args.capture_hotkey,
            quit_hotkey=args.quit_hotkey,
            auto_llm_struct=args.auto_llm_struct,
        )
        return

    row, saved = capture_and_save_current_job(
        debug_port=args.debug_port,
        path=args.output,
        preferred_url=args.preferred_url or None,
        auto_llm_struct=args.auto_llm_struct,
    )
    print(f"已保存 {len(saved)} 条：{row.get(JOB_NAME_COL, '-')}")


if __name__ == "__main__":
    main()
