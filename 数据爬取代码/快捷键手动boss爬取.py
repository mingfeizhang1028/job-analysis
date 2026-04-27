# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:07:43 2026

@author: Rosem
"""

import csv
import json
import os
import re
import time
import urllib.request
from bs4 import BeautifulSoup
import keyboard

DEBUG_PORT = 9222
OUTPUT_FILE = r"P:\零七八碎\找工作的快乐人生\Result\current_selected_job_detail.csv"

# BOSS 薪资私有字符映射
PRIVATE_DIGIT_MAP = {
    "": "9",
    "": "0",
    "": "1",
    "": "2",
    "": "3",
    "": "4",
    "": "5",
    "": "6",
    "": "7",
    "": "8",
}


running = True


def get_tabs():
    with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json", timeout=3) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_cdp(ws_url, method, params=None, msg_id=1):
    import websocket

    ws = websocket.create_connection(ws_url, timeout=8)
    payload = {
        "id": msg_id,
        "method": method,
        "params": params or {}
    }
    ws.send(json.dumps(payload))

    while True:
        result = json.loads(ws.recv())
        if result.get("id") == msg_id:
            ws.close()
            return result


def get_current_page_html():
    tabs = get_tabs()
    page_tabs = [t for t in tabs if t.get("type") == "page"]

    if not page_tabs:
        raise RuntimeError("没有找到可用页面标签页")

    tab = page_tabs[0]
    ws_url = tab["webSocketDebuggerUrl"]

    call_cdp(ws_url, "Page.enable", msg_id=1)
    result = call_cdp(
        ws_url,
        "Runtime.evaluate",
        {
            "expression": "document.documentElement.outerHTML",
            "returnByValue": True
        },
        msg_id=2
    )

    html = result["result"]["result"]["value"]
    return tab["title"], tab["url"], html


def clean_text(text):
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def decode_private_digits(text):
    if not text:
        return ""
    for k, v in PRIVATE_DIGIT_MAP.items():
        text = text.replace(k, v)
    return text


def extract_salary(detail_box, soup):
    salary_raw = ""

    node = detail_box.select_one(".job-detail-info .job-salary")
    if node:
        salary_raw = clean_text(node.get_text(" ", strip=True))

    if not salary_raw:
        node = soup.select_one(".job-card-wrap.active .job-salary")
        if node:
            salary_raw = clean_text(node.get_text(" ", strip=True))

    salary_decoded = decode_private_digits(salary_raw)
    return salary_raw, salary_decoded


def clean_job_desc(detail_box):
    desc_node = detail_box.select_one("p.desc")
    if not desc_node:
        return ""

    for bad in desc_node.select("style, script"):
        bad.decompose()

    for tag in desc_node.find_all(True):
        style = (tag.get("style") or "").replace(" ", "").lower()
        txt = tag.get_text(strip=True)

        if (
            "display:none" in style
            or "visibility:hidden" in style
            or txt in {"boss", "kanzhun", "直聘", "BOSS直聘"}
        ):
            tag.decompose()

    text = desc_node.get_text("\n", strip=True)
    text = decode_private_digits(text)
    text = clean_text(text)
    text = re.sub(r"(BOSS直聘|kanzhun|boss|直聘)", "", text, flags=re.I)
    text = re.sub(r"\n{2,}", "\n", text).strip()

    return text


def extract_company(detail_box, soup):
    boss_info_attr = detail_box.select_one(".boss-info-attr")
    if boss_info_attr:
        txt = clean_text(boss_info_attr.get_text(" ", strip=True))
        if txt:
            return txt.split("·")[0].strip()

    node = soup.select_one(".job-card-wrap.active .boss-name")
    if node:
        return clean_text(node.get_text(" ", strip=True))

    return ""


def extract_detail_url(detail_box, soup):
    node = detail_box.select_one("a.more-job-btn")
    if node and node.get("href"):
        href = node.get("href").strip()
        if href.startswith("http"):
            return href
        return "https://www.zhipin.com" + href

    node = soup.select_one(".job-card-wrap.active .job-name")
    if node and node.get("href"):
        href = node.get("href").strip()
        if href.startswith("http"):
            return href
        return "https://www.zhipin.com" + href

    return ""


def extract_current_job_detail(html, page_url=""):
    soup = BeautifulSoup(html, "html.parser")

    detail_box = soup.select_one(".job-detail-box")
    if not detail_box:
        raise RuntimeError("没有找到当前选中的职位详情区域 .job-detail-box")

    job_name = ""
    node = detail_box.select_one(".job-detail-info .job-name")
    if node:
        job_name = clean_text(node.get_text(" ", strip=True))

    salary_raw, salary_decoded = extract_salary(detail_box, soup)

    tags = detail_box.select(".tag-list li")
    city, exp, edu = "", "", ""

    if len(tags) >= 1:
        city = clean_text(tags[0].get_text(" ", strip=True))
    if len(tags) >= 2:
        exp = clean_text(tags[1].get_text(" ", strip=True))
    if len(tags) >= 3:
        edu = clean_text(tags[2].get_text(" ", strip=True))

    company = extract_company(detail_box, soup)

    address = ""
    node = detail_box.select_one(".job-address-desc")
    if node:
        address = clean_text(node.get_text(" ", strip=True))

    job_desc = clean_job_desc(detail_box)
    detail_url = extract_detail_url(detail_box, soup)

    return {
        "抓取时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "页面URL": page_url,
        "职位名称": job_name,
        "企业名称": company,
        "所在地区": city,
        "薪资原始": salary_raw,
        "薪资解析": salary_decoded,
        "经验要求": exp,
        "学历要求": edu,
        "工作地址": address,
        "岗位详情": job_desc,
        "详情链接": detail_url
    }


def save_to_csv(row, filename=OUTPUT_FILE):
    fieldnames = [
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
        "详情链接"
    ]

    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def capture_current_job():
    try:
        title, url, html = get_current_page_html()
        row = extract_current_job_detail(html, url)
        save_to_csv(row)

        print("\n==============================")
        print("抓取成功，已保存到 CSV：", OUTPUT_FILE)
        print("职位名称：", row["职位名称"])
        print("企业名称：", row["企业名称"])
        print("所在地区：", row["所在地区"])
        print("薪资原始：", row["薪资原始"])
        print("薪资解析：", row["薪资解析"])
        print("经验要求：", row["经验要求"])
        print("学历要求：", row["学历要求"])
        print("工作地址：", row["工作地址"])
        print("详情链接：", row["详情链接"])
        print("==============================\n")
    except Exception as e:
        print("\n抓取失败：", e, "\n")


def quit_program():
    global running
    print("\n收到退出指令，程序结束。")
    running = False


def main():
    global running

    print("程序已启动。")
    print("快捷键说明：")
    print("  Ctrl+Alt+S  -> 抓取当前职位并保存")
    print("  Ctrl+Alt+Q  -> 退出程序")
    print("\n请保持 Chrome 已用 --remote-debugging-port=9222 启动。")

    keyboard.add_hotkey("ctrl+alt+s", capture_current_job)
    keyboard.add_hotkey("ctrl+alt+q", quit_program)

    while running:
        time.sleep(0.5)

    keyboard.unhook_all_hotkeys()


if __name__ == "__main__":
    main()