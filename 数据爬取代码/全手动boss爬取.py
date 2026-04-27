# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:55:54 2026

运行之前win+cmd 的bash中先打开浏览器
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --remote-allow-origins=* --user-data-dir="C:\selenium\manual_chrome"

@author: Rosem
"""

import csv
import json
import os
import re
import time
import urllib.request
from bs4 import BeautifulSoup

DEBUG_PORT = 9222
OUTPUT_FILE = r"P:\找工作的快乐人生\Result\current_selected_job_detail.csv"

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

    # 默认抓第一个 page
    tab = page_tabs[0]
    print(f"\n默认抓取标签页: {tab.get('title', '')[:80]}")
    print(f"页面地址: {tab.get('url', '')}")

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

    # 优先取右侧详情薪资
    node = detail_box.select_one(".job-detail-info .job-salary")
    if node:
        salary_raw = clean_text(node.get_text(" ", strip=True))

    # 右侧没有时，再取左侧当前 active 卡片
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

    # 去掉 style / script
    for bad in desc_node.select("style, script"):
        bad.decompose()

    # 去掉明显反爬噪音节点
    for tag in desc_node.find_all(True):
        style = (tag.get("style") or "").replace(" ", "").lower()
        txt = tag.get_text(strip=True)

        # display:none / visibility:hidden / 明显注入噪音
        if (
            "display:none" in style
            or "visibility:hidden" in style
            or txt in {"boss", "kanzhun", "直聘", "BOSS直聘"}
        ):
            tag.decompose()

    text = desc_node.get_text("\n", strip=True)
    text = decode_private_digits(text)
    text = clean_text(text)

    # 再清理残留噪音词
    text = re.sub(r"(BOSS直聘|kanzhun|boss|直聘)", "", text, flags=re.I)
    text = re.sub(r"\n{2,}", "\n", text).strip()

    return text


def extract_company(detail_box, soup):
    # 优先详情区
    boss_info_attr = detail_box.select_one(".boss-info-attr")
    if boss_info_attr:
        txt = clean_text(boss_info_attr.get_text(" ", strip=True))
        if txt:
            return txt.split("·")[0].strip()

    # 兜底：左侧当前 active 卡片
    node = soup.select_one(".job-card-wrap.active .boss-name")
    if node:
        return clean_text(node.get_text(" ", strip=True))

    return ""


def extract_detail_url(detail_box, soup):
    # 优先详情页“更多”按钮
    node = detail_box.select_one("a.more-job-btn")
    if node and node.get("href"):
        href = node.get("href").strip()
        if href.startswith("http"):
            return href
        return "https://www.zhipin.com" + href

    # 兜底：左侧当前 active 卡片的职位链接
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

    # 职位名称
    job_name = ""
    node = detail_box.select_one(".job-detail-info .job-name")
    if node:
        job_name = clean_text(node.get_text(" ", strip=True))

    # 薪资
    salary_raw, salary_decoded = extract_salary(detail_box, soup)

    # 地区 / 经验 / 学历
    tags = detail_box.select(".tag-list li")
    city, exp, edu = "", "", ""

    if len(tags) >= 1:
        city = clean_text(tags[0].get_text(" ", strip=True))
    if len(tags) >= 2:
        exp = clean_text(tags[1].get_text(" ", strip=True))
    if len(tags) >= 3:
        edu = clean_text(tags[2].get_text(" ", strip=True))

    # 企业名称
    company = extract_company(detail_box, soup)

    # 工作地址
    address = ""
    node = detail_box.select_one(".job-address-desc")
    if node:
        address = clean_text(node.get_text(" ", strip=True))

    # 岗位详情
    job_desc = clean_job_desc(detail_box)

    # 详情链接
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


def main():
    while True:
        cmd = input("\n先手动选中当前职位，回车抓取并保存，q退出：").strip().lower()
        if cmd == "q":
            print("已退出")
            break

        try:
            title, url, html = get_current_page_html()
            row = extract_current_job_detail(html, url)
            save_to_csv(row)

            print("\n已保存到 CSV：", OUTPUT_FILE)
            print("职位名称：", row["职位名称"])
            print("企业名称：", row["企业名称"])
            print("所在地区：", row["所在地区"])
            print("薪资原始：", row["薪资原始"])
            print("薪资解析：", row["薪资解析"])
            print("经验要求：", row["经验要求"])
            print("学历要求：", row["学历要求"])
            print("工作地址：", row["工作地址"])
            print("详情链接：", row["详情链接"])

        except Exception as e:
            print("抓取失败：", e)


if __name__ == "__main__":
    main()