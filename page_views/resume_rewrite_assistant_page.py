import json
import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

import requests
import streamlit as st


DEFAULT_TIMEOUT = 45
DEFAULT_TEMPERATURE = 0.2


DEFAULT_PROMPTS = {
    "company": """你是一名企业研究分析师。请基于抓取到的企业公开资料，整理为可服务于简历改写的企业知识。

输出要求：
1. 公司业务与产品方向
2. 行业定位与核心竞争力
3. 企业价值观、团队文化、人才偏好
4. 和目标岗位相关的关键词
5. 可用于简历表达的素材点
6. 不要编造网页中没有的信息

请用结构化中文输出。""",
    "jd": """你是一名校招岗位分析专家。请解析目标 JD，提取岗位真实筛选标准。

输出要求：
1. 岗位核心职责
2. 必备硬技能
3. 软技能与协作要求
4. 隐含筛选标准
5. 高频关键词
6. 简历中最应该体现的 5 个能力点

请用结构化中文输出。""",
    "diagnosis": """你是一名应届生简历诊断顾问。请综合原始简历、JD 需求画像和企业知识，指出简历和目标岗位之间的差距。

输出要求：
1. 每条诊断必须包含：问题标签、当前问题、依据来源、修改方向
2. 依据来源要标明来自 JD、企业资料、或简历本身
3. 不要否定用户不存在的信息，不要编造经历
4. 优先指出可通过表达优化解决的问题

请输出 5-8 条。""",
    "rewrite": """你是一名中文简历改写专家。请基于原始简历、JD 需求画像、企业知识和诊断结果，生成可直接使用的简历优化内容。

输出格式必须严格包含：
一、匹配建议
- 标签：
- 修改建议：
- 参考话术：
- 依据来源：

二、重点经历改写
- 原表达：
- 改写后：
- 改写理由：

三、可复制的项目经历版本

要求：
1. 不要编造不存在的公司、奖项、指标、经历
2. 可以强化已有经历的表达方式、业务目标、方法链路和结果验证
3. 参考话术必须自然、具体、适合放进中文简历
4. 强调 JD 和企业需求的匹配度
5. 输出中文。""",
    "review": """你是一名严格的简历质检官。请检查改写结果是否符合要求。

检查维度：
1. 是否贴合 JD
2. 是否使用了企业资料中的有效信息
3. 是否存在编造风险
4. 是否过于空泛
5. 是否可以直接复制进简历

请输出：
1. 质检结论：通过 / 需要修改
2. 主要问题
3. 修改建议
4. 最终可用度评分，满分 100。""",
}


@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = DEFAULT_TEMPERATURE


class ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []
        self._current: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "canvas", "nav", "footer"}:
            self._skip_depth += 1
        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "article", "section"}:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "canvas", "nav", "footer"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "article", "section"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if text:
            self._current.append(text)

    def _flush(self) -> None:
        text = " ".join(self._current).strip()
        if len(text) >= 20:
            self._chunks.append(text)
        self._current = []

    def text(self) -> str:
        self._flush()
        return "\n".join(self._chunks)


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def chat_completion(config: ModelConfig, messages: list[dict[str, str]]) -> str:
    if not config.base_url or not config.model:
        raise ValueError("模型 base_url 和 model 不能为空。")
    if not config.api_key:
        raise ValueError("API Key 不能为空。")

    url = f"{normalize_base_url(config.base_url)}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
    }
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def embedding(config: ModelConfig, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    url = f"{normalize_base_url(config.base_url)}/embeddings"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={"model": config.model, "input": texts},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


def fetch_page(url: str) -> dict[str, str]:
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            )
        },
        timeout=20,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding

    parser = ReadableHTMLParser()
    parser.feed(response.text)
    text = clean_text(parser.text())
    return {
        "url": url,
        "title": extract_title(response.text) or url,
        "text": text[:20000],
    }


def extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if not match:
        return ""
    return clean_text(re.sub(r"<[^>]+>", "", match.group(1)))


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_urls(raw_urls: str) -> list[str]:
    urls = []
    for item in re.split(r"[\n,， ]+", raw_urls.strip()):
        item = item.strip()
        if item.startswith(("http://", "https://")):
            urls.append(item)
    return list(dict.fromkeys(urls))


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    normalized = clean_text(text)
    chunks = []
    start = 0
    while start < len(normalized):
        chunk = normalized[start : start + size].strip()
        if len(chunk) >= 80:
            chunks.append(chunk)
        start += max(1, size - overlap)
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    left = math.sqrt(sum(x * x for x in a))
    right = math.sqrt(sum(y * y for y in b))
    if not left or not right:
        return 0.0
    return numerator / (left * right)


def keyword_score(query: str, text: str) -> int:
    tokens = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", query.lower()))
    lowered = text.lower()
    return sum(1 for token in tokens if token in lowered)


def retrieve_context(
    query: str,
    chunks: list[str],
    embedding_config: ModelConfig | None,
    use_embedding: bool,
    top_k: int = 8,
) -> list[str]:
    if not chunks:
        return []
    if use_embedding and embedding_config:
        try:
            vectors = embedding(embedding_config, chunks)
            query_vector = embedding(embedding_config, [query])[0]
            ranked = sorted(
                zip(chunks, vectors),
                key=lambda item: cosine(query_vector, item[1]),
                reverse=True,
            )
            return [item[0] for item in ranked[:top_k]]
        except Exception as exc:
            st.warning(f"Embedding 检索失败，已自动切换为关键词检索：{exc}")

    ranked = sorted(chunks, key=lambda item: keyword_score(query, item), reverse=True)
    return ranked[:top_k]


def build_messages(system_prompt: str, user_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
        },
    ]


def get_model_config(prefix: str, fallback: ModelConfig | None = None) -> ModelConfig:
    base = st.session_state.get(f"{prefix}_base_url") or (fallback.base_url if fallback else "")
    key = st.session_state.get(f"{prefix}_api_key") or (fallback.api_key if fallback else "")
    model = st.session_state.get(f"{prefix}_model") or (fallback.model if fallback else "")
    temp = st.session_state.get(f"{prefix}_temperature", fallback.temperature if fallback else DEFAULT_TEMPERATURE)
    return ModelConfig(base, key, model, float(temp))


def render_model_inputs(prefix: str, label: str, default_model: str = "") -> None:
    with st.expander(label, expanded=False):
        st.text_input("Base URL", key=f"{prefix}_base_url", placeholder="https://api.example.com/v1")
        st.text_input("API Key", key=f"{prefix}_api_key", type="password")
        st.text_input("Model", key=f"{prefix}_model", placeholder=default_model)
        st.slider("Temperature", 0.0, 1.2, DEFAULT_TEMPERATURE, 0.05, key=f"{prefix}_temperature")


def run_analysis(
    resume_text: str,
    jd_text: str,
    pages: list[dict[str, str]],
    prompts: dict[str, str],
    configs: dict[str, ModelConfig],
    use_embedding: bool,
) -> dict[str, str]:
    all_page_text = "\n\n".join(
        f"来源：{page['title']} ({page['url']})\n{page['text']}" for page in pages if page["text"]
    )
    chunks = chunk_text(all_page_text)
    query = f"{jd_text}\n\n{resume_text[:2000]}"
    retrieved = retrieve_context(query, chunks, configs.get("embedding"), use_embedding)
    company_context = "\n\n---\n\n".join(retrieved) if retrieved else all_page_text[:9000]

    company_profile = chat_completion(
        configs["company"],
        build_messages(
            prompts["company"],
            {"企业页面资料": company_context, "目标JD": jd_text},
        ),
    )

    jd_profile = chat_completion(
        configs["jd"],
        build_messages(prompts["jd"], {"目标JD": jd_text, "企业画像": company_profile}),
    )

    diagnosis = chat_completion(
        configs["diagnosis"],
        build_messages(
            prompts["diagnosis"],
            {
                "原始简历": resume_text,
                "JD需求画像": jd_profile,
                "企业知识": company_profile,
                "企业相关片段": company_context,
            },
        ),
    )

    rewrite = chat_completion(
        configs["rewrite"],
        build_messages(
            prompts["rewrite"],
            {
                "原始简历": resume_text,
                "JD需求画像": jd_profile,
                "企业知识": company_profile,
                "诊断结果": diagnosis,
            },
        ),
    )

    review = chat_completion(
        configs["review"],
        build_messages(
            prompts["review"],
            {
                "目标JD": jd_text,
                "企业知识": company_profile,
                "诊断结果": diagnosis,
                "改写结果": rewrite,
            },
        ),
    )

    return {
        "company_context": company_context,
        "company_profile": company_profile,
        "jd_profile": jd_profile,
        "diagnosis": diagnosis,
        "rewrite": rewrite,
        "review": review,
    }


def render_resume_rewrite_assistant_page() -> None:
    st.title("简历改写助手")
    st.caption("针对选中的岗位/JD，调用候选人资料和企业/JD 信息，生成定制化简历改写建议。")

    with st.sidebar:
        st.header("模型配置")
        shared = st.toggle("所有模块使用同一个模型配置", value=True)

        with st.expander("默认模型配置", expanded=True):
            st.text_input("Base URL", key="shared_base_url", placeholder="https://api.deepseek.com/v1")
            st.text_input("API Key", key="shared_api_key", type="password")
            st.text_input("Model", key="shared_model", placeholder="deepseek-chat / qwen-plus / gpt-4.1")
            st.slider("Temperature", 0.0, 1.2, DEFAULT_TEMPERATURE, 0.05, key="shared_temperature")

        if not shared:
            render_model_inputs("company", "企业资料整理模型")
            render_model_inputs("jd", "JD 解析模型")
            render_model_inputs("diagnosis", "简历诊断模型")
            render_model_inputs("rewrite", "简历改写模型")
            render_model_inputs("review", "质检模型")

        st.divider()
        use_embedding = st.toggle("启用 Embedding 知识库检索", value=False)
        if use_embedding:
            render_model_inputs("embedding", "Embedding 模型", default_model="text-embedding-3-small")

    input_col, prompt_col = st.columns([1.05, 0.95], gap="large")

    with input_col:
        st.subheader("输入材料")
        resume_text = st.text_area("原始简历", height=260, placeholder="粘贴你的简历文本...")
        jd_text = st.text_area("目标 JD", height=220, placeholder="粘贴岗位描述...")
        raw_urls = st.text_area(
            "企业资料 URL",
            height=120,
            placeholder="每行一个链接，例如企业介绍、新闻报道、价值观、产品页、招聘页",
        )
        manual_company_text = st.text_area(
            "补充企业资料文本，可选",
            height=120,
            placeholder="如果网页无法抓取，可以把企业介绍或报道正文粘贴到这里。",
        )

    with prompt_col:
        st.subheader("Prompt 设置")
        prompts = {}
        for key, label in [
            ("company", "企业资料整理 Prompt"),
            ("jd", "JD 解析 Prompt"),
            ("diagnosis", "简历诊断 Prompt"),
            ("rewrite", "简历改写 Prompt"),
            ("review", "质检 Prompt"),
        ]:
            prompts[key] = st.text_area(label, value=DEFAULT_PROMPTS[key], height=190)

    run = st.button("开始分析并生成改写建议", type="primary", use_container_width=True)

    if not run:
        st.info("先填写输入材料、模型配置和 Prompt，然后点击开始分析。API Key 只在本次页面会话中使用，不会写入文件。")
        return

    if not resume_text.strip() or not jd_text.strip():
        st.error("原始简历和目标 JD 不能为空。")
        return

    shared_config = get_model_config("shared")
    configs = {
        "company": shared_config if shared else get_model_config("company", shared_config),
        "jd": shared_config if shared else get_model_config("jd", shared_config),
        "diagnosis": shared_config if shared else get_model_config("diagnosis", shared_config),
        "rewrite": shared_config if shared else get_model_config("rewrite", shared_config),
        "review": shared_config if shared else get_model_config("review", shared_config),
        "embedding": get_model_config("embedding", shared_config) if use_embedding else None,
    }

    pages = []
    urls = split_urls(raw_urls)
    progress = st.status("正在抓取和分析材料...", expanded=True)

    with progress:
        for url in urls:
            try:
                st.write(f"抓取页面：{url}")
                pages.append(fetch_page(url))
            except Exception as exc:
                st.warning(f"页面抓取失败：{url}，原因：{exc}")

        if manual_company_text.strip():
            pages.append(
                {
                    "url": "manual://company-notes",
                    "title": "手动补充企业资料",
                    "text": clean_text(manual_company_text),
                }
            )

        if not pages:
            st.warning("没有可用企业资料，将仅基于简历和 JD 生成建议。")
            pages.append({"url": "empty://company", "title": "无企业资料", "text": "用户未提供可用企业资料。"})

        try:
            st.write("调用模型生成企业画像、JD 画像、诊断和改写结果。")
            result = run_analysis(resume_text, jd_text, pages, prompts, configs, use_embedding)
            progress.update(label="分析完成", state="complete", expanded=False)
        except Exception as exc:
            progress.update(label="分析失败", state="error", expanded=True)
            st.error(f"模型调用或分析失败：{exc}")
            return

    st.subheader("分析结果")
    tabs = st.tabs(["企业知识", "JD 画像", "简历诊断", "改写结果", "质检"])

    with tabs[0]:
        st.markdown(result["company_profile"])
        with st.expander("用于生成的企业知识片段"):
            st.text(result["company_context"])

    with tabs[1]:
        st.markdown(result["jd_profile"])

    with tabs[2]:
        st.markdown(result["diagnosis"])

    with tabs[3]:
        st.markdown(result["rewrite"])
        st.download_button(
            "下载改写结果 Markdown",
            data=result["rewrite"],
            file_name="resume_rewrite_result.md",
            mime="text/markdown",
        )

    with tabs[4]:
        st.markdown(result["review"])
