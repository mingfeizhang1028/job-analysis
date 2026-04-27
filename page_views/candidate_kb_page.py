from __future__ import annotations

import streamlit as st

from modules.candidate_kb_loader import (
    SUPPORTED_DOC_TYPES,
    ingest_candidate_document,
    load_document_index,
)
from modules.candidate_kb_chunking import build_document_chunks
from modules.candidate_vector_store import upsert_chunks_to_vector_store, load_vector_store
from modules.resume_loader import supported_resume_types
from modules.llm_settings import get_task_llm_config, render_page_task_llm_settings


def render_candidate_kb_page(processing_options: dict):
    st.header("候选人资料库")
    st.caption("上传/解析简历、项目经历、作品集、个人背景资料，统一入库并建立语义检索索引。")

    with st.container(border=True):
        render_page_task_llm_settings(["kb_embedding"], title="本页模型配置", include_switches=False)

    col1, col2 = st.columns([1, 1])
    with col1:
        doc_type = st.selectbox("材料类型", SUPPORTED_DOC_TYPES, index=0)
        title = st.text_input("材料标题（可选）", value="")
        tags_text = st.text_input("标签（可选，使用逗号分隔）", value="")
        text_input = st.text_area("直接粘贴材料文本", height=180)
        uploaded_file = st.file_uploader(
            "或上传材料文件",
            type=supported_resume_types(),
            key="candidate_kb_upload",
            help="当前支持 txt / md / docx",
        )
        build_embedding = st.checkbox("入库后立即生成向量索引", value=True)

        if st.button("添加到知识库", type="primary"):
            tags = [x.strip() for x in tags_text.replace("，", ",").split(",") if x.strip()]
            ingest_result = ingest_candidate_document(
                text_input=text_input,
                uploaded_file=uploaded_file,
                doc_type=doc_type,
                title=title,
                tags=tags,
            )
            if not ingest_result.get("success"):
                st.error(ingest_result.get("error", "入库失败"))
            else:
                doc_id = ingest_result.get("doc_id")
                st.success(f"材料已入库：{doc_id}")
                chunk_result = build_document_chunks(doc_id)
                if chunk_result.get("success"):
                    st.info(f"已切分为 {chunk_result.get('chunk_count', 0)} 个片段")
                    if build_embedding:
                        embedding_cfg = get_task_llm_config("candidate_embedding")
                        with st.spinner("正在生成 embedding..."):
                            embed_result = upsert_chunks_to_vector_store(
                                chunk_result.get("chunks", []),
                                model=str(embedding_cfg.get("local_model") or processing_options.get("embedding_model", "qwen3-embedding:8b")),
                                ollama_url=str(embedding_cfg.get("local_url") or processing_options.get("embedding_url", "http://localhost:11434/api/embeddings")),
                            )
                        st.success(f"已写入向量库 {embed_result.get('inserted', 0)} 条")
                        if embed_result.get("errors"):
                            st.warning(f"部分片段 embedding 失败：{len(embed_result.get('errors', []))} 条")
                else:
                    st.warning(chunk_result.get("error", "切块失败"))

    with col2:
        index_data = load_document_index()
        st.subheader("已入库材料")
        if index_data:
            st.dataframe(index_data, use_container_width=True, hide_index=True)
        else:
            st.info("当前知识库还没有材料。")

        vector_count = len(load_vector_store())
        st.metric("向量片段数", vector_count)
