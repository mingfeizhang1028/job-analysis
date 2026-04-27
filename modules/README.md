# Modules 变更记录

- 修改文件：app.py
- 修改内容：新增 data/app_cache 处理结果缓存；基于原始数据签名 + 处理配置生成 cache key，优先加载 parquet 处理结果，避免 Streamlit 每次重启都重新全量执行 process_data。
- 影响范围：应用启动速度、跨重启复用处理结果。
- 后续建议：后续可升级为按 job_id/hash 的细粒度增量缓存，而不是仅按整表签名缓存。
