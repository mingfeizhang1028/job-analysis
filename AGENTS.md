# AGENTS.md

## 1. 协作目标

本项目是一个 **求职分析 / 简历优化 / 人岗匹配 / 候选人知识库** 项目。

AI 在本项目中的主要职责：

- 快速定位相关模块
- 避免无关文件全量阅读
- 优先基于现有结构做增量修改
- 输出最小可执行改法
- 保持页面层与业务层边界清晰

---

## 2. 修改前必读顺序

任何修改任务开始前，优先按以下顺序获取上下文：

1. `PROJECT_INDEX.md`
2. `AGENTS.md`
3. 对应的 `module_specs/short/*.md`
4. 目标目录下的相关代码文件

**不要默认全量扫描整个项目。**

---

## 3. 目录职责约定

### 页面层
- 目录：`pages/`
- 职责：页面展示、交互编排、调用业务模块
- 禁止：在页面里堆积复杂分析逻辑、LLM 调用细节、大型数据处理

### 业务逻辑层
- 目录：`modules/`
- 职责：分析逻辑、匹配逻辑、知识库处理、LLM 封装、生成逻辑
- 要求：尽量按功能聚合，减少跨文件重复实现

### 工具层
- 目录：`utils/`
- 职责：轻量通用函数、页面辅助函数、关键词辅助函数
- 禁止：把核心业务塞入 utils

### 数据层
- 目录：`data/`
- 职责：输入数据、缓存、字典、规则、候选人知识库语料
- 禁止：将逻辑代码混入 data 目录

### 模块说明层
- 目录：`module_specs/`
- 职责：为 AI 和开发者提供模块说明
- 规则：不确定实现时，先看 `short/`，不够再看 `full/`

---

## 4. 按任务类型的最小读取策略

### 4.1 页面修改
优先读取：
- `pages/目标页面.py`
- `utils/page_helpers.py`
- 必要的对应 `modules/*.py`

一般不要先读：
- `data/llm_cache/*`
- `lib/*`
- 无关页面
- 全部 modules

### 4.2 模块逻辑修改
优先读取：
- `module_specs/short/对应模块.md`
- `modules/对应模块.py`
- 必要时读取调用它的 `pages/*.py`

### 4.3 数据处理/分析修改
优先读取：
- `modules/data_loader.py`
- `modules/preprocess.py`
- `modules/normalization.py`
- `modules/tag_extraction.py`
- `modules/keyword_analysis.py`
- `data/` 中相关字典/规则文件名

### 4.4 AI / LLM 能力修改
优先读取：
- `modules/llm_client.py`
- `modules/llm_cache.py`
- `modules/llm_jd_structuring.py`
- `modules/llm_resume_structuring.py`
- `modules/llm_skill_extraction.py`
- `modules/ollama_runtime.py`

### 4.5 人岗匹配 / 简历优化 / 职业策略
优先读取：
- `modules/job_resume_matching.py`
- `modules/career_fit_analysis.py`
- `modules/jd_query_builder.py`
- `pages/resume_match_page.py`
- `pages/career_strategy_page.py`

### 4.6 候选人知识库相关
优先读取：
- `modules/candidate_kb_loader.py`
- `modules/candidate_kb_chunking.py`
- `modules/candidate_embedding.py`
- `modules/candidate_vector_store.py`
- `modules/candidate_evidence_retrieval.py`
- `modules/candidate_profile.py`
- `pages/candidate_kb_page.py`

---

## 5. 默认工作方式

收到任务后，按以下顺序处理：

1. 先判断任务类型：
   - 页面修改
   - 模块逻辑修改
   - 数据处理修改
   - Prompt / LLM 改造
   - 架构整理
   - Bug 排查
   - 新功能设计

2. 只读取必要文件。

3. 优先给出：
   - 要改什么
   - 改哪些文件
   - 不需要看哪些文件
   - 最小修改路径
   - 影响范围

4. 默认不做大重构，除非明确要求。

---

## 6. 修改原则

- 优先增量修改，不推翻现有结构
- 优先改已有模块，不重复新增类似模块
- 优先复用已有数据结构和函数
- 页面尽量只做编排，不做重业务计算
- 数据处理逻辑放 `modules/`
- 小型复用函数放 `utils/`
- 缓存逻辑统一走现有 LLM 缓存模块
- 不主动改动 `lib/` 第三方静态资源
- 不主动改动 `outputs/` 产物目录

---

## 7. 输出要求

每次给出修改建议或代码实现时，尽量包含：

### 必须说明
- 修改文件
- 每个文件改什么
- 为什么这么改
- 影响范围

### 若涉及代码实现
尽量给出：
- 函数级建议
- 补丁思路
- 最小可行版本

### 若涉及新功能设计
尽量给出：
- 目标
- 输入/输出
- 模块改动点
- 页面改动点
- MVP 做法

---

## 8. 禁止事项

- 不要默认读取整个项目所有文件
- 不要一次性大规模重命名或迁移文件
- 不要在页面文件里新增大量分析逻辑
- 不要把核心业务随意塞进 `utils/`
- 不要忽略已有 `module_specs/short/`
- 不要无依据创建大量新文档和新目录

---

## 9. 推荐提问方式

为了减少 token 消耗，建议今后发需求时使用如下格式：

```text
任务类型：页面修改 / 模块修改 / 数据处理 / LLM 改造 / Bug 排查 / 新功能设计
目标：xxx
优先查看：xxx
不要查看：xxx
输出方式：方案 / 代码补丁 / 函数设计 / 文档
```

例如：

```text
任务类型：模块修改
目标：优化简历和 JD 的匹配逻辑
优先查看：modules/job_resume_matching.py, modules/jd_query_builder.py
不要查看：pages/network_page.py, lib/
输出方式：先给最小修改方案，再给代码补丁
```

---

## 10. 后续维护建议

当新增模块或页面时，建议同步维护：

- `PROJECT_INDEX.md`
- 对应 `module_specs/short/*.md`

这样后续 AI 才能继续低成本定位上下文。
