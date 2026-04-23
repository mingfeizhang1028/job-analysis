# PROJECT_INDEX.md

## 1. 项目概览

这是一个以 **求职分析 / 简历优化 / 人岗匹配 / 候选人知识库** 为核心的 Python 项目。

从目录结构判断，当前项目主要包含以下能力：

- 岗位数据加载与清洗
- JD 解析与标签分析
- 简历解析与人岗匹配
- 候选人知识库构建与检索
- 职业策略分析
- 网络关系分析与可视化
- LaTeX 简历生成
- 多页面仪表盘展示

项目当前结构基本清晰，采用了：

- `pages/`：页面层
- `modules/`：业务逻辑层
- `utils/`：辅助工具层
- `data/`：数据与缓存
- `module_specs/`：模块说明文档

---

## 2. 根目录结构说明

```text
app_5 - 副本.py              # 当前应用入口（建议后续重命名为 app.py）
pages/                       # 页面层
modules/                     # 核心业务逻辑
utils/                       # 通用辅助函数
data/                        # 数据、缓存、词典、候选人知识库
module_specs/                # 模块说明文档（short/full）
lib/                         # 前端静态依赖/第三方资源
outputs/                     # 输出结果目录
```

---

## 3. 目录职责索引

### 3.1 页面层：`pages/`

当前页面文件：

- `candidate_kb_page.py`：候选人知识库页面
- `career_strategy_page.py`：职业策略分析页面
- `dashboard_page.py`：主仪表盘页面
- `dedup_page.py`：去重分析页面
- `detail_page.py`：岗位/标签等详情页
- `job_profile_page.py`：岗位画像页面
- `latex_resume_page.py`：LaTeX 简历页面
- `network_page.py`：网络关系分析页面
- `resume_match_page.py`：简历匹配页面
- `tag_diagnosis_page.py`：标签诊断页面

**适合处理：**
- 页面布局
- 页面交互
- 页面展示逻辑
- 页面级调用入口

**不建议在这里堆积：**
- 大段数据处理
- LLM 调用细节
- 复杂匹配算法

---

### 3.2 业务逻辑层：`modules/`

按文件名可分为以下几个功能簇：

#### A. 候选人知识库相关
- `candidate_embedding.py`
- `candidate_evidence_retrieval.py`
- `candidate_kb_chunking.py`
- `candidate_kb_loader.py`
- `candidate_profile.py`
- `candidate_vector_store.py`

#### B. 人岗匹配 / 职业分析
- `career_fit_analysis.py`
- `job_resume_matching.py`
- `jd_query_builder.py`

#### C. LLM 能力相关
- `llm_cache.py`
- `llm_client.py`
- `llm_jd_structuring.py`
- `llm_resume_structuring.py`
- `llm_skill_extraction.py`
- `ollama_runtime.py`

#### D. 简历生成相关
- `latex_resume_generator.py`
- `latex_template_renderer.py`

#### E. 数据处理与分析
- `data_loader.py`
- `deduplication.py`
- `keyword_analysis.py`
- `normalization.py`
- `preprocess.py`
- `tag_extraction.py`
- `tag_merge.py`
- `trait_analysis.py`

#### F. 图表 / 网络分析
- `charts.py`
- `network_analysis.py`
- `network_viz.py`

#### G. 配置
- `config.py`
- `resume_loader.py`

**适合处理：**
- 业务规则
- 数据清洗与结构化
- 匹配与分析逻辑
- LLM 封装
- 输出结构生成

---

### 3.3 工具层：`utils/`

- `keyword_helpers.py`：关键词相关辅助函数
- `page_helpers.py`：页面辅助函数

**适合处理：**
- 通用小工具
- 页面共用辅助函数
- 非核心业务的复用逻辑

---

### 3.4 数据层：`data/`

主要包括：

#### 原始/业务数据
- `jobs.xlsx`

#### 词典 / 规则 / 标签体系
- `company_alias.json`
- `job_title_rules.json`
- `llm_skill_alias.json`
- `tag_dict.json`
- `trait_dict.json`
- `stopwords.txt`
- `stopwords_analysis.txt`
- `user_dict.txt`

#### LLM 缓存
- `llm_cache/jd_struct_cache.json`
- `llm_cache/resume_struct_cache.json`
- `llm_cache/skill_cache.json`

#### 候选人知识库
- `candidate_kb/raw/`
- `candidate_kb/parsed/`

**适合处理：**
- 数据文件维护
- 缓存管理
- 字典/规则维护
- 候选人语料管理

---

### 3.5 文档层：`module_specs/`

- `short/`：模块摘要说明，适合 AI 快速理解
- `full/`：模块完整说明，适合深入排查

当前已存在模块说明：

- `app`
- `charts`
- `config`
- `data_loader`
- `deduplication`
- `keyword_analysis`
- `network_analysis`
- `network_viz`
- `normalization`
- `preprocess`
- `tag_extraction`
- `trait_analysis`

**优先建议：**
当不确定模块实现时，先读 `module_specs/short/`，再决定是否读源码。

---

## 4. 按任务类型的最小读取路径

### 4.1 改页面展示
优先看：
- `pages/目标页面.py`
- `utils/page_helpers.py`
- 必要时再看对应 `modules/`

通常不需要先看：
- `data/llm_cache/`
- `lib/`
- 全部 `modules/`

---

### 4.2 改业务分析逻辑
优先看：
- `modules/对应功能模块.py`
- `module_specs/short/对应模块.md`
- 必要时再看上下游页面文件

通常不需要先看：
- 其他无关 page
- `lib/`

---

### 4.3 改 LLM / Prompt / 结构化能力
优先看：
- `modules/llm_client.py`
- `modules/llm_cache.py`
- `modules/llm_jd_structuring.py`
- `modules/llm_resume_structuring.py`
- `modules/llm_skill_extraction.py`
- `modules/ollama_runtime.py`

必要时再看：
- 调用这些模块的页面或分析模块

---

### 4.4 改人岗匹配 / 职业策略
优先看：
- `modules/job_resume_matching.py`
- `modules/career_fit_analysis.py`
- `modules/jd_query_builder.py`
- `pages/resume_match_page.py`
- `pages/career_strategy_page.py`

---

### 4.5 改候选人知识库
优先看：
- `modules/candidate_kb_loader.py`
- `modules/candidate_kb_chunking.py`
- `modules/candidate_embedding.py`
- `modules/candidate_vector_store.py`
- `modules/candidate_evidence_retrieval.py`
- `modules/candidate_profile.py`
- `pages/candidate_kb_page.py`

---

### 4.6 改标签/关键词/画像分析
优先看：
- `modules/tag_extraction.py`
- `modules/tag_merge.py`
- `modules/keyword_analysis.py`
- `modules/trait_analysis.py`
- `modules/normalization.py`
- `modules/preprocess.py`
- `pages/tag_diagnosis_page.py`
- `pages/job_profile_page.py`

---

### 4.7 改网络图 / 图表可视化
优先看：
- `modules/network_analysis.py`
- `modules/network_viz.py`
- `modules/charts.py`
- `pages/network_page.py`
- `pages/dashboard_page.py`

---

## 5. 不建议默认读取的内容

以下内容通常不作为首次阅读目标：

- `data/llm_cache/*`：缓存文件，除非排查缓存问题
- `data/candidate_kb/raw/*`：原始语料较大，除非排查数据源问题
- `lib/*`：第三方静态资源，通常无需改动
- `outputs/*`：输出目录，通常不是逻辑入口
- `module_specs/full/*`：只有在 short 不足时再读

---

## 6. 当前结构上的关键观察

### 优点
- 目录职责已经比较清晰
- 页面与业务逻辑已分层
- 有独立 `module_specs` 文档，适合 AI 协作
- LLM、匹配、标签、网络分析等能力已经按模块拆开

### 目前可优化点
- 根入口文件命名不规范：`app_5 - 副本.py`
- `modules/` 中文件数量较多，后续可按业务子域分组
- `pages/` 页面较多，后续建议补充页面级说明文档
- `module_specs/` 目前没有覆盖所有新模块

---

## 7. 推荐后续维护规则

1. 新增模块时，优先放入已有职责目录。
2. 新增业务模块后，补充 `module_specs/short/` 摘要说明。
3. 新增页面时，尽量保证“页面仅编排，业务逻辑下沉到 modules”。
4. 尽量避免在单个页面文件内堆积大量分析逻辑。
5. 后续建议把 `modules/` 分组为子目录，但先不要大规模重构。

---

## 8. AI 使用建议

今后让 AI 修改代码时，建议先执行以下步骤：

1. 先读 `PROJECT_INDEX.md`
2. 再读 `AGENTS.md`
3. 根据任务类型选择：
   - 页面任务 → `pages/`
   - 业务逻辑任务 → `modules/`
   - 数据任务 → `data/`
   - 不确定实现 → 先读 `module_specs/short/`
4. 只读必要文件，避免全量扫描
