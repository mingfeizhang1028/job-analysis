# AI_EDITING_GUIDE.md

## 1. 目标

本指南用于约束 AI 在 `job_dashboard` 项目中的阅读与修改行为，目标是：

- 减少无关文件读取
- 降低 token 消耗
- 提高定位速度
- 降低误改风险
- 保持项目结构稳定

---

## 2. 总体规则

### 2.1 先索引，后读代码
修改前优先读取：

1. `PROJECT_INDEX.md`
2. `AGENTS.md`
3. `module_specs/short/` 对应模块说明
4. 必要的目标代码文件

### 2.2 不默认全量扫描
除非用户明确要求，否则不要：

- 遍历所有 `.py` 文件内容
- 同时阅读多个无关页面
- 同时阅读多个无关模块
- 读取全部缓存、数据、输出文件

### 2.3 只做最小必要分析
每次任务优先回答：

- 要改什么
- 应该看哪些文件
- 不需要看哪些文件
- 影响范围是什么
- 推荐的最小改法是什么

---

## 3. 按任务类型的读取规则

### 3.1 页面修改
适用场景：
- 页面布局调整
- 组件展示调整
- 页面交互修改
- 页面筛选项/展示项修改

优先读取：
- `pages/目标页面.py`
- `utils/page_helpers.py`
- 必要的业务模块文件

不要默认读取：
- 全部 `pages/`
- 全部 `modules/`
- `data/llm_cache/`
- `lib/`

---

### 3.2 模块逻辑修改
适用场景：
- 算法调整
- 匹配逻辑修改
- 分析指标修改
- 标签提取规则修改

优先读取：
- `module_specs/short/对应模块.md`
- `modules/目标模块.py`
- 直接调用它的页面或上游模块

不要默认读取：
- 无关页面
- 无关模块
- 第三方静态资源

---

### 3.3 数据处理/分析修改
优先读取：
- `modules/data_loader.py`
- `modules/preprocess.py`
- `modules/normalization.py`
- `modules/keyword_analysis.py`
- `modules/tag_extraction.py`
- `modules/trait_analysis.py`
- `data/` 下相关规则或字典文件名

必要时再读取：
- 对应页面文件

---

### 3.4 LLM / 结构化 / Prompt 能力修改
优先读取：
- `modules/llm_client.py`
- `modules/llm_cache.py`
- `modules/llm_jd_structuring.py`
- `modules/llm_resume_structuring.py`
- `modules/llm_skill_extraction.py`
- `modules/ollama_runtime.py`

必要时再读取：
- 调用这些能力的分析模块
- 对应页面文件

---

### 3.5 候选人知识库修改
优先读取：
- `modules/candidate_kb_loader.py`
- `modules/candidate_kb_chunking.py`
- `modules/candidate_embedding.py`
- `modules/candidate_vector_store.py`
- `modules/candidate_evidence_retrieval.py`
- `modules/candidate_profile.py`
- `pages/candidate_kb_page.py`

不要默认读取：
- `data/candidate_kb/raw/*` 全部原始内容

---

### 3.6 人岗匹配 / 职业策略 / 简历生成
优先读取：
- `modules/job_resume_matching.py`
- `modules/career_fit_analysis.py`
- `modules/jd_query_builder.py`
- `modules/latex_resume_generator.py`
- `modules/latex_template_renderer.py`
- `pages/resume_match_page.py`
- `pages/career_strategy_page.py`
- `pages/latex_resume_page.py`

---

### 3.7 网络分析 / 图表展示
优先读取：
- `modules/network_analysis.py`
- `modules/network_viz.py`
- `modules/charts.py`
- `pages/network_page.py`
- `pages/dashboard_page.py`

---

## 4. 修改原则

### 4.1 优先增量修改
- 尽量在原模块中补充逻辑
- 不轻易新增平行重复模块
- 不轻易改变项目总体结构

### 4.2 优先复用已有模块
如果已有：
- 数据加载模块
- LLM 调用模块
- 缓存模块
- 标签分析模块
- 页面辅助函数

则优先复用，不重复造轮子。

### 4.3 页面只做编排
页面层负责：
- 输入收集
- 参数组织
- 调用模块
- 结果展示

复杂逻辑应下沉到 `modules/`。

### 4.4 utils 保持轻量
`utils/` 只放：
- 共用小函数
- 格式化/页面辅助逻辑

不要把核心业务算法放进 `utils/`。

---

## 5. 输出格式建议

每次给出结果时，建议按以下格式输出：

## 一、需求理解
- 任务属于什么类型
- 默认假设是什么

## 二、相关文件定位
- 优先看哪些文件
- 不需要先看哪些文件

## 三、落地方案
- 改哪些文件
- 每个文件改什么
- 最小改法是什么

## 四、风险与优化
- 可能影响哪些模块
- 是否需要补测试或补文档

## 五、下一步建议
- 建议先改哪一步
- 是否需要我继续读取具体文件并直接修改

---

## 6. 常见任务的最小入口

### 修改页面
- 先读：`pages/目标页面.py`

### 修改岗位画像/标签分析
- 先读：`module_specs/short/tag_extraction.md`
- 再读：`modules/tag_extraction.py`

### 修改关键词分析
- 先读：`module_specs/short/keyword_analysis.md`
- 再读：`modules/keyword_analysis.py`

### 修改网络关系分析
- 先读：`module_specs/short/network_analysis.md`
- 再读：`modules/network_analysis.py`

### 修改图表展示
- 先读：`module_specs/short/charts.md`
- 再读：`modules/charts.py`

### 修改去重逻辑
- 先读：`module_specs/short/deduplication.md`
- 再读：`modules/deduplication.py`

### 修改预处理/归一化
- 先读：`module_specs/short/preprocess.md` / `normalization.md`
- 再读：`modules/preprocess.py` / `modules/normalization.py`

---

## 7. 当前项目的管理建议

### 必做
1. 后续把 `app_5 - 副本.py` 重命名为 `app.py`
2. 新增模块后同步补 `module_specs/short/`
3. 新增页面后建议补一个页面级说明文档或索引条目

### 推荐做
1. 未来将 `modules/` 逐步按业务子目录拆分
2. 将候选人知识库、LLM、匹配分析、标签分析分别分组
3. 在 `pages/` 和 `modules/` 下各加一个简短 README

---

## 8. 禁止默认行为

- 不默认读取全部源码
- 不默认读取全部数据文件
- 不默认读取全部缓存文件
- 不默认做结构性重构
- 不默认改入口文件、配置文件和多个页面
- 不在未确认的情况下删除文件

---

## 9. 推荐协作口令

你之后可以直接这样下达任务：

```text
请先阅读 PROJECT_INDEX.md 和 AI_EDITING_GUIDE.md，
再根据任务类型只读取必要文件，不要全量扫描项目。
```

或者：

```text
任务类型：模块逻辑修改
目标：xxx
优先查看：xxx
不要查看：xxx
输出：最小修改方案 + 必要代码补丁
```
