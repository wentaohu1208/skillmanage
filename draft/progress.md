# Progress Log

## Session 1 — 2026-03-21

### 完成的工作

**框架设计（全部完成）**
- [x] 确定核心理念：Skill bank as a Living Ecosystem
- [x] 设计完整lifecycle：Acquisition → Active → Archive → Forgotten
- [x] 去掉Staging，简化为Acquisition三道过滤直接进Active
- [x] 设计增量式Alignment替代批量式
- [x] 设计失败学习机制（从失败轨迹提取warnings）
- [x] 设计Importance Score四维公式
- [x] 设计Compression三优先级（Merge → Distill → Forgetting）
- [x] 设计Archive召回机制
- [x] 设计Forgotten去重黑名单

**数据集调研（全部完成）**
- [x] 确定4个benchmark：ALFWorld, WebShop, MATH, BBH
- [x] 调研数据集分布、数量、metric
- [x] 确定实验分配：哪个实验用哪个数据集
- [x] 确定Non-Stationary实验的phase设计

**参考代码库调研（全部完成）**
- [x] 调研10个相关仓库
- [x] 确定复用策略
- [x] 记录关键文件和参考点

**实验设计（全部完成）**
- [x] 6组实验设计
- [x] Baseline确定
- [x] 多seed shuffle策略
- [x] 模型选择：Qwen2.5-7B-Instruct + GPT-4o-mini

**决策记录**
- [x] 去掉Staging
- [x] Skill Bank = Active
- [x] 增量式Alignment
- [x] 失败学习warnings简化版
- [x] 存储方案：实验用内存+JSON，正式用ChromaDB

**文档输出**
- [x] framework.txt — 整体框架总览
- [x] acquisition.txt — Acquisition详解
- [x] active.txt — Active Area详解
- [x] archive.txt — Archive详解
- [x] forgotten.txt — Forgotten详解
- [x] reference_repos.txt — 参考代码库调研
- [x] dataset.txt — 数据集信息

### Session 2 完成的工作

**Phase 1-4 代码实现（全部完成）**
- [x] Phase 1: 基础架构 — config, core/models, skill_bank, retrieval, storage, embedding, llm, utils（15个文件）
- [x] Phase 2: Acquisition — collector, segmentation, alignment, formalization, failure_learning（5个文件）
- [x] Phase 3: Active管理 — meta_updater, compression, importance, forgetting, manager（5个文件）
- [x] Phase 4: Archive+Forgotten — archive_manager, forgotten_manager（2个文件）
- [x] 基础导入和功能测试通过

**代码统计**: 27个Python文件, ~3000行

**架构决策**:
- LLM抽象: ABC + Factory/Registry, OpenAI兼容接口
- 存储: 内存dict + JSON/npz dump
- Embedding: sentence-transformers懒加载, 默认all-MiniLM-L6-v2
- JSON解析: 三级fallback（直接parse → code block → brace提取）

### Session 3 完成的工作

**Phase 5 Benchmark层（MATH完成）**
- [x] benchmark/base.py — 抽象层: Benchmark/InteractiveBenchmark ABC, TaskInstance, TaskResult, Registry
- [x] benchmark/prompts.py — 4个benchmark的prompt模板
- [x] benchmark/math_bench.py — MATH完整实现:
  - 数据加载: 7科目从HuggingFace加载，标task_type
  - 答案提取: extract_boxed_answer() 处理嵌套大括号
  - 答案比较: is_equiv() 三级比较（字符串→数值→sympy）
  - 全部测试通过
- [x] benchmark/runner.py — AgentRunner完整编排:
  - run_task() 单task lifecycle
  - run_stream() 顺序stream + 定期eval + checkpoint
  - evaluate() 冻结评测
  - 成功/失败Acquisition pipeline集成

### 待开始的工作
- [ ] Phase 5 remaining: BBH, ALFWorld, WebShop benchmark实现
- [ ] Phase 5: 端到端跑通测试（需要LLM接口）
- [ ] Phase 6: 实验运行
- [ ] Phase 7: 分析与论文

### 当前阻塞项
- 端到端测试需要配置LLM（本地vLLM或API key）
- ALFWorld/WebShop需要安装环境

---

## 超参数汇总（当前设定）

| 参数 | 值 | 所属模块 |
|------|-----|---------|
| M (最少记录数) | 3 | Acquisition |
| r (最低复现率) | 0.5 | Acquisition |
| P (失败分析步数阈值) | 2步 | Acquisition |
| K (检索top-K) | 3 | Active |
| threshold (检索最低sim) | 0.3~0.5 | Active |
| budget (token预算) | 实验确定 | Active |
| τ_merge (Merge阈值) | 0.8 | Active |
| θ_archive (降级阈值) | 0.3 | Active |
| T1 (连续低于阈值轮数) | 3 | Active |
| w1,w2,w3,w4 (权重) | 0.3,0.15,0.2,0.35 | Active |
| λ (Recency衰减) | 0.1 | Active |
| N_maintain (维护间隔) | 100 | Active |
| R (召回成功次数) | 2 | Archive |
| max_archive_inactive | 50 | Archive |
| τ (去重sim阈值) | 0.8 | Forgotten |
| D (时间衰减轮数) | 100 | Forgotten |
