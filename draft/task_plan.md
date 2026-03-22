# Skill Lifecycle Management Framework — Implementation Plan

## Project Overview
构建一个完整的Skill Lifecycle Management框架，实现skill的自动提取、管理、压缩、遗忘和召回，在4个benchmark（ALFWorld, WebShop, MATH, BBH）上验证。

**核心理念**: Skill bank不是仓库，而是生态系统。通过Acquisition → Active → Archive → Forgotten的生命周期管理，维持高质量、可控规模的skill bank。

**Base Model**: Qwen2.5-7B-Instruct（主实验待定）+ GPT +gemini等模型（api）

---

## Phase 1: 基础架构搭建
**状态**: [x] 完成
**实现**: src/skillmanage/config.py, core/, utils/, llm/

### 1.1 项目初始化
- [x] 创建项目结构（src/skillmanage/）— 6个子包: core, llm, acquisition, active, archive, utils
- [x] 设置config系统（config.py）— 7个frozen dataclass, 16个超参数

### 1.2 Skill数据结构（core/models.py ~300行）
- [x] Skill, SkillMeta, ActiveSkill — 带to_dict()/from_dict()序列化
- [x] ArchivedSkill, ForgottenSkill
- [x] PatternBuffer, PatternEntry — 增量式alignment用
- [x] Segment, SegmentedTrajectory — 切分结果
- [x] CollectionDecision — acquisition路由结果
- [x] SkillMeta.update_after_use() — running average更新
- [x] Skill.to_prompt_str() — 格式化为agent prompt

### 1.3 SkillBank核心（core/skill_bank.py ~300行）
- [x] 三dict存储: active/archive/forgotten + 各自的embedding dict
- [x] Active操作: add/remove/get + embeddings_matrix + total_tokens + is_over_budget
- [x] Archive操作: add/remove/get + embeddings_matrix
- [x] Forgotten操作: add + embeddings_matrix
- [x] Lifecycle转换: move_active_to_archive, promote_archive_to_active, move_archive_to_forgotten
- [x] max_active_similarity() — 用于Irreplaceability计算

### 1.4 检索（core/retrieval.py ~150行）
- [x] SkillRetriever.retrieve() — Active top-K + Archive fallback
- [x] batch_cosine_similarity + threshold过滤 + top-K排序
- [x] format_skills_for_prompt() — 格式化检索结果

### 1.5 存储（core/storage.py ~170行）
- [x] save_checkpoint() — active/archive/forgotten JSON + embeddings npz + pattern_buffers JSON + meta
- [x] load_checkpoint() — 完整恢复

### 1.6 Embedding（core/embedding.py ~80行）
- [x] EmbeddingModel — 懒加载sentence-transformers
- [x] encode(), encode_skill(), encode_task()
- [x] 默认模型: all-MiniLM-L6-v2 (dim=384)

### 1.7 LLM抽象层（llm/）
- [x] BaseLLMClient ABC — generate() + generate_json()
- [x] Factory/Registry pattern: register_llm_provider + create_llm_client
- [x] OpenAILLMClient — 兼容vLLM/GPT/任何OpenAI-compatible API
- [x] JSON解析: 直接parse → code block提取 → brace提取（三级fallback）
- [x] 所有Prompt模板（llm/prompts.py）— 8个模板

### 1.8 工具函数（utils/）
- [x] generate_skill_id() — UUID-based
- [x] count_tokens() — tiktoken优先, word-based fallback
- [x] cosine_similarity() + batch_cosine_similarity() — numpy实现

**策略选择**:
- 存储: 内存dict + 定期JSON/npz dump（实验阶段）
- Embedding: all-MiniLM-L6-v2（轻量，SkillRL用的Qwen3-Embedding也支持切换）
- LLM: OpenAI兼容接口（支持vLLM本地Qwen、GPT API、Gemini适配）
- 序列化: 每个dataclass自带to_dict/from_dict, 不依赖pickle

---

## Phase 2: Acquisition模块
**状态**: [x] 完成
**实现**: src/skillmanage/acquisition/

### 2.1 收集判断（collector.py ~150行）
- [x] CollectionDecider.decide() — 路由成功/失败
- [x] 成功路径: 裸跑→收集整条 | 用了skill→LLM判断覆盖率→按rate收集
- [x] 失败路径: step数 < P→跳过 | ≥P→分析
- [x] _judge_coverage() — LLM判断 + JSON解析 + 失败fallback

### 2.2 Segmentation（segmentation.py ~100行）
- [x] Segmenter.segment() — 按task_type自动选择interactive/reasoning
- [x] INTERACTIVE_TASK_TYPES集合（ALFWorld 6种 + webshop）
- [x] LLM切分 + JSON解析 + 失败fallback为单segment

### 2.3 Incremental Alignment（alignment.py ~250行）
- [x] PatternBufferManager — 管理所有task_type的PatternBuffer
- [x] add_record() — 语义匹配 → 更新count → 返回extraction candidates
- [x] check_extraction_candidates() — M + r + 未提取 + 非通用 四条件
- [x] _match_to_existing() — embedding相似度匹配（threshold=0.7）
- [x] _is_cross_category_generic() — 跨类别通用性过滤（ratio>0.8）
- [x] find_variants() — 找相似变体用于参数化
- [x] mark_extracted(), get_confidence()

### 2.4 Formalization（formalization.py ~120行）
- [x] Formalizer.formalize() — LLM泛化 + Forgotten去重 + token_cost计算
- [x] 标准skill格式: name, description, precondition, parameters, steps, warnings, confidence, token_cost

### 2.5 失败学习（failure_learning.py ~150行）
- [x] FailureLearner.analyze_failure() — 三种结果:
  - WarningAttachment（附加到已有skill）
  - Skill（独立skill, confidence=0.5）
  - None（不可执行，丢弃）
- [x] _match_to_active_skill() — embedding匹配（threshold=0.5）
- [x] _create_skill_from_warning() — source="failure_analysis"

---

## Phase 3: Active Area管理模块
**状态**: [x] 完成
**实现**: src/skillmanage/active/

### 3.1 Meta更新（meta_updater.py ~50行）
- [x] MetaUpdater.update_after_task() — 根据used_skill_ids更新对应meta
- [x] SkillMeta.update_after_use() — running average reward

### 3.2 Compression — Merge（compression.py 上半部分）
- [x] _find_merge_candidates() — 全pair扫描, sim > τ_merge 且 同task_type
- [x] _execute_merge() — LLM合并 + 合并meta(加权平均) + 合并warnings + 新embedding

### 3.3 Compression — Distill（compression.py 下半部分）
- [x] _find_distill_candidates() — compressed==False, 按token_cost降序
- [x] _execute_distill() — LLM精炼, target=原cost/2, 更新compressed+version

### 3.4 Compression编排（compression.py ~300行）
- [x] Compressor.compress_if_needed() — 优先级: Merge → Distill → 信号Forgetting
- [x] CompressionReport: merges, distills, tokens_saved, still_over_budget

### 3.5 Importance Score（importance.py ~100行）
- [x] ImportanceCalculator.calculate_all() — 四维公式
- [x] Recency: exp(-λ * gap), Frequency: log归一化, Quality: rate*reward
- [x] Irreplaceability: 1 - max_sim (调用SkillBank.max_active_similarity)

### 3.6 Forgetting（forgetting.py ~150行）
- [x] check_natural_forgetting() — 更新low_importance_streak, 返回≥T1的skill IDs
- [x] force_forget() — 按importance排序取最低N个
- [x] execute_degradation() — LLM压缩summary + move_active_to_archive

### 3.7 ActiveManager编排（manager.py ~150行）
- [x] on_round_end() — importance → natural forgetting → compression → force forget
- [x] on_new_skill_added() — 检查预算, 触发compression
- [x] RoundReport: importance_scores, skills_archived, merges, distills, tokens_saved

---

## Phase 4: Archive + Forgotten模块
**状态**: [x] 完成
**实现**: src/skillmanage/archive/

### 4.1 Archive（archive_manager.py ~130行）
- [x] ArchiveManager(forgotten_manager) — 组合ForgottenManager
- [x] record_recall_result() — 成功→recall_count+1, 失败→重置inactive
- [x] Promote: recall_count ≥ R → promote_archive_to_active(重置meta)
- [x] tick_inactive() — 所有archive skill的inactive_rounds+1, ≥max→move to forgotten
- [x] 检索触发: 在SkillRetriever中实现（Active sim < threshold时fallback）

### 4.2 Forgotten（forgotten_manager.py ~80行）
- [x] ForgottenManager.check_dedup() — embedding匹配 + 时间衰减
- [x] 匹配到且 < D轮 → skip（True）
- [x] 匹配到且 ≥ D轮 → allow re-learn（False）
- [x] 未匹配到 → allow（False）

---

## Phase 5: 环境集成
**状态**: [~] 进行中（MATH完成，其余待做）
**实现**: src/skillmanage/benchmark/

### 5.0 Benchmark抽象层（已完成）
- [x] benchmark/base.py — Benchmark ABC + InteractiveBenchmark ABC + TaskInstance + TaskResult + Registry/Factory
- [x] benchmark/prompts.py — 4个benchmark的prompt模板（MATH/BBH/ALFWorld/WebShop）
- [x] benchmark/math_bench.py — MATH完整实现（加载+提取+比较+trajectory）
- [x] benchmark/runner.py — AgentRunner编排（检索→执行→验证→Acquisition→Active管理→Archive tick）
  - run_task(): 单task完整lifecycle
  - run_stream(): 顺序task stream + 定期评测 + checkpoint
  - evaluate(): 冻结skill bank评测
  - 成功路径: segment→alignment→formalize→add to Active
  - 失败路径: failure analysis→attach warning或create skill
- [x] 即插即用设计: Benchmark ABC + Registry + InteractionMode（SINGLE_TURN / MULTI_STEP）

### 5.1 ALFWorld集成

#### 5.1.1 环境搭建
- [x] `pip install alfworld[full]` + `alfworld-download`
- [x] 数据路径: `/data/hwt/alfworld_data` (ALFWORLD_DATA环境变量)

#### 5.1.2 Benchmark实现 (alfworld_bench.py) ✅
- [x] ALFWorldBenchmark(InteractiveBenchmark) — 实现多步交互接口
- [x] load_tasks(): 加载train(3553)/test(134 unseen)
- [x] build_system_prompt(): 包含可用命令列表 + skills
- [x] build_step_prompt(): 历史action + 当前observation
- [x] reset_env() / step(): 对接alfworld TextWorld环境
- [x] check_answer(): 基于环境done flag
- [x] extract_trajectory(): 提取Action步骤列表
- [x] _detect_task_type(): 从observation中检测6种任务类型
- [x] InteractiveBenchmark接口扩展: build_system_prompt + build_step_prompt

#### 5.1.3 Runner多步交互支持 ✅
- [x] runner.py _run_multi_step() 重构:
  - build_system_prompt(skills) 一次，每步 build_step_prompt(task, obs, history)
  - action取第一行（防止LLM输出多行）
  - max_steps从benchmark读取

#### 5.1.4 实验设计 (run_alfworld.py) ✅
- [x] 实验A: No-skill baseline (DeepSeek + Qwen) — 134 unseen直测
- [x] 实验B: With-skill lifecycle (DeepSeek + Qwen)
  - Phase 1: 500 train tasks积累skill bank (checkpoint每100轮)
  - Phase 2: 134 unseen test (冻结skill bank)
- [x] 报告格式: 整体SR + 分6种任务类型SR + skill bank状态
- [x] 输出: /data/hwt/skillmanage/experiments_alfworld/
- [ ] 实验C: Non-Stationary (Phase轮换, 待后续)

#### 5.1.5 数据分布
```
                    Train    Test(unseen)
pick_and_place       790       24
look_at_obj_in_light 308       18
pick_clean_then_place 650      31
pick_heat_then_place  459      23
pick_cool_then_place  533      21
pick_two_obj_and_place 813     17
Total               3,553     134
```

#### 5.1.6 评测协议
- 标准测试集: 134 unseen tasks (和所有paper一致)
- 最大步数: 30步
- 成功判定: 环境done=True
- 指标: SR整体 + SR分6类
- 对比: ReAct baseline(71%), ExpeL(54%), SkillRL(89.9%)

### 5.2 WebShop集成
- [ ] 安装webshop环境
- [ ] 实现agent loop
- [ ] 任务分类（LLM打标签或embedding聚类）
- [ ] reward和SR计算

### 5.3 MATH集成

#### 5.3.1 数据加载 ✅
- [x] 从HuggingFace加载: `load_dataset('EleutherAI/hendrycks_math', subject)`
- [x] 7个科目分别加载后合并: algebra(1744), counting_and_probability(771), geometry(870), intermediate_algebra(1295), number_theory(869), prealgebra(1205), precalculus(746)
- [x] 每条标记task_type = f"{subject}_level{N}" (如"algebra_level5"), 共35种
  - task_type不喂给agent，只用于PatternBuffer分桶和Non-Stationary phase构造
- [x] 原始字段: problem, level, type, solution → 映射到TaskInstance(task_id, instruction, task_type, ground_truth, metadata)
- [x] 支持按subjects和levels过滤
- [x] Train/Test通过split参数选择，limit参数控制数量
- **实现**: benchmark/math_bench.py MathBenchmark.load_tasks()

#### 5.3.2 答案提取（从solution中提取ground truth） ✅
- [x] extract_boxed_answer(text) → 提取最后一个\boxed{}的内容
  - 取最后一个（rfind）：中间可能有标记中间结果的boxed，最终答案在最后
  - 大括号嵌套匹配：depth计数器处理\boxed{\frac{1}{2}}等
- [x] 测试通过: 简单数值(64)、分数(\frac{1}{2})、元组((-3,4))、多个boxed(取最后)、无boxed(返回空)
- **实现**: benchmark/math_bench.py extract_boxed_answer()

#### 5.3.3 Agent输出解析 ✅
- [x] 从agent的CoT输出中提取\boxed{}内容（复用extract_boxed_answer）
- [x] Fallback: _fallback_extract_answer()尝试匹配"the answer is X"、"answer: X"、"= X"等模式
  - 从最后一行往前搜索
- [x] 提取失败 → 返回空字符串 → check_answer返回(False, 0.0)
- **实现**: benchmark/math_bench.py check_answer() + _fallback_extract_answer()

#### 5.3.4 答案比较（三级比较策略） ✅
- [x] 自实现三级比较（未直接复用lm-evaluation-harness，但逻辑等价）
- [x] Level 1: normalize_answer()归一化后字符串比较
  - \dfrac→\frac, \tfrac→\frac
  - 去掉\left \right \! \text{} \textbf{} \mathrm{}
  - 去掉末尾$ . , ; :
  - 多余空格合并
- [x] Level 2: _latex_to_number()数值比较
  - 直接float、\frac{a}{b}→a/b、简单分数a/b、\sqrt{x}→x^0.5
  - 容差1e-6
- [x] Level 3: _sympy_equiv()符号比较
  - parse_latex解析 → simplify(pred-gt)==0
  - sympy不可用时跳过（graceful降级）
- [x] 测试通过: "64"=="64", "0.5"=="\frac{1}{2}", "\frac{2}{4}"=="\frac{1}{2}", "3"!="4", "\dfrac{1}{3}"=="\frac{1}{3}"
- **实现**: benchmark/math_bench.py is_equiv() + normalize_answer() + _latex_to_number() + _sympy_equiv()

#### 5.3.5 Agent Loop（单轮，非交互） ✅
- [x] 完整流程在AgentRunner.run_task()中实现:
  1. SkillRetriever.retrieve() → top-K Active skill（+Archive fallback）
  2. MathBenchmark.build_prompt(task, skills_prompt) → CoT prompt
  3. LLMClient.generate(prompt, system_prompt) → agent输出
  4. MathBenchmark.check_answer() → success/reward
  5. MetaUpdater.update_after_task() → 更新已用skill的meta
  6. CollectionDecider → Segmenter → Alignment → Formalizer (成功路径)
  7. FailureLearner → attach warning或create skill (失败路径)
- [x] Metric: Exact Match Accuracy（evaluate()方法中计算）
- [x] Prompt模板: benchmark/prompts.py MATH_COT_PROMPT + MATH_SYSTEM_PROMPT
- **实现**: benchmark/runner.py AgentRunner

#### 5.3.6 Task Stream构造 — 未实现（Phase 6的实验runner负责）
- [ ] 普通实验: train 7500题随机shuffle
- [ ] Non-Stationary实验: 按phase轮换
  - Phase 1 (1875题): 80% algebra + 20% others
  - Phase 2 (1875题): 80% geometry + 20% others
  - Phase 3 (1875题): 80% number_theory + 20% others
  - Phase 4 (1875题): 80% algebra（回归，测Archive召回）
  - phase内部shuffle，phase顺序固定
- [ ] 每个实验跑3~5个不同random seed
- **注**: task stream构造属于实验设计，由Phase 6的实验runner实现，不在benchmark层

#### 5.3.7 评测 ✅
- [x] AgentRunner.evaluate(test_tasks) → 冻结skill bank评测，返回SR
- [x] AgentRunner.run_stream()内置定期评测: 每eval_interval轮在test上评测
- [x] 日志记录: recent_sr, active/archive/forgotten数量, active_tokens
- [ ] 待Phase 6: learning curve绘图、结果保存
- **实现**: benchmark/runner.py evaluate() + run_stream()

### 5.4 BBH集成
- [ ] 下载数据集（lukaemon/bbh）
- [ ] 自划分train(70%)/test(30%)
- [ ] 实现CoT生成 + 答案提取 + exact match验证
- [ ] 解析23种任务类型标签

### 5.5 LLM接口
- [ ] Qwen2.5-7B-Instruct本地部署（vLLM）
- [ ] OpenAI兼容API封装
- [ ] GPT-4o-mini API封装（补充实验）

**参考**: ReAct的alfworld.ipynb/WebShop.ipynb, SkillRL的environments/

---

## Phase 6: 实验运行
**状态**: [ ] 未开始

### 6.1 实验基础设施
- [ ] 实验runner（task stream → 定期evaluation）
- [ ] checkpoint机制（每N轮save skill bank状态）
- [ ] 多seed支持（3~5个random seed）
- [ ] 日志和metrics记录（skill bank大小、利用率、SR等）
- [ ] learning curve绘图

### 6.2 Baselines实现
- [ ] No Skill baseline
- [ ] Voyager baseline（只增不减）
- [ ] FIFO baseline（固定大小，先进先出）
- [ ] LRU baseline（固定大小，最近最少使用）

### 6.3 实验1: Scaling Analysis（WebShop + MATH）
- [ ] budget ∈ [500, 1000, 1500, 2000, 3000, 5000, ∞]
- [ ] 画performance-size曲线

### 6.4 实验2: Consolidation Ablation（WebShop）
- [ ] 有Acquisition过滤 vs 无过滤（Voyager式）

### 6.5 实验3: Compression Ablation（WebShop）
- [ ] Full vs Merge-Only vs Distill-Only vs No-Compress

### 6.6 实验4: Forgetting策略对比（ALFWorld）
- [ ] Importance-based vs LRU vs LFU vs Random vs No-Forget

### 6.7 实验5: 闭环 vs 单机制（ALFWorld + MATH）
- [ ] 完整lifecycle vs 单机制 vs baseline

### 6.8 实验6: Non-Stationary（ALFWorld + MATH + BBH）
- [ ] Phase轮换设计
- [ ] Archive召回效果验证

---

## Phase 7: 分析与论文
**状态**: [ ] 未开始

### 7.1 结果分析
- [ ] 主表：4个benchmark × 各方法的SR/Acc
- [ ] Scaling曲线图
- [ ] Learning curve图
- [ ] Skill bank dynamics图（Active/Archive/Forgotten数量变化）
- [ ] Ablation表
- [ ] Non-stationary phase图

### 7.2 论文撰写
- [ ] Abstract
- [ ] Introduction（motivation + contribution）
- [ ] Related Work
- [ ] Method（Framework描述）
- [ ] Experiments
- [ ] Analysis & Discussion
- [ ] Conclusion

---

## Key Decisions Log

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-21 | 去掉Staging，Acquisition三道过滤后直接进Active | Acquisition已做质量验证，Staging多余 |
| 2026-03-21 | Skill Bank = 只有Active | 概念清晰，瘦身明确 |
| 2026-03-21 | Archive不参与正常检索，只有Active检索不到时才搜 | 防止Archive干扰Active的检索质量 |
| 2026-03-21 | 增量式Alignment替代批量式 | 解决等待N条记录的空窗期问题 |
| 2026-03-21 | 从失败轨迹提取Warnings | 成功提取steps + 失败提取warnings互补 |
| 2026-03-21 | Warning简化版：当场决定，不用WarningBuffer | 能附加就附加，能转化就转化，否则丢弃 |
| 2026-03-21 | 失败分析阈值：执行步数≥2步 | 第1步就卡住的没有分析价值 |
| 2026-03-21 | 实验阶段用内存+JSON存储，正式用ChromaDB/FAISS | 实验快速迭代，正式持久化 |
| 2026-03-21 | 基础模型Qwen2.5-7B-Instruct | 和SkillRL一致，公平对比 |
