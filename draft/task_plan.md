# Skill Lifecycle Management Framework — Implementation Plan

## Project Overview
构建一个完整的Skill Lifecycle Management框架，实现skill的自动提取、管理、压缩、遗忘和召回，在ALFWorld和MATH上验证。

**核心理念**: Skill bank不是仓库，而是生态系统。通过Acquisition → Active → Archive → Forgotten的生命周期管理，维持高质量、可控规模的skill bank。

**Base Model**: Qwen2.5-7B-Instruct (local vLLM) + DeepSeek-chat (LLM Judge for MATH)

---

## Phase 1: 基础架构搭建 ✅
**状态**: 完成
**实现**: src/skillmanage/config.py, core/, utils/, llm/

- [x] config系统（7个frozen dataclass, 20+超参数）
- [x] Skill数据结构（core/models.py ~300行）
- [x] SkillBank核心（core/skill_bank.py ~300行）
- [x] 检索（core/retrieval.py ~150行）— Active top-K + Archive fallback + format_warnings_for_system
- [x] 存储（core/storage.py ~170行）— JSON/npz checkpoint
- [x] Embedding（core/embedding.py ~80行）— sentence-transformers懒加载
- [x] LLM抽象层（llm/）— OpenAI兼容 + JSON解析四级fallback（含LaTeX escape修复）
- [x] 工具函数（utils/）— UUID, token计数, cosine similarity

---

## Phase 2: Acquisition模块 ✅
**状态**: 完成
**实现**: src/skillmanage/acquisition/

- [x] 收集判断（collector.py）— 覆盖率判断路由
- [x] Segmentation（segmentation.py）— interactive/reasoning两种模式
- [x] Incremental Alignment（alignment.py）— PatternBuffer + 语义匹配 + 跨类别过滤
- [x] Formalization（formalization.py）— LLM泛化 + Forgotten去重
- [x] 失败学习（failure_learning.py）— Warning附加 + Skill修复诊断

---

## Phase 3: Active Area管理模块 ✅
**状态**: 完成
**实现**: src/skillmanage/active/

- [x] Meta更新（meta_updater.py）
- [x] Compression（compression.py）— Merge + Distill + CompressionReport(含merge_details/distill_details)
- [x] Importance Score（importance.py）— 四维公式 [0,1]
- [x] Forgetting（forgetting.py）— Natural(importance/quality_floor) + Forced, 返回Dict[str, str]带原因
- [x] ActiveManager编排（manager.py）— RoundReport(含archive_reasons/merge_details/distill_details)

---

## Phase 4: Archive + Forgotten模块 ✅
**状态**: 完成
**实现**: src/skillmanage/archive/

- [x] Archive（archive_manager.py）— 召回 + Promote + tick_inactive
- [x] Forgotten（forgotten_manager.py）— 去重 + 时间衰减

---

## Phase 5: 环境集成与实验脚本
**状态**: 进行中

### 5.1 MATH集成 ✅
- [x] benchmark/math_bench.py — 完整实现（加载+提取+三级比较: math-verify → string → LLM Judge）
- [x] run_math_qwen.py — Exp1(no-skill) + Exp2(with-skill)
- [x] V1实验已跑完（结果: skill帮倒忙 -2.4% train, -5.1% test）
- [x] V2修复已部署: threshold 0.3→0.5, quality_floor=0.35, max_warnings=5, prompt分离, 关闭failure_skill, skill repair
- [x] V2实验运行中

### 5.2 ALFWorld集成 🔧 当前进行中
- [x] benchmark/alfworld_bench.py — 多步交互实现
  - [x] admissible actions 注入 step prompt
  - [x] think: 拦截（返回 "OK." 不发给 env）
  - [x] Action: 前缀剥离
  - [x] 空 action 守卫
  - [x] 循环检测（3次重复终止）
  - [x] _won fallback = False（不误判 max_steps 超时为成功）
  - [x] task instruction 在每步 step prompt 中
- [x] run_alfworld_qwen.py — Exp A(no-skill) + Exp B(with-skill) + tracker集成
- [x] runner.py _run_multi_step — warnings 注入多步 system prompt
- [ ] **阻塞**: base_config.yaml 路径问题，需要跑 `alfworld-download` 或手动指定 config 路径
- [ ] 运行 Exp A (no-skill baseline)
- [ ] 运行 Exp B (with-skill lifecycle)

### 5.3 ExperimentTracker ✅
- [x] tracker.py — task_log.jsonl + lifecycle_log.jsonl（含 question 字段）
- [x] 即插即用: tracker=None 时零副作用
- [x] 覆盖所有生命周期事件: created/archived/forgotten/recalled/promoted/merged/distilled/repaired/warning

### 5.4 待做
- [ ] WebShop集成
- [ ] BBH集成

---

## Phase 6: 实验运行
**状态**: 部分开始

### 6.1 已完成
- [x] MATH V1 实验（No Skill: 48.7%, With Skill: 46.3%）
- [x] MATH V2 实验运行中
- [ ] ALFWorld baseline（阻塞于 config 路径）

### 6.2 Baselines（需实现）
- [ ] Voyager baseline（只增不减）
- [ ] FIFO baseline
- [ ] LRU baseline
- [ ] Random baseline

### 6.3 核心实验设计
同一套 acquisition，只变 management policy:

| 组 | Management | 证明什么 |
|---|---|---|
| No Skill | — | 下界 |
| Voyager | 只增不减 | lifecycle 必要性 |
| FIFO | 满了删最早 | 简单策略不够 |
| LRU | 删最久没用 | Recency 单维度不够 |
| **Ours** | 完整 lifecycle | 多维度管理优势 |

### 6.4 消融实验
- [ ] Ours - Compression
- [ ] Ours - Forgetting
- [ ] Ours - Archive
- [ ] Ours - Warning

### 6.5 Non-Stationary 实验（ALFWorld 最强 selling point）
- [ ] Phase 轮换: 按任务类型切换分布
- [ ] Phase 4 回归: 验证 Archive 召回能力

---

## Phase 7: 分析与论文
**状态**: 未开始

---

## Key Decisions Log

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-21 | 去掉Staging，直接进Active | Acquisition三道过滤已做质量验证 |
| 2026-03-21 | Archive不参与正常检索 | 防止Archive干扰Active检索质量 |
| 2026-03-21 | 增量式Alignment | 解决批量式等待空窗期 |
| 2026-03-22 | MATH task_type = subject（不含level） | 避免分桶过细导致PatternBuffer稀疏 |
| 2026-03-22 | 答案比较: math-verify → string → LLM Judge | 三级fallback覆盖各种LaTeX格式 |
| 2026-03-23 | V2: threshold 0.3→0.5 | V1 top-3 skill SR(42%) < baseline(48.7%)，检索太宽泛 |
| 2026-03-23 | V2: 关闭 failure skill | V1 failure skill 淘汰率77%，性价比低 |
| 2026-03-23 | V2: max_warnings=5 + LLM合并 | V1 最多43条warning，严重干扰 |
| 2026-03-23 | V2: quality_floor=0.35, min_calls=30 | V1 低质量skill靠frequency留在Active |
| 2026-03-23 | V2: Prompt分离(steps在user, warnings在system) | V1 混在一起干扰推理 |
| 2026-03-23 | V2: Skill Repair(诊断+LLM重写+重试max 3次) | V1 所有version=1，错误steps从未改进 |
| 2026-03-24 | ALFWorld: 加 admissible actions | SkillRL 和标准做法，防止无效命令 |
| 2026-03-24 | ALFWorld: 不加 few-shot | story 是 skill 替代 few-shot |
| 2026-03-24 | 先在 ALFWorld 证明 skill 有效 | 操作型 skill 比推理型更容易被 LLM 遵循 |
| 2026-03-24 | JSON解析加 LaTeX escape修复 | Qwen-7B 输出含 \frac 等破坏 JSON 转义 |

## 当前超参数

### MATH (V2)
| 参数 | 值 |
|------|-----|
| similarity_threshold | 0.5 |
| top_k | 3 |
| token_budget | 2000 |
| min_confidence | 0.3 |
| quality_floor | 0.35 |
| max_warnings | 5 |
| create_failure_skill | False |
| max_skill_retries | 3 |

### ALFWorld
| 参数 | 值 | 备注 |
|------|-----|------|
| temperature | 0.7 | 考虑改为 0 |
| max_tokens | 512 | |
| max_steps | 50 | 标准是49-50 |
| similarity_threshold | 0.3 | |
| top_k | 3 | SkillRL 用 6 |
| token_budget | 2000 | |
| train tasks | 500 | SkillRL 用 3553 (RL) |
| test tasks | 134 | 标准 |
