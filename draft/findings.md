# Findings & Research Notes

## 框架设计发现

### Skill Bank规模存在最优点
- Phase Transition论文（arXiv:2601.04748）：skill selection accuracy在library size超过临界值后骤降
- SkillsBench（arXiv:2602.12670）：2-3个focused skill（+18.6pp）优于4+个skill（+5.9pp）
- 综合文档型skill反而有害（-2.9pp）
- **结论**：需要token预算控制，通过实验找sweet spot

### 现有方法的共同缺陷
- Voyager：只增不减，skill bank无限膨胀
- SkillRL：有分层但没有遗忘和质量门控
- SAGE：skill库每条chain独立，不跨链累积
- AutoSkill：只有Add/Merge/Discard，没有lifecycle
- **Gap**：没有人做完整的skill lifecycle management

### Acquisition设计演化
- 初始：批量式（攒够N条再Alignment）→ 空窗期问题
- 改进：增量式（每来一条更新PatternBuffer）→ 达到M条就提取
- 额外：失败学习（从失败轨迹提取warnings附加到skill）

### Staging Area去除
- 原设计：Acquisition → Staging（试用验证）→ Active
- 问题：Acquisition的三道过滤已经做了质量验证，Staging重复
- 简化：Acquisition（自带验证）→ 直接进Active
- Skill Bank = Active，概念更清晰

### 存储方案
- 实验阶段：方案3（内存dict + 定期dump到JSON/npz）
- 正式部署：方案2（ChromaDB/FAISS + JSON）

---

## 数据集调研

### 四个Benchmark

| 数据集 | Train | Test | 类别数 | Metric | 交互方式 |
|--------|-------|------|--------|--------|---------|
| ALFWorld | 3,553 | 134 | 6种任务类型 | SR | 多步交互 |
| WebShop | ~11,587 | 500 | 按商品类别 | Avg+SR | 多步交互 |
| MATH | 7,500 | 5,000 | 7类×5难度 | EM Acc | 单轮CoT |
| BBH | 无train | 6,511 | 23种推理类型 | EM Acc | 单轮CoT |

### ALFWorld任务类型分布
| 类型 | Train | Test |
|------|-------|------|
| pick_and_place | 790 | 24 |
| look_at_obj_in_light | 308 | 18 |
| pick_clean_then_place | 650 | 31 |
| pick_heat_then_place | 459 | 23 |
| pick_cool_then_place | 533 | 21 |
| pick_two_obj_and_place | 813 | 17 |

### MATH类别分布
| 类别 | Train | Test |
|------|-------|------|
| Algebra | 1,744 | 1,187 |
| Intermediate Algebra | 1,295 | 903 |
| Prealgebra | 1,205 | 871 |
| Number Theory | 869 | 540 |
| Geometry | 870 | 479 |
| Precalculus | 746 | 546 |
| Counting & Probability | 771 | 474 |

### BBH无训练集
- 27个子任务（归为23种），每种约250题
- 需要自划分：70% task stream / 30% test
- 标准评测用3-shot CoT prompting

### 下载方式
- BBH: `load_dataset("lukaemon/bbh")`
- MATH: `load_dataset("hendrycks/competition_math")`
- ALFWorld: `pip install alfworld[full]` + `alfworld-download`
- WebShop: GitHub clone + `./setup.sh -d all`

---

## 参考代码库

### 最值得参考 Top 5

| 仓库 | Stars | 关键文件 | 复用什么 |
|------|-------|---------|---------|
| SkillRL | 472 | skills_only_memory.py, skill_updater.py | 双模式检索、失败evolution |
| Voyager | 6,756 | agents/skill.py | ChromaDB存储、add/retrieve API |
| AutoSkill | 175 | models.py, merge.py, retrieval.py | Lifecycle状态机、混合检索 |
| ExpeL | 208 | expel.py, insight_extraction.py | 双池设计、ADD/EDIT/REMOVE |
| MemEngine | 107 | BaseMemory.py, Forget.py | 模块化架构 |

### 复用策略
- 环境交互层：ReAct/SkillRL
- Skill数据结构：SkillRL格式 + 扩展warnings
- 存储：Voyager模式（内存+JSON）
- 检索：SkillRL双模式
- Acquisition：ExpeL + SkillRL
- Compression：AutoSkill的merge.py
- Forgetting：MemEngine + AutoSkill
- 整体架构：MemEngine模块化设计

---

## 相关论文

### Skill Bank规模与组织
- AgentSkillOS (arXiv:2603.02176): capability-tree + DAG, 90K+ skill
- Phase Transition (arXiv:2601.04748): skill selection accuracy骤降
- Agent Skills Survey (arXiv:2602.12430): skill生态系统综述

### Skill获取与演化
- SkillRL (arXiv:2602.08234): 递归skill演化
- SAGE (arXiv:2512.17102): GRPO强化skill使用策略
- EXIF (arXiv:2506.04287): exploration-first自动skill发现
- AutoSkill (arXiv:2603.01145): Add/Merge/Discard三操作管理

### Skill抽象与压缩
- PolySkill (arXiv:2510.15863): 多态抽象, 1.7x reuse提升
- SkillsBench (arXiv:2602.12670): compact skill >> comprehensive skill

### Agent Memory系统
- Memory Survey (arXiv:2512.13564): Forms/Functions/Dynamics分类
- AgeMem (arXiv:2601.01885): RL学习记忆管理策略
- MemEngine (arXiv:2505.02099): 模块化记忆框架

### 推理Skill相关
- Buffer of Thoughts (2024): thought-template库
- LEGO-Prover (NeurIPS 2023): 渐进式lemma库
- ExpeL (2023): 经验规则积累
