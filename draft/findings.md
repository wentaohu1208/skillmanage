# Findings & Research Notes

## V1 实验发现 (MATH Geometry L4-L5, Qwen-7B)

### 核心结果: Skill 帮倒忙
| | No Skill | With Skill | 差异 |
|---|:---:|:---:|:---:|
| Train SR (598题) | 48.7% | 46.3% | **-2.4%** |
| Test SR (257题) | 47.5% | 42.4% | **-5.1%** |

### 六大根因
1. **检索精度低** (threshold=0.3): Top-3 skill 占62%调用但SR仅42%，比baseline还低
2. **Warning堆积**: 平均11条/skill，最多43条，混在prompt里干扰推理
3. **Failure skill无价值**: 5个avoid_skill，SR=0.46，淘汰率77%
4. **低质量skill不被淘汰**: SR=41%的skill因高frequency保持在Active（马太效应）
5. **Skill description太宽泛**: "determine coordinates"匹配370次
6. **Skill未被修复**: 所有version=1

### V2 修复方案（已部署）
| 修改 | 值 |
|------|-----|
| similarity_threshold | 0.3 → 0.5 |
| quality_floor | 新增, sr<0.35 且 calls≥30 强制降级 |
| max_warnings | 新增, 5条上限 + LLM合并 |
| Prompt分离 | skills在user prompt, warnings在system prompt |
| create_failure_skill | True → False |
| Skill Repair | 新增, 诊断+重写+重试max 3 |

---

## Skill 在 MATH vs ALFWorld 上的根本差异

### MATH (推理型): Skill 可能无增量信息
- 模型训练数据里有海量数学题解，"用海伦公式" 是模型已知知识
- Skill = 重复模型已知信息 = 占prompt空间 + 分散注意力
- 归因困难: 做对了不知道是 skill 帮的还是模型本来就会

### ALFWorld (操作型): Skill 有真实信息增量
- 模型训练数据极少有 "go to fridge 1, take egg 1" 这种操作序列
- Skill = 具体操作策略 = 模型不一定自己能发明
- Warning 更具体: "加热后必须从微波炉取出" 而不是 "注意计算准确性"
- 策略: 先在 ALFWorld 证明系统有效，再考虑 MATH

---

## 与 SkillRL 的核心差异

| 维度 | SkillRL | 我们 |
|------|---------|------|
| Skill 管理 | 只增不减 (55→100) | 完整 lifecycle |
| Skill 来源 | teacher model (o3) 离线蒸馏 | agent 自身轨迹增量提取 |
| 获取时机 | 每个 validation epoch 批量 | 每个 task 后实时 |
| Policy 训练 | SFT + GRPO (改权重) | inference-time (不改权重) |
| 非稳态适应 | 无机制 | Archive 召回 |
| 检索 | 关键词分类匹配 | embedding 语义检索 |

**定位**: 我们不是"更好的SkillRL"，而是"SkillRL缺失的那一块"——lifecycle management。

---

## 公平对比设计

### 必须证明两步
1. **Skill 有正面价值**: With Skill + Lifecycle > No Skill
2. **Lifecycle 有用**: With Skill + Lifecycle > With Skill (Voyager/LRU等)

### 核心实验: 控制 acquisition，只变 management
| 组 | 证明什么 |
|---|---|
| No Skill | 下界 |
| Voyager (只增不减) | lifecycle 必要性 |
| FIFO / LRU / Random | 简单策略不够 |
| **Ours (完整lifecycle)** | 多维度管理优势 |

### 杀手实验: Non-Stationary
Phase 1→2→3→4(回归) 任务分布轮换，Phase 4 只有 Ours 的 Archive 能快速恢复

---

## ALFWorld 参考实现对比

| | ReAct | Reflexion | SkillRL | 我们 |
|---|:---:|:---:|:---:|:---:|
| Temperature | 0 | 0 | 0.4 | 0.7 (考虑改0) |
| Max steps | 50 | 49 | 50 | 50 |
| Few-shot | 2/type | 3/type | 无 | 无 |
| Admissible actions | 否 | 否 | 是 | 是 |
| Think拦截 | 是 | 是 | XML标签 | 是 |
| Success Rate | 71% | 97% (12 trials) | 89.9% | ? |

---

## 相关论文

### Skill Bank规模与组织
- AgentSkillOS (arXiv:2603.02176): capability-tree + DAG, 90K+ skill
- Phase Transition (arXiv:2601.04748): skill selection accuracy骤降
- Agent Skills Survey (arXiv:2602.12430): skill生态系统综述

### Skill获取与演化
- SkillRL (arXiv:2602.08234): 递归skill演化, SFT+GRPO
- SAGE (arXiv:2512.17102): GRPO强化skill使用策略
- AutoSkill (arXiv:2603.01145): Add/Merge/Discard三操作管理

### Skill抽象与压缩
- PolySkill (arXiv:2510.15863): 多态抽象, 1.7x reuse提升
- SkillsBench (arXiv:2602.12670): compact skill >> comprehensive skill

### Agent Memory系统
- Memory Survey (arXiv:2512.13564): Forms/Functions/Dynamics分类
- AgeMem (arXiv:2601.01885): RL学习记忆管理策略
- MemEngine (arXiv:2505.02099): 模块化记忆框架(含forgetting)

### 推理Skill相关
- Buffer of Thoughts (2024): thought-template库
- LEGO-Prover (NeurIPS 2023): 渐进式lemma库
- ExpeL (2023): 经验规则积累

### ALFWorld Agent
- ReAct (ICLR 2023): think+act交替, 71% SR
- Reflexion (NeurIPS 2023): 自反思+重试, 97% SR (12 trials)
- AgentRefine (ICLR 2025): max_turn=30, 超越AgentGym/Agent-FLAN
- ETO (ACL 2024): DPO-based轨迹优化
