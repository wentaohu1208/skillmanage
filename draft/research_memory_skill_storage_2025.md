# Agent Memory & Skill 存储方式调研（2025-2026）

## 一、2025-2026年三大存储范式

### 1. Flat Vector Store（向量存储）
**代表**: Voyager (ChromaDB), 你的框架 (内存dict + embedding)

```
存储: skill embedding → vector DB → cosine similarity检索
结构: [skill_1, skill_2, skill_3, ...] 全部平铺
优势: 简单快速，setup成本低，80%场景够用
劣势: 只靠语义相似度，无法表达skill间的关系和依赖
```

**谁在用**: Voyager, A-MEM, 大部分2023-2024的工作，你的框架

### 2. Hierarchical Structured（层次化结构）
**代表**: SkillRL (General/Task-Specific分层), H-MEM (4层架构)

```
存储: 多级目录/分层dict
结构:
  Level 0: General Skills（通用，每个task都注入）
  Level 1: Task-Specific Skills（按task_type分类）
  Level 2: Instance-Specific（具体示例）

H-MEM的4层:
  Domain Layer → Category Layer → Memory Trace Layer → Episode Layer
```

**优势**: 检索时先粗后细，减少搜索空间；不同层级有不同的生命周期
**劣势**: 需要预定义层级结构，灵活性差

**谁在用**: SkillRL, H-MEM, Memento-Skills (skill文件夹)

### 3. Graph-Based（图结构）
**代表**: A-MEM (Zettelkasten), Zep (时序知识图谱), Cognee

```
存储: 图数据库 / 节点+边
结构:
  Node = 一个skill或memory
  Edge = skill之间的关系（依赖/相似/冲突/演化）

  skill_A --depends_on--> skill_B
  skill_A --similar_to--> skill_C
  skill_A --evolved_from--> skill_A_v1
```

**优势**: 能表达skill间的关系；多跳检索（通过关系找到相关skill）；支持skill组合和依赖
**劣势**: 实现复杂；需要维护图结构；冷启动困难

**谁在用**: A-MEM, Zep, Cognee, From Experience to Strategy (2025)

### 4. File System（文件系统 — 2025新趋势）
**代表**: SKILL.md标准 (Anthropic), Memento-Skills, Claude Skills

```
存储: 文件系统上的markdown文件+目录
结构:
  skills/
  ├── solve_quadratic/
  │   ├── SKILL.md        # 元数据+指令
  │   ├── scripts/        # 可执行代码
  │   ├── references/     # 参考文档
  │   └── examples/       # 示例
  ├── triangle_area/
  │   └── SKILL.md
```

**优势**: Git可追踪、人类可读可编辑、版本控制、无需额外数据库
**劣势**: 检索需要额外的索引层；不适合大规模skill库

**谁在用**: Anthropic Claude Skills, Memento-Skills, AutoSkill, 2026行业趋势

---

## 二、2026年主要趋势

### 趋势1: 混合存储成为标准
```
不再是选一种，而是组合使用：
  - Vector Store: 做语义检索（快速recall）
  - Graph/Structure: 做关系推理（精确precision）
  - File System: 做持久化和版本控制（可追溯）
```

### 趋势2: 从flat向hierarchical演进
```
SimpleMem: 语义无损压缩 → 在线语义合成 → 意图感知检索
SkillRL:   General → Task-Specific → Common Mistakes 三层
H-MEM:     Domain → Category → Trace → Episode 四层
MemSkill:  skill-level（不是step-level），span-level而非turn-level
```

### 趋势3: Skill = 可演化的一等公民
```
SKILL.md标准（Anthropic 2025年发布）:
  skill不只是一段文本描述
  而是一个自包含的目录：SKILL.md + scripts/ + references/

MemSkill: skill通过RL演化
  controller学习选哪个skill → executor执行 → designer演化skill库

Memento-Skills: Read-Write闭环
  读skill → 执行 → 反思 → 写回改进的skill
```

### 趋势4: 文件系统存储的复兴
```
"While the AI industry spent millions building vector databases...
 high-value projects quietly converged on the same boring solution:
 file-based planning for long-running agents."

优势：
  - Git版本控制（skill的每次修改都可追溯）
  - 人类可读可审计
  - 不需要额外数据库
  - 可以commit、revert、branch skill的"知识"
```

---

## 三、各系统存储方式详细对比

| 系统 | 存储方式 | skill格式 | 检索方式 | 演化机制 |
|------|---------|----------|---------|---------|
| **你的框架** | 内存dict + JSON dump | Skill dataclass (steps+warnings) | cosine similarity | Alignment + Importance + Repair |
| **Voyager** | ChromaDB + JSON + .js文件 | code + description | ChromaDB similarity | 只增不减 |
| **SkillRL** | JSON文件分层 | general/task-specific/mistakes | template + embedding双模 | 失败驱动的递归演化 |
| **AutoSkill** | 文件系统SKILL.md | YAML frontmatter + markdown | BM25 + vector混合 | Add/Merge/Discard |
| **A-MEM** | ChromaDB + 链接图 | MemoryNote (content+links) | ChromaDB + link traversal | strengthen/update_neighbor |
| **Memento-Skills** | 文件夹(SKILL.md+code) | markdown + scripts + refs | 可训练router | Read-Write反思闭环 |
| **MemSkill** | span-level skill bank | skill template | 可训练controller(PPO) | designer从失败case演化 |
| **SimpleMem** | 语义压缩memory units | compressed factual units | intent-aware parallel | 在线语义合成 |
| **H-MEM** | 4层hierarchical vectors | multi-level abstractions | layer-by-layer retrieval | 自动抽象和合并 |
| **SKILL.md标准** | 文件系统 | YAML frontmatter + md | 按需加载(lazy) | 手动或agent修改 |

---

## 四、对你框架的借鉴建议

### 建议1: 短期可做 — Skill分层（参考SkillRL）
```
现在: Active里所有skill平铺
改为:
  General Skills:      通用策略（如"验证答案"），每个task都注入
  Domain Skills:       按task_type分组（如geometry的skill），按需检索
  Warnings/Mistakes:   独立的错误教训列表

实现简单：只需要在Skill dataclass加一个 `level: str` 字段
检索时：general始终返回 + domain按embedding检索
```

### 建议2: 中期可做 — SKILL.md文件格式（参考Anthropic标准）
```
现在: Skill是Python dataclass，存为JSON
改为: 每个skill是一个SKILL.md文件

优势:
  - 人类可读可编辑
  - Git追踪skill演化历史（每次repair都是一个commit）
  - 和Anthropic/OpenAI的skill生态对齐
  - 论文里可以展示skill的markdown内容
```

### 建议3: 中期可做 — Skill关系图（参考A-MEM）
```
现在: skill之间唯一的关系是embedding相似度（用于Merge和Irreplaceability）
改为: 显式记录skill间关系

  skill_A --depends_on--> skill_B    （A的步骤引用了B的能力）
  skill_A --conflicts_with--> skill_C （A和C不能同时用）
  skill_A --evolved_from--> skill_A_v1 （A是v1修复后的版本）

用途:
  检索时：找到A也把A依赖的B一起返回
  Forgetting时：A有依赖者就不能随便淘汰
  Repair时：追踪skill的演化链
```

### 建议4: 长期方向 — 可训练的Skill Router（参考MemSkill）
```
现在: cosine similarity（固定，不学习）
改为: controller通过RL学习选哪个skill

MemSkill的做法:
  controller(PPO) → 选skill → executor执行 → reward反馈 → 更新controller

对应你的GRPO想法：用GRPO学习lifecycle决策
```

---

## 五、你现在的存储方式评估

```
你的方案: 内存dict + JSON/npz定期dump

优势:
  ✅ 实验阶段最快最简单
  ✅ 不依赖任何外部数据库
  ✅ checkpoint恢复方便
  ✅ Active/Archive/Forgotten三区管理比大部分系统都完善

和业界差距:
  ⚠️ 没有skill分层（SkillRL有）
  ⚠️ 没有skill关系图（A-MEM有）
  ⚠️ 没有SKILL.md文件格式（Anthropic标准）
  ⚠️ 检索只有cosine similarity（MemSkill有可训练router）
  ⚠️ 没有混合检索（AutoSkill有BM25+vector）

但你的lifecycle管理是独特优势:
  ✅ Acquisition三道过滤（别人没有）
  ✅ Active→Archive→Forgotten三级遗忘（别人大部分只增不减）
  ✅ Importance Score四维评分（别人最多utility score±1）
  ✅ Skill repair闭环（和Memento-Skills类似，但你有retry次数限制）
  ✅ Token预算控制（别人没有）
```

---

## Sources

- [SimpleMem GitHub](https://github.com/aiming-lab/SimpleMem)
- [A-MEM Paper](https://arxiv.org/abs/2502.12110)
- [SkillRL Paper](https://arxiv.org/html/2602.08234v1)
- [MemSkill Paper](https://arxiv.org/abs/2602.02474)
- [Memento-Skills](https://skills.memento.run/)
- [SKILL.md Standard](https://www.mintlify.com/blog/skill-md)
- [H-MEM Paper](https://arxiv.org/abs/2507.22925)
- [Memory in the Age of AI Agents Survey](https://arxiv.org/abs/2512.13564)
- [2026 Memory Stack for Enterprise Agents](https://alok-mishra.com/2026/01/07/a-2026-memory-stack-for-enterprise-agents/)
- [Vector DB vs Graph RAG Comparison](https://machinelearningmastery.com/vector-databases-vs-graph-rag-for-agent-memory-when-to-use-which/)
- [File System Memory Approach](https://dev.to/imaginex/ai-agent-memory-management-when-markdown-files-are-all-you-need-5ekk)
- [Top 6 Memory Frameworks 2026](https://machinelearningmastery.com/the-6-best-ai-agent-memory-frameworks-you-should-try-in-2026/)
- [Zep Knowledge Graph](https://blog.getzep.com/content/files/2025/01/ZEP__USING_KNOWLEDGE_GRAPHS_TO_POWER_LLM_AGENT_MEMORY_2025011700.pdf)
