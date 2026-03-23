# Progress Log

## Session 1 — 2026-03-21
### 完成
- 框架设计: lifecycle 四层架构 (Active→Archive→Forgotten) + Acquisition pipeline
- 数据集调研: ALFWorld, WebShop, MATH, BBH
- 参考代码库调研: 10个仓库 (SkillRL, Voyager, AutoSkill, ExpeL, MemEngine...)
- 实验设计: 6组实验 + baseline
- 文档: framework.txt, acquisition.txt, active.txt, archive.txt, forgotten.txt

## Session 2 — 2026-03-21
### 完成
- Phase 1-4 代码实现 (27个Python文件, ~3000行)
- 基础架构 + Acquisition + Active管理 + Archive/Forgotten
- LLM抽象层 (OpenAI兼容)

## Session 3 — 2026-03-22
### 完成
- Benchmark层: base.py (ABC + Registry), math_bench.py (MATH完整实现), runner.py (AgentRunner)
- MATH答案比较: 三级fallback (normalize→数值→sympy)
- 实验脚本: run_math_deepseek.py, run_math_qwen.py

## Session 4 — 2026-03-22~23
### 完成
- MATH V1 实验跑完: No Skill 48.7%, With Skill 46.3% (**skill帮倒忙**)
- V1 根因分析 (draft/v1.md): 检索精度低、warning堆积、failure skill无价值、低质量skill不淘汰
- 答案比较重构: 换成 math-verify (ANTLR4+SymPy) + string fallback + LLM Judge
- _pre_normalize: 修复逗号处理 (保留千分位, 修复元组)、数字-LaTeX连接
- ALFWorld benchmark 实现 (alfworld_bench.py)
- 实验脚本: run_alfworld_qwen.py, run_alfworld_deepseek.py

## Session 5 — 2026-03-23
### 完成
- V2 修改全部部署:
  - similarity_threshold: 0.3→0.5
  - quality_floor=0.35, min_calls=30
  - max_warnings=5 + LLM合并
  - Prompt分离 (skills在user, warnings在system)
  - create_failure_skill=False
  - Skill Repair (诊断+重写+重试max 3)
- ExperimentTracker 实现 (tracker.py): task_log.jsonl + lifecycle_log.jsonl
- Tracker集成到 runner.py + run_math_qwen.py (即插即用, tracker=None零副作用)

## Session 6 — 2026-03-23~24
### 完成
- Tracker bug修复 (共7个):
  - P2: log_skill_archived 字段错误 → meta snapshot
  - P2: retry_success 始终 False → 移到 retry 后记录
  - P3: recall_count off-by-one → 移到 record_recall_result 后
  - P2: 归档原因 imp>0 判断错误 → archive_reasons 字典传递
  - P3: merged/distilled 事件未落盘 → CompressionReport 携带详情
  - P2: merge 删除的 skill 误记为 archived → merged_consumed 排除
  - P2: on_new_skill_added 压缩事件漏记 → 返回 CompressionReport
  - P2: merge后同轮forced归档漏记 → 改用archive_reasons直接遍历
- forgetting.py: check_natural_forgetting 返回 Dict[str,str] (sid→reason)
- compression.py: distill 日志引用旧变量修复 (skill.token_cost → new_token_cost)
- JSON解析: 加 LaTeX escape 修复 (re.sub 双写非法转义)
- tracker.py: 加 question 字段
- run_math_qwen.py: train_correct/test_correct 初始化

### 完成 (ALFWorld)
- SkillRL 深度调研: 训练流程、检索机制、prompt格式
- ALFWorld 参考实现对比: ReAct/Reflexion/SkillRL
- alfworld_bench.py 6项改进:
  - admissible actions 采集+更新+注入prompt
  - think: 拦截 → "OK." 不发给env
  - Action: 前缀剥离
  - 空action守卫
  - 循环检测 (3次重复终止)
  - _won fallback = False
- prompts.py: step prompt 加 task_instruction + admissible_actions
- runner.py: _run_multi_step warnings注入
- run_alfworld_qwen.py: tracker集成 (no_skill + with_skill)
- alfworld_bench.py: _create_env 改为直接 yaml.safe_load (绕过 generic.load_config argparse冲突)
- admissible_commands IndexError 修复 (空列表安全处理)

### 当前阻塞
- **ALFWorld base_config.yaml 路径**: 需要在服务器跑 `alfworld-download` 生成 config
- MATH V2 实验在服务器运行中

### 下一步
1. 解决 ALFWorld config 路径问题
2. 跑 ALFWorld Exp A (no-skill baseline)
3. 跑 ALFWorld Exp B (with-skill lifecycle)
4. 对比结果: with skill > no skill?
5. 如果 ALFWorld 有效 → 设计 baseline 对比实验 (Voyager/FIFO/LRU)
6. 如果 ALFWorld 无效 → 分析原因，调整策略
