"""End-to-end pipeline test with 30 MATH problems.

Usage:
    API_KEY=sk-xxx python3 test_pipeline.py

    # Or with custom endpoint:
    API_KEY=sk-xxx BASE_URL=https://api.xxx.com/v1 MODEL=deepseek-chat python3 test_pipeline.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Reduce noise from third-party libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# LLM API config (from environment variables)
API_KEY = os.environ.get("API_KEY", "sk-DISQMJtpvWPwvub7Z4xC2IFHyzNt4gEwRB1dJ5fBzkt92wFY")
BASE_URL = os.environ.get("BASE_URL", "https://api.qingyuntop.top/v1")
MODEL = os.environ.get("MODEL", "deepseek-chat")

# Test parameters
NUM_TRAIN_TASKS = 30        # Number of tasks in task stream
NUM_TEST_TASKS = 10         # Number of tasks for evaluation
MATH_SUBJECTS = ["prealgebra"]  # Easiest subject for quick test
MATH_LEVELS = [1, 2]       # Easiest levels
TOKEN_BUDGET = 1000         # Small budget for test
TOP_K = 2                   # Retrieve 2 skills per task
LLM_MAX_TOKENS = 1024      # Max tokens per LLM call
LLM_TEMPERATURE = 0.0      # Deterministic output


def main() -> None:
    if not API_KEY:
        logger.error("Please set API_KEY environment variable")
        logger.error("Example: API_KEY=sk-xxx python3 test_pipeline.py")
        sys.exit(1)

    # --- Imports (after sys.path setup) ---
    from skillmanage.benchmark import AgentRunner, create_benchmark
    from skillmanage.config import (
        EmbeddingConfig,
        LLMConfig,
        RetrievalConfig,
        SkillManageConfig,
    )
    from skillmanage.core.embedding import EmbeddingModel
    from skillmanage.core.skill_bank import SkillBank
    from skillmanage.llm import create_llm_client

    # --- Build config ---
    cfg = SkillManageConfig(
        retrieval=RetrievalConfig(
            top_k=TOP_K,
            similarity_threshold=0.3,
            token_budget=TOKEN_BUDGET,
        ),
        embedding=EmbeddingConfig(model_name="/data/hwt/hf_ckpt/Qwen3-Embedding-0.6B", dimension=1024),
        llm=LLMConfig(
            provider="openai",
            base_url=BASE_URL,
            api_key=API_KEY,
            model_name=MODEL,
        ),
        checkpoint_interval=1,
    )

    # --- Step 1: Init LLM ---
    logger.info("=" * 60)
    logger.info("Step 1: Init LLM (%s @ %s)", MODEL, BASE_URL)
    logger.info("=" * 60)

    llm_client = create_llm_client(
        "openai",
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    # Quick connectivity test
    test_resp = llm_client.generate("What is 2+3? Answer with just the number.")
    logger.info("LLM connectivity OK. Response: %s", test_resp.strip())

    # --- Step 2: Init Embedding ---
    logger.info("=" * 60)
    logger.info("Step 2: Loading embedding model")
    logger.info("=" * 60)

    embedding_model = EmbeddingModel(cfg.embedding)
    # Force load now (so we see any errors early)
    _ = embedding_model.encode("test")
    logger.info("Embedding model loaded OK (dim=%d)", cfg.embedding.dimension)

    # --- Step 3: Init SkillBank (empty) ---
    skill_bank = SkillBank(embedding_dim=1024)
    logger.info("SkillBank initialized (empty): %s", skill_bank.stats())

    # --- Step 4: Load MATH data ---
    logger.info("=" * 60)
    logger.info("Step 4: Loading MATH data (subjects=%s, levels=%s)", MATH_SUBJECTS, MATH_LEVELS)
    logger.info("=" * 60)

    # Use local path for dataset (no network required)
    # Change this to "EleutherAI/hendrycks_math" if you have internet access
    MATH_DATA_PATH = os.environ.get("MATH_DATA_PATH", "/data/hwt/hf_data/math")
    bench = create_benchmark("math", subjects=MATH_SUBJECTS, levels=MATH_LEVELS, dataset_name=MATH_DATA_PATH)
    train_tasks = bench.load_tasks(split="train", limit=NUM_TRAIN_TASKS)
    test_tasks = bench.load_tasks(split="test", limit=NUM_TEST_TASKS)
    logger.info("Loaded: %d train tasks, %d test tasks", len(train_tasks), len(test_tasks))

    # Preview first task
    if train_tasks:
        t = train_tasks[0]
        logger.info("First task: [%s] %s... | GT=%s", t.task_type, t.instruction[:80], t.ground_truth)

    # --- Step 5: Create AgentRunner ---
    runner = AgentRunner(
        benchmark=bench,
        skill_bank=skill_bank,
        embedding_model=embedding_model,
        llm_client=llm_client,
        cfg=cfg,
    )

    # --- Step 6: Run task stream ---
    logger.info("=" * 60)
    logger.info("Step 6: Running %d tasks", len(train_tasks))
    logger.info("=" * 60)

    results = []
    for i, task in enumerate(train_tasks):
        logger.info(
            "\n--- Task %d/%d [%s] ---",
            i + 1, len(train_tasks), task.task_type,
        )
        logger.info("Problem: %s", task.instruction[:120])
        logger.info("Ground truth: %s", task.ground_truth)

        try:
            result = runner.run_task(task, current_round=i)
            results.append(result)

            # Extract just the boxed answer for display
            from skillmanage.benchmark.math_bench import extract_boxed_answer
            pred_answer = extract_boxed_answer(result.agent_answer)

            status = "CORRECT" if result.success else "WRONG"
            logger.info(">>> %s | Predicted: %s | GT: %s", status, pred_answer, task.ground_truth)

        except Exception as e:
            logger.error("Task %d FAILED: %s", i + 1, e, exc_info=True)
            continue

        # Progress every 10 tasks
        if (i + 1) % 10 == 0:
            stats = skill_bank.stats()
            sr = sum(1 for r in results if r.success) / len(results)
            logger.info(
                "\n=== Round %d | SR=%.0f%% | Active=%d | Archive=%d | Forgotten=%d | Tokens=%d ===\n",
                i + 1, 100 * sr,
                stats["active"], stats["archive"], stats["forgotten"], stats["active_tokens"],
            )

    # --- Step 7: Summary ---
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    total = len(results)
    correct = sum(1 for r in results if r.success)
    logger.info("Completed: %d/%d tasks", total, len(train_tasks))
    logger.info("Accuracy:  %d/%d = %.1f%%", correct, total, 100 * correct / total if total else 0)

    stats = skill_bank.stats()
    logger.info("SkillBank: active=%d, archive=%d, forgotten=%d, tokens=%d",
                stats["active"], stats["archive"], stats["forgotten"], stats["active_tokens"])

    if skill_bank.active:
        logger.info("\nActive skills:")
        for sid, askill in skill_bank.active.items():
            logger.info(
                "  [%s] '%s' | calls=%d sr=%.2f tokens=%d | %s",
                sid, askill.skill.name,
                askill.meta.call_count, askill.meta.success_rate, askill.skill.token_cost,
                askill.skill.description[:60],
            )

    # --- Step 8: Eval on test set ---
    if test_tasks:
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION on %d test tasks (frozen skill bank)", len(test_tasks))
        logger.info("=" * 60)

        eval_sr = runner.evaluate(test_tasks)
        logger.info("Test Accuracy: %.1f%%", 100 * eval_sr)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()


'''
  run_task(task, current_round)
      │
      │  ① 检索skill
      │     retriever.retrieve(task.instruction, skill_bank, embedding_model, cfg)
      │     → Active中embedding cosine similarity → top-K
      │     → 如果Active没好的匹配 → fallback搜Archive
      │     → 返回: used_skills列表 (可能为空)
      │
      │  ② 拼prompt + 调LLM
      │     skills_prompt = format_skills_for_prompt(used_skills)
      │     prompt = bench.build_prompt(task, skills_prompt)
      │       → "请解决以下数学问题...\n{skills}\n问题: {problem}"
      │     agent_output = llm_client.generate(prompt, system_prompt)
      │       → 调DeepSeek API，返回CoT文本
      │
      │  ③ 判断对错
      │     success, reward = bench.check_answer(task, agent_output)
      │       → 提取\boxed{} → 三级比较 → True/False
      │
      │  ④ 提取trajectory
      │     trajectory = bench.extract_trajectory(agent_output)
      │       → 按行切分CoT → ["step1", "step2", ...]
      │
      │  ⑤ 如果是Archive召回的skill → 记录召回结果
      │     archive_mgr.record_recall_result(成功/失败)
      │       → 成功次数够R次 → promote回Active
      │
      │  ⑥ 更新已用skill的meta
      │     meta_updater.update_after_task(used_skill_ids, success, reward)
      │       → call_count += 1, success_rate更新, avg_reward更新
      │
      │  ⑦ Acquisition pipeline
      │     collector.decide(success, trajectory, used_skills)
      │       │
      │       ├─ 成功路径:
      │       │    segmenter.segment(trajectory, task_type)
      │       │    alignment.add_record(task_type, segmented)
      │       │      → PatternBuffer更新count和confidence
      │       │    如果有pattern满足提取条件(≥M条, confidence≥r):
      │       │      formalizer.formalize(pattern) → 新Skill
      │       │      skill_bank.add_to_active(新skill)
      │       │
      │       └─ 失败路径:
      │            failure_learner.analyze_failure(task, trajectory)
      │              → 匹配到已有skill → 附加warning
      │              → 没匹配但可执行 → 创建新skill
      │              → 不可执行 → 丢弃
      │
      │  ⑧ Active管理（每轮结束）
      │     active_mgr.on_round_end()
      │       → 算importance score
      │       → 连续T1轮低 → 降级到Archive
      │       → 超预算 → Merge → Distill → 强制Forget
      │
      │  ⑨ Archive维护
      │     archive_mgr.tick_inactive()
      │       → 所有Archive skill的inactive_rounds += 1
      │       → 超过max轮 → 移到Forgotten
      │
      └─ 返回TaskResult


'''
