"""Geometry L4-L5 experiments: no-skill baseline vs skill lifecycle.

Experiment 1 (deepseek-chat): No skill baseline
Experiment 2 (deepseek-chat): With skill lifecycle
Experiment 3 (qwen2.5-7b):   No skill baseline
Experiment 4 (qwen2.5-7b):   With skill lifecycle

Usage:
    python3 run_experiments.py --exp 1      # deepseek no-skill
    python3 run_experiments.py --exp 2      # deepseek with-skill
    python3 run_experiments.py --exp 3      # qwen no-skill
    python3 run_experiments.py --exp 4      # qwen with-skill
    python3 run_experiments.py --exp all    # all experiments
    python3 run_experiments.py --exp ds     # deepseek both (1+2)
    python3 run_experiments.py --exp qwen   # qwen both (3+4)
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# LLM configs
DEEPSEEK_API_KEY = os.environ.get("DS_API_KEY", "sk-DISQMJtpvWPwvub7Z4xC2IFHyzNt4gEwRB1dJ5fBzkt92wFY")
DEEPSEEK_BASE_URL = "https://api.qingyuntop.top/v1"
DEEPSEEK_MODEL = "deepseek-chat"

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "sk-xEX3XN7CU8PMfvbhudubGbKDsp5MTEUAiow3JOxud0JptXr1")
QWEN_BASE_URL = "https://api.qingyuntop.top/v1"
QWEN_MODEL = "Qwen2.5-7B-Instruct"

# Experiment settings
MATH_DATA_PATH = os.environ.get("MATH_DATA_PATH", "/data/hwt/hf_data/math")
SUBJECT = "geometry"
LEVELS = [4, 5]
NUM_TRAIN = None   # None = full (geometry L4: 177 train, L5: 421 train = 598 total)
NUM_TEST = None    # None = full (geometry L4: 125 test, L5: 132 test = 257 total)
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/data/hwt/skillmanage/experiments_geometry")


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def create_llm_client(api_key: str, base_url: str, model: str):
    """Create an LLM client."""
    from skillmanage.llm import create_llm_client as _create
    return _create(
        "openai",
        base_url=base_url,
        api_key=api_key,
        model_name=model,
        temperature=0.0,
        max_tokens=2048,
    )


def create_embedding_model():
    """Create shared embedding model."""
    from skillmanage.config import EmbeddingConfig
    from skillmanage.core.embedding import EmbeddingModel
    emb = EmbeddingModel(
        EmbeddingConfig(model_name="/data/hwt/hf_ckpt/Qwen3-Embedding-0.6B", dimension=1024)
    )
    _ = emb.encode("test")
    return emb


def load_all_tasks():
    """Load geometry L4-L5 train and test tasks (one bench object, called once)."""
    from skillmanage.benchmark import create_benchmark
    bench = create_benchmark("math", subjects=[SUBJECT], levels=LEVELS, dataset_name=MATH_DATA_PATH)
    train = bench.load_tasks(split="train", limit=NUM_TRAIN)
    test = bench.load_tasks(split="test", limit=NUM_TEST)
    logger.info("Loaded %d train + %d test tasks (geometry L4-L5)", len(train), len(test))
    return bench, train, test


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------


def run_no_skill(model_name: str, llm_client, bench, train_tasks, test_tasks, output_dir: str) -> dict:
    """Run without skill bank (pure LLM baseline)."""
    from skillmanage.benchmark.base import TaskResult
    from skillmanage.benchmark.math_bench import extract_boxed_answer

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Running NO-SKILL baseline (%s): %d train, %d test", model_name, len(train_tasks), len(test_tasks))

    def _run_one(task):
        prompt = bench.build_prompt(task, "")
        system_prompt = getattr(bench, "system_prompt", "")
        output = llm_client.generate(prompt, system_prompt=system_prompt)
        success, reward = bench.check_answer(task, output)
        return TaskResult(
            task_id=task.task_id, task_type=task.task_type,
            success=success, reward=reward,
            agent_answer=output, ground_truth=task.ground_truth,
            trajectory=bench.extract_trajectory(output),
            num_steps=0,
        )

    # Train
    train_results = []
    for i, task in enumerate(train_tasks):
        try:
            result = _run_one(task)
            train_results.append(result)
            pred = extract_boxed_answer(result.agent_answer)
            status = "OK" if result.success else "X"
            logger.info("[%s no-skill] Train %d/%d %s | pred=%s gt=%s",
                        model_name, i+1, len(train_tasks), status, pred[:20], task.ground_truth[:20])
        except Exception as e:
            logger.error("[%s no-skill] Train %d FAILED: %s", model_name, i+1, e)

    # Test
    test_results = []
    for i, task in enumerate(test_tasks):
        try:
            result = _run_one(task)
            test_results.append(result)
            pred = extract_boxed_answer(result.agent_answer)
            status = "OK" if result.success else "X"
            logger.info("[%s no-skill] Test %d/%d %s | pred=%s gt=%s",
                        model_name, i+1, len(test_tasks), status, pred[:20], task.ground_truth[:20])
        except Exception as e:
            logger.error("[%s no-skill] Test %d FAILED: %s", model_name, i+1, e)

    train_correct = sum(1 for r in train_results if r.success)
    test_correct = sum(1 for r in test_results if r.success)

    summary = {
        "model": model_name,
        "use_skill": False,
        "subject": SUBJECT,
        "levels": LEVELS,
        "train_total": len(train_tasks),
        "train_processed": len(train_results),
        "train_correct": train_correct,
        "train_sr": train_correct / len(train_tasks) if train_tasks else 0,
        "train_errors": len(train_tasks) - len(train_results),
        "test_total": len(test_tasks),
        "test_processed": len(test_results),
        "test_correct": test_correct,
        "test_sr": test_correct / len(test_tasks) if test_tasks else 0,
        "test_errors": len(test_tasks) - len(test_results),
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("[%s no-skill] Train SR: %d/%d = %.1f%%", model_name,
                train_correct, len(train_results), summary["train_sr"]*100)
    logger.info("[%s no-skill] Test SR:  %d/%d = %.1f%%", model_name,
                test_correct, len(test_results), summary["test_sr"]*100)

    return summary


def run_with_skill(model_name: str, llm_client, embedding_model, bench, train_tasks, test_tasks, output_dir: str) -> dict:
    """Run with skill lifecycle, then evaluate test with final skill bank."""
    from skillmanage.benchmark import AgentRunner
    from skillmanage.benchmark.math_bench import extract_boxed_answer
    from skillmanage.config import RetrievalConfig, SkillManageConfig
    from skillmanage.core.skill_bank import SkillBank
    from skillmanage.core.storage import save_checkpoint

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Running WITH-SKILL (%s): %d train, %d test", model_name, len(train_tasks), len(test_tasks))

    cfg = SkillManageConfig(
        retrieval=RetrievalConfig(top_k=3, similarity_threshold=0.3, token_budget=2000),
        storage_dir=output_dir,
        checkpoint_interval=50,
    )

    skill_bank = SkillBank(embedding_dim=1024)
    runner = AgentRunner(
        benchmark=bench,
        skill_bank=skill_bank,
        embedding_model=embedding_model,
        llm_client=llm_client,
        cfg=cfg,
    )

    # Train (with skill accumulation)
    train_results = []
    for i, task in enumerate(train_tasks):
        try:
            result = runner.run_task(task, current_round=i)
            train_results.append(result)

            pred = extract_boxed_answer(result.agent_answer)
            status = "OK" if result.success else "X"
            stats = skill_bank.stats()
            logger.info("[%s skill] Train %d/%d %s | pred=%s gt=%s | active=%d archive=%d",
                        model_name, i+1, len(train_tasks), status,
                        pred[:20], task.ground_truth[:20],
                        stats["active"], stats["archive"])
        except Exception as e:
            logger.error("[%s skill] Train %d FAILED: %s", model_name, i+1, e)

        # Checkpoint every 50 rounds
        if (i + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, i)

    # Save final checkpoint
    save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, len(train_tasks) - 1)

    # Test (frozen skill bank)
    logger.info("Evaluating on %d test tasks with final skill bank (active=%d)...",
                len(test_tasks), len(skill_bank.active))
    test_sr = runner.evaluate(test_tasks)

    train_correct = sum(1 for r in train_results if r.success)
    stats = skill_bank.stats()

    summary = {
        "model": model_name,
        "use_skill": True,
        "subject": SUBJECT,
        "levels": LEVELS,
        "train_total": len(train_tasks),
        "train_processed": len(train_results),
        "train_correct": train_correct,
        "train_sr": train_correct / len(train_tasks) if train_tasks else 0,
        "train_errors": len(train_tasks) - len(train_results),
        "test_total": len(test_tasks),
        "test_sr": test_sr,
        "active_skills": stats["active"],
        "archive_skills": stats["archive"],
        "forgotten_skills": stats["forgotten"],
        "active_tokens": stats["active_tokens"],
    }

    # Log active skills
    logger.info("[%s skill] Active skills:", model_name)
    for sid, askill in skill_bank.active.items():
        logger.info("  '%s' calls=%d sr=%.2f tokens=%d",
                     askill.skill.name, askill.meta.call_count,
                     askill.meta.success_rate, askill.skill.token_cost)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("[%s skill] Train SR: %d/%d = %.1f%%", model_name,
                train_correct, len(train_results), summary["train_sr"]*100)
    logger.info("[%s skill] Test SR:  %.1f%%", model_name, test_sr*100)
    logger.info("[%s skill] SkillBank: active=%d archive=%d forgotten=%d tokens=%d",
                model_name, stats["active"], stats["archive"], stats["forgotten"], stats["active_tokens"])

    return summary


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def experiment_1(bench, train_tasks, test_tasks):
    """DeepSeek no-skill baseline."""
    logger.info("=" * 60)
    logger.info("EXP 1: DeepSeek-chat NO SKILL (geometry L4-L5)")
    logger.info("=" * 60)
    llm = create_llm_client(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
    llm.generate("test")
    return run_no_skill("deepseek", llm, bench, train_tasks, test_tasks,
                        os.path.join(OUTPUT_BASE, "exp1_deepseek_no_skill"))


def experiment_2(bench, train_tasks, test_tasks):
    """DeepSeek with skill lifecycle."""
    logger.info("=" * 60)
    logger.info("EXP 2: DeepSeek-chat WITH SKILL (geometry L4-L5)")
    logger.info("=" * 60)
    llm = create_llm_client(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
    llm.generate("test")
    emb = create_embedding_model()
    return run_with_skill("deepseek", llm, emb, bench, train_tasks, test_tasks,
                          os.path.join(OUTPUT_BASE, "exp2_deepseek_with_skill"))


def experiment_3(bench, train_tasks, test_tasks):
    """Qwen2.5-7B no-skill baseline."""
    logger.info("=" * 60)
    logger.info("EXP 3: Qwen2.5-7B NO SKILL (geometry L4-L5)")
    logger.info("=" * 60)
    llm = create_llm_client(QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL)
    llm.generate("test")
    return run_no_skill("qwen7b", llm, bench, train_tasks, test_tasks,
                        os.path.join(OUTPUT_BASE, "exp3_qwen_no_skill"))


def experiment_4(bench, train_tasks, test_tasks):
    """Qwen2.5-7B with skill lifecycle."""
    logger.info("=" * 60)
    logger.info("EXP 4: Qwen2.5-7B WITH SKILL (geometry L4-L5)")
    logger.info("=" * 60)
    llm = create_llm_client(QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL)
    llm.generate("test")
    emb = create_embedding_model()
    return run_with_skill("qwen7b", llm, emb, bench, train_tasks, test_tasks,
                          os.path.join(OUTPUT_BASE, "exp4_qwen_with_skill"))


def print_comparison(summaries: dict) -> None:
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON: Geometry L4-L5")
    print("=" * 80)
    print(f"{'Experiment':<35} {'Model':<12} {'Skill':<8} {'Train SR':<12} {'Test SR':<12}")
    print("-" * 80)
    for name, s in summaries.items():
        skill_str = "Yes" if s.get("use_skill") else "No"
        model = s.get("model", "?")
        train_sr = f"{s['train_sr']*100:.1f}% ({s['train_correct']}/{s['train_total']})"
        if "test_sr" in s:
            test_sr = f"{s['test_sr']*100:.1f}%"
        else:
            test_sr = "N/A"
        print(f"{name:<35} {model:<12} {skill_str:<8} {train_sr:<12} {test_sr:<12}")

    # Skill bank info for with-skill experiments
    for name, s in summaries.items():
        if s.get("use_skill"):
            print(f"\n  {name}: active={s.get('active_skills',0)}, archive={s.get('archive_skills',0)}, tokens={s.get('active_tokens',0)}")


def main():
    parser = argparse.ArgumentParser(description="Geometry L4-L5 experiments")
    parser.add_argument("--exp", type=str, default="all",
                        help="1/2/3/4/ds/qwen/all")
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Load tasks once, reuse for all experiments
    logger.info("Loading tasks...")
    bench, train_tasks, test_tasks = load_all_tasks()

    all_summaries = {}

    if args.exp in ("1", "ds", "all"):
        all_summaries["exp1_deepseek_no_skill"] = experiment_1(bench, train_tasks, test_tasks)

    if args.exp in ("2", "ds", "all"):
        all_summaries["exp2_deepseek_with_skill"] = experiment_2(bench, train_tasks, test_tasks)

    if args.exp in ("3", "qwen", "all"):
        all_summaries["exp3_qwen_no_skill"] = experiment_3(bench, train_tasks, test_tasks)

    if args.exp in ("4", "qwen", "all"):
        all_summaries["exp4_qwen_with_skill"] = experiment_4(bench, train_tasks, test_tasks)

    if all_summaries:
        print_comparison(all_summaries)
        with open(os.path.join(OUTPUT_BASE, "all_summaries.json"), "w") as f:
            json.dump(all_summaries, f, indent=2)

    logger.info("\nDone. Results in: %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
