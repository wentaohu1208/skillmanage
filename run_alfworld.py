"""ALFWorld experiments: no-skill baseline vs skill lifecycle.

Experiment A: No skill baseline (DeepSeek + Qwen)
Experiment B: With skill lifecycle (DeepSeek + Qwen)
Experiment D: Weak model (Qwen) comparison

Usage:
    python3 run_alfworld.py --exp a          # No-skill baseline (both models)
    python3 run_alfworld.py --exp b          # With-skill (both models)
    python3 run_alfworld.py --exp a-ds       # No-skill DeepSeek only
    python3 run_alfworld.py --exp b-ds       # With-skill DeepSeek only
    python3 run_alfworld.py --exp a-qwen     # No-skill Qwen only
    python3 run_alfworld.py --exp b-qwen     # With-skill Qwen only
    python3 run_alfworld.py --exp all        # All experiments
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEEPSEEK_API_KEY = os.environ.get("DS_API_KEY", "sk-DISQMJtpvWPwvub7Z4xC2IFHyzNt4gEwRB1dJ5fBzkt92wFY")
DEEPSEEK_BASE_URL = "https://api.qingyuntop.top/v1"
DEEPSEEK_MODEL = "deepseek-chat"

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "sk-xEX3XN7CU8PMfvbhudubGbKDsp5MTEUAiow3JOxud0JptXr1")
QWEN_BASE_URL = "https://api.qingyuntop.top/v1"
QWEN_MODEL = "Qwen2.5-7B-Instruct"

ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA", "/data/hwt/alfworld_data")
NUM_TRAIN = int(os.environ.get("NUM_TRAIN", "500"))  # Train tasks for skill accumulation
MAX_STEPS = 30

OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/data/hwt/skillmanage/experiments_alfworld")

TASK_TYPES = [
    "pick_and_place", "look_at_obj_in_light",
    "pick_clean_then_place", "pick_heat_then_place",
    "pick_cool_then_place", "pick_two_obj_and_place",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_llm(api_key: str, base_url: str, model: str):
    from skillmanage.llm import create_llm_client
    return create_llm_client(
        "openai", base_url=base_url, api_key=api_key,
        model_name=model, temperature=0.0, max_tokens=512,
    )


def create_embedding():
    from skillmanage.config import EmbeddingConfig
    from skillmanage.core.embedding import EmbeddingModel
    emb = EmbeddingModel(
        EmbeddingConfig(model_name="/data/hwt/hf_ckpt/Qwen3-Embedding-0.6B", dimension=1024)
    )
    _ = emb.encode("test")
    return emb


def compute_per_type_sr(results: List[dict]) -> Dict[str, dict]:
    """Compute success rate per task type."""
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        tt = r["task_type"]
        by_type[tt]["total"] += 1
        if r["success"]:
            by_type[tt]["correct"] += 1

    per_type = {}
    for tt in TASK_TYPES:
        d = by_type.get(tt, {"total": 0, "correct": 0})
        sr = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        per_type[tt] = {"total": d["total"], "correct": d["correct"], "sr": sr}
    return per_type


# ---------------------------------------------------------------------------
# No-skill baseline
# ---------------------------------------------------------------------------


def run_no_skill(model_name: str, llm_client, output_dir: str) -> dict:
    """Run 134 unseen ALFWorld tasks without any skill."""
    from skillmanage.benchmark import create_benchmark
    from skillmanage.benchmark.base import TaskResult

    os.makedirs(output_dir, exist_ok=True)
    bench = create_benchmark("alfworld", data_path=ALFWORLD_DATA, max_steps=MAX_STEPS)

    # Load test tasks (134 unseen)
    test_tasks = bench.load_tasks(split="test")
    logger.info("[%s no-skill] Running %d test tasks", model_name, len(test_tasks))

    results = []
    for i, task in enumerate(test_tasks):
        try:
            obs = bench.reset_env(task)
            # Build system prompt AFTER reset (includes task instruction)
            system_prompt = bench.build_system_prompt("")
            history = []
            done = False

            for step_num in range(MAX_STEPS):
                step_prompt = bench.build_step_prompt(task, obs, history)
                action = llm_client.generate(step_prompt, system_prompt=system_prompt)
                action = action.strip().split("\n")[0]

                history.append(f"Action: {action}")
                obs, reward, done = bench.step(action)
                history.append(f"Observation: {obs}")

                if done:
                    break

            task_type = bench.current_task_type
            success, _ = bench.check_answer(task, "")
            result = {
                "task_id": task.task_id,
                "task_type": task_type,
                "success": success,
                "steps": len([h for h in history if h.startswith("Action:")]),
            }
            results.append(result)

            status = "OK" if success else "X"
            logger.info("[%s no-skill] Test %d/%d %s | type=%s | steps=%d",
                        model_name, i+1, len(test_tasks), status, task_type, result["steps"])

        except Exception as e:
            logger.error("[%s no-skill] Test %d FAILED: %s", model_name, i+1, e)

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r["success"])
    per_type = compute_per_type_sr(results)

    summary = {
        "model": model_name,
        "use_skill": False,
        "total": total,
        "correct": correct,
        "overall_sr": correct / total if total else 0,
        "per_task_type": per_type,
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _print_per_type(model_name, "no-skill", per_type, correct, total)
    bench.close()
    return summary


# ---------------------------------------------------------------------------
# With-skill lifecycle
# ---------------------------------------------------------------------------


def run_with_skill(model_name: str, llm_client, embedding_model, output_dir: str) -> dict:
    """Train on N tasks to build skill bank, then test on 134 unseen tasks."""
    from skillmanage.benchmark import AgentRunner, create_benchmark
    from skillmanage.config import RetrievalConfig, SkillManageConfig
    from skillmanage.core.skill_bank import SkillBank
    from skillmanage.core.storage import save_checkpoint

    os.makedirs(output_dir, exist_ok=True)

    # Separate bench objects for train and test (different env splits)
    train_bench = create_benchmark("alfworld", data_path=ALFWORLD_DATA, max_steps=MAX_STEPS)
    test_bench = create_benchmark("alfworld", data_path=ALFWORLD_DATA, max_steps=MAX_STEPS)

    cfg = SkillManageConfig(
        retrieval=RetrievalConfig(top_k=3, similarity_threshold=0.3, token_budget=2000),
        storage_dir=output_dir,
        checkpoint_interval=100,
    )

    skill_bank = SkillBank(embedding_dim=1024)
    runner = AgentRunner(
        benchmark=train_bench, skill_bank=skill_bank,
        embedding_model=embedding_model, llm_client=llm_client, cfg=cfg,
    )

    # Phase 1: Train (accumulate skills)
    train_tasks = train_bench.load_tasks(split="train", limit=NUM_TRAIN)
    logger.info("[%s skill] Training on %d tasks...", model_name, len(train_tasks))

    train_results = []
    for i, task in enumerate(train_tasks):
        try:
            result = runner.run_task(task, current_round=i)
            train_results.append({
                "task_id": result.task_id,
                "task_type": result.task_type or train_bench.current_task_type,
                "success": result.success,
                "steps": result.num_steps,
            })

            stats = skill_bank.stats()
            status = "OK" if result.success else "X"
            logger.info("[%s skill] Train %d/%d %s | type=%s | active=%d archive=%d",
                        model_name, i+1, len(train_tasks), status,
                        train_bench.current_task_type,
                        stats["active"], stats["archive"])
        except Exception as e:
            logger.error("[%s skill] Train %d FAILED: %s", model_name, i+1, e)

        if (i + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, i)

    # Save final skill bank
    save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, len(train_tasks) - 1)
    train_bench.close()

    # Phase 2: Test (frozen skill bank on 134 unseen, SEPARATE env)
    test_tasks = test_bench.load_tasks(split="test")
    logger.info("[%s skill] Testing on %d unseen tasks (active=%d)...",
                model_name, len(test_tasks), len(skill_bank.active))

    from skillmanage.core.retrieval import SkillRetriever
    retriever = SkillRetriever()

    test_results = []
    for i, task in enumerate(test_tasks):
        try:
            # Reset env FIRST to get instruction for retrieval
            obs = test_bench.reset_env(task)

            # Retrieve skills using the actual task instruction
            active_skills, archive_hit = retriever.retrieve(
                test_bench.current_instruction,
                skill_bank, embedding_model, cfg.retrieval,
            )
            used_skills = list(active_skills)
            if archive_hit:
                used_skills = [archive_hit.original_skill_full]

            skills_prompt = SkillRetriever.format_skills_for_prompt(used_skills)
            system_prompt = test_bench.build_system_prompt(skills_prompt)

            history = []
            done = False
            for step_num in range(MAX_STEPS):
                step_prompt = test_bench.build_step_prompt(task, obs, history)
                action = llm_client.generate(step_prompt, system_prompt=system_prompt)
                action = action.strip().split("\n")[0]

                history.append(f"Action: {action}")
                obs, reward, done = test_bench.step(action)
                history.append(f"Observation: {obs}")
                if done:
                    break

            task_type = test_bench.current_task_type
            success, _ = test_bench.check_answer(task, "")
            result = {
                "task_id": task.task_id,
                "task_type": task_type,
                "success": success,
                "steps": len([h for h in history if h.startswith("Action:")]),
                "skills_used": len(used_skills),
            }
            test_results.append(result)

            status = "OK" if success else "X"
            logger.info("[%s skill] Test %d/%d %s | type=%s | skills=%d",
                        model_name, i+1, len(test_tasks), status, task_type, len(used_skills))

        except Exception as e:
            logger.error("[%s skill] Test %d FAILED: %s", model_name, i+1, e)

    # Summary
    train_correct = sum(1 for r in train_results if r["success"])
    test_correct = sum(1 for r in test_results if r["success"])
    test_per_type = compute_per_type_sr(test_results)
    stats = skill_bank.stats()

    summary = {
        "model": model_name,
        "use_skill": True,
        "train_total": len(train_results),
        "train_correct": train_correct,
        "train_sr": train_correct / len(train_results) if train_results else 0,
        "test_total": len(test_results),
        "test_correct": test_correct,
        "test_sr": test_correct / len(test_results) if test_results else 0,
        "per_task_type": test_per_type,
        "skill_bank": stats,
    }

    # Log active skills
    logger.info("[%s skill] Active skills:", model_name)
    for sid, askill in skill_bank.active.items():
        logger.info("  '%s' calls=%d sr=%.2f",
                     askill.skill.name, askill.meta.call_count, askill.meta.success_rate)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _print_per_type(model_name, "with-skill", test_per_type, test_correct, len(test_results))
    test_bench.close()
    return summary


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_per_type(model: str, mode: str, per_type: dict, correct: int, total: int):
    """Print per-task-type results table."""
    print(f"\n=== [{model} {mode}] Per Task Type (test) ===")
    print(f"{'Task Type':<28} {'SR':<12} {'Correct/Total':<15}")
    print("-" * 55)
    for tt in TASK_TYPES:
        d = per_type.get(tt, {"sr": 0, "correct": 0, "total": 0})
        print(f"{tt:<28} {d['sr']*100:>6.1f}%     {d['correct']}/{d['total']}")
    print("-" * 55)
    sr = correct / total if total else 0
    print(f"{'OVERALL':<28} {sr*100:>6.1f}%     {correct}/{total}")


def print_comparison(summaries: dict):
    """Print comparison across experiments."""
    print("\n" + "=" * 80)
    print("ALFWorld RESULTS COMPARISON")
    print("=" * 80)

    # Overall
    print(f"\n{'Experiment':<40} {'Test SR':<15} {'Correct/Total':<15}")
    print("-" * 70)
    for name, s in summaries.items():
        sr = s.get("test_sr", s.get("overall_sr", 0))
        correct = s.get("test_correct", s.get("correct", 0))
        total = s.get("test_total", s.get("total", 0))
        print(f"{name:<40} {sr*100:>6.1f}%       {correct}/{total}")

    # Per type comparison
    print(f"\n{'Task Type':<25}", end="")
    for name in summaries:
        print(f" {name[:15]:<16}", end="")
    print()
    print("-" * (25 + 16 * len(summaries)))
    for tt in TASK_TYPES:
        print(f"{tt:<25}", end="")
        for name, s in summaries.items():
            pt = s.get("per_task_type", {}).get(tt, {})
            sr = pt.get("sr", 0) * 100
            print(f" {sr:>5.1f}%          ", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ALFWorld experiments")
    parser.add_argument("--exp", type=str, default="all",
                        help="a/b/a-ds/a-qwen/b-ds/b-qwen/all")
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    all_summaries = {}

    # Experiment A: No-skill baselines
    if args.exp in ("a", "a-ds", "all"):
        logger.info("=" * 60)
        logger.info("EXP A: DeepSeek NO SKILL")
        logger.info("=" * 60)
        llm = create_llm(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
        llm.generate("test")
        all_summaries["deepseek_no_skill"] = run_no_skill(
            "deepseek", llm, os.path.join(OUTPUT_BASE, "a_deepseek_no_skill"))

    if args.exp in ("a", "a-qwen", "all"):
        logger.info("=" * 60)
        logger.info("EXP A: Qwen NO SKILL")
        logger.info("=" * 60)
        llm = create_llm(QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL)
        llm.generate("test")
        all_summaries["qwen_no_skill"] = run_no_skill(
            "qwen7b", llm, os.path.join(OUTPUT_BASE, "a_qwen_no_skill"))

    # Experiment B: With-skill lifecycle
    if args.exp in ("b", "b-ds", "all"):
        logger.info("=" * 60)
        logger.info("EXP B: DeepSeek WITH SKILL")
        logger.info("=" * 60)
        llm = create_llm(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
        llm.generate("test")
        emb = create_embedding()
        all_summaries["deepseek_with_skill"] = run_with_skill(
            "deepseek", llm, emb, os.path.join(OUTPUT_BASE, "b_deepseek_with_skill"))

    if args.exp in ("b", "b-qwen", "all"):
        logger.info("=" * 60)
        logger.info("EXP B: Qwen WITH SKILL")
        logger.info("=" * 60)
        llm = create_llm(QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL)
        llm.generate("test")
        emb = create_embedding()
        all_summaries["qwen_with_skill"] = run_with_skill(
            "qwen7b", llm, emb, os.path.join(OUTPUT_BASE, "b_qwen_with_skill"))

    if all_summaries:
        print_comparison(all_summaries)
        with open(os.path.join(OUTPUT_BASE, "all_summaries.json"), "w") as f:
            json.dump(all_summaries, f, indent=2)

    logger.info("\nDone. Results in: %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
