"""Run multiple small experiments for comparison.

Experiment 1: Each subject x 30 tasks (with skill lifecycle)
Experiment 2: Same tasks, no skill (baseline)
Experiment 3: Cross-subject skill transfer (subject A's skills used for subject B)

Usage:
    python3 run_experiments.py --exp 1    # Run experiment 1 only
    python3 run_experiments.py --exp 2    # Run experiment 2 only
    python3 run_experiments.py --exp 3    # Run experiment 3 only
    python3 run_experiments.py --exp all  # Run all experiments
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

API_KEY = os.environ.get("API_KEY", "sk-DISQMJtpvWPwvub7Z4xC2IFHyzNt4gEwRB1dJ5fBzkt92wFY")
BASE_URL = os.environ.get("BASE_URL", "https://api.qingyuntop.top/v1")
MODEL = os.environ.get("MODEL", "deepseek-chat")
MATH_DATA_PATH = os.environ.get("MATH_DATA_PATH", "/data/hwt/hf_data/math")

SUBJECTS = ["algebra", "prealgebra", "number_theory", "geometry", "counting_and_probability"]
LEVELS = [1, 2]
NUM_TRAIN = 30
NUM_TEST = 10
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/data/hwt/skillmanage/experiments")


def get_shared_components():
    """Create shared LLM client and embedding model (reused across experiments)."""
    from skillmanage.config import EmbeddingConfig, LLMConfig
    from skillmanage.core.embedding import EmbeddingModel
    from skillmanage.llm import create_llm_client

    llm_client = create_llm_client(
        "openai",
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=MODEL,
        temperature=0.0,
        max_tokens=1024,
    )

    embedding_model = EmbeddingModel(
        EmbeddingConfig(model_name="/data/hwt/hf_ckpt/Qwen3-Embedding-0.6B", dimension=1024)
    )
    # Force load
    _ = embedding_model.encode("test")

    return llm_client, embedding_model


def run_single_subject(subject, llm_client, embedding_model, output_dir, use_skill=True):
    """Run one subject with or without skill lifecycle.

    Args:
        subject: MATH subject name.
        llm_client: LLM client.
        embedding_model: Embedding model.
        output_dir: Where to save checkpoints.
        use_skill: If False, disable skill retrieval and acquisition (baseline).

    Returns:
        Dict with results summary.
    """
    from skillmanage.benchmark import AgentRunner, create_benchmark
    from skillmanage.benchmark.math_bench import extract_boxed_answer
    from skillmanage.config import RetrievalConfig, SkillManageConfig
    from skillmanage.core.skill_bank import SkillBank
    from skillmanage.core.storage import save_checkpoint

    # Config
    cfg = SkillManageConfig(
        retrieval=RetrievalConfig(
            top_k=2 if use_skill else 0,  # top_k=0 means no retrieval
            similarity_threshold=0.3,
            token_budget=1000,
        ),
        storage_dir=output_dir,
        checkpoint_interval=1,
    )

    skill_bank = SkillBank(embedding_dim=1024)
    bench = create_benchmark("math", subjects=[subject], levels=LEVELS, dataset_name=MATH_DATA_PATH)

    train_tasks = bench.load_tasks(split="train", limit=NUM_TRAIN)
    test_tasks = bench.load_tasks(split="test", limit=NUM_TEST)

    runner = AgentRunner(
        benchmark=bench,
        skill_bank=skill_bank,
        embedding_model=embedding_model,
        llm_client=llm_client,
        cfg=cfg,
    )

    # Run task stream
    results = []
    for i, task in enumerate(train_tasks):
        try:
            if use_skill:
                result = runner.run_task(task, current_round=i)
            else:
                # Baseline: just run LLM without any skill
                result = _run_no_skill(task, bench, llm_client)
            results.append(result)

            pred = extract_boxed_answer(result.agent_answer)
            status = "OK" if result.success else "X"
            logger.info(
                "[%s] Task %d/%d %s | pred=%s gt=%s",
                subject, i + 1, len(train_tasks), status,
                pred[:20], task.ground_truth[:20],
            )
        except Exception as e:
            logger.error("[%s] Task %d FAILED: %s", subject, i + 1, e)
            continue

        # Checkpoint every round
        if use_skill and (i + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, i)

    # Save final checkpoint
    os.makedirs(output_dir, exist_ok=True)
    if use_skill:
        save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, len(train_tasks) - 1)

    # Eval on test
    if use_skill and test_tasks:
        eval_sr = runner.evaluate(test_tasks)
    else:
        eval_correct = 0
        for task in test_tasks:
            try:
                r = _run_no_skill(task, bench, llm_client) if not use_skill else runner.run_task(task, 9999)
                if r.success:
                    eval_correct += 1
            except Exception:
                pass
        eval_sr = eval_correct / len(test_tasks) if test_tasks else 0

    # Summary
    train_correct = sum(1 for r in results if r.success)
    summary = {
        "subject": subject,
        "use_skill": use_skill,
        "train_tasks": len(results),
        "train_correct": train_correct,
        "train_sr": train_correct / len(results) if results else 0,
        "test_sr": eval_sr,
        "active_skills": skill_bank.stats()["active"] if use_skill else 0,
        "archive_skills": skill_bank.stats()["archive"] if use_skill else 0,
    }

    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _run_no_skill(task, bench, llm_client):
    """Run a single task without any skill (pure LLM baseline)."""
    from skillmanage.benchmark.base import TaskResult

    prompt = bench.build_prompt(task, "")  # Empty skills_prompt
    system_prompt = getattr(bench, "system_prompt", "")
    agent_output = llm_client.generate(prompt, system_prompt=system_prompt)
    success, reward = bench.check_answer(task, agent_output)
    trajectory = bench.extract_trajectory(agent_output)

    return TaskResult(
        task_id=task.task_id,
        task_type=task.task_type,
        success=success,
        reward=reward,
        trajectory=trajectory,
        agent_answer=agent_output,
        ground_truth=task.ground_truth,
        num_steps=len(trajectory),
    )


def run_with_transferred_skills(
    source_subject, target_subject, llm_client, embedding_model, output_dir
):
    """Run target subject using skills learned from source subject.

    1. Load skill bank from source subject's experiment 1 output.
    2. Run target subject tasks using those skills (no new skill acquisition).

    Returns:
        Dict with results summary.
    """
    from skillmanage.benchmark import AgentRunner, create_benchmark
    from skillmanage.benchmark.math_bench import extract_boxed_answer
    from skillmanage.config import RetrievalConfig, SkillManageConfig
    from skillmanage.core.storage import load_checkpoint

    # Load source skill bank
    source_dir = os.path.join(OUTPUT_BASE, "exp1_per_subject", source_subject)
    try:
        skill_bank, pattern_buffers, _ = load_checkpoint(source_dir, embedding_dim=1024)
        logger.info(
            "Loaded source skills from %s: active=%d",
            source_subject, len(skill_bank.active),
        )
    except Exception as e:
        logger.error("Failed to load source skills from %s: %s", source_dir, e)
        return {"error": str(e)}

    # Run target subject with frozen source skills (no acquisition)
    cfg = SkillManageConfig(
        retrieval=RetrievalConfig(top_k=2, similarity_threshold=0.3, token_budget=1000),
        storage_dir=output_dir,
        checkpoint_interval=1,
    )

    bench = create_benchmark("math", subjects=[target_subject], levels=LEVELS, dataset_name=MATH_DATA_PATH)
    test_tasks = bench.load_tasks(split="test", limit=NUM_TEST)
    train_tasks = bench.load_tasks(split="train", limit=NUM_TRAIN)

    # Run without acquisition — just use source skills
    results = []
    for i, task in enumerate(train_tasks):
        try:
            # Manually retrieve + run (skip acquisition)
            from skillmanage.core.retrieval import SkillRetriever
            retriever = SkillRetriever()
            active_skills, archive_hit = retriever.retrieve(
                task.instruction, skill_bank, embedding_model, cfg.retrieval
            )
            used_skills = list(active_skills)
            if archive_hit:
                used_skills = [archive_hit.original_skill_full]

            skills_prompt = SkillRetriever.format_skills_for_prompt(used_skills)
            prompt = bench.build_prompt(task, skills_prompt)
            system_prompt = getattr(bench, "system_prompt", "")
            agent_output = llm_client.generate(prompt, system_prompt=system_prompt)
            success, reward = bench.check_answer(task, agent_output)

            from skillmanage.benchmark.base import TaskResult
            result = TaskResult(
                task_id=task.task_id, task_type=task.task_type,
                success=success, reward=reward,
                agent_answer=agent_output, ground_truth=task.ground_truth,
                used_skill_ids=[s.skill_id for s in used_skills],
                num_steps=0,
            )
            results.append(result)

            pred = extract_boxed_answer(agent_output)
            status = "OK" if success else "X"
            n_skills = len(used_skills)
            logger.info(
                "[%s→%s] Task %d/%d %s | skills_used=%d | pred=%s gt=%s",
                source_subject, target_subject, i + 1, len(train_tasks),
                status, n_skills, pred[:20], task.ground_truth[:20],
            )
        except Exception as e:
            logger.error("[%s→%s] Task %d FAILED: %s", source_subject, target_subject, i + 1, e)
            continue

    train_correct = sum(1 for r in results if r.success)
    summary = {
        "source_subject": source_subject,
        "target_subject": target_subject,
        "train_tasks": len(results),
        "train_correct": train_correct,
        "train_sr": train_correct / len(results) if results else 0,
        "source_active_skills": len(skill_bank.active),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def experiment_1(llm_client, embedding_model):
    """Exp1: Each subject x 30 tasks with skill lifecycle."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Per-subject with skill lifecycle")
    logger.info("=" * 60)

    all_summaries = {}
    for subject in SUBJECTS:
        logger.info("\n--- Subject: %s ---", subject)
        output_dir = os.path.join(OUTPUT_BASE, "exp1_per_subject", subject)
        summary = run_single_subject(subject, llm_client, embedding_model, output_dir, use_skill=True)
        all_summaries[subject] = summary
        logger.info("Result: train_sr=%.1f%%, test_sr=%.1f%%, active=%d",
                     summary["train_sr"] * 100, summary["test_sr"] * 100, summary["active_skills"])

    # Print comparison table
    print("\n=== Experiment 1: Per-subject with skills ===")
    print(f"{'Subject':<25} {'Train SR':<12} {'Test SR':<12} {'Active':<8} {'Archive':<8}")
    print("-" * 65)
    for subj, s in all_summaries.items():
        print(f"{subj:<25} {s['train_sr']*100:>6.1f}%     {s['test_sr']*100:>6.1f}%     {s['active_skills']:<8} {s['archive_skills']:<8}")

    with open(os.path.join(OUTPUT_BASE, "exp1_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


def experiment_2(llm_client, embedding_model):
    """Exp2: Same tasks, no skill (baseline)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Per-subject NO skill (baseline)")
    logger.info("=" * 60)

    all_summaries = {}
    for subject in SUBJECTS:
        logger.info("\n--- Subject: %s (no skill) ---", subject)
        output_dir = os.path.join(OUTPUT_BASE, "exp2_no_skill", subject)
        summary = run_single_subject(subject, llm_client, embedding_model, output_dir, use_skill=False)
        all_summaries[subject] = summary
        logger.info("Result: train_sr=%.1f%%, test_sr=%.1f%%",
                     summary["train_sr"] * 100, summary["test_sr"] * 100)

    print("\n=== Experiment 2: Per-subject NO skills (baseline) ===")
    print(f"{'Subject':<25} {'Train SR':<12} {'Test SR':<12}")
    print("-" * 50)
    for subj, s in all_summaries.items():
        print(f"{subj:<25} {s['train_sr']*100:>6.1f}%     {s['test_sr']*100:>6.1f}%")

    with open(os.path.join(OUTPUT_BASE, "exp2_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


def experiment_3(llm_client, embedding_model):
    """Exp3: Cross-subject skill transfer (A's skills for B)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Cross-subject skill transfer")
    logger.info("=" * 60)

    # Transfer pairs: source → target
    transfer_pairs = [
        ("algebra", "prealgebra"),           # similar
        ("algebra", "geometry"),             # different
        ("prealgebra", "number_theory"),     # different
        ("geometry", "algebra"),             # reverse
        ("counting_and_probability", "algebra"),  # different
    ]

    all_summaries = {}
    for source, target in transfer_pairs:
        logger.info("\n--- Transfer: %s → %s ---", source, target)
        output_dir = os.path.join(OUTPUT_BASE, "exp3_transfer", f"{source}_to_{target}")
        summary = run_with_transferred_skills(
            source, target, llm_client, embedding_model, output_dir
        )
        all_summaries[f"{source}→{target}"] = summary
        if "error" not in summary:
            logger.info("Result: train_sr=%.1f%%, source_skills=%d",
                         summary["train_sr"] * 100, summary["source_active_skills"])

    print("\n=== Experiment 3: Cross-subject skill transfer ===")
    print(f"{'Transfer':<30} {'Train SR':<12} {'Source Skills':<15}")
    print("-" * 57)
    for pair, s in all_summaries.items():
        if "error" in s:
            print(f"{pair:<30} ERROR: {s['error'][:30]}")
        else:
            print(f"{pair:<30} {s['train_sr']*100:>6.1f}%     {s['source_active_skills']:<15}")

    with open(os.path.join(OUTPUT_BASE, "exp3_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run skill lifecycle experiments")
    parser.add_argument("--exp", type=str, default="all", help="Which experiment: 1, 2, 3, or all")
    args = parser.parse_args()

    logger.info("Initializing shared components...")
    llm_client, embedding_model = get_shared_components()

    # Quick LLM test
    resp = llm_client.generate("What is 2+3? Answer with just the number.")
    logger.info("LLM OK: %s", resp.strip())

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    if args.exp in ("1", "all"):
        experiment_1(llm_client, embedding_model)

    if args.exp in ("2", "all"):
        experiment_2(llm_client, embedding_model)

    if args.exp in ("3", "all"):
        experiment_3(llm_client, embedding_model)

    logger.info("\nAll requested experiments done. Results in: %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
