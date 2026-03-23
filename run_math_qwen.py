"""MATH Geometry L4-L5 experiments with Qwen2.5-7B-Instruct (local vLLM).

Requires: bash launch_vllm.sh to start local vLLM server first.
Uses DeepSeek as LLM judge for answer verification.

Exp 1: No skill baseline
Exp 2: With skill lifecycle

Usage:
    python3 run_math_qwen.py --exp 1        # no-skill
    python3 run_math_qwen.py --exp 2        # with-skill
    python3 run_math_qwen.py --exp all      # both
    python3 run_math_qwen.py --exp 1 --debug  # with debug
"""

import argparse
import json
import logging
import os
import random
import sys

import numpy as np

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
# Configuration — Qwen2.5-7B-Instruct via local vLLM
# ---------------------------------------------------------------------------

# Qwen agent (local vLLM)
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
QWEN_API_KEY = "EMPTY"
QWEN_MODEL = os.environ.get("QWEN_MODEL", "/data/hwt/hf_ckpt/Qwen2.5-7B-Instruct")
MODEL_NAME = "qwen7b"

# Qwen recommended inference config
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 4096
REPETITION_PENALTY = 1.05

# DeepSeek for LLM judge (answer verification)
DS_API_KEY = os.environ.get("DS_API_KEY", "sk-DISQMJtpvWPwvub7Z4xC2IFHyzNt4gEwRB1dJ5fBzkt92wFY")
DS_BASE_URL = "https://api.qingyuntop.top/v1"
DS_MODEL = "deepseek-chat"

# Experiment settings
MATH_DATA_PATH = os.environ.get("MATH_DATA_PATH", "/data/hwt/hf_data/math")
SUBJECT = "geometry"
LEVELS = [4, 5]
NUM_TRAIN = None
NUM_TEST = None
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/data/hwt/skillmanage/outputs/exp_math_qwen_v2")

# Skill lifecycle config
SKILL_TOP_K = 3
SKILL_SIM_THRESHOLD = 0.3
SKILL_TOKEN_BUDGET = 2000
CHECKPOINT_INTERVAL = 1

DEBUG = False


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def create_qwen_llm():
    """Create Qwen LLM client (local vLLM, Qwen-recommended params)."""
    from skillmanage.llm import create_llm_client
    return create_llm_client(
        "openai", base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY,
        model_name=QWEN_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        top_p=TOP_P, repetition_penalty=REPETITION_PENALTY,
    )


def create_judge_llm():
    """Create DeepSeek LLM client for answer verification judge."""
    from skillmanage.llm import create_llm_client
    return create_llm_client(
        "openai", base_url=DS_BASE_URL, api_key=DS_API_KEY,
        model_name=DS_MODEL, temperature=0.0, max_tokens=256,
    )


def create_embedding():
    from skillmanage.config import EmbeddingConfig
    from skillmanage.core.embedding import EmbeddingModel
    emb = EmbeddingModel(EmbeddingConfig(model_name="/data/hwt/hf_ckpt/Qwen3-Embedding-0.6B", dimension=1024))
    _ = emb.encode("test")
    return emb


def load_tasks():
    from skillmanage.benchmark import create_benchmark
    bench = create_benchmark("math", subjects=[SUBJECT], levels=LEVELS, dataset_name=MATH_DATA_PATH)
    train = bench.load_tasks(split="train", limit=NUM_TRAIN)
    test = bench.load_tasks(split="test", limit=NUM_TEST)
    logger.info("Loaded %d train + %d test tasks", len(train), len(test))
    return bench, train, test


def _log_debug(task, result, pred):
    from skillmanage.benchmark.math_bench import _pre_normalize
    gt = task.ground_truth
    logger.info("=" * 40 + " DEBUG " + "=" * 40)
    logger.info("Task ID: %s | Type: %s", task.task_id, task.task_type)
    logger.info("Problem: %s", task.instruction[:200])
    logger.info("GT (raw):   %r", gt)
    logger.info("Pred (raw): %r", pred)
    logger.info("GT (norm):  %r", _pre_normalize(gt))
    logger.info("Pred (norm):%r", _pre_normalize(pred))
    try:
        from math_verify import parse as mv_parse, verify as mv_verify
        gt_p = mv_parse(_pre_normalize(gt))
        pred_p = mv_parse(_pre_normalize(pred))
        logger.info("GT parsed: %s | Pred parsed: %s | Verify: %s", gt_p, pred_p, mv_verify(gt_p, pred_p))
    except Exception as e:
        logger.info("math-verify error: %s", e)
    logger.info("Agent output (last 300):\n%s", result.agent_answer[-300:])
    logger.info("=" * 87)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_no_skill(llm, bench, train_tasks, test_tasks):
    from skillmanage.benchmark.base import TaskResult
    from skillmanage.benchmark.math_bench import extract_boxed_answer

    output_dir = os.path.join(OUTPUT_BASE, "no_skill")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("EXP 1: %s NO SKILL — %d train, %d test", MODEL_NAME, len(train_tasks), len(test_tasks))

    def _run(task):
        prompt = bench.build_prompt(task, "")
        output = llm.generate(prompt, system_prompt=getattr(bench, "system_prompt", ""))
        success, reward = bench.check_answer(task, output)
        return TaskResult(task_id=task.task_id, task_type=task.task_type,
                          success=success, reward=reward, agent_answer=output,
                          ground_truth=task.ground_truth, trajectory=bench.extract_trajectory(output), num_steps=0)

    for label, tasks in [("Train", train_tasks), ("Test", test_tasks)]:
        results = []
        for i, task in enumerate(tasks):
            try:
                r = _run(task)
                results.append(r)
                pred = extract_boxed_answer(r.agent_answer)
                s = "OK" if r.success else "X"
                logger.info("[%s] %s %d/%d %s | pred=%s gt=%s", MODEL_NAME, label, i+1, len(tasks), s, pred[:20], task.ground_truth[:20])
                if DEBUG and not r.success:
                    _log_debug(task, r, pred)
            except Exception as e:
                logger.error("[%s] %s %d FAILED: %s", MODEL_NAME, label, i+1, e)
        correct = sum(1 for r in results if r.success)
        sr = correct / len(tasks) if tasks else 0
        logger.info("[%s] %s SR: %d/%d = %.1f%%", MODEL_NAME, label, correct, len(tasks), sr*100)
        if label == "Train":
            train_correct = correct
        else:
            test_correct = correct

    summary = {"model": MODEL_NAME, "use_skill": False, "subject": SUBJECT, "levels": LEVELS,
               "train_total": len(train_tasks), "train_correct": train_correct, "train_sr": train_correct/len(train_tasks),
               "test_total": len(test_tasks), "test_correct": test_correct, "test_sr": test_correct/len(test_tasks)}
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_with_skill(llm, bench, train_tasks, test_tasks):
    from skillmanage.benchmark import AgentRunner
    from skillmanage.benchmark.math_bench import extract_boxed_answer
    from skillmanage.config import RetrievalConfig, SkillManageConfig
    from skillmanage.core.skill_bank import SkillBank
    from skillmanage.core.storage import save_checkpoint

    output_dir = os.path.join(OUTPUT_BASE, "with_skill")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("EXP 2: %s WITH SKILL — %d train, %d test", MODEL_NAME, len(train_tasks), len(test_tasks))

    cfg = SkillManageConfig(
        retrieval=RetrievalConfig(top_k=SKILL_TOP_K, similarity_threshold=SKILL_SIM_THRESHOLD, token_budget=SKILL_TOKEN_BUDGET),
        storage_dir=output_dir, checkpoint_interval=CHECKPOINT_INTERVAL,
    )
    emb = create_embedding()
    skill_bank = SkillBank(embedding_dim=1024)
    runner = AgentRunner(benchmark=bench, skill_bank=skill_bank, embedding_model=emb, llm_client=llm, cfg=cfg)

    # Train
    train_results = []
    for i, task in enumerate(train_tasks):
        try:
            result = runner.run_task(task, current_round=i)
            train_results.append(result)
            pred = extract_boxed_answer(result.agent_answer)
            s = "OK" if result.success else "X"
            stats = skill_bank.stats()
            logger.info("[%s] Train %d/%d %s | pred=%s gt=%s | active=%d archive=%d",
                        MODEL_NAME, i+1, len(train_tasks), s, pred[:20], task.ground_truth[:20], stats["active"], stats["archive"])
            if DEBUG and not result.success:
                _log_debug(task, result, pred)
        except Exception as e:
            logger.error("[%s] Train %d FAILED: %s", MODEL_NAME, i+1, e)
        if (i + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, i)

    save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, len(train_tasks) - 1)

    # Test
    logger.info("Evaluating %d test tasks (active=%d)...", len(test_tasks), len(skill_bank.active))
    test_sr = runner.evaluate(test_tasks)

    train_correct = sum(1 for r in train_results if r.success)
    stats = skill_bank.stats()
    summary = {"model": MODEL_NAME, "use_skill": True, "subject": SUBJECT, "levels": LEVELS,
               "train_total": len(train_tasks), "train_correct": train_correct, "train_sr": train_correct/len(train_tasks),
               "test_total": len(test_tasks), "test_sr": test_sr,
               "active_skills": stats["active"], "archive_skills": stats["archive"],
               "forgotten_skills": stats["forgotten"], "active_tokens": stats["active_tokens"]}

    logger.info("[%s] Active skills:", MODEL_NAME)
    for sid, a in skill_bank.active.items():
        logger.info("  '%s' calls=%d sr=%.2f tokens=%d", a.skill.name, a.meta.call_count, a.meta.success_rate, a.skill.token_cost)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[%s] Train SR: %.1f%% | Test SR: %.1f%%", MODEL_NAME, summary["train_sr"]*100, test_sr*100)
    return summary


def main():
    parser = argparse.ArgumentParser(description="MATH Qwen experiments (local vLLM)")
    parser.add_argument("--exp", type=str, default="all", help="1/2/all")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    random.seed(42)
    np.random.seed(42)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    bench, train_tasks, test_tasks = load_tasks()

    # Qwen for doing tasks
    logger.info("Connecting to local vLLM at %s ...", QWEN_BASE_URL)
    llm = create_qwen_llm()
    llm.generate("test")
    logger.info("Qwen LLM OK (temp=%.1f, top_p=%.1f, rep_penalty=%.2f)", TEMPERATURE, TOP_P, REPETITION_PENALTY)

    # DeepSeek for LLM judge (answer verification)
    logger.info("Creating DeepSeek LLM judge...")
    judge = create_judge_llm()
    judge.generate("test")
    bench._llm_client = judge

    summaries = {}
    if args.exp in ("1", "all"):
        summaries["no_skill"] = run_no_skill(llm, bench, train_tasks, test_tasks)
    if args.exp in ("2", "all"):
        summaries["with_skill"] = run_with_skill(llm, bench, train_tasks, test_tasks)

    if summaries:
        with open(os.path.join(OUTPUT_BASE, "summaries.json"), "w") as f:
            json.dump(summaries, f, indent=2)
    logger.info("Done. Results in: %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
