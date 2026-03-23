"""ALFWorld experiments with Qwen2.5-7B-Instruct (local vLLM).

Requires: bash launch_vllm.sh to start local vLLM server first.

Exp A: No skill baseline (134 unseen test)
Exp B: With skill lifecycle (500 train + 134 test)

Usage:
    python3 run_alfworld_qwen.py --exp a    # no-skill
    python3 run_alfworld_qwen.py --exp b    # with-skill
    python3 run_alfworld_qwen.py --exp all  # both
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — Qwen2.5-7B-Instruct via local vLLM
# ---------------------------------------------------------------------------

# Qwen agent (local vLLM)
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
QWEN_API_KEY = "EMPTY"
QWEN_MODEL = os.environ.get("QWEN_MODEL", "/data/hwt/hf_ckpt/Qwen2.5-7B-Instruct")
MODEL_NAME = "qwen7b"

# Qwen recommended inference config for agent interaction
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 512
REPETITION_PENALTY = 1.05

# ALFWorld settings
ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA", "/data/hwt/alfworld_data")
NUM_TRAIN = int(os.environ.get("NUM_TRAIN", "500"))
MAX_STEPS = 30
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/data/hwt/skillmanage/exp_alfworld_qwen")

TASK_TYPES = [
    "pick_and_place", "look_at_obj_in_light",
    "pick_clean_then_place", "pick_heat_then_place",
    "pick_cool_then_place", "pick_two_obj_and_place",
]


def create_llm():
    """Create Qwen LLM client (local vLLM, Qwen-recommended params)."""
    from skillmanage.llm import create_llm_client
    return create_llm_client(
        "openai", base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY,
        model_name=QWEN_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        top_p=TOP_P, repetition_penalty=REPETITION_PENALTY,
    )


def create_embedding():
    from skillmanage.config import EmbeddingConfig
    from skillmanage.core.embedding import EmbeddingModel
    emb = EmbeddingModel(EmbeddingConfig(model_name="/data/hwt/hf_ckpt/Qwen3-Embedding-0.6B", dimension=1024))
    _ = emb.encode("test")
    return emb


def compute_per_type_sr(results):
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_type[r["task_type"]]["total"] += 1
        if r["success"]:
            by_type[r["task_type"]]["correct"] += 1
    return {tt: {"total": d["total"], "correct": d["correct"],
                 "sr": d["correct"]/d["total"] if d["total"] > 0 else 0}
            for tt in TASK_TYPES for d in [by_type.get(tt, {"total": 0, "correct": 0})]}


def print_per_type(per_type, correct, total):
    print(f"\n{'Task Type':<28} {'SR':<12} {'Correct/Total':<15}")
    print("-" * 55)
    for tt in TASK_TYPES:
        d = per_type.get(tt, {"sr": 0, "correct": 0, "total": 0})
        print(f"{tt:<28} {d['sr']*100:>6.1f}%     {d['correct']}/{d['total']}")
    print("-" * 55)
    sr = correct / total if total else 0
    print(f"{'OVERALL':<28} {sr*100:>6.1f}%     {correct}/{total}")


def run_no_skill(llm):
    from skillmanage.benchmark import create_benchmark
    output_dir = os.path.join(OUTPUT_BASE, "no_skill")
    os.makedirs(output_dir, exist_ok=True)
    bench = create_benchmark("alfworld", data_path=ALFWORLD_DATA, max_steps=MAX_STEPS)
    test_tasks = bench.load_tasks(split="test")
    logger.info("[%s] NO-SKILL: %d test tasks", MODEL_NAME, len(test_tasks))

    results = []
    for i, task in enumerate(test_tasks):
        try:
            obs = bench.reset_env(task)
            system_prompt = bench.build_system_prompt("")
            history, done = [], False
            for _ in range(MAX_STEPS):
                action = llm.generate(bench.build_step_prompt(task, obs, history), system_prompt=system_prompt)
                action = action.strip().split("\n")[0]
                history.append(f"Action: {action}")
                obs, reward, done = bench.step(action)
                history.append(f"Observation: {obs}")
                if done:
                    break
            success, _ = bench.check_answer(task, "")
            results.append({"task_id": task.task_id, "task_type": bench.current_task_type,
                            "success": success, "steps": len([h for h in history if h.startswith("Action:")])})
            logger.info("[%s] Test %d/%d %s | type=%s", MODEL_NAME, i+1, len(test_tasks),
                        "OK" if success else "X", bench.current_task_type)
        except Exception as e:
            logger.error("[%s] Test %d FAILED: %s", MODEL_NAME, i+1, e)
            results.append({"task_id": task.task_id, "task_type": "unknown", "success": False, "steps": 0})

    correct = sum(1 for r in results if r["success"])
    per_type = compute_per_type_sr(results)
    summary = {"model": MODEL_NAME, "use_skill": False, "total": len(results),
               "correct": correct, "overall_sr": correct/len(results) if results else 0, "per_task_type": per_type}
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_per_type(per_type, correct, len(results))
    bench.close()
    return summary


def run_with_skill(llm):
    from skillmanage.benchmark import AgentRunner, create_benchmark
    from skillmanage.config import RetrievalConfig, SkillManageConfig
    from skillmanage.core.retrieval import SkillRetriever
    from skillmanage.core.skill_bank import SkillBank
    from skillmanage.core.storage import save_checkpoint

    output_dir = os.path.join(OUTPUT_BASE, "with_skill")
    os.makedirs(output_dir, exist_ok=True)

    train_bench = create_benchmark("alfworld", data_path=ALFWORLD_DATA, max_steps=MAX_STEPS)
    test_bench = create_benchmark("alfworld", data_path=ALFWORLD_DATA, max_steps=MAX_STEPS)
    emb = create_embedding()

    cfg = SkillManageConfig(
        retrieval=RetrievalConfig(top_k=3, similarity_threshold=0.3, token_budget=2000),
        storage_dir=output_dir, checkpoint_interval=100,
    )
    skill_bank = SkillBank(embedding_dim=1024)
    runner = AgentRunner(benchmark=train_bench, skill_bank=skill_bank, embedding_model=emb, llm_client=llm, cfg=cfg)

    # Train
    train_tasks = train_bench.load_tasks(split="train", limit=NUM_TRAIN)
    logger.info("[%s] WITH-SKILL: Training on %d tasks", MODEL_NAME, len(train_tasks))
    for i, task in enumerate(train_tasks):
        try:
            result = runner.run_task(task, current_round=i)
            stats = skill_bank.stats()
            logger.info("[%s] Train %d/%d %s | type=%s | active=%d", MODEL_NAME, i+1, len(train_tasks),
                        "OK" if result.success else "X", train_bench.current_task_type, stats["active"])
        except Exception as e:
            logger.error("[%s] Train %d FAILED: %s", MODEL_NAME, i+1, e)
        if (i + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, i)
    save_checkpoint(skill_bank, runner.alignment.buffers, output_dir, len(train_tasks) - 1)
    train_bench.close()

    # Test
    test_tasks = test_bench.load_tasks(split="test")
    logger.info("[%s] Testing on %d tasks (active=%d)", MODEL_NAME, len(test_tasks), len(skill_bank.active))
    retriever = SkillRetriever()
    results = []
    for i, task in enumerate(test_tasks):
        try:
            obs = test_bench.reset_env(task)
            active_skills, archive_hit = retriever.retrieve(test_bench.current_instruction, skill_bank, emb, cfg.retrieval)
            used = list(active_skills)
            if archive_hit:
                used = [archive_hit.original_skill_full]
            skills_prompt = SkillRetriever.format_skills_for_prompt(used)
            system_prompt = test_bench.build_system_prompt(skills_prompt)
            history, done = [], False
            for _ in range(MAX_STEPS):
                action = llm.generate(test_bench.build_step_prompt(task, obs, history), system_prompt=system_prompt)
                action = action.strip().split("\n")[0]
                history.append(f"Action: {action}")
                obs, reward, done = test_bench.step(action)
                history.append(f"Observation: {obs}")
                if done:
                    break
            success, _ = test_bench.check_answer(task, "")
            results.append({"task_id": task.task_id, "task_type": test_bench.current_task_type,
                            "success": success, "skills_used": len(used)})
            logger.info("[%s] Test %d/%d %s | type=%s | skills=%d", MODEL_NAME, i+1, len(test_tasks),
                        "OK" if success else "X", test_bench.current_task_type, len(used))
        except Exception as e:
            logger.error("[%s] Test %d FAILED: %s", MODEL_NAME, i+1, e)
            results.append({"task_id": task.task_id, "task_type": "unknown", "success": False, "skills_used": 0})

    correct = sum(1 for r in results if r["success"])
    per_type = compute_per_type_sr(results)
    stats = skill_bank.stats()
    summary = {"model": MODEL_NAME, "use_skill": True, "total": len(results),
               "correct": correct, "overall_sr": correct/len(results) if results else 0,
               "per_task_type": per_type, "skill_bank": stats}
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_per_type(per_type, correct, len(results))
    test_bench.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="ALFWorld Qwen experiments (local vLLM)")
    parser.add_argument("--exp", type=str, default="all", help="a/b/all")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    logger.info("Connecting to local vLLM at %s ...", QWEN_BASE_URL)
    llm = create_llm()
    llm.generate("test")
    logger.info("Qwen LLM OK (temp=%.1f, top_p=%.1f)", TEMPERATURE, TOP_P)

    summaries = {}
    if args.exp in ("a", "all"):
        summaries["no_skill"] = run_no_skill(llm)
    if args.exp in ("b", "all"):
        summaries["with_skill"] = run_with_skill(llm)

    if summaries:
        with open(os.path.join(OUTPUT_BASE, "summaries.json"), "w") as f:
            json.dump(summaries, f, indent=2)
    logger.info("Done. Results in: %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
