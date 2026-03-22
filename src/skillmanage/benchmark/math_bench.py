"""MATH benchmark implementation."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import Benchmark, InteractionMode, TaskInstance, register_benchmark
from .prompts import MATH_COT_PROMPT, MATH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@register_benchmark("math")
class MathBenchmark(Benchmark):
    """MATH benchmark (Hendrycks et al., 2021).

    7 subjects x 5 difficulty levels. Single-turn CoT evaluation.
    Answers in \\boxed{} format. Metric: Exact Match Accuracy.

    Args:
        subjects: Filter to specific subjects (None = all 7).
        levels: Filter to specific levels 1-5 (None = all).
        dataset_name: HuggingFace dataset identifier.
    """

    def __init__(
        self,
        subjects: Optional[List[str]] = None,
        levels: Optional[List[int]] = None,
        dataset_name: str = "EleutherAI/hendrycks_math",
    ) -> None:
        self._subjects = subjects or SUBJECTS
        self._levels = levels
        self._dataset_name = dataset_name

    @property
    def name(self) -> str:
        return "math"

    def get_interaction_mode(self) -> InteractionMode:
        return InteractionMode.SINGLE_TURN

    def load_tasks(
        self, split: str = "train", limit: Optional[int] = None
    ) -> List[TaskInstance]:
        """Load MATH tasks from HuggingFace.

        Loads each subject separately and merges. Each task is tagged with
        task_type = "{subject}_level{N}" for PatternBuffer bucketing.

        Args:
            split: 'train' or 'test'.
            limit: Max total tasks (None = all).

        Returns:
            List of TaskInstance.
        """
        from datasets import load_dataset

        tasks: List[TaskInstance] = []
        task_counter = 0

        for subject in self._subjects:
            logger.info("Loading MATH/%s/%s", subject, split)
            # Support local path (no network) or HuggingFace hub name
            try:
                ds = load_dataset(self._dataset_name, subject, split=split)
            except Exception:
                # Fallback: try loading as local directory
                import os
                local_path = os.path.join(self._dataset_name, subject)
                logger.info("HuggingFace hub failed, trying local: %s", local_path)
                ds = load_dataset(local_path, split=split)

            for example in ds:
                level_str = example["level"]  # e.g., "Level 5"
                level_num = _extract_level_number(level_str)

                if self._levels and level_num not in self._levels:
                    continue

                task_type = f"{subject}_level{level_num}"
                ground_truth = extract_boxed_answer(example["solution"])

                if not ground_truth:
                    logger.debug("Skipping task with no boxed answer: %s", example["problem"][:50])
                    continue

                task = TaskInstance(
                    task_id=f"math_{subject}_{task_counter}",
                    instruction=example["problem"],
                    task_type=task_type,
                    ground_truth=ground_truth,
                    metadata={
                        "subject": subject,
                        "level": level_num,
                        "level_str": level_str,
                        "full_solution": example["solution"],
                    },
                )
                tasks.append(task)
                task_counter += 1

                if limit and len(tasks) >= limit:
                    break

            if limit and len(tasks) >= limit:
                break

        logger.info("Loaded %d MATH tasks (%s split)", len(tasks), split)
        return tasks

    def build_prompt(self, task: TaskInstance, skills_prompt: str) -> str:
        """Build CoT prompt for a MATH problem."""
        return MATH_COT_PROMPT.format(
            skills_prompt=skills_prompt,
            instruction=task.instruction,
        )

    @property
    def system_prompt(self) -> str:
        """System prompt for MATH."""
        return MATH_SYSTEM_PROMPT

    def check_answer(
        self, task: TaskInstance, agent_output: str
    ) -> Tuple[bool, float]:
        """Check agent answer against ground truth.

        Uses HuggingFace math-verify for robust LaTeX comparison.
        Falls back to string normalization if math-verify unavailable.

        Args:
            task: Task with ground truth.
            agent_output: Agent's full CoT output.

        Returns:
            (success, reward) where reward is 0.0 or 1.0.
        """
        predicted = extract_boxed_answer(agent_output)
        if not predicted:
            predicted = _fallback_extract_answer(agent_output)

        if not predicted:
            return False, 0.0

        correct = is_equiv(predicted, task.ground_truth)
        return correct, 1.0 if correct else 0.0

    def extract_trajectory(self, agent_output: str) -> List[str]:
        """Extract reasoning steps from CoT output.

        Groups consecutive non-empty lines into paragraphs (split by blank lines).
        Each paragraph becomes one trajectory step, representing a reasoning stage
        rather than individual lines. This prevents over-fragmentation.

        If no blank lines found (single paragraph), splits by sentence-ending patterns
        and groups into chunks of ~3 sentences.
        """
        text = agent_output.strip()

        # Try splitting by blank lines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if len(paragraphs) >= 2:
            # Good paragraph structure, use it
            return paragraphs

        # No blank lines — split by lines and group into chunks
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) <= 4:
            return lines

        # Group every 3 lines into one step
        chunk_size = max(len(lines) // 4, 2)  # Aim for ~4 steps
        steps = []
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i : i + chunk_size])
            steps.append(chunk)
        return steps

    def get_task_types(self) -> List[str]:
        """Return all 35 task_type combinations (7 subjects x 5 levels)."""
        return [f"{s}_level{l}" for s in SUBJECTS for l in range(1, 6)]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{} content from text.

    Handles nested braces (e.g., \\boxed{\\frac{1}{2}}).
    Takes the LAST \\boxed{} since intermediate results may be boxed too.

    Args:
        text: Text containing \\boxed{answer}.

    Returns:
        Content inside \\boxed{}, or empty string if not found.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""

    # Match nested braces
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return ""

    return text[start : i - 1].strip()


def _fallback_extract_answer(agent_output: str) -> str:
    """Fallback answer extraction when no \\boxed{} found.

    Tries patterns like "the answer is X", "answer: X", etc.
    """
    patterns = [
        r"[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)",
        r"[Aa]nswer[:\s]*(.+?)(?:\.|$)",
        r"= (.+?)$",
    ]
    lines = agent_output.strip().split("\n")
    # Search from last line backwards
    for line in reversed(lines):
        for pattern in patterns:
            match = re.search(pattern, line.strip())
            if match:
                answer = match.group(1).strip().rstrip(".")
                answer = answer.strip("$")
                return answer
    return ""


# ---------------------------------------------------------------------------
# Answer comparison — uses HuggingFace math-verify with fallback
# ---------------------------------------------------------------------------

# Try to import math-verify (pip install math-verify[antlr4_13_2])
_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import parse as mv_parse, verify as mv_verify
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    logger.warning(
        "math-verify not installed. Using basic string comparison. "
        "Install with: pip install math-verify[antlr4_13_2]"
    )


def is_equiv(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer is equivalent to ground truth.

    Requires math-verify. Raises ImportError if not installed.

    Args:
        predicted: Predicted answer string.
        ground_truth: Ground truth answer string.

    Returns:
        True if answers are equivalent.
    """
    if not _MATH_VERIFY_AVAILABLE:
        raise ImportError(
            "math-verify is required for MATH answer comparison. "
            "Install with: pip install math-verify[antlr4_13_2]"
        )
    return _math_verify_equiv(predicted, ground_truth)


def _pre_normalize(s: str) -> str:
    """Light normalization before passing to math-verify.

    Fixes formatting issues that math-verify can't handle,
    e.g. '(18,-18)' -> '(18, -18)'.
    """
    # Normalize comma spacing for tuples: "(18,-18)" -> "(18, -18)"
    # Only add space when comma is followed by minus sign (not for "1,000,000")
    s = re.sub(r",\s*(?=-)", ", ", s)
    s = " ".join(s.split())
    return s


def _math_verify_equiv(predicted: str, ground_truth: str) -> bool:
    """Compare using HuggingFace math-verify."""
    try:
        pred_norm = _pre_normalize(predicted)
        gt_norm = _pre_normalize(ground_truth)
        gold = mv_parse(gt_norm)
        answer = mv_parse(pred_norm)
        return mv_verify(gold, answer)
    except Exception as e:
        logger.debug("math-verify parse/verify failed: %s. Treating as not equal.", e)
        return False


def _extract_level_number(level_str: str) -> int:
    """Extract level number from 'Level N' string.

    Args:
        level_str: e.g., "Level 5", "Level ?"

    Returns:
        Level number (1-5), defaults to 3 if unparseable.
    """
    match = re.search(r"(\d+)", level_str)
    if match:
        return int(match.group(1))
    return 3
