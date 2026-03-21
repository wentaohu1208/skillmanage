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

        Splits by newlines, filters empty lines.
        Each non-empty line becomes a trajectory step.
        """
        lines = agent_output.strip().split("\n")
        steps = [line.strip() for line in lines if line.strip()]
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
        # Also try \boxed without backslash escape
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

    Uses HuggingFace math-verify if available (ANTLR4 + SymPy cascade).
    Falls back to basic string normalization otherwise.

    Args:
        predicted: Predicted answer string.
        ground_truth: Ground truth answer string.

    Returns:
        True if answers are equivalent.
    """
    if _MATH_VERIFY_AVAILABLE:
        return _math_verify_equiv(predicted, ground_truth)
    return _fallback_equiv(predicted, ground_truth)


def _math_verify_equiv(predicted: str, ground_truth: str) -> bool:
    """Compare using HuggingFace math-verify."""
    try:
        gold = mv_parse(ground_truth)
        answer = mv_parse(predicted)
        return mv_verify(gold, answer)
    except Exception as e:
        logger.debug("math-verify failed: %s. Falling back to string comparison.", e)
        return _fallback_equiv(predicted, ground_truth)


def _fallback_equiv(predicted: str, ground_truth: str) -> bool:
    """Basic fallback: normalize strings then compare.

    Used when math-verify is not installed.
    """
    pred = _normalize_basic(predicted)
    gt = _normalize_basic(ground_truth)

    # Direct string match
    if pred == gt:
        return True

    # Try numeric comparison
    try:
        pred_val = float(pred)
        gt_val = float(gt)
        return abs(pred_val - gt_val) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


def _normalize_basic(answer: str) -> str:
    """Basic LaTeX normalization for fallback comparison."""
    s = answer.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", " ").replace("\\ ", " ")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # Remove \text{X} -> X, \textbf{X} -> X, etc.
    for cmd in ["\\textbf", "\\textit", "\\mathrm", "\\mathbf", "\\text"]:
        while cmd + "{" in s:
            start = s.find(cmd + "{")
            brace = start + len(cmd)
            depth, j = 0, brace
            while j < len(s):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth == 0:
                s = s[:start] + s[brace + 1 : j] + s[j + 1 :]
            else:
                break

    s = s.strip("$.,;: ")
    s = " ".join(s.split())
    return s


def _extract_level_number(level_str: str) -> int:
    """Extract level number from 'Level N' string.

    Args:
        level_str: e.g., "Level 5", "Level ?"

    Returns:
        Level number (1-5), defaults to 3 if unparseable.
    """
    match = re.search(r"(\d)", level_str)
    if match:
        return int(match.group(1))
    return 3
