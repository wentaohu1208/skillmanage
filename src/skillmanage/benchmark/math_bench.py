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
            ds = load_dataset(self._dataset_name, subject, split=split)

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

        Uses three-level comparison:
        1. Normalized string match
        2. Numeric value comparison
        3. Sympy symbolic comparison (if available)

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
# Answer comparison (three-level)
# ---------------------------------------------------------------------------


def is_equiv(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer is equivalent to ground truth.

    Three-level comparison:
    1. Normalized string match
    2. Numeric value comparison (tolerance 1e-6)
    3. Sympy symbolic comparison (if sympy available)

    Args:
        predicted: Predicted answer string.
        ground_truth: Ground truth answer string.

    Returns:
        True if answers are equivalent.
    """
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)

    # Level 1: Direct string match
    if pred == gt:
        return True

    # Level 2: Numeric comparison
    try:
        pred_val = _latex_to_number(pred)
        gt_val = _latex_to_number(gt)
        if pred_val is not None and gt_val is not None:
            return abs(pred_val - gt_val) < 1e-6
    except (ValueError, ZeroDivisionError, OverflowError):
        pass

    # Level 3: Sympy symbolic comparison
    try:
        return _sympy_equiv(pred, gt)
    except Exception:
        pass

    return False


def normalize_answer(answer: str) -> str:
    """Normalize a LaTeX answer string for comparison.

    Args:
        answer: Raw answer string.

    Returns:
        Normalized answer.
    """
    s = answer.strip()

    # Remove common LaTeX wrappers
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "")
    s = s.replace("\\,", " ")
    s = s.replace("\\ ", " ")

    # Normalize fraction commands
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")

    # Remove text formatting
    for cmd in ["\\text", "\\textbf", "\\textit", "\\mathrm", "\\mathbf"]:
        s = s.replace(cmd, "")

    # Remove dollar signs and trailing punctuation
    s = s.strip("$.,;: ")

    # Collapse whitespace
    s = " ".join(s.split())

    return s


def _latex_to_number(latex: str) -> Optional[float]:
    """Try to convert a LaTeX expression to a float.

    Handles: integers, decimals, fractions, negative numbers.

    Args:
        latex: Normalized LaTeX string.

    Returns:
        Float value, or None if conversion fails.
    """
    s = latex.strip()

    # Direct number
    try:
        return float(s)
    except ValueError:
        pass

    # \frac{a}{b}
    frac_match = re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            return num / den
        except (ValueError, ZeroDivisionError):
            pass

    # Simple fraction a/b
    if "/" in s and "\\" not in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass

    # \sqrt{x}
    sqrt_match = re.match(r"\\sqrt\{(\d+)\}", s)
    if sqrt_match:
        try:
            return float(sqrt_match.group(1)) ** 0.5
        except ValueError:
            pass

    return None


def _sympy_equiv(pred: str, gt: str) -> bool:
    """Use sympy to check symbolic equivalence.

    Args:
        pred: Normalized predicted answer.
        gt: Normalized ground truth.

    Returns:
        True if symbolically equivalent.

    Raises:
        Exception: If sympy parsing fails.
    """
    from sympy.parsing.latex import parse_latex

    pred_expr = parse_latex(pred)
    gt_expr = parse_latex(gt)

    diff = pred_expr - gt_expr
    try:
        from sympy import simplify
        return simplify(diff) == 0
    except Exception:
        return pred_expr.equals(gt_expr)


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
