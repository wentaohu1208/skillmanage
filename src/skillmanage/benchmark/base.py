"""Benchmark ABC, TaskInstance, TaskResult, and registry."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InteractionMode(Enum):
    """How the agent interacts with the benchmark."""

    SINGLE_TURN = "single_turn"  # MATH, BBH: one prompt -> one response
    MULTI_STEP = "multi_step"    # ALFWorld, WebShop: iterative loop


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskInstance:
    """One task from a benchmark (immutable input)."""

    task_id: str
    instruction: str
    task_type: str
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of running one task (mutable, populated during execution)."""

    task_id: str
    task_type: str
    success: bool = False
    reward: float = 0.0
    trajectory: List[str] = field(default_factory=list)
    used_skill_ids: List[str] = field(default_factory=list)
    agent_answer: str = ""
    ground_truth: str = ""
    num_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: Dict[str, Type[Benchmark]] = {}


def register_benchmark(name: str):
    """Decorator to register a benchmark implementation.

    Args:
        name: Benchmark name (e.g., 'math', 'bbh', 'alfworld', 'webshop').
    """
    def decorator(cls: Type[Benchmark]) -> Type[Benchmark]:
        BENCHMARK_REGISTRY[name] = cls
        logger.debug("Registered benchmark: %s", name)
        return cls
    return decorator


def create_benchmark(name: str, **kwargs: Any) -> Benchmark:
    """Create a benchmark by name.

    Args:
        name: Benchmark name.
        **kwargs: Benchmark-specific arguments.

    Returns:
        Configured benchmark instance.

    Raises:
        ValueError: If benchmark not registered.
    """
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {list(BENCHMARK_REGISTRY.keys())}"
        )
    return BENCHMARK_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class Benchmark(ABC):
    """Abstract base for all benchmarks.

    Defines the interface for data loading, prompt building,
    answer checking, and trajectory extraction.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""

    @abstractmethod
    def get_interaction_mode(self) -> InteractionMode:
        """Return interaction mode (single-turn or multi-step)."""

    @abstractmethod
    def load_tasks(
        self, split: str = "train", limit: Optional[int] = None
    ) -> List[TaskInstance]:
        """Load tasks from the benchmark dataset.

        Args:
            split: 'train' or 'test'.
            limit: Max number of tasks to load (None = all).

        Returns:
            List of TaskInstance.
        """

    @abstractmethod
    def build_prompt(self, task: TaskInstance, skills_prompt: str) -> str:
        """Build the agent prompt for a task.

        Args:
            task: The task to solve.
            skills_prompt: Pre-formatted skills section from SkillRetriever.

        Returns:
            Complete prompt string.
        """

    @abstractmethod
    def check_answer(
        self, task: TaskInstance, agent_output: str
    ) -> Tuple[bool, float]:
        """Check agent output against ground truth.

        Args:
            task: The task with ground truth.
            agent_output: Raw agent output.

        Returns:
            Tuple of (success, reward). Reward is 0-1.
        """

    @abstractmethod
    def extract_trajectory(self, agent_output: str) -> List[str]:
        """Extract reasoning/action steps from agent output.

        Used by the Acquisition pipeline for skill extraction.

        Args:
            agent_output: Raw agent output.

        Returns:
            List of step strings.
        """

    @abstractmethod
    def get_task_types(self) -> List[str]:
        """Return all possible task_type labels for this benchmark."""


class InteractiveBenchmark(Benchmark):
    """Extended interface for multi-step benchmarks (ALFWorld, WebShop)."""

    def get_interaction_mode(self) -> InteractionMode:
        return InteractionMode.MULTI_STEP

    @abstractmethod
    def build_system_prompt(self, skills_prompt: str) -> str:
        """Build system prompt with skills (called once per task).

        Args:
            skills_prompt: Pre-formatted skills section.

        Returns:
            System prompt string.
        """

    @abstractmethod
    def build_step_prompt(
        self, task: TaskInstance, observation: str, history: List[str]
    ) -> str:
        """Build prompt for one step (called each step).

        Args:
            task: The current task.
            observation: Current environment observation.
            history: List of previous "Action: ..." and "Observation: ..." strings.

        Returns:
            Step prompt string.
        """

    @abstractmethod
    def reset_env(self, task: TaskInstance) -> str:
        """Reset environment for a task.

        Args:
            task: The task to set up.

        Returns:
            Initial observation string.
        """

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool]:
        """Execute one action in the environment.

        Args:
            action: Action string.

        Returns:
            Tuple of (observation, reward, done).
        """
