"""Benchmark abstraction layer: plug-and-play benchmark switching."""

from .base import (
    BENCHMARK_REGISTRY,
    Benchmark,
    InteractiveBenchmark,
    InteractionMode,
    TaskInstance,
    TaskResult,
    create_benchmark,
    register_benchmark,
)
from .alfworld_bench import ALFWorldBenchmark  # noqa: F401 (registers in BENCHMARK_REGISTRY)
from .math_bench import MathBenchmark  # noqa: F401 (registers in BENCHMARK_REGISTRY)
from .runner import AgentRunner

__all__ = [
    "Benchmark",
    "InteractiveBenchmark",
    "InteractionMode",
    "TaskInstance",
    "TaskResult",
    "AgentRunner",
    "BENCHMARK_REGISTRY",
    "register_benchmark",
    "create_benchmark",
]
