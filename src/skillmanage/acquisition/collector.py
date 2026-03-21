"""Collection decision: route success/failure trajectories."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..config import AcquisitionConfig
from ..core.models import CollectionDecision, Skill
from ..llm.base import BaseLLMClient
from ..llm.prompts import COVERAGE_JUDGMENT_PROMPT

logger = logging.getLogger(__name__)


class CollectionDecider:
    """Decides what to collect from a completed task.

    Routes to success path (extract skill steps) or failure path (extract warnings).
    For success with skill usage, judges coverage to decide what's new.
    """

    def decide(
        self,
        success: bool,
        trajectory: List[str],
        used_skills: List[Skill],
        llm_client: BaseLLMClient,
        cfg: AcquisitionConfig,
    ) -> CollectionDecision:
        """Make collection decision for a completed task.

        Args:
            success: Whether the task succeeded.
            trajectory: List of action/reasoning steps.
            used_skills: Skills that were retrieved and used.
            llm_client: LLM client for coverage judgment.
            cfg: Acquisition configuration.

        Returns:
            CollectionDecision with routing info.
        """
        if success:
            return self._decide_success(trajectory, used_skills, llm_client)
        return self._decide_failure(trajectory, cfg)

    def _decide_success(
        self,
        trajectory: List[str],
        used_skills: List[Skill],
        llm_client: BaseLLMClient,
    ) -> CollectionDecision:
        """Handle successful task."""
        if not used_skills:
            # No skill used (bare run) -> collect everything
            logger.debug("Bare run success -> collect full trajectory")
            return CollectionDecision(
                path="success",
                full_trajectory=trajectory,
                coverage_rate=0.0,
            )

        # Judge coverage
        coverage = self._judge_coverage(trajectory, used_skills, llm_client)
        rate = coverage.get("coverage_rate", 0.0)

        if rate >= 1.0:
            # Fully covered -> nothing new
            logger.debug("Full coverage (%.1f%%) -> skip collection", rate * 100)
            return CollectionDecision(path="skip", coverage_rate=rate)

        if rate > 0.5:
            # Partially covered -> collect uncovered segments
            uncovered = coverage.get("uncovered_steps", [])
            # LLM may return string indices, ensure int
            uncovered_int = []
            for idx in uncovered:
                try:
                    uncovered_int.append(int(idx))
                except (ValueError, TypeError):
                    continue
            uncovered_text = [trajectory[i] for i in uncovered_int if i < len(trajectory)]
            logger.debug(
                "Partial coverage (%.1f%%) -> collect %d uncovered steps",
                rate * 100, len(uncovered_text),
            )
            return CollectionDecision(
                path="success",
                segments_to_collect=uncovered_text,
                coverage_rate=rate,
            )

        # Low coverage -> collect full trajectory
        logger.debug("Low coverage (%.1f%%) -> collect full trajectory", rate * 100)
        return CollectionDecision(
            path="success",
            full_trajectory=trajectory,
            coverage_rate=rate,
        )

    def _decide_failure(
        self, trajectory: List[str], cfg: AcquisitionConfig
    ) -> CollectionDecision:
        """Handle failed task."""
        if len(trajectory) < cfg.min_failure_steps:
            logger.debug(
                "Failure with %d steps < %d -> skip",
                len(trajectory), cfg.min_failure_steps,
            )
            return CollectionDecision(path="skip")

        logger.debug("Failure with %d steps -> analyze", len(trajectory))
        return CollectionDecision(
            path="failure",
            full_trajectory=trajectory,
            failure_info={"step_count": len(trajectory)},
        )

    def _judge_coverage(
        self,
        trajectory: List[str],
        used_skills: List[Skill],
        llm_client: BaseLLMClient,
    ) -> Dict[str, Any]:
        """Judge how much of the trajectory is covered by used skills.

        Args:
            trajectory: Agent's execution steps.
            used_skills: Skills that were used.
            llm_client: LLM for judgment.

        Returns:
            Dict with coverage_rate, covered_steps, uncovered_steps.
        """
        # Format trajectory and skills for prompt
        traj_str = "\n".join(f"  step{i}: {s}" for i, s in enumerate(trajectory))
        skill_names = ", ".join(s.name for s in used_skills)
        skill_steps = "\n".join(
            f"  [{s.name}]: " + " -> ".join(s.steps) for s in used_skills
        )

        prompt = COVERAGE_JUDGMENT_PROMPT.format(
            skill_name=skill_names,
            skill_steps=skill_steps,
            trajectory=traj_str,
        )

        try:
            result = llm_client.generate_json(prompt)
            return {
                "coverage_rate": float(result.get("coverage_rate", 0.0)),
                "covered_steps": result.get("covered_steps", []),
                "uncovered_steps": result.get("uncovered_steps", []),
            }
        except (ValueError, KeyError) as e:
            logger.warning("Coverage judgment failed: %s. Assuming no coverage.", e)
            return {
                "coverage_rate": 0.0,
                "covered_steps": [],
                "uncovered_steps": list(range(len(trajectory))),
            }
