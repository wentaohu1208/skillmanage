"""Trajectory segmentation into meaningful parts."""

from __future__ import annotations

import logging
from typing import List

from ..core.models import Segment, SegmentedTrajectory
from ..llm.base import BaseLLMClient
from ..llm.prompts import SEGMENTATION_INTERACTIVE_PROMPT, SEGMENTATION_REASONING_PROMPT

logger = logging.getLogger(__name__)

# Task types that use interactive segmentation
INTERACTIVE_TASK_TYPES = {
    "pick_and_place", "look_at_obj_in_light",
    "pick_clean_then_place", "pick_heat_then_place",
    "pick_cool_then_place", "pick_two_obj_and_place",
    "webshop",
}


class Segmenter:
    """Segments trajectories into meaningful parts.

    Interactive tasks (ALFWorld/WebShop): split by environment state changes.
    Reasoning tasks (MATH/BBH): split by reasoning stage.
    """

    def segment(
        self,
        trajectory: List[str],
        task_type: str,
        llm_client: BaseLLMClient,
    ) -> SegmentedTrajectory:
        """Segment a trajectory.

        Args:
            trajectory: List of steps.
            task_type: Task type label.
            llm_client: LLM for segmentation.

        Returns:
            SegmentedTrajectory with segments.
        """
        if self._is_interactive(task_type):
            return self._segment_interactive(trajectory, task_type, llm_client)
        return self._segment_reasoning(trajectory, task_type, llm_client)

    def _is_interactive(self, task_type: str) -> bool:
        """Check if task type uses interactive segmentation."""
        return task_type.lower() in INTERACTIVE_TASK_TYPES

    def _segment_interactive(
        self,
        trajectory: List[str],
        task_type: str,
        llm_client: BaseLLMClient,
    ) -> SegmentedTrajectory:
        """Segment an interactive task trajectory."""
        traj_str = "\n".join(f"  step{i+1}: {s}" for i, s in enumerate(trajectory))
        prompt = SEGMENTATION_INTERACTIVE_PROMPT.format(trajectory=traj_str)

        try:
            result = llm_client.generate_json(prompt)
            segments = [
                Segment(steps=s["steps"], subgoal=s["subgoal"])
                for s in result.get("segments", [])
            ]
        except (ValueError, KeyError) as e:
            logger.warning("Interactive segmentation failed: %s. Using single segment.", e)
            segments = [Segment(steps=trajectory, subgoal="complete task")]

        return SegmentedTrajectory(segments=segments, task_type=task_type)

    def _segment_reasoning(
        self,
        trajectory: List[str],
        task_type: str,
        llm_client: BaseLLMClient,
    ) -> SegmentedTrajectory:
        """Segment a reasoning chain (CoT)."""
        traj_str = "\n".join(f"  step{i+1}: {s}" for i, s in enumerate(trajectory))
        prompt = SEGMENTATION_REASONING_PROMPT.format(trajectory=traj_str)

        try:
            result = llm_client.generate_json(prompt)
            segments = [
                Segment(steps=s["steps"], subgoal=s["subgoal"])
                for s in result.get("segments", [])
            ]
        except (ValueError, KeyError) as e:
            logger.warning("Reasoning segmentation failed: %s. Using single segment.", e)
            segments = [Segment(steps=trajectory, subgoal="solve problem")]

        return SegmentedTrajectory(segments=segments, task_type=task_type)
