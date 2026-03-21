"""Meta updates after task execution."""

from __future__ import annotations

import logging
from typing import List

from ..core.models import ActiveSkill
from ..core.skill_bank import SkillBank

logger = logging.getLogger(__name__)


class MetaUpdater:
    """Updates skill metadata after task execution."""

    def update_after_task(
        self,
        skill_bank: SkillBank,
        used_skill_ids: List[str],
        success: bool,
        reward: float,
        current_round: int,
    ) -> None:
        """Update meta for skills that were used in a task.

        Args:
            skill_bank: The skill bank.
            used_skill_ids: IDs of skills actually used (not just retrieved).
            success: Whether the task succeeded.
            reward: Task reward (0-1).
            current_round: Current round number.
        """
        for skill_id in used_skill_ids:
            active_skill = skill_bank.get_active_skill(skill_id)
            if active_skill is None:
                continue
            active_skill.meta.update_after_use(success, reward, current_round)
            logger.debug(
                "Updated meta for '%s': count=%d rate=%.2f reward=%.2f",
                active_skill.skill.name,
                active_skill.meta.call_count,
                active_skill.meta.success_rate,
                active_skill.meta.avg_reward,
            )
