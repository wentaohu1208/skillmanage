"""Forgetting: degrade low-importance skills to Archive."""

from __future__ import annotations

import logging
from typing import Dict, List

from ..config import ActiveConfig
from ..core.embedding import EmbeddingModel
from ..core.models import ArchivedSkill, Skill
from ..core.skill_bank import SkillBank
from ..llm.base import BaseLLMClient
from ..llm.prompts import ARCHIVE_COMPRESS_PROMPT

logger = logging.getLogger(__name__)


class ForgettingManager:
    """Manages skill degradation from Active to Archive.

    Two triggers:
    1. Natural: importance below threshold for T1 consecutive rounds.
    2. Forced: budget overflow after Merge+Distill.
    """

    def check_natural_forgetting(
        self,
        skill_bank: SkillBank,
        importance_scores: Dict[str, float],
        cfg: ActiveConfig,
    ) -> List[str]:
        """Find skills that should be naturally degraded.

        Updates low_importance_streak and returns IDs of skills
        that have been below threshold for T1 consecutive rounds.

        Args:
            skill_bank: The skill bank.
            importance_scores: Current importance scores.
            cfg: Active configuration.

        Returns:
            List of skill IDs to degrade.
        """
        to_degrade = []

        for skill_id, active_skill in list(skill_bank.active.items()):
            score = importance_scores.get(skill_id, 0.0)

            # Quality floor: force degrade skills with low success rate
            if (active_skill.meta.call_count >= cfg.quality_floor_min_calls
                    and active_skill.meta.success_rate < cfg.quality_floor):
                to_degrade.append(skill_id)
                logger.info(
                    "Quality floor: '%s' (sr=%.2f, calls=%d)",
                    active_skill.skill.name,
                    active_skill.meta.success_rate,
                    active_skill.meta.call_count,
                )
                continue

            if score < cfg.archive_threshold:
                active_skill.meta.low_importance_streak += 1
                if active_skill.meta.low_importance_streak >= cfg.consecutive_rounds:
                    to_degrade.append(skill_id)
                    logger.info(
                        "Natural forgetting: '%s' (score=%.3f, streak=%d)",
                        active_skill.skill.name, score,
                        active_skill.meta.low_importance_streak,
                    )
            else:
                active_skill.meta.low_importance_streak = 0

        return to_degrade

    def force_forget(
        self,
        skill_bank: SkillBank,
        importance_scores: Dict[str, float],
        n_to_remove: int,
    ) -> List[str]:
        """Force-remove the N lowest importance skills (budget-driven).

        Args:
            skill_bank: The skill bank.
            importance_scores: Current importance scores.
            n_to_remove: Number of skills to remove.

        Returns:
            List of skill IDs to degrade.
        """
        sorted_skills = sorted(
            importance_scores.items(), key=lambda x: x[1]
        )
        to_degrade = [sid for sid, _ in sorted_skills[:n_to_remove]]

        for sid in to_degrade:
            active = skill_bank.get_active_skill(sid)
            name = active.skill.name if active else sid
            logger.info(
                "Forced forgetting: '%s' (score=%.3f)",
                name, importance_scores.get(sid, 0),
            )

        return to_degrade

    def execute_degradation(
        self,
        skill_bank: SkillBank,
        skill_ids: List[str],
        llm_client: BaseLLMClient,
        embedding_model: EmbeddingModel,
        current_round: int,
    ) -> List[ArchivedSkill]:
        """Execute degradation: compress and move skills to Archive.

        Args:
            skill_bank: The skill bank.
            skill_ids: Skills to degrade.
            llm_client: For compressing skill descriptions.
            embedding_model: For computing summary embeddings.
            current_round: Current round number.

        Returns:
            List of created ArchivedSkills.
        """
        archived = []
        for skill_id in skill_ids:
            active_skill = skill_bank.get_active_skill(skill_id)
            if active_skill is None:
                continue

            # Compress skill description
            summary = self._compress_for_archive(active_skill.skill, llm_client)
            summary_emb = embedding_model.encode(summary)

            # Move to archive
            result = skill_bank.move_active_to_archive(
                skill_id, summary, summary_emb, current_round
            )
            if result:
                archived.append(result)

        return archived

    def _compress_for_archive(self, skill: Skill, llm_client: BaseLLMClient) -> str:
        """Compress a skill into a one-sentence summary for Archive storage."""
        prompt = ARCHIVE_COMPRESS_PROMPT.format(
            skill_name=skill.name,
            skill_description=skill.description,
            skill_steps="\n".join(skill.steps),
        )

        try:
            return llm_client.generate(prompt).strip()
        except Exception as e:
            logger.warning("Archive compression failed: %s. Using description.", e)
            return skill.description
