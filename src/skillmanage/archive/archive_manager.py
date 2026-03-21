"""Archive management: recall flow, promote, and cleanup."""

from __future__ import annotations

import logging
from typing import Optional

from ..config import ArchiveConfig, ForgottenConfig
from ..core.embedding import EmbeddingModel
from ..core.models import ArchivedSkill
from ..core.skill_bank import SkillBank
from .forgotten_manager import ForgottenManager

logger = logging.getLogger(__name__)


class ArchiveManager:
    """Manages Archive area operations.

    - Record recall results (success/failure)
    - Promote back to Active after R successful recalls
    - Tick inactive rounds and clean up to Forgotten
    """

    def __init__(self, forgotten_manager: ForgottenManager) -> None:
        self._forgotten = forgotten_manager

    def record_recall_result(
        self,
        skill_bank: SkillBank,
        archived_skill_id: str,
        success: bool,
        current_round: int,
        embedding_model: EmbeddingModel,
        cfg: ArchiveConfig,
    ) -> bool:
        """Record the result of recalling an archived skill.

        Args:
            skill_bank: The skill bank.
            archived_skill_id: ID of the recalled skill.
            success: Whether the task succeeded with this skill.
            current_round: Current round.
            embedding_model: For computing embedding on promote.
            cfg: Archive configuration.

        Returns:
            True if skill was promoted back to Active.
        """
        archived = skill_bank.get_archived_skill(archived_skill_id)
        if archived is None:
            return False

        # Reset inactive counter (skill was needed, whether or not it worked)
        archived.inactive_rounds = 0

        if success:
            archived.recall_count += 1
            logger.info(
                "Recall success for '%s' (count=%d/%d)",
                archived.original_skill_full.name,
                archived.recall_count,
                cfg.recall_success_threshold,
            )

            if archived.recall_count >= cfg.recall_success_threshold:
                # Promote back to Active
                emb = embedding_model.encode_skill(archived.original_skill_full)
                skill_bank.promote_archive_to_active(
                    archived_skill_id, emb, current_round
                )
                return True
        else:
            logger.info(
                "Recall failure for '%s' (inactive reset)",
                archived.original_skill_full.name,
            )

        return False

    def tick_inactive(
        self,
        skill_bank: SkillBank,
        current_round: int,
        archive_cfg: ArchiveConfig,
        forgotten_cfg: ForgottenConfig,
    ) -> int:
        """Increment inactive rounds for all Archive skills and clean up.

        Skills that exceed max_inactive_rounds are moved to Forgotten.

        Args:
            skill_bank: The skill bank.
            current_round: Current round.
            archive_cfg: Archive configuration.
            forgotten_cfg: Forgotten configuration (passed to forgotten manager).

        Returns:
            Number of skills moved to Forgotten.
        """
        to_forget = []
        for skill_id, archived in list(skill_bank.archive.items()):
            archived.inactive_rounds += 1
            if archived.inactive_rounds >= archive_cfg.max_inactive_rounds:
                to_forget.append(skill_id)

        count = 0
        for skill_id in to_forget:
            result = skill_bank.move_archive_to_forgotten(skill_id, current_round)
            if result:
                count += 1
                logger.info(
                    "Moved '%s' from Archive to Forgotten (inactive %d rounds)",
                    result.name, archive_cfg.max_inactive_rounds,
                )

        return count
