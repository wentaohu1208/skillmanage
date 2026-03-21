"""Forgotten area management: deduplication blacklist with time decay."""

from __future__ import annotations

import logging

import numpy as np

from ..config import ForgottenConfig
from ..core.embedding import EmbeddingModel
from ..core.models import Skill
from ..core.skill_bank import SkillBank
from ..utils.similarity import batch_cosine_similarity

logger = logging.getLogger(__name__)


class ForgottenManager:
    """Manages the Forgotten area (dedup blacklist).

    Prevents re-learning skills that were previously learned and forgotten.
    Allows re-learning after sufficient time has passed (time decay).
    """

    def check_dedup(
        self,
        new_skill: Skill,
        skill_bank: SkillBank,
        embedding_model: EmbeddingModel,
        current_round: int,
        cfg: ForgottenConfig = ForgottenConfig(),
    ) -> bool:
        """Check if a new skill should be skipped due to Forgotten dedup.

        Args:
            new_skill: Skill about to enter Active.
            skill_bank: The skill bank (for Forgotten pool).
            embedding_model: For semantic matching.
            current_round: Current round.
            cfg: Forgotten configuration.

        Returns:
            True if skill should be SKIPPED (recently forgotten duplicate).
            False if skill should be allowed.
        """
        ids, matrix = skill_bank.get_forgotten_embeddings_matrix()
        if len(ids) == 0:
            return False

        new_emb = embedding_model.encode_skill(new_skill)
        sims = batch_cosine_similarity(matrix, new_emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < cfg.dedup_threshold:
            return False  # No match in Forgotten

        # Found a match — check time decay
        matched_id = ids[best_idx]
        forgotten_skill = skill_bank.forgotten.get(matched_id)
        if forgotten_skill is None:
            return False

        rounds_since_forgotten = current_round - forgotten_skill.forgotten_at

        if rounds_since_forgotten < cfg.time_decay_rounds:
            logger.info(
                "Dedup: '%s' matches forgotten '%s' (sim=%.2f, forgotten %d rounds ago < D=%d). Skipping.",
                new_skill.name, forgotten_skill.name,
                best_sim, rounds_since_forgotten, cfg.time_decay_rounds,
            )
            return True  # Recently forgotten, skip

        logger.info(
            "Dedup: '%s' matches forgotten '%s' (sim=%.2f, forgotten %d rounds ago >= D=%d). Allowing re-learn.",
            new_skill.name, forgotten_skill.name,
            best_sim, rounds_since_forgotten, cfg.time_decay_rounds,
        )
        return False  # Forgotten long ago, allow re-learning
