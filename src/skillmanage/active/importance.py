"""Importance Score calculation (4 dimensions)."""

from __future__ import annotations

import logging
import math
from typing import Dict

from ..config import ActiveConfig
from ..core.skill_bank import SkillBank

logger = logging.getLogger(__name__)


class ImportanceCalculator:
    """Calculates importance scores for all Active skills.

    I(s,t) = w1*Recency + w2*Frequency + w3*Quality + w4*Irreplaceability

    All dimensions normalized to [0, 1].
    """

    def calculate_all(
        self,
        skill_bank: SkillBank,
        current_round: int,
        cfg: ActiveConfig,
    ) -> Dict[str, float]:
        """Calculate importance scores for all Active skills.

        Args:
            skill_bank: The skill bank.
            current_round: Current round number.
            cfg: Active configuration with weights and decay.

        Returns:
            Dict mapping skill_id to importance score.
        """
        scores: Dict[str, float] = {}
        max_count = self._get_max_call_count(skill_bank)

        for skill_id, active_skill in skill_bank.active.items():
            meta = active_skill.meta

            r = self._recency(meta.last_used_at, current_round, cfg.recency_decay)
            f = self._frequency(meta.call_count, max_count)
            q = self._quality(meta.success_rate, meta.avg_reward)
            ir = self._irreplaceability(skill_id, skill_bank)

            score = (
                cfg.weight_recency * r
                + cfg.weight_frequency * f
                + cfg.weight_quality * q
                + cfg.weight_irreplaceability * ir
            )
            scores[skill_id] = score

        return scores

    def _recency(self, last_used_at: int, current_round: int, decay: float) -> float:
        """Recency = exp(-lambda * (current - last_used)).

        Range: [0, 1]. 1.0 if just used, approaches 0 as rounds pass.
        """
        gap = max(0, current_round - last_used_at)
        return math.exp(-decay * gap)

    def _frequency(self, call_count: int, max_count: int) -> float:
        """Frequency = log(1 + count) / log(1 + max_count).

        Normalized to [0, 1]. Log-scaled to prevent high-frequency dominance.
        """
        if max_count <= 0:
            return 0.0
        numerator = math.log(1 + call_count)
        denominator = math.log(1 + max_count)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _quality(self, success_rate: float, avg_reward: float) -> float:
        """Quality = success_rate * avg_reward.

        Range: [0, 1]. Both dimensions must be good.
        """
        return success_rate * avg_reward

    def _irreplaceability(self, skill_id: str, skill_bank: SkillBank) -> float:
        """Irreplaceability = 1 - max_similarity_to_other_skills.

        Range: [0, 1]. 1.0 means completely unique (no similar skill).
        """
        max_sim = skill_bank.max_active_similarity(skill_id)
        return 1.0 - max_sim

    def _get_max_call_count(self, skill_bank: SkillBank) -> int:
        """Get maximum call count across all Active skills (for normalization)."""
        if not skill_bank.active:
            return 0
        return max(s.meta.call_count for s in skill_bank.active.values())
