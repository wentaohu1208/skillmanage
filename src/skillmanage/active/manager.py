"""ActiveManager: orchestrates per-round Active area operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import ActiveConfig, RetrievalConfig
from ..core.embedding import EmbeddingModel
from ..core.skill_bank import SkillBank
from ..llm.base import BaseLLMClient
from .compression import Compressor
from .forgetting import ForgettingManager
from .importance import ImportanceCalculator

logger = logging.getLogger(__name__)


@dataclass
class RoundReport:
    """Report of Active area operations for a round."""

    importance_scores: dict
    skills_archived: int = 0
    merges: int = 0
    distills: int = 0
    tokens_saved: int = 0


class ActiveManager:
    """Top-level orchestrator for Active area per-round operations.

    Runs at the end of each round:
    1. Calculate importance scores
    2. Check natural forgetting (consecutive low importance)
    3. If over budget: run compression cycle (merge -> distill -> force forget)
    """

    def __init__(self) -> None:
        self._importance = ImportanceCalculator()
        self._compressor = Compressor()
        self._forgetting = ForgettingManager()

    def on_round_end(
        self,
        skill_bank: SkillBank,
        current_round: int,
        llm_client: BaseLLMClient,
        embedding_model: EmbeddingModel,
        active_cfg: ActiveConfig,
        retrieval_cfg: RetrievalConfig,
    ) -> RoundReport:
        """Run end-of-round Active area maintenance.

        Args:
            skill_bank: The skill bank.
            current_round: Current round number.
            llm_client: For compression LLM calls.
            embedding_model: For embedding computation.
            active_cfg: Active configuration.
            retrieval_cfg: Retrieval configuration (for budget).

        Returns:
            Report of actions taken.
        """
        report = RoundReport(importance_scores={})

        if not skill_bank.active:
            return report

        # 1. Calculate importance scores
        scores = self._importance.calculate_all(skill_bank, current_round, active_cfg)
        report.importance_scores = scores

        # 2. Check natural forgetting
        to_degrade = self._forgetting.check_natural_forgetting(
            skill_bank, scores, active_cfg
        )
        if to_degrade:
            archived = self._forgetting.execute_degradation(
                skill_bank, to_degrade, llm_client, embedding_model, current_round
            )
            report.skills_archived = len(archived)

        # 3. Compression if over budget
        if skill_bank.is_over_budget(retrieval_cfg):
            comp_report = self._compressor.compress_if_needed(
                skill_bank, embedding_model, llm_client, active_cfg, retrieval_cfg
            )
            report.merges = comp_report.merges
            report.distills = comp_report.distills
            report.tokens_saved = comp_report.tokens_saved

            # If still over budget after merge+distill, force forget
            if comp_report.still_over_budget:
                # Recalculate scores after compression changes
                scores = self._importance.calculate_all(
                    skill_bank, current_round, active_cfg
                )
                overage = skill_bank.get_total_active_tokens() - retrieval_cfg.token_budget
                avg_cost = max(
                    skill_bank.get_total_active_tokens() // max(len(skill_bank.active), 1),
                    1,
                )
                n_to_remove = max(1, overage // avg_cost + 1)

                force_ids = self._forgetting.force_forget(skill_bank, scores, n_to_remove)
                archived = self._forgetting.execute_degradation(
                    skill_bank, force_ids, llm_client, embedding_model, current_round
                )
                report.skills_archived += len(archived)

        return report

    def on_new_skill_added(
        self,
        skill_bank: SkillBank,
        current_round: int,
        llm_client: BaseLLMClient,
        embedding_model: EmbeddingModel,
        active_cfg: ActiveConfig,
        retrieval_cfg: RetrievalConfig,
    ) -> None:
        """Check budget after a new skill is added.

        If over budget, triggers compression cycle.
        """
        if skill_bank.is_over_budget(retrieval_cfg):
            logger.info("New skill pushed over budget, running compression")
            self._compressor.compress_if_needed(
                skill_bank, embedding_model, llm_client, active_cfg, retrieval_cfg
            )
