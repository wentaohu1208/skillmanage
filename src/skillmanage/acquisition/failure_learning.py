"""Failure learning: extract warnings from failed trajectories."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from ..config import AcquisitionConfig
from ..core.embedding import EmbeddingModel
from ..core.models import Skill
from ..core.skill_bank import SkillBank
from ..llm.base import BaseLLMClient
from ..llm.prompts import FAILURE_ANALYSIS_PROMPT
from ..utils import count_tokens, generate_skill_id
from ..utils.similarity import batch_cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class WarningAttachment:
    """Result: attach a warning to an existing skill."""

    skill_id: str
    warning_text: str


class FailureLearner:
    """Extracts warnings from failed trajectories.

    Three outcomes:
    1. Warning matches existing Active skill -> attach to that skill
    2. Warning is actionable but no match -> create independent skill
    3. Warning is not actionable -> discard
    """

    def analyze_failure(
        self,
        task_description: str,
        trajectory: List[str],
        failure_point: str,
        skill_bank: SkillBank,
        llm_client: BaseLLMClient,
        embedding_model: EmbeddingModel,
        cfg: AcquisitionConfig,
    ) -> Optional[Union[WarningAttachment, Skill]]:
        """Analyze a failed trajectory and extract warning.

        Args:
            task_description: Task that was attempted.
            trajectory: Execution steps.
            failure_point: Description of where/why it failed.
            skill_bank: For matching warnings to existing skills.
            llm_client: LLM for failure analysis.
            embedding_model: For semantic matching.
            cfg: Configuration.

        Returns:
            WarningAttachment if matched to existing skill,
            new Skill if actionable but unmatched,
            None if not actionable.
        """
        # Check minimum steps
        if len(trajectory) < cfg.min_failure_steps:
            return None

        # LLM analyzes failure
        traj_str = "\n".join(f"  step{i+1}: {s}" for i, s in enumerate(trajectory))
        prompt = FAILURE_ANALYSIS_PROMPT.format(
            task=task_description,
            trajectory=traj_str,
            failure_point=failure_point,
        )

        try:
            result = llm_client.generate_json(prompt)
        except ValueError as e:
            logger.warning("Failure analysis LLM failed: %s", e)
            return None

        warning_text = result.get("warning", "")
        actionable = result.get("actionable", False)

        if not warning_text:
            return None

        # Try to match to existing Active skill
        matched_skill_id = self._match_to_active_skill(
            warning_text, skill_bank, embedding_model
        )

        if matched_skill_id:
            logger.info(
                "Warning matched to skill %s: '%s'",
                matched_skill_id, warning_text[:50],
            )
            return WarningAttachment(
                skill_id=matched_skill_id, warning_text=warning_text
            )

        # No match -> check if actionable
        if not actionable:
            logger.debug("Warning not actionable, discarding: '%s'", warning_text[:50])
            return None

        # Create independent skill from warning
        skill = self._create_skill_from_warning(warning_text, result, cfg)
        logger.info("Created skill from failure: '%s'", skill.name)
        return skill

    def _match_to_active_skill(
        self,
        warning_text: str,
        skill_bank: SkillBank,
        embedding_model: EmbeddingModel,
        threshold: float = 0.5,
    ) -> Optional[str]:
        """Match warning to an existing Active skill by semantic similarity.

        Returns:
            Skill ID if matched, None otherwise.
        """
        ids, matrix = skill_bank.get_active_embeddings_matrix()
        if len(ids) == 0:
            return None

        warning_emb = embedding_model.encode(warning_text)
        sims = batch_cosine_similarity(matrix, warning_emb)
        best_idx = int(np.argmax(sims))

        if sims[best_idx] >= threshold:
            return ids[best_idx]
        return None

    def _create_skill_from_warning(
        self,
        warning_text: str,
        analysis: dict,
        cfg: AcquisitionConfig,
    ) -> Skill:
        """Create an independent skill from an actionable warning."""
        task_types = analysis.get("applicable_task_types", [])
        task_type = task_types[0] if task_types else ""

        return Skill(
            skill_id=generate_skill_id("sk"),
            name=f"avoid_{warning_text[:20].replace(' ', '_').lower()}",
            description=warning_text,
            steps=[warning_text],
            warnings=[],
            source="failure_analysis",
            task_type=task_type,
            confidence=cfg.low_confidence_init,
            token_cost=count_tokens(warning_text),
        )
