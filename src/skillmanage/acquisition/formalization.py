"""Formalization: convert patterns to standard skill format."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..config import AcquisitionConfig
from ..core.embedding import EmbeddingModel
from ..core.models import PatternBuffer, PatternEntry, Skill
from ..core.skill_bank import SkillBank
from ..llm.base import BaseLLMClient
from ..llm.prompts import FORMALIZATION_PROMPT
from ..utils import count_tokens, generate_skill_id

logger = logging.getLogger(__name__)


class Formalizer:
    """Converts extracted patterns into standard skill format.

    Also handles Forgotten deduplication check before finalizing.
    """

    def formalize(
        self,
        pattern: PatternEntry,
        task_type: str,
        variants: List[str],
        confidence: float,
        llm_client: BaseLLMClient,
        skill_bank: SkillBank,
        embedding_model: EmbeddingModel,
        forgotten_manager,  # Forward ref to avoid circular import
        current_round: int,
        cfg: AcquisitionConfig,
    ) -> Optional[Skill]:
        """Convert a pattern to a Skill and check dedup.

        Args:
            pattern: Pattern entry to formalize.
            task_type: Source task type.
            variants: Variant descriptions found during alignment.
            confidence: Current confidence of the pattern.
            llm_client: LLM client for formalization.
            skill_bank: For Forgotten dedup check.
            embedding_model: For embedding computation.
            forgotten_manager: For dedup check.
            current_round: Current round.
            cfg: Configuration.

        Returns:
            New Skill, or None if dedup rejects it.
        """
        # Generate skill via LLM
        variant_str = ", ".join(variants) if variants else "none"
        prompt = FORMALIZATION_PROMPT.format(
            pattern_description=pattern.description,
            confidence=f"{confidence:.2f}",
            variants=variant_str,
            task_type=task_type,
        )

        try:
            result = llm_client.generate_json(prompt)
        except ValueError as e:
            logger.warning("Formalization LLM failed: %s", e)
            return None

        # Build Skill object
        skill_id = generate_skill_id("sk")
        name = result.get("name", pattern.description[:30])
        description = result.get("description", pattern.description)
        steps = result.get("steps", [])
        parameters = result.get("parameters", [])
        precondition = result.get("precondition", "")

        # Calculate token cost
        skill_text = f"{description} {' '.join(steps)}"
        token_cost = count_tokens(skill_text)

        skill = Skill(
            skill_id=skill_id,
            name=name,
            description=description,
            precondition=precondition,
            parameters=parameters,
            steps=steps,
            warnings=[],
            source="experience_distillation",
            task_type=task_type,
            confidence=confidence,
            token_cost=token_cost,
        )

        # Check Forgotten dedup
        if forgotten_manager is not None:
            should_skip = forgotten_manager.check_dedup(
                skill, skill_bank, embedding_model, current_round
            )
            if should_skip:
                logger.info(
                    "Skill '%s' rejected by Forgotten dedup", skill.name
                )
                return None

        logger.info(
            "Formalized skill '%s' (%s) confidence=%.2f tokens=%d",
            skill.name, skill.skill_id, confidence, token_cost,
        )
        return skill
