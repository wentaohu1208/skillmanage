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
from ..llm.prompts import FAILURE_ANALYSIS_PROMPT, SKILL_FAILURE_DIAGNOSIS_PROMPT, SKILL_REPAIR_PROMPT
from ..utils import count_tokens, generate_skill_id
from ..utils.similarity import batch_cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class WarningAttachment:
    """Result: attach a warning to an existing skill."""

    skill_id: str
    warning_text: str


@dataclass
class SkillRepairResult:
    """Result: a skill was diagnosed as faulty and repaired."""

    skill_id: str
    diagnosis: str  # "B" = skill fault
    reason: str


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

    def diagnose_skill_failure(
        self,
        task_description: str,
        trajectory: List[str],
        used_skill: Skill,
        ground_truth: str,
        agent_answer: str,
        llm_client: BaseLLMClient,
    ) -> Optional[SkillRepairResult]:
        """Diagnose whether a failure is due to skill fault or model limitation.

        Args:
            task_description: The task.
            trajectory: Execution steps.
            used_skill: The skill that was used.
            ground_truth: Correct answer.
            agent_answer: Agent's wrong answer.
            llm_client: LLM for diagnosis.

        Returns:
            SkillRepairResult if skill is faulty (diagnosis="B"), None if model error.
        """
        traj_str = "\n".join(f"  step{i+1}: {s}" for i, s in enumerate(trajectory))
        prompt = SKILL_FAILURE_DIAGNOSIS_PROMPT.format(
            task=task_description,
            skill_name=used_skill.name,
            skill_steps="\n".join(used_skill.steps),
            trajectory=traj_str,
            ground_truth=ground_truth,
            agent_answer=agent_answer,
        )

        try:
            result = llm_client.generate_json(prompt)
        except ValueError as e:
            logger.warning("Skill failure diagnosis failed: %s", e)
            return None

        diagnosis = result.get("diagnosis", "A")
        reason = result.get("reason", "")

        if diagnosis == "B":
            logger.info(
                "Skill '%s' diagnosed as faulty: %s",
                used_skill.name, reason[:80],
            )
            return SkillRepairResult(
                skill_id=used_skill.skill_id,
                diagnosis=diagnosis,
                reason=reason,
            )

        logger.debug(
            "Skill '%s' is correct, model execution error: %s",
            used_skill.name, reason[:80],
        )
        return None

    def repair_skill(
        self,
        skill: Skill,
        task_description: str,
        failure_reason: str,
        ground_truth: str,
        llm_client: BaseLLMClient,
    ) -> Skill:
        """Repair a faulty skill by rewriting its steps.

        Args:
            skill: The skill to repair.
            task_description: The task that exposed the fault.
            failure_reason: Why the skill failed.
            ground_truth: Correct answer.
            llm_client: LLM for repair.

        Returns:
            New Skill object with repaired steps (version incremented).
        """
        prompt = SKILL_REPAIR_PROMPT.format(
            skill_name=skill.name,
            skill_steps="\n".join(skill.steps),
            skill_warnings="\n".join(skill.warnings) or "none",
            task=task_description,
            failure_reason=failure_reason,
            ground_truth=ground_truth,
        )

        try:
            result = llm_client.generate_json(prompt)
        except ValueError as e:
            logger.warning("Skill repair LLM failed: %s. Keeping original.", e)
            return skill

        new_steps = result.get("steps", skill.steps)
        new_warnings = result.get("warnings", list(skill.warnings))

        repaired = Skill(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.description,
            precondition=skill.precondition,
            parameters=skill.parameters,
            steps=new_steps,
            warnings=new_warnings,
            source=skill.source,
            task_type=skill.task_type,
            confidence=skill.confidence,
            token_cost=count_tokens(skill.description + " " + " ".join(new_steps)),
        )

        logger.info(
            "Repaired skill '%s': %d steps -> %d steps",
            skill.name, len(skill.steps), len(new_steps),
        )
        return repaired

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
            name=f"avoid_{warning_text[:50].replace(' ', '_').lower().strip('_')}",
            description=warning_text,
            steps=[warning_text],
            warnings=[],
            source="failure_analysis",
            task_type=task_type,
            confidence=cfg.low_confidence_init,
            token_cost=count_tokens(warning_text),
        )
