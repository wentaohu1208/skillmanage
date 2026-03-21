"""Skill retrieval from Active (with Archive fallback)."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..config import RetrievalConfig
from ..utils.similarity import batch_cosine_similarity
from .embedding import EmbeddingModel
from .models import ArchivedSkill, Skill
from .skill_bank import SkillBank

logger = logging.getLogger(__name__)


class SkillRetriever:
    """Retrieves relevant skills for a given task.

    Searches Active first. If no good match (max sim < threshold),
    falls back to searching Archive.
    """

    def retrieve(
        self,
        task_description: str,
        skill_bank: SkillBank,
        embedding_model: EmbeddingModel,
        cfg: RetrievalConfig,
    ) -> Tuple[List[Skill], Optional[ArchivedSkill]]:
        """Retrieve skills for a task.

        Args:
            task_description: Task instruction text.
            skill_bank: The skill bank to search.
            embedding_model: Model for encoding.
            cfg: Retrieval configuration.

        Returns:
            Tuple of (active_skills, archive_hit). archive_hit is non-None
            only if Active had no good match and Archive was searched.
        """
        task_emb = embedding_model.encode_task(task_description)

        active_skills, max_sim = self._retrieve_from_active(
            task_emb, skill_bank, cfg
        )

        if active_skills:
            return active_skills, None

        # Fallback to Archive
        if max_sim < cfg.similarity_threshold:
            archive_hit = self._retrieve_from_archive(
                task_emb, skill_bank, cfg
            )
            if archive_hit:
                logger.info(
                    "Archive fallback: found '%s'",
                    archive_hit.original_skill_full.name,
                )
                return [], archive_hit

        return [], None

    def _retrieve_from_active(
        self,
        task_emb: np.ndarray,
        skill_bank: SkillBank,
        cfg: RetrievalConfig,
    ) -> Tuple[List[Skill], float]:
        """Retrieve top-K skills from Active area.

        Returns:
            Tuple of (matched_skills, max_similarity).
        """
        ids, matrix = skill_bank.get_active_embeddings_matrix()
        if len(ids) == 0:
            return [], 0.0

        sims = batch_cosine_similarity(matrix, task_emb)
        max_sim = float(np.max(sims))

        # Get top-K above threshold
        above_threshold = [
            (ids[i], float(sims[i]))
            for i in range(len(ids))
            if sims[i] >= cfg.similarity_threshold
        ]
        above_threshold.sort(key=lambda x: x[1], reverse=True)
        top_k = above_threshold[: cfg.top_k]

        skills = []
        for skill_id, sim in top_k:
            active_skill = skill_bank.get_active_skill(skill_id)
            if active_skill:
                skills.append(active_skill.skill)
                logger.debug(
                    "Retrieved '%s' (sim=%.3f)", active_skill.skill.name, sim
                )

        return skills, max_sim

    def _retrieve_from_archive(
        self,
        task_emb: np.ndarray,
        skill_bank: SkillBank,
        cfg: RetrievalConfig,
    ) -> Optional[ArchivedSkill]:
        """Search Archive for a matching skill.

        Returns:
            Best matching ArchivedSkill, or None.
        """
        ids, matrix = skill_bank.get_archive_embeddings_matrix()
        if len(ids) == 0:
            return None

        sims = batch_cosine_similarity(matrix, task_emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= cfg.similarity_threshold:
            return skill_bank.get_archived_skill(ids[best_idx])
        return None

    @staticmethod
    def format_skills_for_prompt(skills: List[Skill]) -> str:
        """Format retrieved skills for inclusion in agent prompt.

        Args:
            skills: List of skills to format.

        Returns:
            Formatted string for prompt insertion.
        """
        if not skills:
            return ""
        parts = ["The following skills may be useful:\n"]
        for i, skill in enumerate(skills, 1):
            parts.append(f"--- Skill {i} ---")
            parts.append(skill.to_prompt_str())
            parts.append("")
        return "\n".join(parts)
