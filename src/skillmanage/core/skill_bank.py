"""Central SkillBank: Active, Archive, and Forgotten storage."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import RetrievalConfig
from ..utils.similarity import batch_cosine_similarity
from .models import ActiveSkill, ArchivedSkill, ForgottenSkill, Skill, SkillMeta

logger = logging.getLogger(__name__)


class SkillBank:
    """Central storage for all skill lifecycle states.

    Active skills are the Skill Bank — the only source for agent retrieval.
    Archive stores compressed backups of degraded skills.
    Forgotten stores minimal info for deduplication.

    Embeddings are stored as numpy arrays separately from skill objects,
    indexed by skill_id for efficient batch similarity computation.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self.active: Dict[str, ActiveSkill] = {}
        self.archive: Dict[str, ArchivedSkill] = {}
        self.forgotten: Dict[str, ForgottenSkill] = {}

        self._embedding_dim = embedding_dim
        self._active_embeddings: Dict[str, np.ndarray] = {}
        self._archive_embeddings: Dict[str, np.ndarray] = {}
        self._forgotten_embeddings: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Active operations
    # ------------------------------------------------------------------

    def add_to_active(
        self,
        skill: Skill,
        embedding: np.ndarray,
        meta: Optional[SkillMeta] = None,
    ) -> None:
        """Add a skill to the Active area.

        Args:
            skill: Skill to add.
            embedding: Pre-computed embedding vector.
            meta: Optional initial metadata.
        """
        if skill.skill_id in self.active:
            logger.warning("Skill %s already in Active, overwriting", skill.skill_id)
        active_skill = ActiveSkill(skill=skill, meta=meta or SkillMeta())
        self.active[skill.skill_id] = active_skill
        self._active_embeddings[skill.skill_id] = embedding
        logger.info(
            "Added skill '%s' (%s) to Active. Total: %d",
            skill.name, skill.skill_id, len(self.active),
        )

    def remove_from_active(self, skill_id: str) -> Optional[ActiveSkill]:
        """Remove a skill from Active.

        Args:
            skill_id: ID of skill to remove.

        Returns:
            The removed ActiveSkill, or None if not found.
        """
        active_skill = self.active.pop(skill_id, None)
        self._active_embeddings.pop(skill_id, None)
        if active_skill:
            logger.info("Removed skill %s from Active. Total: %d", skill_id, len(self.active))
        return active_skill

    def get_active_skill(self, skill_id: str) -> Optional[ActiveSkill]:
        """Get an Active skill by ID."""
        return self.active.get(skill_id)

    def get_active_embedding(self, skill_id: str) -> Optional[np.ndarray]:
        """Get embedding for an Active skill."""
        return self._active_embeddings.get(skill_id)

    def get_active_embeddings_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Get all Active embeddings as a matrix.

        Returns:
            Tuple of (skill_ids, embeddings_matrix) where matrix is (N, dim).
        """
        if not self._active_embeddings:
            return [], np.zeros((0, self._embedding_dim))
        ids = list(self._active_embeddings.keys())
        matrix = np.array([self._active_embeddings[sid] for sid in ids])
        return ids, matrix

    def update_active_embedding(self, skill_id: str, embedding: np.ndarray) -> None:
        """Update embedding for an Active skill."""
        if skill_id in self.active:
            self._active_embeddings[skill_id] = embedding

    def get_total_active_tokens(self) -> int:
        """Get total token cost of all Active skills."""
        return sum(s.skill.token_cost for s in self.active.values())

    def is_over_budget(self, cfg: RetrievalConfig) -> bool:
        """Check if Active area exceeds token budget."""
        return self.get_total_active_tokens() > cfg.token_budget

    def all_active_skills(self) -> List[ActiveSkill]:
        """Get all Active skills."""
        return list(self.active.values())

    def active_skill_ids(self) -> List[str]:
        """Get all Active skill IDs."""
        return list(self.active.keys())

    def max_active_similarity(self, skill_id: str) -> float:
        """Find maximum similarity between a skill and all other Active skills.

        Used for Irreplaceability calculation.

        Args:
            skill_id: Target skill ID.

        Returns:
            Maximum cosine similarity to any other Active skill.
        """
        emb = self._active_embeddings.get(skill_id)
        if emb is None:
            return 0.0
        max_sim = 0.0
        for other_id, other_emb in self._active_embeddings.items():
            if other_id == skill_id:
                continue
            sim = float(np.dot(emb, other_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(other_emb) + 1e-8
            ))
            max_sim = max(max_sim, sim)
        return max_sim

    # ------------------------------------------------------------------
    # Archive operations
    # ------------------------------------------------------------------

    def add_to_archive(
        self, archived_skill: ArchivedSkill, summary_embedding: np.ndarray
    ) -> None:
        """Add a skill to Archive."""
        sid = archived_skill.original_skill_id
        self.archive[sid] = archived_skill
        self._archive_embeddings[sid] = summary_embedding
        logger.info("Added skill %s to Archive. Total: %d", sid, len(self.archive))

    def remove_from_archive(self, skill_id: str) -> Optional[ArchivedSkill]:
        """Remove a skill from Archive."""
        archived = self.archive.pop(skill_id, None)
        self._archive_embeddings.pop(skill_id, None)
        return archived

    def get_archived_skill(self, skill_id: str) -> Optional[ArchivedSkill]:
        """Get an archived skill by ID."""
        return self.archive.get(skill_id)

    def get_archive_embeddings_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Get all Archive embeddings as a matrix."""
        if not self._archive_embeddings:
            return [], np.zeros((0, self._embedding_dim))
        ids = list(self._archive_embeddings.keys())
        matrix = np.array([self._archive_embeddings[sid] for sid in ids])
        return ids, matrix

    def all_archived_skills(self) -> List[ArchivedSkill]:
        """Get all archived skills."""
        return list(self.archive.values())

    # ------------------------------------------------------------------
    # Forgotten operations
    # ------------------------------------------------------------------

    def add_to_forgotten(self, forgotten_skill: ForgottenSkill, summary_embedding: np.ndarray) -> None:
        """Add a skill to Forgotten."""
        sid = forgotten_skill.skill_id
        self.forgotten[sid] = forgotten_skill
        self._forgotten_embeddings[sid] = summary_embedding
        logger.info("Added skill %s to Forgotten. Total: %d", sid, len(self.forgotten))

    def get_forgotten_embeddings_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Get all Forgotten embeddings as a matrix."""
        if not self._forgotten_embeddings:
            return [], np.zeros((0, self._embedding_dim))
        ids = list(self._forgotten_embeddings.keys())
        matrix = np.array([self._forgotten_embeddings[sid] for sid in ids])
        return ids, matrix

    def all_forgotten_skills(self) -> List[ForgottenSkill]:
        """Get all forgotten skills."""
        return list(self.forgotten.values())

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def move_active_to_archive(
        self, skill_id: str, summary: str, summary_embedding: np.ndarray, current_round: int
    ) -> Optional[ArchivedSkill]:
        """Move a skill from Active to Archive.

        Args:
            skill_id: Skill to move.
            summary: Compressed summary text.
            summary_embedding: Embedding of the summary.
            current_round: Current round number.

        Returns:
            The created ArchivedSkill, or None if skill not found.
        """
        active_skill = self.remove_from_active(skill_id)
        if active_skill is None:
            return None
        archived = ArchivedSkill(
            skill_summary=summary,
            original_skill_id=skill_id,
            original_skill_full=active_skill.skill,
            archived_at=current_round,
            last_used_at=active_skill.meta.last_used_at,
            recall_count=0,
            inactive_rounds=0,
        )
        self.add_to_archive(archived, summary_embedding)
        return archived

    def promote_archive_to_active(
        self, skill_id: str, embedding: np.ndarray, current_round: int
    ) -> Optional[ActiveSkill]:
        """Promote a skill from Archive back to Active.

        Args:
            skill_id: Skill to promote.
            embedding: Embedding for the restored skill.
            current_round: Current round number.

        Returns:
            The created ActiveSkill, or None if not found in archive.
        """
        archived = self.remove_from_archive(skill_id)
        if archived is None:
            return None
        skill = archived.original_skill_full
        meta = SkillMeta(last_used_at=current_round)
        self.add_to_active(skill, embedding, meta)
        logger.info("Promoted skill %s from Archive to Active", skill_id)
        return self.active[skill_id]

    def move_archive_to_forgotten(
        self, skill_id: str, current_round: int
    ) -> Optional[ForgottenSkill]:
        """Move a skill from Archive to Forgotten.

        Args:
            skill_id: Skill to move.
            current_round: Current round number.

        Returns:
            The created ForgottenSkill, or None if not found.
        """
        # Save embedding BEFORE removal (remove_from_archive deletes it)
        summary_emb = self._archive_embeddings.get(skill_id)
        archived = self.remove_from_archive(skill_id)
        if archived is None:
            return None
        forgotten = ForgottenSkill(
            skill_id=skill_id,
            name=archived.original_skill_full.name,
            summary=archived.skill_summary,
            forgotten_at=current_round,
        )
        # Reuse archive summary embedding for forgotten dedup
        emb = summary_emb if summary_emb is not None else np.zeros(self._embedding_dim)
        self.add_to_forgotten(forgotten, emb)
        logger.info("Moved skill %s from Archive to Forgotten", skill_id)
        return forgotten

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Get counts for each area."""
        return {
            "active": len(self.active),
            "archive": len(self.archive),
            "forgotten": len(self.forgotten),
            "active_tokens": self.get_total_active_tokens(),
        }
