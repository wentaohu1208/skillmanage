"""Incremental Alignment: PatternBuffer management and extraction."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import AcquisitionConfig
from ..core.embedding import EmbeddingModel
from ..core.models import PatternBuffer, PatternEntry, SegmentedTrajectory
from ..utils import generate_skill_id
from ..utils.similarity import batch_cosine_similarity

logger = logging.getLogger(__name__)


class PatternBufferManager:
    """Manages PatternBuffers for incremental alignment.

    Each task type has its own PatternBuffer. New segments are matched
    against existing patterns; confidence is updated incrementally.
    """

    def __init__(self) -> None:
        self.buffers: Dict[str, PatternBuffer] = {}

    def get_or_create_buffer(self, task_type: str) -> PatternBuffer:
        """Get or create a PatternBuffer for a task type."""
        if task_type not in self.buffers:
            self.buffers[task_type] = PatternBuffer(task_type=task_type)
        return self.buffers[task_type]

    def add_record(
        self,
        task_type: str,
        segmented: SegmentedTrajectory,
        embedding_model: EmbeddingModel,
        cfg: AcquisitionConfig,
    ) -> List[PatternEntry]:
        """Add a segmented trajectory record and return extraction candidates.

        Args:
            task_type: Task type label.
            segmented: Segmented trajectory.
            embedding_model: For semantic matching.
            cfg: Acquisition configuration.

        Returns:
            List of PatternEntry candidates ready for extraction.
        """
        buf = self.get_or_create_buffer(task_type)
        buf.total_records += 1

        # Match each segment against existing patterns
        for segment in segmented.segments:
            subgoal = segment.subgoal
            matched = self._match_to_existing(
                subgoal, buf, embedding_model, cfg.pattern_match_threshold
            )
            if matched is not None:
                matched.count += 1
                logger.debug(
                    "Matched '%s' to existing pattern '%s' (count=%d)",
                    subgoal, matched.description, matched.count,
                )
            else:
                new_entry = PatternEntry(
                    pattern_id=generate_skill_id("pat"),
                    description=subgoal,
                    count=1,
                )
                buf.patterns.append(new_entry)
                logger.debug("New pattern: '%s'", subgoal)

        # Check extraction candidates
        return self.check_extraction_candidates(task_type, cfg)

    def check_extraction_candidates(
        self, task_type: str, cfg: AcquisitionConfig
    ) -> List[PatternEntry]:
        """Check which patterns are ready for extraction.

        Conditions:
        1. total_records >= M (minimum records)
        2. confidence >= r (minimum confidence)
        3. Not already extracted
        4. Not a cross-category generic pattern

        Args:
            task_type: Task type to check.
            cfg: Acquisition configuration.

        Returns:
            List of patterns ready for extraction.
        """
        buf = self.buffers.get(task_type)
        if buf is None or buf.total_records < cfg.min_records:
            return []

        candidates = []
        for pattern in buf.patterns:
            if pattern.extracted:
                continue
            if pattern.pattern_id in buf.extracted_pattern_ids:
                continue

            confidence = buf.get_confidence(pattern)
            if confidence < cfg.min_confidence:
                continue

            if self._is_cross_category_generic(pattern, cfg):
                logger.debug(
                    "Skipping generic pattern '%s' (appears across categories)",
                    pattern.description,
                )
                continue

            candidates.append(pattern)

        return candidates

    def mark_extracted(self, task_type: str, pattern_id: str) -> None:
        """Mark a pattern as extracted."""
        buf = self.buffers.get(task_type)
        if buf is None:
            return
        buf.extracted_pattern_ids.add(pattern_id)
        for p in buf.patterns:
            if p.pattern_id == pattern_id:
                p.extracted = True
                break

    def get_confidence(self, task_type: str, pattern_id: str) -> float:
        """Get current confidence for a pattern."""
        buf = self.buffers.get(task_type)
        if buf is None:
            return 0.0
        for p in buf.patterns:
            if p.pattern_id == pattern_id:
                return buf.get_confidence(p)
        return 0.0

    def find_variants(
        self,
        task_type: str,
        pattern: PatternEntry,
        embedding_model: EmbeddingModel,
        threshold: float = 0.6,
    ) -> List[str]:
        """Find variant patterns similar to the given pattern.

        Used to parameterize skills (e.g., method={factoring|completing_square}).

        Args:
            task_type: Task type.
            pattern: The pattern to find variants for.
            embedding_model: For semantic matching.
            threshold: Similarity threshold for variants.

        Returns:
            List of variant descriptions.
        """
        buf = self.buffers.get(task_type)
        if buf is None:
            return []

        target_emb = embedding_model.encode(pattern.description)
        variants = []
        for p in buf.patterns:
            if p.pattern_id == pattern.pattern_id:
                continue
            p_emb = embedding_model.encode(p.description)
            sim = float(np.dot(target_emb, p_emb))
            if threshold <= sim < 0.95:  # Similar but not identical
                variants.append(p.description)
        return variants

    def _match_to_existing(
        self,
        subgoal: str,
        buf: PatternBuffer,
        embedding_model: EmbeddingModel,
        threshold: float,
    ) -> Optional[PatternEntry]:
        """Match a subgoal to an existing pattern via semantic similarity."""
        if not buf.patterns:
            return None

        subgoal_emb = embedding_model.encode(subgoal)
        descriptions = [p.description for p in buf.patterns]
        pattern_embs = embedding_model.encode(descriptions)

        sims = batch_cosine_similarity(pattern_embs, subgoal_emb)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            return buf.patterns[best_idx]
        return None

    def _is_cross_category_generic(
        self, pattern: PatternEntry, cfg: AcquisitionConfig
    ) -> bool:
        """Check if a pattern appears uniformly across many categories.

        A pattern that appears in >80% of categories is too generic.
        """
        if len(self.buffers) < 2:
            return False

        # Check how many categories have this pattern (by description similarity)
        # Simplified: check exact description match
        categories_with_pattern = 0
        for task_type, buf in self.buffers.items():
            for p in buf.patterns:
                if p.description.lower() == pattern.description.lower():
                    categories_with_pattern += 1
                    break

        ratio = categories_with_pattern / len(self.buffers)
        return ratio > cfg.cross_category_ratio
