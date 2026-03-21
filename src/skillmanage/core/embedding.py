"""Embedding model wrapper for skill and task encoding."""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np

from ..config import EmbeddingConfig
from .models import Skill

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper around sentence-transformers for encoding skills and tasks.

    Args:
        cfg: Embedding configuration.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._cfg = cfg
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self._cfg.model_name)
            self._model = SentenceTransformer(self._cfg.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into embedding vector(s).

        Args:
            texts: Single text or list of texts.

        Returns:
            Array of shape (dim,) for single text or (N, dim) for list.
        """
        self._load_model()
        if isinstance(texts, str):
            texts = [texts]
            result = self._model.encode(texts, normalize_embeddings=True)
            return result[0]
        return self._model.encode(texts, normalize_embeddings=True)

    def encode_skill(self, skill: Skill) -> np.ndarray:
        """Encode a skill's description into an embedding.

        Args:
            skill: Skill to encode.

        Returns:
            Embedding vector of shape (dim,).
        """
        text = skill.description
        if skill.precondition:
            text = f"{text}. Precondition: {skill.precondition}"
        return self.encode(text)

    def encode_task(self, task_description: str) -> np.ndarray:
        """Encode a task description into an embedding.

        Args:
            task_description: Task instruction text.

        Returns:
            Embedding vector of shape (dim,).
        """
        return self.encode(task_description)

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._cfg.dimension
