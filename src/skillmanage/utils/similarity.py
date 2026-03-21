"""Cosine similarity helpers."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: Vector of shape (dim,).
        b: Vector of shape (dim,).

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(
    matrix: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between a matrix of vectors and a single vector.

    Args:
        matrix: Matrix of shape (N, dim).
        vector: Vector of shape (dim,).

    Returns:
        Array of similarities of shape (N,).
    """
    if matrix.shape[0] == 0:
        return np.array([])
    norms = np.linalg.norm(matrix, axis=1)
    vec_norm = np.linalg.norm(vector)
    if vec_norm == 0:
        return np.zeros(matrix.shape[0])
    dots = matrix @ vector
    denom = norms * vec_norm
    denom = np.where(denom == 0, 1.0, denom)
    return dots / denom
