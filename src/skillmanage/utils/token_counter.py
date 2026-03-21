"""Token counting utility."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ENCODER = None


def _get_encoder():
    """Lazy-load tiktoken encoder."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.debug("tiktoken not available, using word-based estimate")
            _ENCODER = "fallback"
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count tokens in text.

    Uses tiktoken if available, otherwise word-based estimate (words * 1.3).

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    encoder = _get_encoder()
    if encoder == "fallback":
        return int(len(text.split()) * 1.3)
    return len(encoder.encode(text))
