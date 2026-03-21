"""Abstract LLM client interface with factory/registry pattern."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)

LLM_REGISTRY: Dict[str, Type[BaseLLMClient]] = {}


def register_llm_provider(name: str):
    """Decorator to register an LLM provider.

    Args:
        name: Provider name (e.g., 'openai', 'gemini').
    """
    def decorator(cls: Type[BaseLLMClient]) -> Type[BaseLLMClient]:
        LLM_REGISTRY[name] = cls
        logger.debug("Registered LLM provider: %s", name)
        return cls
    return decorator


def create_llm_client(provider: str, **kwargs: Any) -> BaseLLMClient:
    """Create an LLM client by provider name.

    Args:
        provider: Provider name (e.g., 'openai').
        **kwargs: Provider-specific arguments.

    Returns:
        Configured LLM client.

    Raises:
        ValueError: If provider not registered.
    """
    if provider not in LLM_REGISTRY:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available: {list(LLM_REGISTRY.keys())}"
        )
    return LLM_REGISTRY[provider](**kwargs)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text.
        """

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> Dict:
        """Generate and parse JSON from a prompt.

        Args:
            prompt: User prompt (should request JSON output).
            system_prompt: System prompt.
            **kwargs: Additional generation parameters.

        Returns:
            Parsed JSON dict.
        """
