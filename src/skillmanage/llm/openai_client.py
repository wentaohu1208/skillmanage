"""OpenAI-compatible LLM client (works with vLLM, GPT, Gemini via adapter)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from .base import BaseLLMClient, register_llm_provider

logger = logging.getLogger(__name__)


@register_llm_provider("openai")
class OpenAILLMClient(BaseLLMClient):
    """LLM client using OpenAI-compatible API.

    Works with:
    - vLLM-served Qwen2.5-7B-Instruct
    - OpenAI GPT-4o-mini
    - Any OpenAI-compatible endpoint

    Args:
        base_url: API base URL.
        api_key: API key.
        model_name: Model name.
        temperature: Generation temperature.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        repetition_penalty: float = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty

    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        """Generate text from a prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build extra_body for vLLM-specific params (e.g., repetition_penalty)
        extra_body = {}
        rep_penalty = kwargs.get("repetition_penalty", self._repetition_penalty)
        if rep_penalty is not None:
            extra_body["repetition_penalty"] = rep_penalty

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            top_p=kwargs.get("top_p", self._top_p),
            **({"extra_body": extra_body} if extra_body else {}),
        )
        return response.choices[0].message.content or ""

    def generate_json(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> Dict:
        """Generate and parse JSON from a prompt."""
        full_system = system_prompt or ""
        if "json" not in full_system.lower() and "json" not in prompt.lower():
            full_system += "\nRespond with valid JSON only."

        text = self.generate(prompt, full_system, **kwargs)
        return _parse_json_from_text(text)


def _parse_json_from_text(text: str) -> Dict:
    """Extract and parse JSON from LLM output.

    Handles cases where LLM wraps JSON in markdown code blocks.

    Args:
        text: LLM output text.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON found.
    """
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}...")
