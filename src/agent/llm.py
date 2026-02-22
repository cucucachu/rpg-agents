"""LLM factory for the GM agent pipeline."""

import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)


def get_llm(
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.7,
) -> Any:
    """Get the LLM based on provider."""
    if provider == "anthropic":
        model = model or "claude-haiku-4-5-20251001"
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        api_key_preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "(missing or short)"
        logger.info(f"Creating ChatAnthropic: model={model}, api_key={api_key_preview}, key_length={len(api_key)}")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=4096,
        )
    else:  # openai
        model = model or "gpt-4o"
        return ChatOpenAI(
            model=model,
            temperature=temperature,
        )
