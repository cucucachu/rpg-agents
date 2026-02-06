"""Tests for GM prompts."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.prompts import GM_SYSTEM_PROMPT, GM_SYSTEM_PROMPT_MINIMAL


def test_system_prompt_exists():
    """Verify system prompt is defined."""
    assert GM_SYSTEM_PROMPT is not None
    assert len(GM_SYSTEM_PROMPT) > 100


def test_system_prompt_contains_key_sections():
    """Verify system prompt contains essential guidance."""
    assert "Game Master" in GM_SYSTEM_PROMPT
    assert "Tool Usage" in GM_SYSTEM_PROMPT
    assert "load_session" in GM_SYSTEM_PROMPT
    assert "roll_dice" in GM_SYSTEM_PROMPT
    assert "world.settings" in GM_SYSTEM_PROMPT


def test_minimal_prompt_exists():
    """Verify minimal prompt is defined."""
    assert GM_SYSTEM_PROMPT_MINIMAL is not None
    assert len(GM_SYSTEM_PROMPT_MINIMAL) < len(GM_SYSTEM_PROMPT)
