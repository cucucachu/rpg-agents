"""Agent module — LangGraph GM agent pipeline.

Architecture:
- load_context:     Deterministic DB fetch (world state, events, messages)
- historian:        Read-only search to enrich context
- barrister:        Mechanics analysis (no tools)
- gm_agent:         Main game master (narration, dice, combat)
- capture_response: Capture GM output before post-processing
- bard:             Record new entities (parallel)
- accountant:       Sync state changes (parallel)
- scribe:           Record events and chronicles (parallel)
"""

from .prompts import GM_SYSTEM_PROMPT, GM_SYSTEM_PROMPT_MINIMAL, SCRIBE_SYSTEM_PROMPT


def create_gm_agent(*args, **kwargs):
    """Create a GM agent graph. Lazily imports heavy dependencies."""
    from .graph import create_gm_agent as _create
    return _create(*args, **kwargs)


def GMAgent(*args, **kwargs):
    """Create a GMAgent wrapper. Lazily imports heavy dependencies."""
    from .graph import GMAgent as _GMAgent
    return _GMAgent(*args, **kwargs)


__all__ = ["create_gm_agent", "GMAgent", "GM_SYSTEM_PROMPT", "GM_SYSTEM_PROMPT_MINIMAL", "SCRIBE_SYSTEM_PROMPT"]
