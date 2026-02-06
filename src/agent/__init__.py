"""Agent module - LangGraph GM agent with separated concerns.

Architecture:
- load_context: Deterministic node that fetches world state, events, messages
- gm_agent: Main game master agent with all game tools  
- scribe_agent: Post-processing agent that creates events and chronicles
"""

from .prompts import GM_SYSTEM_PROMPT, GM_SYSTEM_PROMPT_MINIMAL, SCRIBE_SYSTEM_PROMPT

# Lazy imports for heavy dependencies
def create_gm_agent(*args, **kwargs):
    """Create a GM agent. Lazily imports dependencies."""
    from .gm_agent import create_gm_agent as _create
    return _create(*args, **kwargs)

def GMAgent(*args, **kwargs):
    """Create a GMAgent wrapper. Lazily imports dependencies."""
    from .gm_agent import GMAgent as _GMAgent
    return _GMAgent(*args, **kwargs)

__all__ = ["create_gm_agent", "GMAgent", "GM_SYSTEM_PROMPT", "GM_SYSTEM_PROMPT_MINIMAL", "SCRIBE_SYSTEM_PROMPT"]
