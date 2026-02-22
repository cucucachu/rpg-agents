"""Barrister sub-agent — mechanics analysis (no tools, read-only)."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..prompts import BARRISTER_SYSTEM_PROMPT
from ..state import GMAgentState

logger = logging.getLogger(__name__)


def barrister_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Barrister agent."""
    world_context = state.get("world_context")
    enriched_context = state.get("enriched_context")
    messages = list(state.get("messages", []))
    history_count = state.get("history_message_count", 0)

    logger.debug("[barrister_init] Building context for Barrister")

    barrister_messages = [SystemMessage(content=BARRISTER_SYSTEM_PROMPT)]
    if world_context:
        barrister_messages.append(SystemMessage(
            content=f"[WORLD STATE]\n{world_context}\n[END WORLD STATE]"
        ))
    if enriched_context:
        barrister_messages.append(SystemMessage(
            content=f"[ENRICHED CONTEXT]\n{enriched_context}\n[END ENRICHED CONTEXT]"
        ))
    if history_count < len(messages):
        player_msg = messages[history_count]
        if hasattr(player_msg, "content"):
            barrister_messages.append(SystemMessage(
                content=f"[PLAYER MESSAGE]\n{player_msg.content}\n[END PLAYER MESSAGE]"
            ))

    barrister_messages.append(HumanMessage(
        content=(
            "Survey the current situation. Identify every mechanic that could trigger this turn — "
            "the player's action, any NPC or creature actions, environmental effects, ongoing "
            "conditions, and any other rules that apply. Produce a mechanics brief covering all of "
            "them. If nothing mechanical applies, return an empty brief."
        )
    ))
    return {"barrister_messages": barrister_messages}


def create_barrister_agent_node(llm):
    """Create the Barrister agent node that analyzes mechanics for the current turn."""

    async def barrister_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Survey the turn situation and produce a mechanics brief for the GM."""
        messages = list(state.get("barrister_messages", []))

        logger.debug(f"[barrister] Invoking with {len(messages)} messages")

        response = await llm.ainvoke(messages)
        logger.debug(f"[barrister] response content_len={len(response.content or '')}")

        return {"barrister_messages": [response]}

    return barrister_agent_node


def compile_barrister_context_node(state: GMAgentState) -> dict[str, Any]:
    """Capture Barrister output into mechanics_context for the GM."""
    messages = state.get("barrister_messages", [])
    parts = [msg.content for msg in messages if isinstance(msg, AIMessage) and msg.content]
    mechanics_context = "\n\n".join(parts) if parts else ""
    logger.debug(f"[compile_barrister] mechanics_context len={len(mechanics_context)}")
    return {"mechanics_context": mechanics_context}
