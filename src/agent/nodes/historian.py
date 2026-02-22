"""Historian sub-agent — read-only context enrichment via search tools."""

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..prompts import HISTORIAN_SYSTEM_PROMPT
from ..state import GMAgentState

logger = logging.getLogger(__name__)

HISTORIAN_TOOL_NAMES = {
    "search_lore", "find_characters", "find_locations", "search_locations", "find_events",
    "find_quests", "find_factions", "get_entity", "get_chronicle_details",
    "get_location_contents", "find_nearby_locations", "get_character_inventory",
}


def historian_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Historian agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    events_context = state.get("events_context")
    messages = list(state.get("messages", []))
    history_count = state.get("history_message_count", 0)

    logger.debug("[historian_init] Building context for historian")

    historian_messages = [SystemMessage(content=HISTORIAN_SYSTEM_PROMPT)]
    if world_id:
        historian_messages.append(SystemMessage(
            content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"
        ))
    if world_context:
        historian_messages.append(SystemMessage(
            content=f"[WORLD STATE]\n{world_context}\n[END WORLD STATE]"
        ))
    if events_context:
        historian_messages.append(SystemMessage(
            content=f"[RECENT EVENTS]\n{events_context}\n[END RECENT EVENTS]"
        ))

    # Inject the prior GM response so the historian can proactively look up NPCs
    # the GM is likely to re-narrate this turn, not just those the player mentioned
    if history_count > 0:
        prior_msg = messages[history_count - 1]
        if isinstance(prior_msg, AIMessage) and prior_msg.content:
            historian_messages.append(SystemMessage(
                content=f"[PRIOR GM RESPONSE]\n{prior_msg.content}\n[END PRIOR GM RESPONSE]"
            ))

    if history_count < len(messages):
        player_msg = messages[history_count]
        if hasattr(player_msg, "content"):
            historian_messages.append(SystemMessage(
                content=f"[PLAYER MESSAGE]\n{player_msg.content}\n[END PLAYER MESSAGE]"
            ))

    historian_messages.append(HumanMessage(
        content=(
            "Identify any characters, locations, lore, or events referenced above that lack detail. "
            "Search for additional information and return a brief JSON summary of what you found "
            "(or empty {} if nothing to add)."
        )
    ))

    return {"historian_messages": historian_messages}


def create_historian_agent_node(llm_with_historian_tools, db):
    """Create the Historian agent node that searches for relevant context."""

    async def historian_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Analyze context and user message, search for additional detail, return findings."""
        messages = list(state.get("historian_messages", []))

        logger.debug(f"[historian] Invoking with {len(messages)} messages")

        response = await llm_with_historian_tools.ainvoke(messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        has_tools = bool(tool_calls)
        logger.debug(
            f"[historian] response: tool_calls={has_tools}, "
            f"tools={[tc['name'] for tc in tool_calls]}, "
            f"content_len={len(response.content or '')}"
        )

        return {"historian_messages": [response]}

    return historian_agent_node


def compile_historian_context_node(state: GMAgentState) -> dict[str, Any]:
    """Capture historian output (and tool results) into enriched_context for the GM."""
    messages = state.get("historian_messages", [])
    parts = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", None)
            if content:
                if isinstance(content, list):
                    content = str(content)
                parts.append(content)
        elif isinstance(msg, ToolMessage):
            content = getattr(msg, "content", None) or str(msg)
            if isinstance(content, list):
                content = str(content)
            if len(str(content)) > 2000:
                content = str(content)[:2000] + "..."
            parts.append(f"{getattr(msg, 'name', 'tool')}: {content}")
    enriched_context = "\n\n".join(parts) if parts else "{}"
    logger.debug(f"[compile_historian] enriched_context len={len(enriched_context)}, parts={len(parts)}")
    return {"enriched_context": enriched_context}


def should_continue_historian(state: GMAgentState) -> Literal["historian_tools", "compile_historian"]:
    """Route from Historian agent to tools or compile context."""
    messages = state.get("historian_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Historian] Routing to tools: {tool_names}")
        logger.debug("[should_continue_historian] -> historian_tools")
        return "historian_tools"

    logger.info("[Historian] No tool calls, compiling context")
    logger.debug("[should_continue_historian] -> compile_historian")
    return "compile_historian"
