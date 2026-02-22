"""World Creator sub-agent — conversational world-building during creation phase."""

import logging
from typing import Any, Literal

from langchain_core.messages import SystemMessage

from ..prompts import WORLD_CREATOR_SYSTEM_PROMPT
from ..state import GMAgentState

logger = logging.getLogger(__name__)

WORLD_CREATOR_TOOL_NAMES = {
    # Search/query
    "search_lore", "find_characters", "find_locations", "find_events",
    "find_quests", "find_factions", "get_entity", "get_location_contents", "find_nearby_locations",
    # Create/record (NPCs, locations, factions, lore)
    "set_lore", "set_location", "set_faction",
    "create_npc", "update_npc", "spawn_enemies",
    "set_item_blueprint", "set_ability_blueprint",
    # World basics
    "update_world_basics", "start_game",
    # Dice/random
    "roll_table", "coin_flip",
}


def world_creator_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for World Creator agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    messages = list(state.get("messages", []))

    logger.debug("[world_creator_init] Building context for World Creator")

    system_messages = [SystemMessage(content=WORLD_CREATOR_SYSTEM_PROMPT)]

    if world_context:
        system_messages.append(SystemMessage(
            content=f"[WORLD STATE]\n{world_context}\n[END WORLD STATE]"
        ))

    if world_id:
        system_messages.append(SystemMessage(
            content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"
        ))

    world_creator_messages = system_messages + messages

    return {"world_creator_messages": world_creator_messages}


def create_world_creator_agent_node(llm_with_tools, db):
    """Create the World Creator agent node for conversational world building."""

    async def world_creator_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Run the world creator agent for this turn."""
        messages = list(state.get("world_creator_messages", []))
        world_id = state.get("world_id")

        logger.info(f"[World Creator] Running for world {world_id}")
        logger.info(f"[world_creator] Invoking LLM with {len(messages)} messages")

        try:
            response = await llm_with_tools.ainvoke(messages)
        except Exception as e:
            logger.exception(f"[world_creator] LLM invocation failed: {type(e).__name__}: {e}")
            raise

        tool_calls = getattr(response, "tool_calls", None) or []
        logger.info(
            f"[world_creator] LLM responded: tool_calls={[tc['name'] for tc in tool_calls]}, "
            f"content_len={len(response.content or '')}"
        )

        return {"world_creator_messages": [response]}

    return world_creator_agent_node


def should_continue_world_creator(state: GMAgentState) -> Literal["world_creator_tools", "persist_response"]:
    """Route from world_creator agent to tools or persist response."""
    messages = state.get("world_creator_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[World Creator] Routing to tools: {tool_names}")
        return "world_creator_tools"

    logger.info("[World Creator] Response complete, persisting")
    return "persist_response"
