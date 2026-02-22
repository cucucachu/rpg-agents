"""Character Creator sub-agent — conversational PC creation plus persist_response helper."""

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage

from ..prompts import CHAR_CREATOR_SYSTEM_PROMPT
from ..state import GMAgentState

logger = logging.getLogger(__name__)

CHAR_CREATOR_TOOL_NAMES = {
    # Update PC
    "update_pc_basics", "set_attributes", "set_skills", "grant_abilities",
    # Finalize
    "finalize_character",
    # Dice
    "roll_dice", "roll_stat_array", "roll_table",
    # Search
    "search_lore", "find_locations", "get_entity",
}


def char_creator_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Character Creator agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    messages = list(state.get("messages", []))

    logger.debug("[char_creator_init] Building context for Character Creator")

    system_messages = [SystemMessage(content=CHAR_CREATOR_SYSTEM_PROMPT)]

    if world_context:
        system_messages.append(SystemMessage(
            content=f"[WORLD STATE]\n{world_context}\n[END WORLD STATE]"
        ))

    if world_id:
        system_messages.append(SystemMessage(
            content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"
        ))

    char_creator_messages = system_messages + messages

    return {"char_creator_messages": char_creator_messages}


def create_char_creator_agent_node(llm_with_tools, db):
    """Create the Character Creator agent node for conversational character building."""

    async def char_creator_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Run the character creator agent for this turn."""
        messages = list(state.get("char_creator_messages", []))
        world_id = state.get("world_id")

        logger.info(f"[Char Creator] Running for world {world_id}")
        logger.debug(f"[char_creator] Invoking with {len(messages)} messages")

        response = await llm_with_tools.ainvoke(messages)

        tool_calls = getattr(response, "tool_calls", None) or []
        logger.debug(
            f"[char_creator] tool_calls={[tc['name'] for tc in tool_calls]}, "
            f"content_len={len(response.content or '')}"
        )

        return {"char_creator_messages": [response]}

    return char_creator_agent_node


def should_continue_char_creator(state: GMAgentState) -> Literal["char_creator_tools", "persist_response"]:
    """Route from char_creator agent to tools or persist response."""
    messages = state.get("char_creator_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Char Creator] Routing to tools: {tool_names}")
        return "char_creator_tools"

    logger.info("[Char Creator] Response complete, persisting")
    return "persist_response"


def persist_response_node(state: GMAgentState) -> dict[str, Any]:
    """Capture the last AIMessage as gm_final_response for creation paths."""
    world_creator_messages = state.get("world_creator_messages", [])
    char_creator_messages = state.get("char_creator_messages", [])

    messages = world_creator_messages if world_creator_messages else char_creator_messages

    gm_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                gm_response = msg.content
                break

    logger.info(f"[persist_response] Captured response: {gm_response[:100] if gm_response else '(empty)'}...")

    return {"gm_final_response": gm_response}
