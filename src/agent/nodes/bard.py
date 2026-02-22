"""Bard sub-agent — records new entities (NPCs, locations, lore, factions) from GM narrative."""

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from ..prompts import BARD_SYSTEM_PROMPT
from ..state import GMAgentState

logger = logging.getLogger(__name__)

BARD_TOOL_NAMES = {
    # Search (check if entities exist before creating)
    "find_characters",
    # Create/update
    "create_npc", "update_npc", "set_location",
    "set_lore", "set_faction",
}


def bard_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Bard agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    gm_response = state.get("gm_final_response", "")
    gm_tool_calls = state.get("gm_tool_calls", [])

    logger.debug("[bard_init] Building context for Bard")

    bard_messages = [SystemMessage(content=BARD_SYSTEM_PROMPT)]
    if world_id:
        bard_messages.append(SystemMessage(
            content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"
        ))
    if world_context:
        bard_messages.append(SystemMessage(
            content=f"[WORLD STATE]\n{world_context}\n[END WORLD STATE]"
        ))
    if gm_tool_calls:
        tool_summary = "\n".join([
            f"- {tc.get('name')}({tc.get('args', {})})" for tc in gm_tool_calls
        ])
        bard_messages.append(SystemMessage(
            content=f"[TOOLS GM ALREADY CALLED - do NOT duplicate]\n{tool_summary}\n[END]"
        ))
    if gm_response:
        bard_messages.append(SystemMessage(
            content=f"[GM'S NARRATIVE]\n{gm_response}\n[END NARRATIVE]"
        ))
    bard_messages.append(HumanMessage(
        content=(
            "Identify any NEW named NPCs, locations, lore, or factions mentioned in the narrative. "
            "First, search for them to check if they exist. Then create NEW entities or update "
            "EXISTING entities with new information. Batch your searches first, then batch your "
            "creates/updates. If nothing new to record, respond with 'No new entities to record.'"
        )
    ))

    return {"bard_messages": bard_messages}


def create_bard_agent_node(llm_with_bard_tools, db):
    """Create the Bard agent node that records new entities from the GM's narrative."""

    async def bard_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Review GM's narrative and create/update records for NPCs, locations, lore, factions."""
        messages = list(state.get("bard_messages", []))

        logger.debug(f"[bard] Invoking with {len(messages)} messages")

        response = await llm_with_bard_tools.ainvoke(messages)
        bard_tools = getattr(response, "tool_calls", None) or []
        logger.info(
            f"[Bard] Routing to {'tools' if bard_tools else 'END'}: "
            f"{[tc['name'] for tc in bard_tools] if bard_tools else 'none'}"
        )

        return {"bard_messages": [response]}

    return bard_agent_node


def should_continue_bard(state: GMAgentState) -> Literal["bard_tools", "__end__"]:
    """Route from Bard agent to tools or END (runs in parallel with accountant and scribe)."""
    messages = state.get("bard_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Bard] Routing to tools: {tool_names}")
        logger.debug("[should_continue_bard] -> bard_tools")
        return "bard_tools"

    logger.info("[Bard] No entities to record, done")
    logger.debug("[should_continue_bard] -> END")
    return END
