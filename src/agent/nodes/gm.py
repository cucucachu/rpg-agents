"""GM sub-agent — main narration, dice, and combat management."""

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..prompts import GM_SYSTEM_PROMPT
from ..state import GMAgentState

logger = logging.getLogger(__name__)

GM_TOOL_NAMES = {
    "roll_dice", "roll_table", "coin_flip", "roll_stat_array", "percentile_roll",
    "start_encounter", "get_encounter", "get_active_encounter",
    "add_combatant", "set_initiative", "remove_combatant", "next_turn", "end_encounter",
}


def _extract_this_turn_content(messages: list[BaseMessage], history_count: int) -> str:
    """Extract content from this turn's messages (everything after history).

    Returns a formatted string describing what happened this turn.
    The history_count tells us how many messages were from history (before this turn).
    This turn = messages[history_count:] (current HumanMessage + GM responses/tools)
    """
    this_turn = messages[history_count:] if history_count < len(messages) else messages[-1:]

    parts = []
    for msg in this_turn:
        msg_type = type(msg).__name__
        content = msg.content if hasattr(msg, "content") else ""

        if msg_type == "HumanMessage":
            parts.append(f"PLAYER: {content}")
        elif msg_type == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls]
                parts.append(f"GM TOOL CALLS: {', '.join(tool_names)}")
            if content:
                parts.append(f"GM RESPONSE: {content}")
        elif msg_type == "ToolMessage":
            tool_name = msg.name if hasattr(msg, "name") else "tool"
            result_preview = content[:200] + "..." if len(content) > 200 else content
            parts.append(f"TOOL RESULT ({tool_name}): {result_preview}")

    return "\n\n".join(parts) if parts else "No significant activity this turn."


def gm_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for GM agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    events_context = state.get("events_context")
    enriched_context = state.get("enriched_context")
    messages = list(state.get("messages", []))

    logger.debug("[gm_init] Building context for GM agent")

    system_messages = [SystemMessage(content=GM_SYSTEM_PROMPT)]

    if world_context:
        system_messages.append(SystemMessage(
            content=(
                "[CURRENT WORLD STATE]\n"
                "Note: 'player_characters' are controlled by real players. "
                "NEVER write their actions, dialogue, or reactions.\n"
                f"{world_context}\n[END WORLD STATE]"
            )
        ))

    if events_context:
        system_messages.append(SystemMessage(
            content=(
                "[EVENTS SINCE LAST CHRONICLE - These are the canonical events that have occurred. "
                f"Your narrative MUST be consistent with these.]\n{events_context}\n[END EVENTS]"
            )
        ))

    if enriched_context:
        system_messages.append(SystemMessage(
            content=(
                f"[ENRICHED CONTEXT - Additional detail from the Historian.]\n"
                f"{enriched_context}\n[END ENRICHED CONTEXT]"
            )
        ))

    mechanics_context = state.get("mechanics_context")
    if mechanics_context:
        system_messages.append(SystemMessage(
            content=(
                "[MECHANICS BRIEF - Barrister's analysis of required checks, DCs, modifiers, "
                f"and triggered mechanics for this turn.]\n{mechanics_context}\n[END MECHANICS BRIEF]"
            )
        ))

    if world_id:
        system_messages.append(SystemMessage(
            content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"
        ))

    gm_messages = system_messages + messages
    logger.debug(
        f"[gm_init] Built {len(system_messages)} system + {len(messages)} conversation "
        f"= {len(gm_messages)} total"
    )

    return {"gm_messages": gm_messages}


def create_gm_agent_node(llm_with_tools, db):
    """Create the GM agent node."""

    async def gm_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Process messages and generate GM response."""
        messages = list(state.get("gm_messages", []))

        logger.debug(f"[gm_agent] Invoking with {len(messages)} messages")

        response = await llm_with_tools.ainvoke(messages)

        tool_calls = getattr(response, "tool_calls", None) or []
        logger.debug(
            f"[gm_agent] response: tool_calls={len(tool_calls)} "
            f"{[tc['name'] for tc in tool_calls]}, "
            f"content_len={len(response.content or '')}"
        )

        return {"gm_messages": [response]}

    return gm_agent_node


def capture_gm_response_node(state: GMAgentState) -> dict[str, Any]:
    """Capture the GM's final response and tool calls before transitioning to post-processing agents.

    This ensures we have:
    1. The GM's response stored separately from subsequent agent messages
    2. A record of all tool calls the GM made (for Accountant to avoid duplicates)
    """
    messages = state.get("gm_messages", [])

    gm_response = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                gm_response = msg.content
                break

    gm_tool_calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                gm_tool_calls.append({
                    "name": tc.get("name"),
                    "args": tc.get("args", {})
                })

    if gm_response:
        logger.info(f"[Capture] Captured GM response: {gm_response[:100]}...")
        logger.debug(f"[capture_response] gm_final_response len={len(gm_response)}")
    else:
        logger.warning("[Capture] No GM response found to capture")

    if gm_tool_calls:
        tool_names = [tc["name"] for tc in gm_tool_calls]
        logger.info(f"[Capture] GM made {len(gm_tool_calls)} tool calls: {tool_names}")
        logger.debug(f"[capture_response] gm_tool_calls: {tool_names}")

    return {
        "gm_final_response": gm_response,
        "gm_tool_calls": gm_tool_calls,
    }


def should_continue_gm(state: GMAgentState) -> Literal["gm_tools", "capture_response"]:
    """Route from GM agent to tools or capture response."""
    messages = state.get("gm_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[GM] Routing to tools: {tool_names}")
        logger.debug("[should_continue_gm] -> gm_tools")
        return "gm_tools"

    logger.info("[GM] Response complete, capturing response")
    logger.debug("[should_continue_gm] -> capture_response")
    return "capture_response"
