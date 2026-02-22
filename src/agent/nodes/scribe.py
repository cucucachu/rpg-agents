"""Scribe sub-agent — records events and chronicles for the current turn."""

import json
import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from ..prompts import SCRIBE_SYSTEM_PROMPT
from ..state import GMAgentState
from .gm import _extract_this_turn_content

logger = logging.getLogger(__name__)

SCRIBE_TOOL_NAMES = {"record_event", "set_chronicle"}


def scribe_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Scribe agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    events_context = state.get("events_context")
    messages = list(state.get("messages", []))
    history_count = state.get("history_message_count", 0)
    first_event_id = state.get("first_event_id", "")
    last_event_id = state.get("last_event_id", "")
    current_game_time = state.get("current_game_time", 0)

    logger.debug("[scribe_init] Building context for Scribe")

    scribe_messages = [SystemMessage(content=SCRIBE_SYSTEM_PROMPT)]

    if world_id:
        scribe_messages.append(SystemMessage(
            content=f"world_id for tool calls: {world_id}"
        ))

    context_parts = []
    if world_context:
        context_parts.append(f"WORLD STATE:\n{world_context}")
    if events_context:
        context_parts.append(f"PREVIOUS EVENTS (already recorded, do not duplicate):\n{events_context}")
    if context_parts:
        scribe_messages.append(SystemMessage(
            content="=== CONTEXT FROM BEFORE THIS TURN ===\n\n" + "\n\n---\n\n".join(context_parts)
        ))

    this_turn_content = _extract_this_turn_content(messages, history_count)

    # gm_final_response lives in state (not in messages), so append it explicitly
    gm_final_response = state.get("gm_final_response", "")
    if gm_final_response:
        this_turn_content = this_turn_content + f"\n\nGM RESPONSE:\n{gm_final_response}"

    scribe_messages.append(SystemMessage(
        content=(
            "=== THIS TURN (what just happened - record this) ===\n\n"
            "The following is a transcript of the player's action and the GM's response. "
            "This is NOT your communication - this is what you must record as events.\n\n"
            f"{this_turn_content}"
        )
    ))

    event_count = len(json.loads(events_context)) if events_context else 0
    logger.debug(f"[scribe_init] current_game_time={current_game_time}, event_count={event_count}")

    days = current_game_time // 86400
    remaining = current_game_time % 86400
    hours = remaining // 3600
    minutes = (remaining % 3600) // 60
    time_str = f"Day {days + 1}, {hours:02d}:{minutes:02d}"

    chronicle_info = ""
    if first_event_id and last_event_id:
        chronicle_info = (
            f"\n\nIf creating a chronicle: use start_event_id={first_event_id} and "
            f"end_event_id={last_event_id} to link the {event_count} events."
        )

    scribe_messages.append(HumanMessage(
        content=(
            f"**CURRENT GAME TIME: {current_game_time} seconds ({time_str})**\n\n"
            f"Record events for this turn. Each event's game_time MUST be > {current_game_time}.\n"
            f"Add estimated duration to current time (e.g., combat round +6s, dialogue +30-60s, travel +1800s).\n"
            f"If 15+ previous events exist, consider creating a chronicle summary.{chronicle_info}"
        )
    ))

    logger.info(f"[scribe_init] Processing turn with {len(messages) - history_count} new messages")

    return {"scribe_messages": scribe_messages}


def create_scribe_agent_node(llm_with_scribe_tools, db):
    """Create the Scribe agent node that records events and chronicles."""

    async def scribe_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Review this turn and create events/chronicles."""
        messages = list(state.get("scribe_messages", []))

        logger.debug(f"[scribe] Invoking with {len(messages)} messages")

        response = await llm_with_scribe_tools.ainvoke(messages)

        scribe_tools = getattr(response, "tool_calls", None) or []
        has_tool_calls = bool(scribe_tools)
        logger.info(
            f"[Scribe] Response: has_tool_calls={has_tool_calls}, "
            f"content_length={len(response.content) if response.content else 0}"
        )
        logger.debug(
            f"[scribe] response tool_calls={[tc['name'] for tc in scribe_tools]}, "
            f"content_preview={repr((response.content or '')[:150])}"
        )

        return {"scribe_messages": [response]}

    return scribe_agent_node


def should_continue_scribe(state: GMAgentState) -> Literal["scribe_tools", "__end__"]:
    """Route from Scribe agent to tools or END (runs in parallel with bard and accountant).

    Note: scribe_tools goes directly to END (no loop back) — the Scribe records
    events and chronicles in a single batch to prevent duplicate entries on retry.
    """
    messages = state.get("scribe_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Scribe] Routing to tools: {tool_names}")
        logger.debug("[should_continue_scribe] -> scribe_tools")
        return "scribe_tools"

    logger.info("[Scribe] Complete, done")
    logger.debug("[should_continue_scribe] -> END")
    return END
