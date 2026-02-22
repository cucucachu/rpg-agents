"""Accountant sub-agent — syncs game state changes (damage, healing, status, items) to the DB."""

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from ..prompts import ACCOUNTANT_SYSTEM_PROMPT
from ..state import GMAgentState

logger = logging.getLogger(__name__)

ACCOUNTANT_TOOL_NAMES = {
    # Character state
    "deal_damage", "heal", "apply_statuses", "remove_status",
    "set_attributes", "set_skills", "set_level",
    "grant_abilities", "revoke_ability",
    # Movement
    "move_character",
    # Items
    "give_item", "drop_item", "set_item_quantity", "spawn_item", "destroy_item",
    "set_item_attribute", "apply_item_status", "remove_item_status",
}


def accountant_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Accountant agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    gm_response = state.get("gm_final_response", "")
    gm_tool_calls = state.get("gm_tool_calls", [])

    logger.debug("[accountant_init] Building context for Accountant")

    accountant_messages = [SystemMessage(content=ACCOUNTANT_SYSTEM_PROMPT)]

    if world_id:
        accountant_messages.append(SystemMessage(
            content=f"world_id for tool calls: {world_id}"
        ))

    if world_context:
        accountant_messages.append(SystemMessage(
            content=f"=== WORLD STATE (use these IDs for tool calls) ===\n\n{world_context}"
        ))

    if gm_response:
        accountant_messages.append(SystemMessage(
            content=f"=== GM'S NARRATIVE (sync state changes from this) ===\n\n{gm_response}"
        ))

    if gm_tool_calls:
        tool_call_summary = "\n".join([
            f"- {tc['name']}({', '.join(f'{k}={v}' for k, v in tc['args'].items())})"
            for tc in gm_tool_calls
        ])
        accountant_messages.append(SystemMessage(
            content=f"=== TOOLS GM ALREADY CALLED (do NOT duplicate) ===\n\n{tool_call_summary}"
        ))

    accountant_messages.append(HumanMessage(
        content=(
            "Review the GM's narrative above and sync any state changes to the database.\n\n"
            "Look for:\n"
            "- Damage dealt (HP reductions)\n"
            "- Healing received\n"
            "- Status effects applied or removed\n"
            "- Items given, dropped, or created\n"
            "- Character movement\n\n"
            "Only sync changes that are CLEARLY stated in the narrative and were NOT already "
            "handled by the GM's tool calls.\n\n"
            "If no state changes need syncing, respond with 'No state changes to sync.'"
        )
    ))

    logger.info(f"[accountant_init] GM made {len(gm_tool_calls)} tool calls")

    return {"accountant_messages": accountant_messages}


def create_accountant_agent_node(llm_with_accountant_tools, db):
    """Create the Accountant agent node that syncs game state to the database.

    The Accountant reviews the GM's narrative and makes tool calls to persist
    any state changes (damage, healing, status effects, movement, items) that
    the GM narrated but didn't explicitly call tools for.
    """

    async def accountant_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Review GM's response and sync any state changes to the database."""
        messages = list(state.get("accountant_messages", []))

        logger.debug(f"[accountant] Invoking with {len(messages)} messages")

        response = await llm_with_accountant_tools.ainvoke(messages)

        acct_tools = getattr(response, "tool_calls", None) or []
        has_tool_calls = bool(acct_tools)
        logger.info(
            f"[Accountant] Response: has_tool_calls={has_tool_calls}, "
            f"content_length={len(response.content) if response.content else 0}"
        )
        logger.debug(f"[accountant] response tool_calls={[tc['name'] for tc in acct_tools]}")

        return {"accountant_messages": [response]}

    return accountant_agent_node


def should_continue_accountant(state: GMAgentState) -> Literal["accountant_tools", "__end__"]:
    """Route from Accountant agent to tools or END (runs in parallel with bard and scribe).

    Note: accountant_tools goes directly to END (no loop back) — the Accountant syncs
    all state changes in a single batch to prevent duplicate operations on retry.
    """
    messages = state.get("accountant_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Accountant] Routing to tools: {tool_names}")
        logger.debug("[should_continue_accountant] -> accountant_tools")
        return "accountant_tools"

    logger.info("[Accountant] No state changes, done")
    logger.debug("[should_continue_accountant] -> END")
    return END
