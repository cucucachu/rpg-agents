"""GM Agent implementation using LangGraph with separated concerns.

Architecture:
1. load_context - Deterministic node that fetches world state, events, messages
2. historian - Enriches context with search (lore, characters, locations, events)
3. compile_historian - Captures historian output as enriched_context
4. gm_agent - Main game master agent (narration, dice, combat management)
5. capture_response - Captures GM response and tool calls
6. bard - Records new NPCs, locations, lore, factions from narrative
7. accountant_agent - Syncs game state (damage, healing, status, items) to database
8. scribe_agent - Records events and chronicles

Flow:
load_context -> historian <-> historian_tools -> compile_historian -> gm_agent <-> gm_tools -> capture_response -> bard <-> bard_tools -> (if creation_in_progress: END else: accountant -> ... -> scribe -> END)
"""

import logging
import time
import json
import os
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .prompts import (
    GM_SYSTEM_PROMPT,
    HISTORIAN_SYSTEM_PROMPT,
    BARD_SYSTEM_PROMPT,
    SCRIBE_SYSTEM_PROMPT,
    ACCOUNTANT_SYSTEM_PROMPT,
    WORLD_CREATOR_SYSTEM_PROMPT,
    CHAR_CREATOR_SYSTEM_PROMPT,
)
from .mcp_tools import get_mcp_tools, get_mcp_client

# Debug mode - controls verbose state logging and debug-level logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

logger = logging.getLogger(__name__)

# Configure logger based on DEBUG environment variable
if DEBUG:
    logger.setLevel(logging.DEBUG)
    # Also set the root logger to DEBUG if not already configured
    root_logger = logging.getLogger()
    if root_logger.level == logging.NOTSET or root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)


# ============================================================================
# State Schema (using TypedDict for proper LangGraph message handling)
# ============================================================================

class GMAgentState(TypedDict, total=False):
    """State for the GM agent pipeline.
    
    Uses TypedDict instead of Pydantic BaseModel to ensure proper
    message handling by LangGraph. The 'messages' field uses the
    add_messages reducer to properly accumulate messages.
    """
    # Core identifiers
    world_id: str
    
    # Messages (conversation history, read-only) - uses operator.add to accumulate messages
    messages: Annotated[list[BaseMessage], operator.add]

    # Per-agent isolated message lists for ReAct loops and context isolation
    historian_messages: Annotated[list[BaseMessage], operator.add]
    gm_messages: Annotated[list[BaseMessage], operator.add]
    bard_messages: Annotated[list[BaseMessage], operator.add]
    accountant_messages: Annotated[list[BaseMessage], operator.add]
    scribe_messages: Annotated[list[BaseMessage], operator.add]
    world_creator_messages: Annotated[list[BaseMessage], operator.add]
    char_creator_messages: Annotated[list[BaseMessage], operator.add]
    
    # How many messages in the initial state were history (before this turn)
    # Used by scribe to identify which messages are "this turn"
    history_message_count: int
    
    # Loaded context (populated by load_context node)
    world_context: str  # JSON string of world data
    events_context: str  # JSON string of events since last chronicle
    enriched_context: str  # JSON string from Historian (additional findings)
    last_chronicle_id: str  # ID of most recent chronicle
    first_event_id: str  # First event ID since last chronicle (for linking)
    last_event_id: str  # Last event ID since last chronicle (for linking)
    current_game_time: int  # Game time in seconds from last event (for scribe to track time)
    
    # GM's final response (captured before scribe runs)
    # This is what gets persisted to the database
    gm_final_response: str
    
    # GM's tool calls (captured for Accountant to avoid duplicates)
    # Contains list of tool call dicts with 'name' and 'args'
    gm_tool_calls: list[dict]
    
    # World creation phase: when True, route to world_creator agent
    creation_in_progress: bool
    
    # Character creation phase: when True, route to char_creator agent
    needs_character_creation: bool


# ============================================================================
# LLM Factory
# ============================================================================

def get_llm(
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.7
) -> Any:
    """Get the LLM based on provider."""
    if provider == "anthropic":
        model = model or "claude-haiku-4-5-20251001"
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        api_key_preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "(missing or short)"
        logger.info(f"Creating ChatAnthropic: model={model}, api_key={api_key_preview}, key_length={len(api_key)}")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=4096,
        )
    else:  # openai
        model = model or "gpt-4o"
        return ChatOpenAI(
            model=model,
            temperature=temperature,
        )


# ============================================================================
# Context Loading (Deterministic)
# ============================================================================

async def load_world_context(db, world_id: str) -> dict:
    """Load world state from database."""
    from bson import ObjectId
    
    result = {
        "world": None,
        "player_characters": [],  # PCs controlled by players - NEVER act or speak for these
        "active_quests": [],
        "active_encounter": None,
    }
    
    # Load world
    world_doc = await db.worlds.find_one({"_id": ObjectId(world_id)})
    if world_doc:
        result["world"] = {
            "id": str(world_doc["_id"]),
            "name": world_doc.get("name"),
            "description": world_doc.get("description"),
            "settings": world_doc.get("settings", {}),
            "creation_in_progress": world_doc.get("creation_in_progress", False),
        }
    
    # Load player characters
    pc_cursor = db.characters.find({"world_id": world_id, "is_player_character": True})
    async for doc in pc_cursor:
        char = {
            "id": str(doc["_id"]),
            "name": doc.get("name"),
            "description": doc.get("description"),
            "level": doc.get("level", 1),
            "location_id": doc.get("location_id"),
            "attributes": doc.get("attributes", []),
            "statuses": doc.get("statuses", []),
            "abilities": [a.get("name") for a in doc.get("abilities", [])],
        }
        result["player_characters"].append(char)
    
    # Load active quests
    quest_cursor = db.quests.find({"world_id": world_id, "status": "active"})
    async for doc in quest_cursor:
        quest = {
            "id": str(doc["_id"]),
            "name": doc.get("name"),
            "description": doc.get("description"),
            "progress": doc.get("progress"),
        }
        result["active_quests"].append(quest)
    
    # Load active encounter if any
    encounter_doc = await db.encounters.find_one({"world_id": world_id, "status": "active"})
    if encounter_doc:
        result["active_encounter"] = {
            "id": str(encounter_doc["_id"]),
            "name": encounter_doc.get("name"),
            "round": encounter_doc.get("round", 1),
            "turn_order": encounter_doc.get("turn_order", []),
            "current_turn": encounter_doc.get("current_turn", 0),
        }
    
    return result


async def load_events_since_chronicle(db, world_id: str) -> tuple[list[dict], str | None, int]:
    """Load all events since the last chronicle (or all events if no chronicle).
    
    Returns:
        tuple: (events_list, last_chronicle_id, max_game_time)
        - events_list: Events since the last chronicle for context
        - last_chronicle_id: ID of the most recent chronicle (or None)
        - max_game_time: The highest game_time seen in the world (for Scribe continuity)
    """
    from bson import ObjectId
    
    # Find the most recent chronicle
    last_chronicle = await db.chronicles.find_one(
        {"world_id": world_id},
        sort=[("_id", -1)]  # Most recent by creation time
    )
    
    last_chronicle_id = str(last_chronicle["_id"]) if last_chronicle else None
    chronicle_game_time_end = last_chronicle.get("game_time_end", 0) if last_chronicle else 0
    
    # Build query for events since chronicle (for context)
    query = {"world_id": world_id}
    if last_chronicle:
        # Events created after the chronicle
        query["_id"] = {"$gt": last_chronicle["_id"]}
    
    # Load events sorted by creation time (oldest first for narrative order)
    events = []
    event_cursor = db.events.find(query).sort("_id", 1)
    async for doc in event_cursor:
        events.append({
            "id": str(doc["_id"]),
            "name": doc.get("name"),
            "description": doc.get("description"),
            "participants": doc.get("participants"),
            "changes": doc.get("changes"),
            "tags": doc.get("tags", []),
            "game_time": doc.get("game_time", 0),
        })
    
    # Get the MAX game_time across ALL events in this world
    # This ensures Scribe always builds on top of the highest time seen,
    # even after a chronicle is created
    max_game_time_doc = await db.events.find_one(
        {"world_id": world_id},
        sort=[("game_time", -1)]  # Highest game_time first
    )
    max_event_game_time = max_game_time_doc.get("game_time", 0) if max_game_time_doc else 0
    
    # Use the highest of: chronicle end time, or max event time
    max_game_time = max(chronicle_game_time_end, max_event_game_time)
    
    return events, last_chronicle_id, max_game_time


async def load_message_pairs(db, world_id: str, pairs: int = 10) -> list[dict]:
    """Load the last N message pairs (player + GM = 2 messages per pair)."""
    # Get messages in reverse chronological order
    messages = []
    cursor = db.messages.find({"world_id": world_id}).sort("_id", -1).limit(pairs * 2)
    
    async for doc in cursor:
        messages.append({
            "role": "player" if doc.get("message_type") == "player" else "gm",
            "content": doc.get("content"),
            "character_name": doc.get("character_name"),
        })
    
    # Reverse to chronological order
    messages.reverse()
    return messages


# ============================================================================
# Node: Load Context
# ============================================================================

async def create_load_context_node(db):
    """Create the load_context node with database access."""
    
    async def load_context_node(state: GMAgentState) -> dict[str, Any]:
        """Deterministically load all context needed for the GM agent."""
        world_id = state.get("world_id")
        
        if not world_id:
            logger.warning("No world_id in state, skipping context load")
            return {}
        
        logger.info(f"Loading context for world {world_id}")
        logger.debug(f"[load_context] world_id={world_id}")
        
        # Load world state
        world_context = await load_world_context(db, world_id)
        logger.debug(f"[load_context] world_context: {len(world_context.get('player_characters', []))} PCs, {len(world_context.get('active_quests', []))} quests, encounter={bool(world_context.get('active_encounter'))}")
        
        # Load events since last chronicle (and get max game_time from all events)
        events, last_chronicle_id, max_game_time = await load_events_since_chronicle(db, world_id)
        
        # Extract first and last event IDs for chronicle linking
        first_event_id = events[0]["id"] if events else ""
        last_event_id = events[-1]["id"] if events else ""
        
        # Use max_game_time which considers ALL events (not just since chronicle)
        # This ensures the Scribe always builds on top of the highest time seen
        current_game_time = max_game_time
        
        logger.info(f"Loaded context: {len(events)} events since chronicle {last_chronicle_id}, max_game_time={current_game_time}")
        logger.debug(f"[load_context] first_event_id={first_event_id}, last_event_id={last_event_id}")
        
        creation_in_progress = world_context.get("world", {}).get("creation_in_progress", False)
        
        return {
            "world_context": json.dumps(world_context, indent=2),
            "events_context": json.dumps(events, indent=2),
            "last_chronicle_id": last_chronicle_id,
            "first_event_id": first_event_id,
            "last_event_id": last_event_id,
            "current_game_time": current_game_time,
            "creation_in_progress": creation_in_progress,
        }
    
    return load_context_node


# ============================================================================
# Node: Historian Agent (context enrichment - read-only search)
# ============================================================================

def create_historian_agent_node(llm_with_historian_tools, db):
    """Create the Historian agent node that searches for relevant context."""

    async def historian_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Analyze context and user message, search for additional detail, return findings."""
        messages = list(state.get("historian_messages", []))
        world_id = state.get("world_id")

        logger.debug(f"[historian] Invoking with {len(messages)} messages")

        response = await llm_with_historian_tools.ainvoke(messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        has_tools = bool(tool_calls)
        logger.debug(f"[historian] response: tool_calls={has_tools}, tools={[tc['name'] for tc in tool_calls]}, content_len={len(response.content or '')}")

        return {"historian_messages": [response]}

    return historian_agent_node


def compile_historian_context_node(state: GMAgentState) -> dict[str, Any]:
    """Capture historian output (and tool results) into enriched_context for the GM."""
    messages = state.get("historian_messages", [])
    # All historian messages except the initial system messages
    parts = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", None)
            if content:
                # Ensure content is a string (it might be a list of content blocks)
                if isinstance(content, list):
                    content = str(content)
                parts.append(content)
        elif isinstance(msg, ToolMessage):
            content = getattr(msg, "content", None) or str(msg)
            # Ensure content is a string
            if isinstance(content, list):
                content = str(content)
            if len(str(content)) > 2000:
                content = str(content)[:2000] + "..."
            parts.append(f"{getattr(msg, 'name', 'tool')}: {content}")
    enriched_context = "\n\n".join(parts) if parts else "{}"
    logger.debug(f"[compile_historian] enriched_context len={len(enriched_context)}, parts={len(parts)}")
    return {"enriched_context": enriched_context}


# ============================================================================
# Node: GM Agent
# ============================================================================

def create_gm_agent_node(llm_with_tools, db):
    """Create the GM agent node."""
    
    async def gm_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Process messages and generate GM response."""
        messages = list(state.get("gm_messages", []))
        world_id = state.get("world_id")
        
        logger.debug(f"[gm_agent] Invoking with {len(messages)} messages")

        # Invoke LLM
        response = await llm_with_tools.ainvoke(messages)
        
        tool_calls = getattr(response, "tool_calls", None) or []
        logger.debug(f"[gm_agent] response: tool_calls={len(tool_calls)} {[tc['name'] for tc in tool_calls]}, content_len={len(response.content or '')}")
        
        return {"gm_messages": [response]}
    
    return gm_agent_node


# ============================================================================
# Node: Scribe Agent
# ============================================================================

def _extract_this_turn_content(messages: list[BaseMessage], history_count: int) -> str:
    """
    Extract content from this turn's messages (everything after history).
    
    Returns a formatted string describing what happened this turn.
    The history_count tells us how many messages were from history (before this turn).
    This turn = messages[history_count:] (current HumanMessage + GM responses/tools)
    """
    # Messages from this turn start after history
    this_turn = messages[history_count:] if history_count < len(messages) else messages[-1:]
    
    parts = []
    for msg in this_turn:
        msg_type = type(msg).__name__
        content = msg.content if hasattr(msg, "content") else ""
        
        if msg_type == "HumanMessage":
            parts.append(f"PLAYER: {content}")
        elif msg_type == "AIMessage":
            # Check if it has tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls]
                parts.append(f"GM TOOL CALLS: {', '.join(tool_names)}")
            if content:
                parts.append(f"GM RESPONSE: {content}")
        elif msg_type == "ToolMessage":
            # Summarize tool results briefly
            tool_name = msg.name if hasattr(msg, "name") else "tool"
            # Truncate long tool results
            result_preview = content[:200] + "..." if len(content) > 200 else content
            parts.append(f"TOOL RESULT ({tool_name}): {result_preview}")
    
    return "\n\n".join(parts) if parts else "No significant activity this turn."


# ============================================================================
# Node: Accountant Agent (state sync)
# ============================================================================

def create_accountant_agent_node(llm_with_accountant_tools, db):
    """
    Create the Accountant agent node that syncs game state to the database.
    
    The Accountant reviews the GM's narrative and makes tool calls to persist
    any state changes (damage, healing, status effects, movement, items) that
    the GM narrated but didn't explicitly call tools for.
    
    Args:
        llm_with_accountant_tools: LLM with accountant tools bound
        db: Database connection for activity logging
    """
    
    async def accountant_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Review GM's response and sync any state changes to the database."""
        messages = list(state.get("accountant_messages", []))
        world_id = state.get("world_id")
        
        logger.debug(f"[accountant] Invoking with {len(messages)} messages")
        
        # Invoke accountant LLM
        response = await llm_with_accountant_tools.ainvoke(messages)
        
        # Debug logging
        acct_tools = getattr(response, "tool_calls", None) or []
        has_tool_calls = bool(acct_tools)
        logger.info(f"[Accountant] Response: has_tool_calls={has_tool_calls}, content_length={len(response.content) if response.content else 0}")
        logger.debug(f"[accountant] response tool_calls={[tc['name'] for tc in acct_tools]}")
        
        return {"accountant_messages": [response]}
    
    return accountant_agent_node


# ============================================================================
# Node: Bard Agent (entity recording)
# ============================================================================

def create_bard_agent_node(llm_with_bard_tools, db):
    """Create the Bard agent node that records new entities from the GM's narrative."""

    async def bard_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Review GM's narrative and create/update records for NPCs, locations, lore, factions."""
        messages = list(state.get("bard_messages", []))
        world_id = state.get("world_id")

        logger.debug(f"[bard] Invoking with {len(messages)} messages")

        response = await llm_with_bard_tools.ainvoke(messages)
        bard_tools = getattr(response, "tool_calls", None) or []
        logger.info(f"[Bard] Routing to {'tools' if bard_tools else 'accountant'}: {[tc['name'] for tc in bard_tools] if bard_tools else 'none'}")

        return {"bard_messages": [response]}

    return bard_agent_node


def create_scribe_agent_node(llm_with_scribe_tools, db):
    """
    Create the Scribe agent node that records events and chronicles.
    
    Args:
        llm_with_scribe_tools: LLM with scribe tools bound
        db: Database connection for activity logging
    """
    
    async def scribe_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Review this turn and create events/chronicles."""
        messages = list(state.get("scribe_messages", []))
        world_id = state.get("world_id")
        
        logger.debug(f"[scribe] Invoking with {len(messages)} messages")
        
        # Invoke scribe LLM
        response = await llm_with_scribe_tools.ainvoke(messages)
        
        # Debug logging
        scribe_tools = getattr(response, "tool_calls", None) or []
        has_tool_calls = bool(scribe_tools)
        logger.info(f"[Scribe] Response: has_tool_calls={has_tool_calls}, content_length={len(response.content) if response.content else 0}")
        logger.debug(f"[scribe] response tool_calls={[tc['name'] for tc in scribe_tools]}, content_preview={repr((response.content or '')[:150])}")
        
        return {"scribe_messages": [response]}
    
    return scribe_agent_node


# ============================================================================
# Tool Node Factory
# ============================================================================

def create_logging_tool_node(tools, db, node_name: str = "tools", messages_key: str = "messages"):
    """Create a tool node with logging that reads/writes to a specific message stream.
    
    Args:
        tools: List of tools available to this node
        db: Database connection for activity logging
        node_name: Display name for logging
        messages_key: State key for the message list (e.g. "messages", "bard_messages")
    """
    base_tool_node = ToolNode(tools)
    
    async def logging_tool_node(state: GMAgentState) -> dict[str, Any]:
        """Wrapper that logs tool execution timing."""
        messages = state.get(messages_key, [])
        world_id = state.get("world_id")
        last_message = messages[-1] if messages else None
        
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls
            tool_names = [tc["name"] for tc in tool_calls]
            
            start_time = time.time()
            logger.info(f"[{node_name}] Executing {len(tool_calls)} tool(s): {tool_names}")
            logger.debug(f"[{node_name}] tool_calls args (truncated): {[(tc.get('name'), str(tc.get('args', {}))[:80]) for tc in tool_calls]}")
            
            # Execute via the base ToolNode (always uses "messages" key)
            result = await base_tool_node.ainvoke({"messages": messages})
            
            elapsed = time.time() - start_time
            logger.info(f"[{node_name}] Tool execution completed in {elapsed:.2f}s")
            
            # Log individual results
            if "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage):
                        content_preview = str(msg.content)[:50] if len(str(msg.content)) > 50 else str(msg.content)
                        logger.info(f"  [{node_name}] Tool result [{msg.name}]: {content_preview}")
            
            # Re-map "messages" -> messages_key for correct state field
            if "messages" in result:
                return {messages_key: result["messages"]}
            return result
        
        # No tool calls - invoke base node
        result = await base_tool_node.ainvoke({"messages": messages})
        if "messages" in result:
            return {messages_key: result["messages"]}
        return result
    
    return logging_tool_node


# ============================================================================
# Capture GM Response Node
# ============================================================================

def capture_gm_response_node(state: GMAgentState) -> dict[str, Any]:
    """Capture the GM's final response and tool calls before transitioning to accountant/scribe.
    
    This ensures we have:
    1. The GM's response stored separately from subsequent agent messages
    2. A record of all tool calls the GM made (for Accountant to avoid duplicates)
    """
    messages = state.get("gm_messages", [])
    
    # Find the last AI message that has content and no tool calls (the final response)
    gm_response = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                gm_response = msg.content
                break
    
    # Collect ALL tool calls made by GM (for Accountant to know what's already done)
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
        "gm_tool_calls": gm_tool_calls
    }


# ============================================================================
# Init Nodes (build initial message context for each agent)
# ============================================================================

def historian_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Historian agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    events_context = state.get("events_context")
    messages = list(state.get("messages", []))
    history_count = state.get("history_message_count", 0)
    
    logger.debug(f"[historian_init] Building context for historian")
    
    historian_messages = [SystemMessage(content=HISTORIAN_SYSTEM_PROMPT)]
    if world_id:
        historian_messages.append(SystemMessage(content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"))
    if world_context:
        historian_messages.append(SystemMessage(content=f"[WORLD STATE]\n{world_context}\n[END WORLD STATE]"))
    if events_context:
        historian_messages.append(SystemMessage(content=f"[RECENT EVENTS]\n{events_context}\n[END RECENT EVENTS]"))

    # Inject the prior GM response so the historian can proactively look up NPCs
    # the GM is likely to re-narrate this turn, not just those the player mentioned
    if history_count > 0:
        prior_msg = messages[history_count - 1]
        if isinstance(prior_msg, AIMessage) and prior_msg.content:
            historian_messages.append(SystemMessage(
                content=f"[PRIOR GM RESPONSE]\n{prior_msg.content}\n[END PRIOR GM RESPONSE]"
            ))

    # Current turn user message
    if history_count < len(messages):
        player_msg = messages[history_count]
        if hasattr(player_msg, "content"):
            historian_messages.append(SystemMessage(content=f"[PLAYER MESSAGE]\n{player_msg.content}\n[END PLAYER MESSAGE]"))
    
    historian_messages.append(HumanMessage(content="Identify any characters, locations, lore, or events referenced above that lack detail. Search for additional information and return a brief JSON summary of what you found (or empty {} if nothing to add)."))
    
    return {"historian_messages": historian_messages}


def gm_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for GM agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    events_context = state.get("events_context")
    enriched_context = state.get("enriched_context")
    messages = list(state.get("messages", []))
    
    logger.debug(f"[gm_init] Building context for GM agent")
    
    # Build system messages
    system_messages = [SystemMessage(content=GM_SYSTEM_PROMPT)]
    
    # Inject world context
    if world_context:
        world_ctx = SystemMessage(
            content=f"[CURRENT WORLD STATE]\nNote: 'player_characters' are controlled by real players. NEVER write their actions, dialogue, or reactions.\n{world_context}\n[END WORLD STATE]"
        )
        system_messages.append(world_ctx)
    
    # Inject events context (critical for narrative consistency)
    if events_context:
        events_ctx = SystemMessage(
            content=f"[EVENTS SINCE LAST CHRONICLE - These are the canonical events that have occurred. Your narrative MUST be consistent with these.]\n{events_context}\n[END EVENTS]"
        )
        system_messages.append(events_ctx)

    # Inject enriched context from Historian (additional findings)
    if enriched_context:
        system_messages.append(
            SystemMessage(content=f"[ENRICHED CONTEXT - Additional detail from the Historian.]\n{enriched_context}\n[END ENRICHED CONTEXT]")
        )

    # Inject world_id for tool calls
    if world_id:
        world_id_ctx = SystemMessage(
            content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"
        )
        system_messages.append(world_id_ctx)
    
    # Combine system messages with conversation
    gm_messages = system_messages + messages
    logger.debug(f"[gm_init] Built {len(system_messages)} system + {len(messages)} conversation = {len(gm_messages)} total")
    
    return {"gm_messages": gm_messages}


def bard_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Bard agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    gm_response = state.get("gm_final_response", "")
    gm_tool_calls = state.get("gm_tool_calls", [])
    
    logger.debug(f"[bard_init] Building context for Bard")
    
    bard_messages = [SystemMessage(content=BARD_SYSTEM_PROMPT)]
    if world_id:
        bard_messages.append(SystemMessage(content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"))
    if world_context:
        bard_messages.append(SystemMessage(content=f"[WORLD STATE]\n{world_context}\n[END WORLD STATE]"))
    if gm_tool_calls:
        tool_summary = "\n".join([f"- {tc.get('name')}({tc.get('args', {})})" for tc in gm_tool_calls])
        bard_messages.append(SystemMessage(content=f"[TOOLS GM ALREADY CALLED - do NOT duplicate]\n{tool_summary}\n[END]"))
    if gm_response:
        bard_messages.append(SystemMessage(content=f"[GM'S NARRATIVE]\n{gm_response}\n[END NARRATIVE]"))
    bard_messages.append(HumanMessage(content="Identify any NEW named NPCs, locations, lore, or factions mentioned in the narrative. First, search for them to check if they exist. Then create NEW entities or update EXISTING entities with new information. Batch your searches first, then batch your creates/updates. If nothing new to record, respond with 'No new entities to record.'"))
    
    return {"bard_messages": bard_messages}


def accountant_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Accountant agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    gm_response = state.get("gm_final_response", "")
    gm_tool_calls = state.get("gm_tool_calls", [])
    
    logger.debug(f"[accountant_init] Building context for Accountant")
    
    # 1. System prompt
    accountant_messages = [SystemMessage(content=ACCOUNTANT_SYSTEM_PROMPT)]
    
    # 2. World ID for tool calls
    if world_id:
        accountant_messages.append(SystemMessage(
            content=f"world_id for tool calls: {world_id}"
        ))
    
    # 3. World state (characters with IDs, locations, etc.)
    if world_context:
        accountant_messages.append(SystemMessage(
            content=f"=== WORLD STATE (use these IDs for tool calls) ===\n\n{world_context}"
        ))
    
    # 4. GM's final response (what we need to sync)
    if gm_response:
        accountant_messages.append(SystemMessage(
            content=f"=== GM'S NARRATIVE (sync state changes from this) ===\n\n{gm_response}"
        ))
    
    # 5. Tools the GM already called (don't duplicate these)
    if gm_tool_calls:
        tool_call_summary = "\n".join([
            f"- {tc['name']}({', '.join(f'{k}={v}' for k, v in tc['args'].items())})"
            for tc in gm_tool_calls
        ])
        accountant_messages.append(SystemMessage(
            content=f"=== TOOLS GM ALREADY CALLED (do NOT duplicate) ===\n\n{tool_call_summary}"
        ))
    
    # 6. Instruction to do the work
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
    
    logger.debug(f"[scribe_init] Building context for Scribe")
    
    # 1. System prompt
    scribe_messages = [SystemMessage(content=SCRIBE_SYSTEM_PROMPT)]
    
    # 2. World ID for tool calls
    if world_id:
        scribe_messages.append(SystemMessage(
            content=f"world_id for tool calls: {world_id}"
        ))
    
    # 3. Context from BEFORE this turn (world state + previous events)
    context_parts = []
    
    if world_context:
        context_parts.append(f"WORLD STATE:\n{world_context}")
    
    if events_context:
        context_parts.append(f"PREVIOUS EVENTS (already recorded, do not duplicate):\n{events_context}")
    
    if context_parts:
        scribe_messages.append(SystemMessage(
            content="=== CONTEXT FROM BEFORE THIS TURN ===\n\n" + "\n\n---\n\n".join(context_parts)
        ))
    
    # 4. THIS TURN's content - extracted and formatted clearly
    this_turn_content = _extract_this_turn_content(messages, history_count)

    # Append the GM's final narrative response so the scribe sees what actually happened.
    # gm_final_response is captured by capture_gm_response_node and lives in state,
    # not in the messages list, which is why _extract_this_turn_content misses it.
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
    
    # 5. Event IDs for chronicle linking and game time
    event_count = len(json.loads(events_context)) if events_context else 0
    logger.debug(f"[scribe_init] current_game_time={current_game_time}, event_count={event_count}")
    
    # Format game time for readability
    days = current_game_time // 86400
    remaining = current_game_time % 86400
    hours = remaining // 3600
    minutes = (remaining % 3600) // 60
    time_str = f"Day {days + 1}, {hours:02d}:{minutes:02d}"
    
    chronicle_info = ""
    if first_event_id and last_event_id:
        chronicle_info = f"\n\nIf creating a chronicle: use start_event_id={first_event_id} and end_event_id={last_event_id} to link the {event_count} events."
    
    # 6. User message asking scribe to do its job
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


def world_creator_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for World Creator agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    messages = list(state.get("messages", []))
    
    logger.debug(f"[world_creator_init] Building context for World Creator")
    
    # Build system messages with context
    system_messages = [SystemMessage(content=WORLD_CREATOR_SYSTEM_PROMPT)]
    
    # Inject world context
    if world_context:
        world_ctx = SystemMessage(content=f"[WORLD STATE]\n{world_context}\n[END WORLD STATE]")
        system_messages.append(world_ctx)
    
    # Inject world_id for tool calls
    if world_id:
        system_messages.append(SystemMessage(content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"))
    
    # Combine with conversation
    world_creator_messages = system_messages + messages
    
    return {"world_creator_messages": world_creator_messages}


def char_creator_init_node(state: GMAgentState) -> dict[str, Any]:
    """Build initial message context for Character Creator agent."""
    world_id = state.get("world_id")
    world_context = state.get("world_context")
    messages = list(state.get("messages", []))
    
    logger.debug(f"[char_creator_init] Building context for Character Creator")
    
    # Build system messages with context
    system_messages = [SystemMessage(content=CHAR_CREATOR_SYSTEM_PROMPT)]
    
    # Inject world context (includes character info)
    if world_context:
        world_ctx = SystemMessage(content=f"[WORLD STATE]\n{world_context}\n[END WORLD STATE]")
        system_messages.append(world_ctx)
    
    # Inject world_id for tool calls
    if world_id:
        system_messages.append(SystemMessage(content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"))
    
    # Combine with conversation
    char_creator_messages = system_messages + messages
    
    return {"char_creator_messages": char_creator_messages}


# ============================================================================
# Routing Functions
# ============================================================================

def route_entry(state: GMAgentState) -> Literal["world_creator", "char_creator", "historian"]:
    """Route from load_context to the appropriate agent based on creation flags."""
    if state.get("creation_in_progress", False):
        logger.info("[route_entry] creation_in_progress=True -> world_creator")
        return "world_creator"
    if state.get("needs_character_creation", False):
        logger.info("[route_entry] needs_character_creation=True -> char_creator")
        return "char_creator"
    logger.debug("[route_entry] -> historian (normal gameplay)")
    return "historian"


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


def should_continue_historian(state: GMAgentState) -> Literal["historian_tools", "compile_historian"]:
    """Route from Historian agent to tools or compile context."""
    messages = state.get("historian_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Historian] Routing to tools: {tool_names}")
        logger.debug(f"[should_continue_historian] -> historian_tools")
        return "historian_tools"

    logger.info("[Historian] No tool calls, compiling context")
    logger.debug(f"[should_continue_historian] -> compile_historian")
    return "compile_historian"


def should_continue_bard(state: GMAgentState) -> Literal["bard_tools", "accountant"]:
    """Route from Bard agent to tools or accountant."""
    messages = state.get("bard_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Bard] Routing to tools: {tool_names}")
        logger.debug(f"[should_continue_bard] -> bard_tools")
        return "bard_tools"

    logger.info("[Bard] No entities to record, routing to accountant")
    logger.debug(f"[should_continue_bard] -> accountant")
    return "accountant"


def should_continue_gm(state: GMAgentState) -> Literal["gm_tools", "capture_response"]:
    """Route from GM agent to tools or capture response."""
    messages = state.get("gm_messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[GM] Routing to tools: {tool_names}")
        logger.debug(f"[should_continue_gm] -> gm_tools")
        return "gm_tools"

    logger.info("[GM] Response complete, capturing response")
    logger.debug(f"[should_continue_gm] -> capture_response")
    return "capture_response"


def should_continue_accountant(state: GMAgentState) -> Literal["accountant_tools", "scribe"]:
    """Route from Accountant agent to tools or scribe."""
    messages = state.get("accountant_messages", [])
    last_message = messages[-1] if messages else None
    
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Accountant] Routing to tools: {tool_names}")
        logger.debug(f"[should_continue_accountant] -> accountant_tools")
        return "accountant_tools"
    
    logger.info("[Accountant] No state changes, routing to scribe")
    logger.debug(f"[should_continue_accountant] -> scribe")
    return "scribe"


def should_continue_scribe(state: GMAgentState) -> Literal["scribe_tools", "cleanup"]:
    """Route from Scribe agent to tools or cleanup."""
    messages = state.get("scribe_messages", [])
    last_message = messages[-1] if messages else None
    
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Scribe] Routing to tools: {tool_names}")
        logger.debug(f"[should_continue_scribe] -> scribe_tools")
        return "scribe_tools"
    
    logger.info("[Scribe] Complete, routing to cleanup")
    logger.debug(f"[should_continue_scribe] -> cleanup")
    return "cleanup"


# ============================================================================
# Node: World Creator Agent (for world creation phase)
# ============================================================================

def create_world_creator_agent_node(llm_with_tools, db):
    """Create the World Creator agent node for conversational world building."""
    
    async def world_creator_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Run the world creator agent for this turn."""
        messages = list(state.get("world_creator_messages", []))
        world_id = state.get("world_id")
        
        logger.info(f"[World Creator] Running for world {world_id}")
        
        logger.info(f"[world_creator] Invoking LLM with {len(messages)} messages")
        
        # Invoke LLM
        try:
            response = await llm_with_tools.ainvoke(messages)
        except Exception as e:
            logger.exception(f"[world_creator] LLM invocation failed: {type(e).__name__}: {e}")
            raise
        
        tool_calls = getattr(response, "tool_calls", None) or []
        logger.info(f"[world_creator] LLM responded: tool_calls={[tc['name'] for tc in tool_calls]}, content_len={len(response.content or '')}")
        
        return {"world_creator_messages": [response]}
    
    return world_creator_agent_node


# ============================================================================
# Node: Character Creator Agent (for character creation phase)
# ============================================================================

def create_char_creator_agent_node(llm_with_tools, db):
    """Create the Character Creator agent node for conversational character building."""
    
    async def char_creator_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Run the character creator agent for this turn."""
        messages = list(state.get("char_creator_messages", []))
        world_id = state.get("world_id")
        
        logger.info(f"[Char Creator] Running for world {world_id}")
        
        logger.debug(f"[char_creator] Invoking with {len(messages)} messages")
        
        # Invoke LLM
        response = await llm_with_tools.ainvoke(messages)
        
        tool_calls = getattr(response, "tool_calls", None) or []
        logger.debug(f"[char_creator] tool_calls={[tc['name'] for tc in tool_calls]}, content_len={len(response.content or '')}")
        
        return {"char_creator_messages": [response]}
    
    return char_creator_agent_node


# ============================================================================
# Node: Persist Response (for creation paths)
# ============================================================================

def persist_response_node(state: GMAgentState) -> dict[str, Any]:
    """Grab the last AIMessage content and persist it as gm_final_response (for creation paths)."""
    # Check both world_creator_messages and char_creator_messages
    world_creator_messages = state.get("world_creator_messages", [])
    char_creator_messages = state.get("char_creator_messages", [])
    
    # Use whichever is populated
    messages = world_creator_messages if world_creator_messages else char_creator_messages
    
    # Find the last AI response (not tool calls)
    gm_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                gm_response = msg.content
                break
    
    logger.info(f"[persist_response] Captured response: {gm_response[:100] if gm_response else '(empty)'}...")
    
    return {"gm_final_response": gm_response}


# ============================================================================
# Node: Debug State Logger
# ============================================================================



# ============================================================================
# Graph Factory
# ============================================================================

async def create_gm_agent(
    db,  # Database connection
    mcp_url: str = "http://localhost:8080",
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.7,
) -> tuple[StateGraph, dict]:
    """
    Create the GM agent graph with separated concerns.

    Flow:
    load_context -> ... -> bard <-> bard_tools -> (creation_in_progress ? END : accountant -> ... -> scribe -> END)

    Agent Responsibilities:
    - Historian: Read-only search to enrich context (lore, characters, locations, events)
    - Bard: Record new entities from GM narrative (NPCs, locations, lore, factions)
    - GM: Narration, dice rolls, combat management, quest/world updates, character creation
    - Accountant: State sync (damage, healing, status effects, movement, items)
    - Scribe: Event recording and chronicle creation
    
    Note: Neither Accountant nor Scribe loop back after tools execute.
    This prevents duplicate operations if rate limiting causes LLM retries.
    Each can complete their work in a single batch of tool calls.
    
    The capture_response node stores the GM's final response and tool calls
    so the Accountant knows what state changes were already persisted.
    
    Args:
        db: MongoDB database connection
        mcp_url: URL of the rpg-mcp server
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (uses provider default if not specified)
        temperature: LLM temperature
    
    Returns:
        Tuple of (compiled_graph, config_dict)
    """
    # Get LLMs
    gm_llm = get_llm(provider, model, temperature)
    scribe_llm = get_llm(provider, model, temperature=0.3)  # Lower temperature for scribe
    
    # Get all MCP tools
    all_tools = await get_mcp_tools(mcp_url)

    # =========================================================================
    # Tool Separation: Historian (read-only) | Bard (entity creation) | GM | Accountant | Scribe
    # =========================================================================

    # Historian tools (read-only search)
    historian_tool_names = {
        "search_lore", "find_characters", "find_locations", "search_locations", "find_events",
        "find_quests", "find_factions", "get_entity", "get_chronicle_details",
        "get_location_contents", "find_nearby_locations", "get_character_inventory",
    }
    historian_tools = [t for t in all_tools if t.name in historian_tool_names]

    # Bard tools (search + create/update entities)
    bard_tool_names = {
        # Search tools (check if entities exist before creating)
        "find_characters",
        # Create/update tools
        "create_npc", "update_npc", "set_location",
        "set_lore", "set_faction",
    }
    bard_tools = [t for t in all_tools if t.name in bard_tool_names]

    # World Creator tools (world-building only, NPC-specific tools)
    world_creator_tool_names = {
        # Search/query
        "search_lore", "find_characters", "find_locations", "find_events",
        "find_quests", "find_factions", "get_entity", "get_location_contents", "find_nearby_locations",
        # Create/record (NPCs, locations, factions, lore)
        "set_lore", "set_location", "set_faction",
        "create_npc", "update_npc", "spawn_enemies",  # NPC-specific tools
        "set_item_blueprint", "set_ability_blueprint",
        # World basics
        "update_world_basics", "start_game",
        # Dice/random
        "roll_table", "coin_flip",
    }
    world_creator_tools = [t for t in all_tools if t.name in world_creator_tool_names]

    # Character Creator tools (PC-specific tools for stats + finalize)
    char_creator_tool_names = {
        # Update PC
        "update_pc_basics", "set_attributes", "set_skills", "grant_abilities",
        # Finalize
        "finalize_character",
        # Dice
        "roll_dice", "roll_stat_array", "roll_table",
        # Search
        "search_lore", "find_locations", "get_entity",
    }
    char_creator_tools = [t for t in all_tools if t.name in char_creator_tool_names]

    # Scribe-only tools (event and chronicle management)
    scribe_tool_names = {"record_event", "set_chronicle"}
    scribe_tools = [t for t in all_tools if t.name in scribe_tool_names]

    # Accountant tools (state sync - damage, healing, status, movement, items)
    accountant_tool_names = {
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
    accountant_tools = [t for t in all_tools if t.name in accountant_tool_names]

    # GM gets dice + encounters. Context is pre-loaded; Bard/Accountant/Scribe handle state.
    # World creation and character creation are handled by dedicated agents upstream.
    gm_tool_names = {
        "roll_dice", "roll_table", "coin_flip", "roll_stat_array", "percentile_roll",
        "start_encounter", "get_encounter", "get_active_encounter",
        "add_combatant", "set_initiative", "remove_combatant", "next_turn", "end_encounter",
    }
    gm_tools = [t for t in all_tools if t.name in gm_tool_names]

    logger.info(f"Tool separation: {len(historian_tools)} Historian, {len(bard_tools)} Bard, {len(gm_tools)} GM, {len(accountant_tools)} Accountant, {len(scribe_tools)} Scribe, {len(world_creator_tools)} WorldCreator, {len(char_creator_tools)} CharCreator")

    # Create LLMs with different temperatures
    historian_llm = get_llm(provider, model, temperature=0.3)
    bard_llm = get_llm(provider, model, temperature=0.3)
    accountant_llm = get_llm(provider, model, temperature=0.3)

    # Bind tools to LLMs
    historian_llm_with_tools = historian_llm.bind_tools(historian_tools)
    bard_llm_with_tools = bard_llm.bind_tools(bard_tools)
    gm_llm_with_tools = gm_llm.bind_tools(gm_tools)
    accountant_llm_with_tools = accountant_llm.bind_tools(accountant_tools)
    scribe_llm_with_tools = scribe_llm.bind_tools(scribe_tools)
    world_creator_llm = get_llm(provider, model, temperature=0.7)
    world_creator_llm_with_tools = world_creator_llm.bind_tools(world_creator_tools)
    char_creator_llm = get_llm(provider, model, temperature=0.7)
    char_creator_llm_with_tools = char_creator_llm.bind_tools(char_creator_tools)
    
    # Create nodes
    load_context_node = await create_load_context_node(db)
    historian_agent_node = create_historian_agent_node(historian_llm_with_tools, db)
    historian_tools_node = create_logging_tool_node(historian_tools, db, "Historian", "historian_messages")
    gm_agent_node = create_gm_agent_node(gm_llm_with_tools, db)
    gm_tools_node = create_logging_tool_node(gm_tools, db, "GM", "gm_messages")
    accountant_agent_node = create_accountant_agent_node(accountant_llm_with_tools, db)
    accountant_tools_node = create_logging_tool_node(accountant_tools, db, "Accountant", "accountant_messages")
    bard_agent_node = create_bard_agent_node(bard_llm_with_tools, db)
    bard_tools_node = create_logging_tool_node(bard_tools, db, "Bard", "bard_messages")
    scribe_agent_node = create_scribe_agent_node(scribe_llm_with_tools, db)
    scribe_tools_node = create_logging_tool_node(scribe_tools, db, "Scribe", "scribe_messages")
    world_creator_agent_node = create_world_creator_agent_node(world_creator_llm_with_tools, db)
    world_creator_tools_node = create_logging_tool_node(world_creator_tools, db, "WorldCreator", "world_creator_messages")
    char_creator_agent_node = create_char_creator_agent_node(char_creator_llm_with_tools, db)
    char_creator_tools_node = create_logging_tool_node(char_creator_tools, db, "CharCreator", "char_creator_messages")

    # Build the graph
    workflow = StateGraph(GMAgentState)

    # Add nodes
    workflow.add_node("load_context", load_context_node)
    
    # Init nodes
    workflow.add_node("historian_init", historian_init_node)
    workflow.add_node("gm_init", gm_init_node)
    workflow.add_node("bard_init", bard_init_node)
    workflow.add_node("accountant_init", accountant_init_node)
    workflow.add_node("scribe_init", scribe_init_node)
    workflow.add_node("world_creator_init", world_creator_init_node)
    workflow.add_node("char_creator_init", char_creator_init_node)
    
    # Agent nodes
    workflow.add_node("historian", historian_agent_node)
    workflow.add_node("historian_tools", historian_tools_node)
    workflow.add_node("compile_historian", compile_historian_context_node)
    workflow.add_node("gm_agent", gm_agent_node)
    workflow.add_node("gm_tools", gm_tools_node)
    workflow.add_node("capture_response", capture_gm_response_node)
    workflow.add_node("bard", bard_agent_node)
    workflow.add_node("bard_tools", bard_tools_node)
    workflow.add_node("accountant", accountant_agent_node)
    workflow.add_node("accountant_tools", accountant_tools_node)
    workflow.add_node("scribe", scribe_agent_node)
    workflow.add_node("scribe_tools", scribe_tools_node)
    workflow.add_node("world_creator", world_creator_agent_node)
    workflow.add_node("world_creator_tools", world_creator_tools_node)
    workflow.add_node("char_creator", char_creator_agent_node)
    workflow.add_node("char_creator_tools", char_creator_tools_node)
    workflow.add_node("persist_response", persist_response_node)

    # Set entry point
    workflow.set_entry_point("load_context")

    # Add edges
    # Flow: load_context -> route_entry -> (world_creator_init | char_creator_init | historian_init) -> ...
    workflow.add_conditional_edges(
        "load_context",
        route_entry,
        {
            "world_creator": "world_creator_init",
            "char_creator": "char_creator_init",
            "historian": "historian_init",
        }
    )
    
    # World creator path: world_creator_init -> world_creator <-> world_creator_tools -> persist_response -> END
    workflow.add_edge("world_creator_init", "world_creator")
    workflow.add_conditional_edges(
        "world_creator",
        should_continue_world_creator,
        {
            "world_creator_tools": "world_creator_tools",
            "persist_response": "persist_response",
        }
    )
    workflow.add_edge("world_creator_tools", "world_creator")
    
    # Char creator path: char_creator_init -> char_creator <-> char_creator_tools -> persist_response -> END
    workflow.add_edge("char_creator_init", "char_creator")
    workflow.add_conditional_edges(
        "char_creator",
        should_continue_char_creator,
        {
            "char_creator_tools": "char_creator_tools",
            "persist_response": "persist_response",
        }
    )
    workflow.add_edge("char_creator_tools", "char_creator")
    
    # Persist response -> END (for creation paths)
    workflow.add_edge("persist_response", END)
    
    # Normal gameplay path: historian_init -> historian <-> historian_tools -> compile_historian -> gm_init -> gm_agent -> ...
    workflow.add_edge("historian_init", "historian")
    workflow.add_conditional_edges(
        "historian",
        should_continue_historian,
        {
            "historian_tools": "historian_tools",
            "compile_historian": "compile_historian",
        }
    )
    workflow.add_edge("historian_tools", "historian")
    workflow.add_edge("compile_historian", "gm_init")
    
    # GM path: gm_init -> gm_agent <-> gm_tools -> capture_response -> bard_init -> bard -> ...
    workflow.add_edge("gm_init", "gm_agent")
    workflow.add_conditional_edges(
        "gm_agent",
        should_continue_gm,
        {
            "gm_tools": "gm_tools",
            "capture_response": "capture_response",
        }
    )
    workflow.add_edge("gm_tools", "gm_agent")

    # After capturing response, go to bard_init
    workflow.add_edge("capture_response", "bard_init")
    
    # Bard path: bard_init -> bard <-> bard_tools -> accountant_init -> accountant -> ...
    workflow.add_edge("bard_init", "bard")
    workflow.add_conditional_edges(
        "bard",
        should_continue_bard,
        {
            "bard_tools": "bard_tools",
            "accountant": "accountant_init",
        }
    )
    # Bard tools loop back to bard for ReAct pattern (search -> create/update)
    workflow.add_edge("bard_tools", "bard")
    
    # Accountant path: accountant_init -> accountant -> accountant_tools -> scribe_init -> scribe -> ...
    workflow.add_edge("accountant_init", "accountant")
    workflow.add_conditional_edges(
        "accountant",
        should_continue_accountant,
        {
            "accountant_tools": "accountant_tools",
            "scribe": "scribe_init",
        }
    )
    # NOTE: Accountant tools go directly to scribe_init (no loop back to accountant)
    # This prevents duplicate state sync if rate limiting causes retries
    # The Accountant can sync all state changes in a single batch of tool calls
    workflow.add_edge("accountant_tools", "scribe_init")
    
    # Scribe path: scribe_init -> scribe -> scribe_tools -> END
    workflow.add_edge("scribe_init", "scribe")
    workflow.add_conditional_edges(
        "scribe",
        should_continue_scribe,
        {
            "scribe_tools": "scribe_tools",
            "cleanup": END,
        }
    )
    # NOTE: Scribe tools go directly to END (no loop back to scribe)
    # This prevents duplicate event recording if rate limiting causes retries
    # The Scribe can record events + chronicles in a single batch of tool calls
    workflow.add_edge("scribe_tools", END)
    
    # Compile
    graph = workflow.compile()
    logger.info("Compiled GM agent graph: load_context -> ... -> bard -> (creation_in_progress? END : accountant -> scribe -> END)")
    
    config = {
        "provider": provider,
        "model": model or ("claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o"),
        "mcp_url": mcp_url,
        "gm_tools_count": len(gm_tools),
        "accountant_tools_count": len(accountant_tools),
        "scribe_tools_count": len(scribe_tools),
    }
    
    return graph, config


# ============================================================================
# High-Level Wrapper
# ============================================================================

class GMAgent:
    """High-level wrapper for the GM agent."""
    
    def __init__(
        self,
        mcp_url: str = "http://localhost:8080",
        provider: str = "anthropic",
        model: str | None = None,
    ):
        self.mcp_url = mcp_url
        self.provider = provider
        self.model = model
        self._graph = None
        self._config = None
        self._db = None
    
    async def initialize(self, db) -> None:
        """Initialize the agent with database connection.
        
        Args:
            db: MongoDB database connection for context loading
        """
        self._db = db
        self._graph, self._config = await create_gm_agent(
            db=db,
            mcp_url=self.mcp_url,
            provider=self.provider,
            model=self.model,
        )
    
    @property
    def is_initialized(self) -> bool:
        return self._graph is not None
    
    async def chat(
        self,
        message: str,
        thread_id: str | None = None,
        world_id: str | None = None,
    ) -> str:
        """
        Simple chat method for backwards compatibility.
        
        Note: This method works best with a world_id. Without it,
        context loading will be limited.
        
        Args:
            message: Player's message
            thread_id: Ignored (for backwards compatibility)
            world_id: World ID for context loading
        
        Returns:
            GM's response text
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize(db) first.")
        
        response_text = ""
        async for event in self.stream_chat(
            message=message,
            world_id=world_id or "",
            history=None,
        ):
            messages = event.get("messages", [])
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        content = msg.content
                        if content and not content.lower().startswith("events recorded"):
                            response_text = content
        
        return response_text
    
    async def stream_chat(
        self,
        message: str,
        world_id: str,
        history: list[dict] | None = None,
        needs_character_creation: bool = False,
    ):
        """
        Stream responses from the GM.
        
        Context is loaded deterministically:
        1. World state (characters, quests, encounters)
        2. All events since last chronicle
        3. Recent message history (passed in)
        
        Args:
            message: Current user message
            world_id: World ID for context loading
            history: Recent conversation history as list of dicts:
                     [{"role": "player"|"gm", "content": "...", "character_name": "..."}]
            needs_character_creation: If True, route to character creator agent
        
        Yields:
            Events as they occur (tool calls, responses, etc.)
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize(db) first.")
        
        # Build messages from history
        messages = []
        
        if history:
            for msg in history:
                if msg["role"] == "player":
                    char_name = msg.get("character_name", "Player")
                    content = f"**Submitted by {char_name}**\n\n{msg['content']}"
                    messages.append(HumanMessage(content=content))
                elif msg["role"] == "gm":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Track how many messages are history (before this turn)
        history_message_count = len(messages)
        
        # Add the current message (this is where "this turn" begins)
        messages.append(HumanMessage(content=message))
        
        # Create input state as plain dict (TypedDict, not Pydantic)
        # history_message_count tells the scribe where history ends and this turn begins
        input_state: GMAgentState = {
            "world_id": world_id,
            "messages": messages,
            "history_message_count": history_message_count,
            "needs_character_creation": needs_character_creation,
        }
        
        logger.info(f"Running agent with {len(messages)} messages (history={len(history) if history else 0})")
        
        async for event in self._graph.astream(input_state, stream_mode="values"):
            yield event
