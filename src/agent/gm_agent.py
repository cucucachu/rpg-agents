"""GM Agent implementation using LangGraph with separated concerns.

Architecture:
1. load_context - Deterministic node that fetches world state, events, messages
2. gm_agent - Main game master agent with all game tools
3. capture_response - Captures GM response before scribe runs
4. scribe_agent - Post-processing agent that creates events and chronicles
5. cleanup - Cleans up agent activity records from the database
"""

import logging
import time
import json
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .prompts import GM_SYSTEM_PROMPT, SCRIBE_SYSTEM_PROMPT
from .mcp_tools import get_mcp_tools, get_mcp_client

logger = logging.getLogger(__name__)


# ============================================================================
# Activity Logging Helper
# ============================================================================

async def log_activity(db, world_id: str, node_name: str, activity_type: str, content: str) -> None:
    """Log agent activity to the database for streaming to UI.
    
    Args:
        db: MongoDB database connection
        world_id: The world being processed
        node_name: Which graph node is logging (load_context, gm_agent, scribe, etc.)
        activity_type: Type of activity (thinking, tool_call, tool_result, response, status)
        content: The activity text (truncated for display)
    """
    try:
        # Truncate content to 100 chars for display
        display_content = content[:100] if len(content) > 100 else content
        
        await db.agent_activity.insert_one({
            "world_id": world_id,
            "node_name": node_name,
            "activity_type": activity_type,
            "content": display_content,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        # Don't let activity logging errors break the agent
        logger.warning(f"Failed to log activity: {e}")


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
    
    # Messages (conversation) - uses operator.add to accumulate messages
    messages: Annotated[list[BaseMessage], operator.add]
    
    # How many messages in the initial state were history (before this turn)
    # Used by scribe to identify which messages are "this turn"
    history_message_count: int
    
    # Loaded context (populated by load_context node)
    world_context: str  # JSON string of world data
    events_context: str  # JSON string of events since last chronicle
    last_chronicle_id: str  # ID of most recent chronicle
    first_event_id: str  # First event ID since last chronicle (for linking)
    last_event_id: str  # Last event ID since last chronicle (for linking)
    current_game_time: int  # Game time in seconds from last event (for scribe to track time)
    
    # GM's final response (captured before scribe runs)
    # This is what gets persisted to the database
    gm_final_response: str


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
        "characters": [],
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
            "game_time": world_doc.get("game_time", 0),
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
        result["characters"].append(char)
    
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


async def load_events_since_chronicle(db, world_id: str) -> tuple[list[dict], str | None]:
    """Load all events since the last chronicle (or all events if no chronicle)."""
    from bson import ObjectId
    
    # Find the most recent chronicle
    last_chronicle = await db.chronicles.find_one(
        {"world_id": world_id},
        sort=[("_id", -1)]  # Most recent by creation time
    )
    
    last_chronicle_id = str(last_chronicle["_id"]) if last_chronicle else None
    
    # Build query for events
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
    
    return events, last_chronicle_id


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
        await log_activity(db, world_id, "load_context", "status", "Loading world context...")
        
        # Load world state
        world_context = await load_world_context(db, world_id)
        
        # Load events since last chronicle
        events, last_chronicle_id = await load_events_since_chronicle(db, world_id)
        
        # Extract first and last event IDs for chronicle linking
        first_event_id = events[0]["id"] if events else ""
        last_event_id = events[-1]["id"] if events else ""
        
        # Get current game time from last event (or 0 if no events)
        current_game_time = events[-1].get("game_time", 0) if events else 0
        
        logger.info(f"Loaded context: {len(events)} events since chronicle {last_chronicle_id}, game_time={current_game_time}")
        await log_activity(db, world_id, "load_context", "status", f"Loaded {len(events)} events, world ready")
        
        return {
            "world_context": json.dumps(world_context, indent=2),
            "events_context": json.dumps(events, indent=2),
            "last_chronicle_id": last_chronicle_id,
            "first_event_id": first_event_id,
            "last_event_id": last_event_id,
            "current_game_time": current_game_time,
        }
    
    return load_context_node


# ============================================================================
# Node: GM Agent
# ============================================================================

def create_gm_agent_node(llm_with_tools, db):
    """Create the GM agent node."""
    
    async def gm_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Process messages and generate GM response."""
        messages = list(state.get("messages", []))
        world_id = state.get("world_id")
        
        # Log thinking activity
        if world_id:
            await log_activity(db, world_id, "gm_agent", "thinking", "GM is thinking...")
        
        # Build system messages
        system_messages = [SystemMessage(content=GM_SYSTEM_PROMPT)]
        
        # Inject world context
        world_context = state.get("world_context")
        if world_context:
            world_ctx = SystemMessage(
                content=f"[CURRENT WORLD STATE]\n{world_context}\n[END WORLD STATE]"
            )
            system_messages.append(world_ctx)
        
        # Inject events context (critical for narrative consistency)
        events_context = state.get("events_context")
        if events_context:
            events_ctx = SystemMessage(
                content=f"[EVENTS SINCE LAST CHRONICLE - These are the canonical events that have occurred. Your narrative MUST be consistent with these.]\n{events_context}\n[END EVENTS]"
            )
            system_messages.append(events_ctx)
        
        # Inject world_id for tool calls
        if world_id:
            world_id_ctx = SystemMessage(
                content=f"[WORLD ID]\nworld_id for all tool calls: {world_id}\n[END WORLD ID]"
            )
            system_messages.append(world_id_ctx)
        
        # Combine system messages with conversation
        full_messages = system_messages + messages
        
        # Invoke LLM
        response = await llm_with_tools.ainvoke(full_messages)
        
        # Log the response
        if world_id:
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_names = [tc["name"] for tc in response.tool_calls]
                await log_activity(db, world_id, "gm_agent", "tool_call", f"Calling: {', '.join(tool_names)}")
            elif response.content:
                preview = response.content[:50].replace('\n', ' ')
                await log_activity(db, world_id, "gm_agent", "response", preview)
        
        return {"messages": [response]}
    
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


def create_scribe_agent_node(llm_with_scribe_tools, db):
    """
    Create the Scribe agent node that records events and chronicles.
    
    Args:
        llm_with_scribe_tools: LLM with scribe tools bound
        db: Database connection for activity logging
    """
    
    async def scribe_agent_node(state: GMAgentState) -> dict[str, Any]:
        """Review this turn and create events/chronicles."""
        messages = list(state.get("messages", []))
        world_id = state.get("world_id")
        
        # Get history count from state (set when stream_chat is called)
        hist_count = state.get("history_message_count", 0)
        
        # Log scribe activity
        if world_id:
            await log_activity(db, world_id, "scribe", "thinking", "Scribe recording this turn...")
        
        # =====================================================================
        # Build scribe context - ONLY what the scribe needs
        # =====================================================================
        
        # 1. System prompt
        scribe_messages = [SystemMessage(content=SCRIBE_SYSTEM_PROMPT)]
        
        # 2. World ID for tool calls
        if world_id:
            scribe_messages.append(SystemMessage(
                content=f"world_id for tool calls: {world_id}"
            ))
        
        # 3. Context from BEFORE this turn (world state + previous events)
        context_parts = []
        
        world_context = state.get("world_context")
        if world_context:
            context_parts.append(f"WORLD STATE:\n{world_context}")
        
        events_context = state.get("events_context")
        if events_context:
            context_parts.append(f"PREVIOUS EVENTS (already recorded, do not duplicate):\n{events_context}")
        
        if context_parts:
            scribe_messages.append(SystemMessage(
                content="=== CONTEXT FROM BEFORE THIS TURN ===\n\n" + "\n\n---\n\n".join(context_parts)
            ))
        
        # 4. THIS TURN's content - extracted and formatted clearly
        #    These are NOT the scribe's thoughts - this is what happened in the game
        this_turn_content = _extract_this_turn_content(messages, hist_count)
        
        scribe_messages.append(SystemMessage(
            content=(
                "=== THIS TURN (what just happened - record this) ===\n\n"
                "The following is a transcript of the player's action and the GM's response. "
                "This is NOT your communication - this is what you must record as events.\n\n"
                f"{this_turn_content}"
            )
        ))
        
        # 5. Event IDs for chronicle linking and game time
        first_event_id = state.get("first_event_id", "")
        last_event_id = state.get("last_event_id", "")
        current_game_time = state.get("current_game_time", 0)
        event_count = len(json.loads(events_context)) if events_context else 0
        
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
                f"Current game time: {current_game_time} seconds ({time_str})\n\n"
                f"Record events for this turn. Estimate game_time for each event by adding estimated duration to current time.\n"
                f"If 15+ previous events exist, consider creating a chronicle summary.{chronicle_info}"
            )
        ))
        
        logger.info(f"[Scribe] Processing turn: {len(messages)} total messages, {hist_count} history, this_turn has {len(messages) - hist_count} messages")
        
        # Invoke scribe LLM
        response = await llm_with_scribe_tools.ainvoke(scribe_messages)
        
        # Debug logging
        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
        logger.info(f"[Scribe] Response: has_tool_calls={bool(has_tool_calls)}, content_length={len(response.content) if response.content else 0}")
        if response.content:
            logger.info(f"[Scribe] Content: {response.content[:200]}")
        if has_tool_calls:
            logger.info(f"[Scribe] Tool calls: {[tc['name'] for tc in response.tool_calls]}")
        
        # Log the response to activity stream
        if world_id:
            if has_tool_calls:
                tool_names = [tc["name"] for tc in response.tool_calls]
                await log_activity(db, world_id, "scribe", "tool_call", f"Recording: {', '.join(tool_names)}")
            elif response.content:
                preview = response.content[:50].replace('\n', ' ')
                await log_activity(db, world_id, "scribe", "status", preview)
        
        return {"messages": [response]}
    
    return scribe_agent_node


# ============================================================================
# Tool Node Factory
# ============================================================================

def create_logging_tool_node(tools, db, node_name: str = "tools"):
    """Create a tool node with logging."""
    base_tool_node = ToolNode(tools)
    
    async def logging_tool_node(state: GMAgentState) -> dict[str, Any]:
        """Wrapper that logs tool execution timing."""
        messages = state.get("messages", [])
        world_id = state.get("world_id")
        last_message = messages[-1] if messages else None
        
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls
            tool_names = [tc["name"] for tc in tool_calls]
            
            start_time = time.time()
            logger.info(f"[{node_name}] Executing {len(tool_calls)} tool(s): {tool_names}")
            
            # Log activity
            if world_id:
                await log_activity(db, world_id, node_name.lower(), "tool_call", f"Executing: {', '.join(tool_names)}")
            
            # Execute via the base ToolNode
            result = await base_tool_node.ainvoke({"messages": messages})
            
            elapsed = time.time() - start_time
            logger.info(f"[{node_name}] Tool execution completed in {elapsed:.2f}s")
            
            # Log individual results
            if "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage):
                        content_preview = str(msg.content)[:50] if len(str(msg.content)) > 50 else str(msg.content)
                        logger.info(f"  [{node_name}] Tool result [{msg.name}]: {content_preview}")
                        
                        # Log activity for tool result
                        if world_id:
                            await log_activity(db, world_id, node_name.lower(), "tool_result", f"{msg.name}: {content_preview}")
            
            return result
        
        return await base_tool_node.ainvoke({"messages": messages})
    
    return logging_tool_node


# ============================================================================
# Capture GM Response Node
# ============================================================================

def capture_gm_response_node(state: GMAgentState) -> dict[str, Any]:
    """Capture the GM's final response before transitioning to scribe.
    
    This ensures we have the GM's response stored separately from
    any subsequent scribe messages.
    """
    messages = state.get("messages", [])
    
    # Find the last AI message that has content and no tool calls
    gm_response = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                gm_response = msg.content
                break
    
    if gm_response:
        logger.info(f"[Capture] Captured GM response: {gm_response[:100]}...")
    else:
        logger.warning("[Capture] No GM response found to capture")
    
    return {"gm_final_response": gm_response}


# ============================================================================
# Routing Functions
# ============================================================================

def should_continue_gm(state: GMAgentState) -> Literal["gm_tools", "capture_response"]:
    """Route from GM agent to tools or capture response."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[GM] Routing to tools: {tool_names}")
        return "gm_tools"
    
    logger.info("[GM] Response complete, capturing response")
    return "capture_response"


def should_continue_scribe(state: GMAgentState) -> Literal["scribe_tools", "cleanup"]:
    """Route from Scribe agent to tools or cleanup."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[Scribe] Routing to tools: {tool_names}")
        return "scribe_tools"
    
    logger.info("[Scribe] Complete, routing to cleanup")
    return "cleanup"


# ============================================================================
# Node: Cleanup Activity
# ============================================================================

async def create_cleanup_node(db):
    """Create the cleanup node that deletes activity records."""
    
    async def cleanup_node(state: GMAgentState) -> dict[str, Any]:
        """Delete all agent activity records for this world after the turn completes."""
        world_id = state.get("world_id")
        
        if world_id:
            try:
                result = await db.agent_activity.delete_many({"world_id": world_id})
                logger.info(f"[Cleanup] Deleted {result.deleted_count} activity records for world {world_id}")
            except Exception as e:
                logger.warning(f"[Cleanup] Failed to delete activity records: {e}")
        
        # Return empty dict - no state changes
        return {}
    
    return cleanup_node


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
    load_context -> gm_agent <-> gm_tools -> capture_response -> scribe <-> scribe_tools -> cleanup -> END
    
    The capture_response node stores the GM's final response in state.gm_final_response
    before the scribe runs. This ensures the GM's response is preserved separately
    from any scribe messages.
    
    The cleanup node deletes all agent_activity records for this world.
    
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
    
    # Scribe-only tools (event and chronicle management)
    scribe_tool_names = {"record_event", "set_chronicle"}
    scribe_tools = [t for t in all_tools if t.name in scribe_tool_names]
    
    # GM tools = all tools EXCEPT scribe tools (separation of concerns)
    # The GM focuses on gameplay; the Scribe handles record-keeping
    gm_tools = [t for t in all_tools if t.name not in scribe_tool_names]
    
    logger.info(f"Loaded {len(gm_tools)} GM tools, {len(scribe_tools)} scribe tools (separated)")
    
    # Bind tools to LLMs
    gm_llm_with_tools = gm_llm.bind_tools(gm_tools)
    scribe_llm_with_tools = scribe_llm.bind_tools(scribe_tools)
    
    # Create nodes (pass db for activity logging)
    load_context_node = await create_load_context_node(db)
    gm_agent_node = create_gm_agent_node(gm_llm_with_tools, db)
    gm_tools_node = create_logging_tool_node(gm_tools, db, "GM")
    scribe_agent_node = create_scribe_agent_node(scribe_llm_with_tools, db)
    scribe_tools_node = create_logging_tool_node(scribe_tools, db, "Scribe")
    cleanup_node = await create_cleanup_node(db)
    
    # Build the graph
    workflow = StateGraph(GMAgentState)
    
    # Add nodes
    workflow.add_node("load_context", load_context_node)
    workflow.add_node("gm_agent", gm_agent_node)
    workflow.add_node("gm_tools", gm_tools_node)
    workflow.add_node("capture_response", capture_gm_response_node)
    workflow.add_node("scribe", scribe_agent_node)
    workflow.add_node("scribe_tools", scribe_tools_node)
    workflow.add_node("cleanup", cleanup_node)
    
    # Set entry point
    workflow.set_entry_point("load_context")
    
    # Add edges
    # Flow: load_context -> gm_agent <-> gm_tools -> capture_response -> scribe <-> scribe_tools -> cleanup -> END
    workflow.add_edge("load_context", "gm_agent")
    
    workflow.add_conditional_edges(
        "gm_agent",
        should_continue_gm,
        {
            "gm_tools": "gm_tools",
            "capture_response": "capture_response",
        }
    )
    workflow.add_edge("gm_tools", "gm_agent")
    workflow.add_edge("capture_response", "scribe")
    
    workflow.add_conditional_edges(
        "scribe",
        should_continue_scribe,
        {
            "scribe_tools": "scribe_tools",
            "cleanup": "cleanup",
        }
    )
    workflow.add_edge("scribe_tools", "scribe")
    workflow.add_edge("cleanup", END)
    
    # Compile
    graph = workflow.compile()
    logger.info("Compiled GM agent graph: load_context -> gm_agent -> capture_response -> scribe -> cleanup")
    
    config = {
        "provider": provider,
        "model": model or ("claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o"),
        "mcp_url": mcp_url,
        "gm_tools_count": len(gm_tools),
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
                    content = msg["content"]
                    if not content.startswith(f"[{char_name}]"):
                        content = f"[{char_name}]: {content}"
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
        }
        
        logger.info(f"Running agent with {len(messages)} messages (history={len(history) if history else 0})")
        
        async for event in self._graph.astream(input_state, stream_mode="values"):
            yield event
