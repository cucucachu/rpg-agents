"""Message endpoints for chat functionality.

This module implements a clean separation between:
- UI Messages: What users see in chat (stored in MongoDB)
- Agent Context: Minimal context for LLM reasoning (built fresh each call)

Key endpoints:
- POST /api/worlds/{world_id}/messages - Send message (non-blocking, returns 202)
- GET /api/worlds/{world_id}/messages/stream - SSE stream of new messages
- GET /api/worlds/{world_id}/messages - Get message history
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from bson import ObjectId

from ..db import get_db
from ..models import Message, Trace, serialize_messages, User
from .auth import get_current_user, get_current_user_optional
from .broadcast import get_broadcaster
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/worlds", tags=["messages"])

# Removed: NEW_PLAYER_MESSAGE - character creation now handled by dedicated agent

# ============================================================================
# Per-World Lock Management
# ============================================================================

_world_locks: dict[str, asyncio.Lock] = {}
_world_processing: dict[str, bool] = {}  # Track if world is currently processing
_world_lock_holder: dict[str, dict] = {}  # Track who holds the lock {user_id, character_name}


def get_world_lock(world_id: str) -> asyncio.Lock:
    """Get or create a lock for a world."""
    if world_id not in _world_locks:
        _world_locks[world_id] = asyncio.Lock()
        _world_processing[world_id] = False
    return _world_locks[world_id]


def is_world_processing(world_id: str) -> bool:
    """Check if a world is currently processing a message."""
    return _world_processing.get(world_id, False)


def get_world_lock_holder(world_id: str) -> dict | None:
    """Get info about who holds the lock for a world."""
    if is_world_processing(world_id):
        return _world_lock_holder.get(world_id)
    return None


# ============================================================================
# Request/Response Models
# ============================================================================

class SendMessageRequest(BaseModel):
    """Request to send a message."""
    content: str = Field(..., min_length=1, max_length=10000)


class MessageResponse(BaseModel):
    """A message in the response."""
    id: str
    world_id: str
    user_id: str | None
    character_name: str
    display_name: str | None  # User's display name (for player messages)
    content: str
    message_type: str
    created_at: str


class SendMessageResponse(BaseModel):
    """Response from sending a message."""
    user_message: MessageResponse
    gm_message: MessageResponse | None = None
    status: str = "processing"  # "processing" or "complete"


class MessageHistoryResponse(BaseModel):
    """Response with message history."""
    messages: list[MessageResponse]
    total: int
    has_more: bool


# ============================================================================
# Agent Reference (set by server.py)
# ============================================================================

_gm_agent = None


def set_gm_agent(agent):
    """Set the GM agent reference."""
    global _gm_agent
    _gm_agent = agent
    logger.info("GM Agent set for messages module")


# ============================================================================
# Helper Functions
# ============================================================================

async def get_recent_messages(db, world_id: str, limit: int = 10) -> list[dict]:
    """Get recent messages for agent context."""
    cursor = db.messages.find(
        {"world_id": world_id}
    ).sort("created_at", -1).limit(limit)
    
    messages = []
    async for doc in cursor:
        msg = Message.from_doc(doc)
        messages.append({
            "role": "gm" if msg.message_type == "gm" else "player",
            "content": msg.content,
            "character_name": msg.character_name,
        })
    
    # Reverse to get chronological order
    messages.reverse()
    return messages


async def get_user_character_name(db, user_id: str, world_id: str) -> str:
    """Get the character name for a user in a world."""
    # Check world access for character assignment
    access = await db.world_access.find_one({
        "user_id": user_id,
        "world_id": world_id,
    })
    
    if access and access.get("character_id"):
        char = await db.characters.find_one({"_id": ObjectId(access["character_id"])})
        if char:
            return char.get("name", "Player")
    
    # Fallback to user display name
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if user:
        return user.get("display_name", "Player")
    
    return "Player"


class AgentResult:
    """Result from running the GM agent."""
    def __init__(self, response_text: str, created_character_ids: list[str], final_state: dict | None = None):
        self.response_text = response_text
        self.created_character_ids = created_character_ids
        self.final_state = final_state


# Removed: extract_created_character_ids - PCs are now created deterministically at world create/join time


async def run_agent(world_id: str, user_message: str, character_name: str, needs_character_creation: bool, db) -> AgentResult:
    """Run the GM agent and return the response.
    
    The agent graph now handles context loading deterministically:
    - World state (characters, quests, encounters)
    - All events since last chronicle
    - Recent message history
    
    After the GM agent responds, a Scribe agent records events and chronicles.
    The GM's final response is captured in state.gm_final_response before the
    scribe runs, ensuring we persist the correct message.
    
    Returns:
        AgentResult with response_text and list of created player character IDs
    """
    global _gm_agent
    
    if not _gm_agent:
        raise HTTPException(status_code=503, detail="GM agent not available")
    
    if not _gm_agent.is_initialized:
        await _gm_agent.initialize(db)
    
    # Get recent message pairs for context (10 pairs = 20 messages max)
    # The agent graph also loads world state and events automatically
    recent_messages = await get_recent_messages(db, world_id, limit=20)
    
    # Format the new message with character context
    formatted_message = f"**Submitted by {character_name}**\n\n{user_message}"
    
    # Run agent graph (load_context -> gm_agent -> capture_response -> scribe -> cleanup)
    # The gm_final_response is captured before scribe runs
    response_text = ""
    final_state = None
    
    async for event in _gm_agent.stream_chat(
        message=formatted_message,
        world_id=world_id,
        history=recent_messages,
        needs_character_creation=needs_character_creation,
    ):
        # Keep track of the final state
        final_state = event
    
    # Get the GM's response from the captured state field
    if final_state:
        response_text = final_state.get("gm_final_response", "")
        
        # Fallback: if gm_final_response wasn't captured, try to find it in messages
        if not response_text:
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        content = msg.content
                        # Skip scribe confirmations
                        if content and not content.lower().startswith("events recorded"):
                            response_text = content
                            break
    
    return AgentResult(response_text=response_text, created_character_ids=[], final_state=final_state)


# ============================================================================
# Background Processing
# ============================================================================

async def process_message_background(
    world_id: str,
    user_id: str,
    user_message_content: str,
    character_name: str,
    character_id: str | None,
    user_message_id: str | None = None,
):
    """
    Background task to process a message and run the GM agent.
    
    This runs asynchronously after the user message is saved,
    allowing the API to return immediately. Progress is communicated
    via SSE broadcasts.
    """
    lock = get_world_lock(world_id)
    db = await get_db()
    broadcaster = get_broadcaster()
    
    async with lock:
        _world_processing[world_id] = True
        _world_lock_holder[world_id] = {
            "user_id": user_id,
            "character_name": character_name,
        }
        
        try:
            # Broadcast processing started
            await broadcaster.broadcast(world_id, {
                "type": "processing_started",
                "locked_by": user_id,
                "character_name": character_name,
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            # Check if user needs character creation
            needs_character_creation = False
            if character_id:
                char_doc = await db.characters.find_one({"_id": ObjectId(character_id)})
                if char_doc and char_doc.get("creation_in_progress", False):
                    needs_character_creation = True
                    logger.info(f"User {user_id} character {character_id} has creation_in_progress=True")
            else:
                needs_character_creation = True
                logger.warning(f"User {user_id} has no character_id in world {world_id}")
            
            # Run the agent
            try:
                agent_result = await run_agent(
                    world_id=world_id,
                    user_message=user_message_content,
                    character_name=character_name,
                    needs_character_creation=needs_character_creation,
                    db=db,
                )
                gm_response = agent_result.response_text
            except Exception as e:
                logger.exception(f"Agent error in background processing: {e}")
                gm_response = f"[The GM encounters a moment of confusion...] (Error: {str(e)[:100]})"
            
            # Save GM message
            gm_msg = Message(
                world_id=world_id,
                user_id=None,
                character_name="Game Master",
                content=gm_response,
                message_type="gm",
            )
            result = await db.messages.insert_one(gm_msg.to_doc())
            gm_msg.id = str(result.inserted_id)

            logger.info(f"Saved GM message {gm_msg.id} in world {world_id} (background)")

            # Persist agent trace and backlink both messages
            if user_message_id and agent_result.final_state:
                try:
                    state = agent_result.final_state
                    agent_message_keys = [
                        "historian_messages", "gm_messages", "bard_messages",
                        "accountant_messages", "scribe_messages",
                        "world_creator_messages", "char_creator_messages",
                    ]
                    agent_messages_serialized: dict = {}
                    for key in agent_message_keys:
                        msgs = state.get(key)
                        if msgs:
                            agent_name = key.replace("_messages", "")
                            agent_messages_serialized[agent_name] = serialize_messages(msgs)

                    if state.get("creation_in_progress"):
                        route = "world_creator"
                    elif state.get("needs_character_creation"):
                        route = "char_creator"
                    else:
                        route = "gm"

                    trace = Trace(
                        world_id=world_id,
                        user_message_id=user_message_id,
                        gm_message_id=gm_msg.id,
                        route=route,
                        gm_final_response=gm_response,
                        agent_messages=agent_messages_serialized,
                    )
                    trace_result = await db.traces.insert_one(trace.to_doc())
                    trace_id = str(trace_result.inserted_id)

                    await db.messages.update_one(
                        {"_id": ObjectId(user_message_id)},
                        {"$set": {"trace_id": trace_id}},
                    )
                    await db.messages.update_one(
                        {"_id": ObjectId(gm_msg.id)},
                        {"$set": {"trace_id": trace_id}},
                    )
                    logger.info(f"Saved trace {trace_id} for world {world_id}")
                except Exception as trace_err:
                    logger.exception(f"Failed to persist trace: {trace_err}")

            # Broadcast processing complete - UI will refresh via SSE
            now = datetime.utcnow().isoformat()
            await broadcaster.broadcast(world_id, {
                "type": "processing_complete",
                "messages_updated_at": now,
                "events_updated_at": now,
            })
            
        except Exception as e:
            logger.exception(f"Error in background message processing: {e}")
            # Broadcast error event so UI can show error state
            await broadcaster.broadcast(world_id, {
                "type": "processing_error",
                "error": str(e)[:200],
                "timestamp": datetime.utcnow().isoformat(),
            })
        finally:
            _world_processing[world_id] = False
            _world_lock_holder.pop(world_id, None)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/{world_id}/messages", response_model=SendMessageResponse, status_code=202)
async def send_message(
    world_id: str,
    request: SendMessageRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    Send a message to the GM agent (non-blocking).
    
    This endpoint:
    1. Checks if world is already processing (returns 409 if busy)
    2. Saves the user message to DB
    3. Queues background processing of the GM response
    4. Returns immediately with 202 Accepted
    
    The GM response is processed asynchronously. Clients receive
    updates via the SSE /worlds/{world_id}/updates/stream endpoint:
    - processing_started: GM is thinking
    - processing_complete: New messages available (triggers UI refresh)
    - processing_error: Something went wrong
    
    This non-blocking design provides instant feedback while ensuring
    sequential message ordering via the per-world lock.
    """
    db = await get_db()
    user_id = current_user.id
    
    # Check world access
    access = await db.world_access.find_one({
        "user_id": user_id,
        "world_id": world_id,
    })
    if not access:
        raise HTTPException(status_code=403, detail="No access to this world")
    
    # Check if already processing (non-blocking check)
    if is_world_processing(world_id):
        raise HTTPException(
            status_code=409,
            detail="World is currently processing a message. Please wait."
        )
    
    # Get character name for this user
    character_name = await get_user_character_name(db, user_id, world_id)
    
    # Save user message immediately
    user_msg = Message(
        world_id=world_id,
        user_id=user_id,
        character_name=character_name,
        content=request.content,
        message_type="player",
    )
    result = await db.messages.insert_one(user_msg.to_doc())
    user_msg.id = str(result.inserted_id)
    
    logger.info(f"Saved user message {user_msg.id} in world {world_id}, queuing background processing")
    
    # Queue background processing
    background_tasks.add_task(
        process_message_background,
        world_id=world_id,
        user_id=user_id,
        user_message_content=request.content,
        character_name=character_name,
        character_id=access.get("character_id"),
        user_message_id=user_msg.id,
    )
    
    # Return immediately with user message and processing status
    return SendMessageResponse(
        user_message=MessageResponse(
            id=user_msg.id,
            world_id=user_msg.world_id,
            user_id=user_msg.user_id,
            character_name=user_msg.character_name,
            display_name=current_user.display_name,
            content=user_msg.content,
            message_type=user_msg.message_type,
            created_at=user_msg.created_at.isoformat(),
        ),
        gm_message=None,  # Will be available after background processing
        status="processing",
    )


@router.get("/{world_id}/messages/stream")
async def stream_messages(
    world_id: str,
    request: Request,
    token: str | None = None,  # Accept token as query param for EventSource
    current_user: User | None = Depends(get_current_user_optional),
):
    """
    SSE stream for messages - now a simple keep-alive ping stream.
    
    NOTE: This endpoint no longer uses MongoDB Change Streams (which require 
    replica sets). Instead, the UI uses the /updates/stream endpoint to receive
    processing_started/processing_complete events and refreshes data accordingly.
    
    This endpoint is kept for backwards compatibility and connection keep-alive.
    
    Events:
    - status: Processing status (sent on connect)
    - ping: Keep-alive (every 30s)
    """
    from .auth import decode_token
    
    db = await get_db()
    
    # Handle auth - either from header (current_user) or query param (token)
    user_id = None
    if current_user:
        user_id = current_user.id
    elif token:
        try:
            payload = decode_token(token)
            user_id = payload.get("sub")
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Check world access
    access = await db.world_access.find_one({
        "user_id": user_id,
        "world_id": world_id,
    })
    if not access:
        raise HTTPException(status_code=403, detail="No access to this world")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events - simple ping stream."""
        # Send initial processing status
        yield f"data: {json.dumps({'type': 'status', 'processing': is_world_processing(world_id)})}\n\n"
        
        ping_interval = 30
        
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break
            
            # Wait and send periodic ping
            await asyncio.sleep(ping_interval)
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/{world_id}/messages", response_model=MessageHistoryResponse)
async def get_messages(
    world_id: str,
    limit: int = 50,
    before: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """
    Get message history for a world.
    
    Args:
        world_id: World ID
        limit: Max messages to return (default 50)
        before: Get messages before this message ID (for pagination)
    """
    db = await get_db()
    user_id = current_user.id
    
    # Check world access
    access = await db.world_access.find_one({
        "user_id": user_id,
        "world_id": world_id,
    })
    if not access:
        raise HTTPException(status_code=403, detail="No access to this world")
    
    # Build query
    query = {"world_id": world_id}
    if before:
        try:
            before_doc = await db.messages.find_one({"_id": ObjectId(before)})
            if before_doc:
                query["created_at"] = {"$lt": before_doc["created_at"]}
        except Exception:
            pass
    
    # Get total count
    total = await db.messages.count_documents({"world_id": world_id})
    
    # Get messages
    cursor = db.messages.find(query).sort("created_at", -1).limit(limit + 1)
    
    # Collect messages and user IDs
    raw_messages = []
    user_ids = set()
    async for doc in cursor:
        msg = Message.from_doc(doc)
        raw_messages.append(msg)
        if msg.user_id:
            user_ids.add(msg.user_id)
    
    # Fetch user display names
    user_display_names: dict[str, str] = {}
    if user_ids:
        async for user_doc in db.users.find({"_id": {"$in": [ObjectId(uid) for uid in user_ids]}}):
            user_display_names[str(user_doc["_id"])] = user_doc.get("display_name", "")
    
    # Build response
    messages = []
    for msg in raw_messages:
        messages.append(MessageResponse(
            id=msg.id,
            world_id=msg.world_id,
            user_id=msg.user_id,
            character_name=msg.character_name,
            display_name=user_display_names.get(msg.user_id) if msg.user_id else None,
            content=msg.content,
            message_type=msg.message_type,
            created_at=msg.created_at.isoformat(),
        ))
    
    # Check if there are more
    has_more = len(messages) > limit
    if has_more:
        messages = messages[:limit]
    
    # Reverse to chronological order
    messages.reverse()
    
    return MessageHistoryResponse(
        messages=messages,
        total=total,
        has_more=has_more,
    )


@router.get("/{world_id}/messages/status")
async def get_processing_status(
    world_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get the current processing status for a world."""
    db = await get_db()
    user_id = current_user.id
    
    # Check world access
    access = await db.world_access.find_one({
        "user_id": user_id,
        "world_id": world_id,
    })
    if not access:
        raise HTTPException(status_code=403, detail="No access to this world")
    
    return {
        "world_id": world_id,
        "processing": is_world_processing(world_id),
    }
