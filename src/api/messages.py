"""Message endpoints for chat functionality.

This module implements a clean separation between:
- UI Messages: What users see in chat (stored in MongoDB)
- Agent Context: Minimal context for LLM reasoning (built fresh each call)

Key endpoints:
- POST /api/worlds/{world_id}/messages - Send message (blocking, with lock)
- GET /api/worlds/{world_id}/messages/stream - SSE stream of new messages
- GET /api/worlds/{world_id}/messages - Get message history
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from bson import ObjectId

from ..db import get_db
from ..models import Message, User
from .auth import get_current_user, get_current_user_optional
from .broadcast import get_broadcaster
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/worlds", tags=["messages"])

# Static message for users without a character - referenced in GM prompt
NEW_PLAYER_MESSAGE = "NEW_PLAYER_JOINING: I am a new player joining this world. Please help me create a character."

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
    def __init__(self, response_text: str, created_character_ids: list[str]):
        self.response_text = response_text
        self.created_character_ids = created_character_ids


def extract_created_character_ids(messages: list) -> list[str]:
    """Extract character IDs from create_character tool results."""
    character_ids = []
    
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "create_character":
            # Tool result format: "Created character: {json}"
            content = msg.content
            if content and "Created character:" in content:
                try:
                    # Extract JSON part after "Created character: "
                    json_str = content.split("Created character: ", 1)[1]
                    char_data = json.loads(json_str)
                    if char_data.get("id") and char_data.get("is_player_character"):
                        character_ids.append(char_data["id"])
                        logger.info(f"Found created PC: {char_data['id']} ({char_data.get('name', 'unknown')})")
                except (json.JSONDecodeError, IndexError, KeyError) as e:
                    logger.warning(f"Failed to parse create_character result: {e}")
    
    return character_ids


async def run_agent(world_id: str, user_message: str, character_name: str, db) -> AgentResult:
    """Run the GM agent and return the response.
    
    The agent graph now handles context loading deterministically:
    - World state (characters, quests, encounters)
    - All events since last chronicle
    - Recent message history
    
    After the GM agent responds, a Scribe agent records events and chronicles.
    The GM's final response is captured in state.gm_final_response before the
    scribe runs, ensuring we persist the correct message.
    
    Agent activity is logged to the agent_activity collection and can be
    streamed via the /activity/stream endpoint.
    
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
    ):
        # Keep track of the final state
        final_state = event
    
    # Extract created character IDs from tool messages
    created_character_ids = []
    
    # Get the GM's response from the captured state field
    if final_state:
        response_text = final_state.get("gm_final_response", "")
        
        # Extract any created player characters from tool results
        messages = final_state.get("messages", [])
        created_character_ids = extract_created_character_ids(messages)
        
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
    
    return AgentResult(response_text=response_text, created_character_ids=created_character_ids)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/{world_id}/messages", response_model=SendMessageResponse)
async def send_message(
    world_id: str,
    request: SendMessageRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Send a message to the GM agent.
    
    This endpoint:
    1. Acquires a per-world lock (returns 409 if busy)
    2. Saves the user message to DB
    3. Runs the agent synchronously (blocks until complete)
    4. Saves the GM response to DB
    5. Returns both messages
    
    The blocking design ensures sequential message ordering
    and prevents race conditions in multi-user scenarios.
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
    
    # Try to acquire the world lock
    lock = get_world_lock(world_id)
    
    # Check if already processing (non-blocking check)
    if is_world_processing(world_id):
        raise HTTPException(
            status_code=409,
            detail="World is currently processing a message. Please wait."
        )
    
    # Try to acquire lock with timeout
    try:
        acquired = lock.locked()
        if acquired:
            raise HTTPException(
                status_code=409,
                detail="World is currently processing a message. Please wait."
            )
        
        async with lock:
            _world_processing[world_id] = True
            broadcaster = get_broadcaster()
            
            try:
                # Check if user has a character assigned
                has_character = bool(access.get("character_id"))
                
                # Get character name for this user
                character_name = await get_user_character_name(db, user_id, world_id)
                
                # Determine the message to send to the agent
                # If user has no character, substitute with the new player message
                if has_character:
                    agent_message = request.content
                else:
                    agent_message = NEW_PLAYER_MESSAGE
                    logger.info(f"User {user_id} has no character in world {world_id}, triggering character creation flow")
                
                # Track who holds the lock and broadcast processing started
                _world_lock_holder[world_id] = {
                    "user_id": user_id,
                    "character_name": character_name,
                }
                await broadcaster.broadcast(world_id, {
                    "type": "processing_started",
                    "locked_by": user_id,
                    "character_name": character_name,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
                # Save user message (save the ORIGINAL content, not the substituted message)
                user_msg = Message(
                    world_id=world_id,
                    user_id=user_id,
                    character_name=character_name,
                    content=request.content,
                    message_type="player",
                )
                result = await db.messages.insert_one(user_msg.to_doc())
                user_msg.id = str(result.inserted_id)
                
                logger.info(f"Saved user message {user_msg.id} in world {world_id}")
                
                # Run the agent (with substituted message if no character)
                try:
                    agent_result = await run_agent(
                        world_id=world_id,
                        user_message=agent_message,
                        character_name=character_name,
                        db=db,
                    )
                    gm_response = agent_result.response_text
                    
                    # If user had no character and one was created, link them
                    if not has_character and agent_result.created_character_ids:
                        # Link the first created PC to this user
                        new_char_id = agent_result.created_character_ids[0]
                        await db.world_access.update_one(
                            {"user_id": user_id, "world_id": world_id},
                            {"$set": {"character_id": new_char_id}}
                        )
                        logger.info(f"Linked user {user_id} to character {new_char_id} in world {world_id}")
                        
                        # Update character_name for the response (it was "Player" or display_name before)
                        char_doc = await db.characters.find_one({"_id": ObjectId(new_char_id)})
                        if char_doc:
                            character_name = char_doc.get("name", character_name)
                            # Update the user message we already saved with the correct character name
                            await db.messages.update_one(
                                {"_id": ObjectId(user_msg.id)},
                                {"$set": {"character_name": character_name}}
                            )
                            user_msg.character_name = character_name
                            logger.info(f"Updated message {user_msg.id} with character name: {character_name}")
                            
                except Exception as e:
                    logger.error(f"Agent error: {e}")
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
                
                logger.info(f"Saved GM message {gm_msg.id} in world {world_id}")
                
                # Broadcast processing complete with timestamps for refresh
                now = datetime.utcnow().isoformat()
                await broadcaster.broadcast(world_id, {
                    "type": "processing_complete",
                    "messages_updated_at": now,
                    "events_updated_at": now,
                })
                
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
                    gm_message=MessageResponse(
                        id=gm_msg.id,
                        world_id=gm_msg.world_id,
                        user_id=gm_msg.user_id,
                        character_name=gm_msg.character_name,
                        display_name=None,
                        content=gm_msg.content,
                        message_type=gm_msg.message_type,
                        created_at=gm_msg.created_at.isoformat(),
                    ),
                )
            finally:
                _world_processing[world_id] = False
                _world_lock_holder.pop(world_id, None)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
