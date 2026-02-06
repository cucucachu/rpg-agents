"""Events API - Fetches game events from the MCP database."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import asyncio
import json

from ..db import get_db
from ..models import User
from .auth import get_current_user

router = APIRouter(prefix="/worlds", tags=["events"])


class EventResponse(BaseModel):
    """A game event."""
    id: str
    world_id: str
    game_time: int
    location_id: Optional[str] = None
    name: str
    description: str = ""
    participants: str = ""
    changes: str = ""
    tags: list[str] = []
    created_at: Optional[datetime] = None


class EventsListResponse(BaseModel):
    """List of events."""
    events: list[EventResponse]
    total: int
    has_more: bool


@router.get("/{world_id}/events", response_model=EventsListResponse)
async def get_events(
    world_id: str,
    limit: int = Query(default=30, ge=1, le=200),
    before: Optional[str] = Query(default=None, description="Get events before this event ID (for pagination)"),
    current_user: User = Depends(get_current_user),
):
    """
    Get events for a world.
    
    Returns the most recent events first, then reverses for chronological display.
    Use 'before' parameter to load older events (pagination).
    """
    from bson import ObjectId
    
    db = await get_db()
    
    # Check world access
    access = await db.world_access.find_one({
        "user_id": current_user.id,
        "world_id": world_id
    })
    if not access:
        raise HTTPException(status_code=403, detail="No access to this world")
    
    # Get total count
    total = await db.events.count_documents({"world_id": world_id})
    
    # Build query - filter by before if provided
    query = {"world_id": world_id}
    if before:
        try:
            # Get events older than the 'before' event
            query["_id"] = {"$lt": ObjectId(before)}
        except Exception:
            pass
    
    # Get events sorted by _id descending (newest first), fetch limit+1 to check for more
    cursor = db.events.find(query).sort("_id", -1).limit(limit + 1)
    
    events = []
    async for doc in cursor:
        events.append(EventResponse(
            id=str(doc["_id"]),
            world_id=doc["world_id"],
            game_time=doc.get("game_time", 0),
            location_id=doc.get("location_id"),
            name=doc.get("name", ""),
            description=doc.get("description", ""),
            participants=doc.get("participants", ""),
            changes=doc.get("changes", ""),
            tags=doc.get("tags", []),
            created_at=doc.get("created_at"),
        ))
    
    # Check if there are more (older) events
    has_more = len(events) > limit
    if has_more:
        events = events[:limit]
    
    # Reverse to chronological order (oldest first for display)
    events.reverse()
    
    return EventsListResponse(events=events, total=total, has_more=has_more)


@router.get("/{world_id}/events/stream")
async def stream_events(
    world_id: str,
    token: str = Query(..., description="JWT token for authentication"),
):
    """
    SSE stream of events for a world.
    Uses MongoDB Change Streams to watch for new events.
    """
    from .auth import decode_token
    
    # Validate token
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    db = await get_db()
    
    # Check world access
    access = await db.world_access.find_one({
        "user_id": user_id,
        "world_id": world_id
    })
    if not access:
        raise HTTPException(status_code=403, detail="No access to this world")
    
    async def event_generator():
        """Generate SSE events."""
        # Send initial connection event
        yield f"event: connected\ndata: {json.dumps({'world_id': world_id})}\n\n"
        
        # Watch for changes
        pipeline = [
            {"$match": {
                "operationType": "insert",
                "fullDocument.world_id": world_id
            }}
        ]
        
        try:
            async with db.events.watch(pipeline) as stream:
                async for change in stream:
                    doc = change["fullDocument"]
                    event_data = {
                        "id": str(doc["_id"]),
                        "world_id": doc["world_id"],
                        "game_time": doc.get("game_time", 0),
                        "location_id": doc.get("location_id"),
                        "name": doc.get("name", ""),
                        "description": doc.get("description", ""),
                        "participants": doc.get("participants", ""),
                        "changes": doc.get("changes", ""),
                        "tags": doc.get("tags", []),
                    }
                    yield f"event: event\ndata: {json.dumps(event_data)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
