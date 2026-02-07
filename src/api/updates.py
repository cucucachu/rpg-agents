"""
World updates SSE endpoint for multi-user synchronization.

This endpoint broadcasts notifications when:
- Processing starts (world locked by a user)
- Processing completes (new messages/events available)

Unlike other SSE endpoints that use MongoDB Change Streams,
this uses an in-memory pub/sub for lightweight notifications.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..db import get_db
from .auth import get_current_user_optional, User, decode_token
from .broadcast import get_broadcaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/worlds", tags=["updates"])


@router.get("/{world_id}/updates/stream")
async def stream_updates(
    world_id: str,
    request: Request,
    token: str | None = None,  # Accept token as query param for EventSource
    current_user: User | None = Depends(get_current_user_optional),
):
    """
    SSE stream of world updates for multi-user synchronization.
    
    Uses in-memory pub/sub for lightweight notifications.
    
    Note: Since EventSource doesn't support custom headers, 
    token can be passed as query parameter.
    
    Events:
    - connected: Initial connection confirmation
    - processing_started: Another user started interacting with GM
      {type: "processing_started", locked_by: user_id, character_name: str, timestamp: ISO8601}
    - processing_complete: GM finished responding, new data available
      {type: "processing_complete", messages_updated_at: ISO8601, events_updated_at: ISO8601}
    - ping: Keep-alive (every 30s)
    """
    db = await get_db()
    broadcaster = get_broadcaster()
    
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
        """Generate SSE events for world updates."""
        queue = None
        try:
            # Subscribe to updates for this world
            queue = await broadcaster.subscribe(world_id)
            
            # Send connected event
            connected_event = {
                "type": "connected",
                "world_id": world_id,
                "subscriber_count": broadcaster.get_subscriber_count(world_id),
            }
            yield f"data: {json.dumps(connected_event)}\n\n"
            
            ping_interval = 30
            last_ping = asyncio.get_event_loop().time()
            
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info(f"[Updates] Client disconnected from world {world_id}")
                    break
                
                # Try to get next event from queue with timeout
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=5.0
                    )
                    
                    # Send the broadcast event
                    yield f"data: {json.dumps(event)}\n\n"
                    
                except asyncio.TimeoutError:
                    pass
                
                # Send periodic ping
                now = asyncio.get_event_loop().time()
                if now - last_ping > ping_interval:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    last_ping = now
        
        except Exception as e:
            logger.error(f"[Updates] SSE stream error for world {world_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        finally:
            # Clean up subscription
            if queue:
                await broadcaster.unsubscribe(world_id, queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
