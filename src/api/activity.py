"""Agent activity streaming API."""

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..db import get_db
from ..models import AgentActivity
from .auth import get_current_user_optional, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/worlds", tags=["activity"])


@router.get("/{world_id}/activity/stream")
async def stream_activity(
    world_id: str,
    request: Request,
    token: str | None = None,  # Accept token as query param for EventSource
    current_user: User | None = Depends(get_current_user_optional),
):
    """
    SSE stream of agent activity for a world.
    
    Uses MongoDB Change Streams to watch for new activity records.
    Activity records are created by agent nodes during processing.
    
    Note: Since EventSource doesn't support custom headers, 
    token can be passed as query parameter.
    
    Events:
    - activity: New agent activity record
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
        """Generate SSE events for agent activity."""
        try:
            # Watch for changes to the agent_activity collection
            pipeline = [
                {"$match": {
                    "operationType": "insert",
                    "fullDocument.world_id": world_id,
                }}
            ]
            
            async with db.agent_activity.watch(pipeline, full_document="updateLookup") as stream:
                ping_interval = 30
                last_ping = asyncio.get_event_loop().time()
                
                while True:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break
                    
                    # Try to get next change with timeout
                    try:
                        change = await asyncio.wait_for(
                            stream.try_next(),
                            timeout=5.0
                        )
                        
                        if change:
                            doc = change.get("fullDocument")
                            if doc:
                                activity = AgentActivity.from_doc(doc)
                                event_data = {
                                    "type": "activity",
                                    "activity": activity.to_public(),
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                    
                    except asyncio.TimeoutError:
                        pass
                    
                    # Send periodic ping
                    now = asyncio.get_event_loop().time()
                    if now - last_ping > ping_interval:
                        yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                        last_ping = now
        
        except Exception as e:
            logger.error(f"Activity SSE stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
