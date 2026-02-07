"""
WorldBroadcaster: In-memory pub/sub for real-time world updates.

This provides a lightweight notification system for multi-user synchronization.
Each world has a set of subscriber queues that receive broadcast events.
"""

import asyncio
import logging
from typing import Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class WorldBroadcaster:
    """
    Manages per-world pub/sub for SSE notifications.
    
    Usage:
        # Subscribe a client
        queue = await broadcaster.subscribe(world_id)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            await broadcaster.unsubscribe(world_id, queue)
        
        # Broadcast to all clients in a world
        await broadcaster.broadcast(world_id, {"type": "processing_started", ...})
    """
    
    def __init__(self):
        # Dict[world_id, Set[asyncio.Queue]]
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def subscribe(self, world_id: str) -> asyncio.Queue:
        """
        Subscribe to updates for a world.
        Returns a queue that will receive broadcast events.
        """
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers[world_id].add(queue)
            subscriber_count = len(self._subscribers[world_id])
        
        logger.info(f"[Broadcast] New subscriber for world {world_id}, total: {subscriber_count}")
        return queue
    
    async def unsubscribe(self, world_id: str, queue: asyncio.Queue) -> None:
        """
        Unsubscribe from world updates.
        """
        async with self._lock:
            self._subscribers[world_id].discard(queue)
            subscriber_count = len(self._subscribers[world_id])
            
            # Clean up empty world entries
            if not self._subscribers[world_id]:
                del self._subscribers[world_id]
        
        logger.info(f"[Broadcast] Subscriber left world {world_id}, remaining: {subscriber_count}")
    
    async def broadcast(self, world_id: str, event: dict[str, Any]) -> int:
        """
        Broadcast an event to all subscribers of a world.
        Returns the number of subscribers that received the event.
        """
        async with self._lock:
            subscribers = list(self._subscribers.get(world_id, set()))
        
        if not subscribers:
            logger.debug(f"[Broadcast] No subscribers for world {world_id}")
            return 0
        
        # Send to all subscribers (non-blocking)
        delivered = 0
        for queue in subscribers:
            try:
                queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning(f"[Broadcast] Queue full for world {world_id}, dropping event")
        
        logger.info(f"[Broadcast] Sent event '{event.get('type')}' to {delivered} subscribers in world {world_id}")
        return delivered
    
    def get_subscriber_count(self, world_id: str) -> int:
        """Get the number of subscribers for a world."""
        return len(self._subscribers.get(world_id, set()))
    
    def get_all_world_ids(self) -> list[str]:
        """Get all world IDs with active subscribers."""
        return list(self._subscribers.keys())


# Global singleton instance
_broadcaster: WorldBroadcaster | None = None


def get_broadcaster() -> WorldBroadcaster:
    """Get the global broadcaster instance."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = WorldBroadcaster()
        logger.info("[Broadcast] Initialized WorldBroadcaster")
    return _broadcaster
