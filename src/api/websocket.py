"""
DEPRECATED: WebSocket handler for real-time chat.

This module is deprecated in favor of the new SSE-based architecture.
Use the messages module endpoints instead:
- POST /api/worlds/{world_id}/messages - Send message (blocking)
- GET /api/worlds/{world_id}/messages/stream - SSE stream

This stub is kept for backward compatibility during transition.
"""

import warnings
from fastapi import APIRouter

warnings.warn(
    "websocket module is deprecated. Use messages module instead.",
    DeprecationWarning,
    stacklevel=2,
)

router = APIRouter()

# Stub for backward compatibility
_gm_agent = None


def set_gm_agent(agent):
    """Deprecated: Set the GM agent reference."""
    global _gm_agent
    _gm_agent = agent
    # No-op - this module no longer handles agent communication
