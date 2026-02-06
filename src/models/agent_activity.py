"""Agent activity model for streaming internal agent state."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class AgentActivity(BaseModel):
    """A single agent activity record for streaming to the UI."""
    
    id: str | None = None
    world_id: str
    node_name: str  # Which graph node generated this (load_context, gm_agent, scribe, etc.)
    activity_type: str  # "thinking", "tool_call", "tool_result", "response", "status"
    content: str  # The activity text/preview
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "AgentActivity":
        """Create from MongoDB document."""
        return cls(
            id=str(doc["_id"]),
            world_id=doc["world_id"],
            node_name=doc["node_name"],
            activity_type=doc["activity_type"],
            content=doc["content"],
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
        )
    
    def to_doc(self) -> dict[str, Any]:
        """Convert to MongoDB document (without _id)."""
        return {
            "world_id": self.world_id,
            "node_name": self.node_name,
            "activity_type": self.activity_type,
            "content": self.content,
            "created_at": self.created_at,
        }
    
    def to_public(self) -> dict[str, Any]:
        """Convert to public API response."""
        return {
            "id": self.id,
            "world_id": self.world_id,
            "node_name": self.node_name,
            "activity_type": self.activity_type,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }
