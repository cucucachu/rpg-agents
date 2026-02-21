"""Message model for chat history."""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field
from bson import ObjectId


class Message(BaseModel):
    """A chat message in a world."""
    
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
    
    id: Optional[str] = Field(default=None, alias="_id")
    world_id: str
    user_id: Optional[str] = None  # None for GM messages
    character_name: str  # Display name (e.g., "Kael" or "Game Master")
    content: str
    message_type: Literal["player", "gm", "system"] = "player"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None  # Links to the associated agent trace
    
    def to_doc(self) -> dict:
        """Convert to MongoDB document."""
        doc = self.model_dump(by_alias=True, exclude_none=True)
        if doc.get("_id"):
            doc["_id"] = ObjectId(doc["_id"])
        else:
            doc.pop("_id", None)
        return doc
    
    @classmethod
    def from_doc(cls, doc: dict) -> "Message":
        """Create from MongoDB document."""
        if doc.get("_id"):
            doc["_id"] = str(doc["_id"])
        return cls(**doc)
    
    def to_public(self) -> dict:
        """Return public message data."""
        return {
            "id": self.id,
            "world_id": self.world_id,
            "user_id": self.user_id,
            "character_name": self.character_name,
            "content": self.content,
            "message_type": self.message_type,
            "created_at": self.created_at.isoformat(),
        }
