"""WorldAccess model - links users to worlds and their characters.

Roles:
- god: Admin/creator of the world. Can manage world settings, create invites, etc.
- mortal: Player in the world. Has a character and can interact with the game.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field
from bson import ObjectId


class WorldAccess(BaseModel):
    """Links a user to a world with a specific role.
    
    Roles:
        god: Admin/creator. Can manage world, create invites, see everything.
        mortal: Player. Has a character and plays the game.
    """
    
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
    
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    world_id: str
    character_id: Optional[str] = None  # The PC this user plays (for mortals)
    role: Literal["god", "mortal"] = "mortal"
    invited_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_doc(self) -> dict:
        """Convert to MongoDB document."""
        doc = self.model_dump(by_alias=True, exclude_none=True)
        if doc.get("_id"):
            doc["_id"] = ObjectId(doc["_id"])
        else:
            doc.pop("_id", None)
        return doc
    
    @classmethod
    def from_doc(cls, doc: dict) -> "WorldAccess":
        """Create from MongoDB document."""
        if doc.get("_id"):
            doc["_id"] = str(doc["_id"])
        return cls(**doc)
