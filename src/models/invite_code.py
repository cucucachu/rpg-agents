"""InviteCode model for registration gating.

Invite codes are used to control who can create accounts.
They are NOT tied to any specific world.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from bson import ObjectId
import secrets


def generate_code() -> str:
    """Generate a random code."""
    return secrets.token_urlsafe(8)


class InviteCode(BaseModel):
    """An invite code for creating an account.
    
    This is purely for registration gating - not tied to any world.
    """
    
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
    
    id: Optional[str] = Field(default=None, alias="_id")
    code: str = Field(default_factory=generate_code)
    created_by: Optional[str] = None  # user_id or "system"
    max_uses: Optional[int] = None  # None = unlimited
    uses: int = 0
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def is_valid(self) -> bool:
        """Check if the invite code is still valid."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        if self.max_uses is not None and self.uses >= self.max_uses:
            return False
        return True
    
    def to_doc(self) -> dict:
        """Convert to MongoDB document."""
        doc = self.model_dump(by_alias=True, exclude_none=True)
        if doc.get("_id"):
            doc["_id"] = ObjectId(doc["_id"])
        else:
            doc.pop("_id", None)
        return doc
    
    @classmethod
    def from_doc(cls, doc: dict) -> "InviteCode":
        """Create from MongoDB document."""
        if doc.get("_id"):
            doc["_id"] = str(doc["_id"])
        return cls(**doc)
