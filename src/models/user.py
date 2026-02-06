"""User model."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from bson import ObjectId


class User(BaseModel):
    """A registered user."""
    
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
    
    id: Optional[str] = Field(default=None, alias="_id")
    email: str
    password_hash: str
    display_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_doc(self) -> dict:
        """Convert to MongoDB document."""
        doc = self.model_dump(by_alias=True, exclude_none=True)
        if doc.get("_id"):
            doc["_id"] = ObjectId(doc["_id"])
        else:
            doc.pop("_id", None)
        return doc
    
    @classmethod
    def from_doc(cls, doc: dict) -> "User":
        """Create from MongoDB document."""
        if doc.get("_id"):
            doc["_id"] = str(doc["_id"])
        return cls(**doc)
    
    def to_public(self) -> dict:
        """Return public-safe user data (no password hash)."""
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "created_at": self.created_at.isoformat(),
        }
