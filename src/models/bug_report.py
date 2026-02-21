"""Bug report model for user-submitted issues linked to agent traces."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from bson import ObjectId


class BugReport(BaseModel):
    """A user-submitted bug report linked to an agent trace."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: Optional[str] = Field(default=None, alias="_id")
    trace_id: str
    user_id: str
    description: str

    def to_doc(self) -> dict:
        """Convert to MongoDB document."""
        doc = self.model_dump(by_alias=True, exclude_none=True)
        if doc.get("_id"):
            doc["_id"] = ObjectId(doc["_id"])
        else:
            doc.pop("_id", None)
        doc["trace_id"] = ObjectId(doc["trace_id"])
        return doc

    @classmethod
    def from_doc(cls, doc: dict) -> "BugReport":
        """Create from MongoDB document."""
        doc = dict(doc)
        if doc.get("_id"):
            doc["_id"] = str(doc["_id"])
        if doc.get("trace_id"):
            doc["trace_id"] = str(doc["trace_id"])
        return cls(**doc)

    def to_public(self) -> dict:
        """Return public bug report data."""
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "description": self.description,
        }
