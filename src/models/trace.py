"""Trace model for persisting agent pipeline state after each GM run."""

from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field
from bson import ObjectId
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage


def serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """Serialize a list of LangChain BaseMessages to plain dicts for MongoDB storage."""
    result = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            entry: dict[str, Any] = {
                "type": "ai",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            }
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "args": tc.get("args", {}),
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append({
                "type": "tool",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                "tool_call_id": getattr(msg, "tool_call_id", None),
                "name": getattr(msg, "name", None),
            })
        elif isinstance(msg, SystemMessage):
            result.append({
                "type": "system",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            })
        elif isinstance(msg, HumanMessage):
            result.append({
                "type": "human",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            })
        else:
            result.append({
                "type": type(msg).__name__.lower(),
                "content": str(getattr(msg, "content", "")),
            })
    return result


class Trace(BaseModel):
    """Persisted state from a single GM agent pipeline run."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: Optional[str] = Field(default=None, alias="_id")
    world_id: str
    user_message_id: str
    gm_message_id: str
    route: str  # "gm" | "world_creator" | "char_creator"
    gm_final_response: str
    agent_messages: dict[str, list[dict]]  # agent name -> serialized messages
    bug_report_ids: list[str] = Field(default_factory=list)

    def to_doc(self) -> dict:
        """Convert to MongoDB document."""
        doc = self.model_dump(by_alias=True, exclude_none=True)
        if doc.get("_id"):
            doc["_id"] = ObjectId(doc["_id"])
        else:
            doc.pop("_id", None)
        doc["user_message_id"] = ObjectId(doc["user_message_id"])
        doc["gm_message_id"] = ObjectId(doc["gm_message_id"])
        if doc.get("bug_report_ids"):
            doc["bug_report_ids"] = [ObjectId(bid) for bid in doc["bug_report_ids"]]
        return doc

    @classmethod
    def from_doc(cls, doc: dict) -> "Trace":
        """Create from MongoDB document."""
        doc = dict(doc)
        if doc.get("_id"):
            doc["_id"] = str(doc["_id"])
        if doc.get("user_message_id"):
            doc["user_message_id"] = str(doc["user_message_id"])
        if doc.get("gm_message_id"):
            doc["gm_message_id"] = str(doc["gm_message_id"])
        if doc.get("bug_report_ids"):
            doc["bug_report_ids"] = [str(bid) for bid in doc["bug_report_ids"]]
        return cls(**doc)

    def to_public(self) -> dict:
        """Return public trace data."""
        return {
            "id": self.id,
            "world_id": self.world_id,
            "user_message_id": self.user_message_id,
            "gm_message_id": self.gm_message_id,
            "route": self.route,
            "gm_final_response": self.gm_final_response,
            "agent_messages": self.agent_messages,
            "bug_report_ids": self.bug_report_ids,
        }
