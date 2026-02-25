"""Shared utilities for agent nodes."""

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


def extract_entity_ids(data: Any) -> dict[str, str]:
    """Recursively extract name→id pairs from any JSON structure.

    Handles both list responses (find_* tools) and single-object responses.
    Accepts both 'id' and '_id' field names since MongoDB serialization isn't consistent.
    """
    result: dict[str, str] = {}
    if isinstance(data, dict):
        name = data.get("name")
        entity_id = data.get("id") or data.get("_id")
        if name and entity_id:
            result[str(name)] = str(entity_id)
        for value in data.values():
            result.update(extract_entity_ids(value))
    elif isinstance(data, list):
        for item in data:
            result.update(extract_entity_ids(item))
    return result


def extract_gm_tool_calls(messages: list) -> list[dict]:
    """Extract all tool call name+args made by the GM from gm_messages.

    Used by the accountant to build its 'TOOLS GM ALREADY CALLED' block.
    """
    tool_calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc.get("name"), "args": tc.get("args", {})})
    return tool_calls


def extract_gm_roll_results(messages: list) -> list[dict]:
    """Extract roll_dice call+result pairs from gm_messages.

    Used by the scribe to build its MECHANICS THIS TURN block.
    Each returned dict: {notation, reason, total, details}
    """
    tool_results: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tid = getattr(msg, "tool_call_id", None)
            if tid:
                tool_results[tid] = msg.content

    rolls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc.get("name") == "roll_dice":
                    raw = tool_results.get(tc.get("id", ""), "")
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = {}
                    rolls.append({
                        "notation": tc["args"].get("notation", ""),
                        "reason":   tc["args"].get("reason", ""),
                        "total":    parsed.get("total", "?"),
                        "details":  parsed.get("details", raw[:120]),
                    })
    return rolls
