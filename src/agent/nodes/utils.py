"""Shared utilities for agent nodes."""

from typing import Any


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
