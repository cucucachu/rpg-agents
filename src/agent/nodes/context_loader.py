"""Context loading node — deterministic DB fetches that bootstrap every turn."""

import json
import logging
from typing import Any

from ..state import GMAgentState

logger = logging.getLogger(__name__)


async def load_world_context(db, world_id: str) -> dict:
    """Load world state from database."""
    from bson import ObjectId

    result = {
        "world": None,
        "player_characters": [],  # PCs controlled by players - NEVER act or speak for these
        "active_quests": [],
        "active_encounter": None,
    }

    world_doc = await db.worlds.find_one({"_id": ObjectId(world_id)})
    if world_doc:
        result["world"] = {
            "id": str(world_doc["_id"]),
            "name": world_doc.get("name"),
            "description": world_doc.get("description"),
            "settings": world_doc.get("settings", {}),
            "creation_in_progress": world_doc.get("creation_in_progress", False),
        }

    pc_cursor = db.characters.find({"world_id": world_id, "is_player_character": True})
    async for doc in pc_cursor:
        result["player_characters"].append({
            "id": str(doc["_id"]),
            "name": doc.get("name"),
            "description": doc.get("description"),
            "level": doc.get("level", 1),
            "location_id": doc.get("location_id"),
            "attributes": doc.get("attributes", []),
            "statuses": doc.get("statuses", []),
            "abilities": [a.get("name") for a in doc.get("abilities", [])],
        })

    quest_cursor = db.quests.find({"world_id": world_id, "status": "active"})
    async for doc in quest_cursor:
        result["active_quests"].append({
            "id": str(doc["_id"]),
            "name": doc.get("name"),
            "description": doc.get("description"),
            "progress": doc.get("progress"),
        })

    encounter_doc = await db.encounters.find_one({"world_id": world_id, "status": "active"})
    if encounter_doc:
        result["active_encounter"] = {
            "id": str(encounter_doc["_id"]),
            "name": encounter_doc.get("name"),
            "round": encounter_doc.get("round", 1),
            "turn_order": encounter_doc.get("turn_order", []),
            "current_turn": encounter_doc.get("current_turn", 0),
        }

    return result


async def load_events_since_chronicle(db, world_id: str) -> tuple[list[dict], str | None, int]:
    """Load all events since the last chronicle (or all events if no chronicle).

    Returns:
        tuple: (events_list, last_chronicle_id, max_game_time)
        - events_list: Events since the last chronicle for context
        - last_chronicle_id: ID of the most recent chronicle (or None)
        - max_game_time: The highest game_time seen in the world (for Scribe continuity)
    """
    last_chronicle = await db.chronicles.find_one(
        {"world_id": world_id},
        sort=[("_id", -1)]
    )

    last_chronicle_id = str(last_chronicle["_id"]) if last_chronicle else None
    chronicle_game_time_end = last_chronicle.get("game_time_end", 0) if last_chronicle else 0

    query = {"world_id": world_id}
    if last_chronicle:
        query["_id"] = {"$gt": last_chronicle["_id"]}

    events = []
    event_cursor = db.events.find(query).sort("_id", 1)
    async for doc in event_cursor:
        events.append({
            "id": str(doc["_id"]),
            "name": doc.get("name"),
            "description": doc.get("description"),
            "participants": doc.get("participants"),
            "changes": doc.get("changes"),
            "tags": doc.get("tags", []),
            "game_time": doc.get("game_time", 0),
        })

    # Get the MAX game_time across ALL events in this world so Scribe always
    # builds on top of the highest time seen, even after a chronicle is created.
    max_game_time_doc = await db.events.find_one(
        {"world_id": world_id},
        sort=[("game_time", -1)]
    )
    max_event_game_time = max_game_time_doc.get("game_time", 0) if max_game_time_doc else 0
    max_game_time = max(chronicle_game_time_end, max_event_game_time)

    return events, last_chronicle_id, max_game_time


async def load_message_pairs(db, world_id: str, pairs: int = 10) -> list[dict]:
    """Load the last N message pairs (player + GM = 2 messages per pair)."""
    messages = []
    cursor = db.messages.find({"world_id": world_id}).sort("_id", -1).limit(pairs * 2)

    async for doc in cursor:
        messages.append({
            "role": "player" if doc.get("message_type") == "player" else "gm",
            "content": doc.get("content"),
            "character_name": doc.get("character_name"),
        })

    messages.reverse()
    return messages


async def create_load_context_node(db):
    """Create the load_context node with database access."""

    async def load_context_node(state: GMAgentState) -> dict[str, Any]:
        """Deterministically load all context needed for the GM agent."""
        world_id = state.get("world_id")

        if not world_id:
            logger.warning("No world_id in state, skipping context load")
            return {}

        logger.info(f"Loading context for world {world_id}")
        logger.debug(f"[load_context] world_id={world_id}")

        world_context = await load_world_context(db, world_id)
        logger.debug(
            f"[load_context] world_context: {len(world_context.get('player_characters', []))} PCs, "
            f"{len(world_context.get('active_quests', []))} quests, "
            f"encounter={bool(world_context.get('active_encounter'))}"
        )

        events, last_chronicle_id, max_game_time = await load_events_since_chronicle(db, world_id)

        first_event_id = events[0]["id"] if events else ""
        last_event_id = events[-1]["id"] if events else ""
        current_game_time = max_game_time

        logger.info(
            f"Loaded context: {len(events)} events since chronicle {last_chronicle_id}, "
            f"max_game_time={current_game_time}"
        )
        logger.debug(f"[load_context] first_event_id={first_event_id}, last_event_id={last_event_id}")

        creation_in_progress = world_context.get("world", {}).get("creation_in_progress", False)

        return {
            "world_context": json.dumps(world_context, indent=2),
            "events_context": json.dumps(events, indent=2),
            "last_chronicle_id": last_chronicle_id,
            "first_event_id": first_event_id,
            "last_event_id": last_event_id,
            "current_game_time": current_game_time,
            "creation_in_progress": creation_in_progress,
        }

    return load_context_node
