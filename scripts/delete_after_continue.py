"""
Delete messages and events from a world starting at and including a given message.

Identify the cutoff message in the database (e.g. by world and content), then pass
its 24-char ObjectId. The script lists what would be deleted; only deletes after
confirmation.

Usage (from rpg-agents project root):
  Dry run (list only, no delete):
    python scripts/delete_after_continue.py <message_id>

  Delete after confirming:
    python scripts/delete_after_continue.py <message_id> --confirm
    python scripts/delete_after_continue.py <message_id> -y

  In Docker (from repo root with docker-compose):
    docker compose run --rm agents python scripts/delete_after_continue.py <message_id> -y

  With env:
    MONGO_URL=mongodb://localhost:27017 MONGO_DB=rpg_mcp python scripts/delete_after_continue.py <message_id> --confirm
"""

import os
import sys

from pymongo import MongoClient
from bson import ObjectId


def main():
    args = [a for a in sys.argv[1:] if a not in ("--confirm", "-y")]
    if not args:
        print("Usage: delete_after_continue.py <message_id> [--confirm | -y]", file=sys.stderr)
        print("  message_id: 24-char hex ObjectId of the message to start deleting from (inclusive)", file=sys.stderr)
        return 1

    message_id = args[0].strip()
    if len(message_id) != 24 or not all(c in "0123456789abcdef" for c in message_id.lower()):
        print("Invalid message_id: must be a 24-character hex string.", file=sys.stderr)
        return 1

    do_delete = "--confirm" in sys.argv or "-y" in sys.argv

    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "rpg_mcp")

    print(f"Connecting to {mongo_url} / {db_name} ...")
    client = MongoClient(mongo_url)
    db = client[db_name]

    # 1. Find the cutoff message by ObjectId
    try:
        cutoff_msg = db.messages.find_one({"_id": ObjectId(message_id)})
    except Exception as e:
        print(f"Invalid message_id: {e}", file=sys.stderr)
        return 1

    if not cutoff_msg:
        print(f"Message not found: {message_id}", file=sys.stderr)
        return 1

    world_id = cutoff_msg.get("world_id")
    if not world_id:
        print("Message has no world_id.", file=sys.stderr)
        return 1

    continue_oid = cutoff_msg["_id"]
    cutoff_ts = continue_oid.generation_time
    print(f"Found message: id={continue_oid}  world_id={world_id}  oid_ts={cutoff_ts}")
    print(f"  type={cutoff_msg.get('message_type')}  char={cutoff_msg.get('character_name')}")
    content = (cutoff_msg.get("content") or "")[:80]
    if len(cutoff_msg.get("content") or "") > 80:
        content += "..."
    print(f"  content: {content}\n")

    # 2. Identify messages to delete (ObjectId timestamp >= cutoff message's)
    messages_to_delete = []
    for doc in db.messages.find({"world_id": world_id}).sort("created_at", 1):
        if doc["_id"].generation_time >= cutoff_ts:
            messages_to_delete.append(doc)

    print("--- MESSAGES TO DELETE (ObjectId timestamp >= cutoff message _id) ---")
    if not messages_to_delete:
        print("  (none)")
    else:
        for m in messages_to_delete:
            oid_ts = m["_id"].generation_time
            content_preview = (m.get("content") or "")[:60].replace("\n", " ")
            if len(m.get("content") or "") > 60:
                content_preview += "..."
            print(f"  id={m['_id']}  oid_ts={oid_ts}  type={m.get('message_type')}  char={m.get('character_name')}")
            print(f"    content: {content_preview}")
    print(f"  Total: {len(messages_to_delete)} message(s)\n")

    # 3. Identify events to delete (ObjectId timestamp >= cutoff)
    events_to_delete = []
    for doc in db.events.find({"world_id": world_id}):
        if doc["_id"].generation_time >= cutoff_ts:
            events_to_delete.append(doc)

    events_to_delete.sort(key=lambda d: d["_id"].generation_time)

    print("--- EVENTS TO DELETE (ObjectId timestamp >= cutoff message _id) ---")
    if not events_to_delete:
        print("  (none)")
    else:
        for e in events_to_delete:
            oid_ts = e["_id"].generation_time
            desc_preview = (e.get("description") or "")[:50].replace("\n", " ")
            if len(e.get("description") or "") > 50:
                desc_preview += "..."
            print(f"  id={e['_id']}  oid_ts={oid_ts}  game_time={e.get('game_time')}  name={e.get('name')}")
            print(f"    description: {desc_preview}")
    print(f"  Total: {len(events_to_delete)} event(s)\n")

    # 4. Confirm before deleting
    if not do_delete:
        print("No deletion performed. Run with --confirm (prompt) or -y (delete without prompt) to delete the above.")
        return 0

    if "-y" not in sys.argv:
        confirm = input("Proceed with deletion? (y/n): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return 0

    # 5. Delete messages
    for doc in messages_to_delete:
        db.messages.delete_one({"_id": doc["_id"]})
    print(f"Deleted {len(messages_to_delete)} message(s).")

    # 6. Delete events
    for doc in events_to_delete:
        db.events.delete_one({"_id": doc["_id"]})
    print(f"Deleted {len(events_to_delete)} event(s).")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
