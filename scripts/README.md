# Utility scripts

Run from the **rpg-agents** project root, or inside the Docker container (see below).

## delete_after_continue.py

Deletes messages and events from a world starting at and including a given message. You must identify the cutoff message in the database and pass its 24-char ObjectId.

- **Dry run (list only):** `python scripts/delete_after_continue.py <message_id>`
- **Prompt before delete:** `python scripts/delete_after_continue.py <message_id> --confirm`
- **Delete without prompt:** `python scripts/delete_after_continue.py <message_id> -y`

Uses `MONGO_URL` and `MONGO_DB` from the environment (defaults: `mongodb://localhost:27017`, `rpg_mcp`).

### Running in Docker

From the **repository root** (where `docker-compose.yml` defines `agents` and `mongo`):

```bash
# Dry run (replace MESSAGE_ID with the actual 24-char hex id)
docker compose run --rm agents python scripts/delete_after_continue.py MESSAGE_ID

# Delete after confirmation
docker compose run --rm agents python scripts/delete_after_continue.py MESSAGE_ID --confirm

# Delete without prompt
docker compose run --rm agents python scripts/delete_after_continue.py MESSAGE_ID -y
```

The `agents` container has `MONGO_URL=mongodb://mongo:27017` and `MONGO_DB=rpg_mcp` set, so it connects to the stack’s MongoDB.
