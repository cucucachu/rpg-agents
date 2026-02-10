# Brand New World Flow

How a newly created world behaves from creation through first character and first play.

---

## 1. World creation (API)

**Endpoint:** `POST /worlds` with `{ "name": "My World" }`

- Creates a **world** document: `name`, `description=""`, `settings={}`, `created_by=user_id`. No `game_time` (derived from events).
- Creates **world_access** for the creator: `user_id`, `world_id`, `role="god"`, **no `character_id`** (creator is not yet playing a character).

So: new world is empty (no characters, no events, no chronicles, no locations/lore), and the creator has no PC yet.

---

## 2. First message in the world

When **any** user (including the creator) sends a message in that world:

1. **Lock:** World is locked so only one message is processed at a time.
2. **Character check:** `has_character = bool(access.get("character_id"))` → **false** for the creator (and any other user without a character).
3. **Message substitution:** If the user has no character, the content sent to the agent is **replaced** with:
   - `NEW_PLAYER_MESSAGE` = `"NEW_PLAYER_JOINING: I am a new player joining this world. Please help me create a character."`
   - The **original** user message is still stored in the DB for the chat UI.
4. **Character name:** From `get_user_character_name()` → no character, so the user’s `display_name` or `"Player"`.
5. **Agent run:** The pipeline runs with:
   - `load_context` → world (name, description, **settings={}**), **player_characters=[]**, no events, no chronicles.
   - Historian → little or nothing to add.
   - GM receives: empty world, no PCs, and the substituted message: `**Submitted by {display_name}**\n\nNEW_PLAYER_JOINING: ...`.

---

## 3. GM behavior (new / empty world)

From the GM prompt:

- **If `world.settings` is empty/missing (new world):**  
  “Describe the setting and ask about game system and tone; world setup is handled elsewhere.”
- **New player joining:**  
  Guide them through character creation (name, concept, class, stats from `world.settings`). When the GM has what they need, call **`create_player_character`** with `world_id`, `name`, `description`, and optionally `location_id`, `level`, `attributes`, `skills`, `abilities`, then narrate their arrival.

So in practice:

- **First turn (brand new world):** GM often has no game system in `world.settings`, so they may describe the world as new and ask what kind of game/setting the player wants, **without** calling `create_player_character` yet.
- **Later turn(s):** Once the GM has enough (e.g. name, concept, maybe a simple system), they call **`create_player_character`** with full stats. That creates the PC in the DB.

Note: The GM has **no** tool to set `world.settings` or `world.description`. “World setup is handled elsewhere” means outside this flow (e.g. future UI or manual DB). So for a brand new world, the GM can only narrate and ask questions until they’re ready to create the PC.

---

## 4. Post-processing: linking user ↔ character

After the agent run:

- **`extract_created_character_ids(messages)`** looks for:
  - **`create_player_character`** tool results: content `"Created player character: {json}"` → parse JSON, take `id`.
  - **`create_character`** tool results (legacy): content `"Created character: {json}"` and `is_player_character === true` → take `id`.
- If the user **had no character** and at least one such ID was found:
  - **Link user to PC:** `world_access.character_id = created_character_ids[0]`.
  - **Fix message and display:** Load the new character’s name, update the saved user message’s `character_name` to that name.

So: the **first** time the GM creates a PC in that run (via `create_player_character` or legacy `create_character` with PC flag), that PC is attached to the user who sent the message (the one who triggered the “new player” flow).

---

## 5. Subsequent messages (same user)

Once `world_access.character_id` is set:

- `has_character` is true.
- No more message substitution; the user’s real message goes to the agent.
- `character_name` comes from the linked character.
- Normal play: GM has context (world, that PC, any events/chronicles the Scribe has recorded).

---

## 6. Summary (brand new world)

| Step | What happens |
|------|----------------|
| 1. Create world | API creates world (empty) and world_access (god, no character_id). |
| 2. First message | User has no character → message replaced with NEW_PLAYER_JOINING; agent sees empty world + “new player” message. |
| 3. GM first reply | Often: describe new world, ask for game/setting (no `world.settings` to read). May or may not create a PC yet. |
| 4. If GM created a PC | Post-processing finds `create_player_character` result, sets world_access.character_id, updates message character_name. |
| 5. Next message (same user) | If linked: real message, normal play. If not linked: again NEW_PLAYER_JOINING until GM calls create_player_character. |
| 6. Later players | Same pattern: join world → no character_id → first message substituted → GM creates PC with create_player_character → post-processing links them. |

**Gap:** The GM cannot persist `world.settings` or world description; they can only narrate and create PCs. So “brand new world” flow is: GM guides in chat until they have enough to call `create_player_character`; then the pipeline links that PC to the user automatically.
