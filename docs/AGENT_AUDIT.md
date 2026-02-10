# Agent System Audit – The Shattered Realms

**Audit date:** Based on current DB state after recent play.  
**World ID:** `6984c087af881c7f64c1a783`

---

## 1. Database snapshot

| Collection       | Count (this world) | Notes                    |
|------------------|--------------------|--------------------------|
| **events**       | 133                | Scribe writes here       |
| **chronicles**   | 17                 | Scribe writes here       |
| **lore**         | 5                  | Bard can write here      |
| **characters**   | 12                 | 2 PC, 10 NPC             |
| **locations**    | 5                  | Bard/GM can create       |
| **items**        | 7                  | Accountant can create     |
| **agent_activity** | 0 (all worlds)   | Cleared each turn (see §4) |

**World document:** `game_time` has been removed; game time is now derived from events. The pipeline uses **max `game_time` across events** for Scribe continuity.

---

## 2. Scribe – events and chronicles

**Verdict: Working well.**

- **Events:** 133 total. Newest events match recent play (e.g. “Morning Arrives – Party Recovers Enough to Move” at game_time 86,460, “Kess Warns of Intelligent Threat”, “Thorne Senses the Distant Presence”, dialogue with Kess about Rifts).
- **Chronicles:** 17 total. Latest: **“The Rift’s Intelligence Revealed”** (game_time_start 43,152 → game_time_end 86,460), summarizing the post-battle conversation with Kess, Thorne’s sensing the presence to the north, rest, and morning departure.
- **Chronicle detail:** The new chronicle correctly links 6 events and has a clear summary (revelation about intelligent Rifts, rest, then preparation to move north).

**Conclusion:** Scribe is creating new events and new chronicles and linking them appropriately.

---

## 3. Bard – lore and characters

**Verdict: Partially working; no new lore from recent play.**

- **Lore:** Still **5** entries (all from the manual “flesh out the world” pass: The Cataclysm, Ancient Roads, The Rift and Its Creatures, Ashfall’s Losses, The Old Archive). No new lore entries created by the Bard during recent play.
- **Characters:** 12 total. The four NPCs we added (Kess, Garrett, Marta, Scarred Merchant Woman) are present. No new named NPCs appeared in the recent session, so the Bard had nothing new to create on the character side.
- **Gap:** Recent narrative contained strong **lore-worthy content** (e.g. Kess present during the Cataclysm 30 years ago; scholar of old magics; Rifts as deliberate doorways; Rifts increasing in frequency; self-sealing Rift being unprecedented). That is good material for `set_lore`, but no new lore documents were added. So either:
  - The Bard is not treating “revelations in dialogue” as lore to record, or
  - It is not receiving/using that part of the narrative, or
  - It is instructed to avoid duplicating and is conservatively skipping.

**Recommendation:** Review Bard prompt and context (e.g. whether it sees full GM narrative and whether “significant lore reveals (history, legends)” is explicitly tied to dialogue/exposition). Consider adding an example of creating a lore entry from a character’s revelation.

---

## 4. Accountant – character state (e.g. HP)

**Verdict: Partially working; HP updated but incomplete.**

- **Thorne Ashwood:** HP **2**, AC 11. No `max` on HP in attributes. Events say he was stabilized (was at -5), then rested; “recovered enough to move” fits low HP.
- **Lyra Nightwhisper:** HP **2**, max **8**, AC 13. Same pattern: stabilized, rested, “wounded and weak” but able to move.

So the Accountant is updating HP to reflect “wounded but mobile” after stabilization and rest. Possible issues:

- **Thorne:** No `max` for HP. If the narrative or GM never stated “max HP” explicitly, the Accountant may not set it; worth confirming whether `set_attributes` is called with `max` when available.
- **Consistency:** Both at 2 HP is plausible; if the narrative implied different healing (e.g. one character got more care), that might not have been synced.

**Recommendation:** Ensure Accountant prompt or logic calls `set_attributes` with both `value` and `max` for HP when the narrative or world state implies a known max (e.g. from class/level). Optionally add a rule: “If you set HP, set max_hp when known.”

---

## 5. Agent logs (agent_activity)

**Verdict: Not auditable after the fact.**

- **Behavior:** Each agent node (load_context, historian, gm_agent, capture_response, bard, accountant, scribe, tools) writes to `agent_activity` during the run. The **cleanup** node then runs and **deletes all `agent_activity` for that world**.
- **Result:** `agent_activity` is empty after every turn. There is no persistent record of “what did the Scribe/Bard/Accountant do this turn?” in the DB.

**Ways to audit agent behavior:**

1. **Live:** Watch the activity SSE stream (`GET /worlds/{world_id}/activity/stream`) while a message is being processed.
2. **Persistent log (future):** Add a separate collection (e.g. `agent_turn_log`) that stores a summary per turn (node names, tool names, counts) and is **not** cleared by cleanup, or append to a log document instead of delete.

**Recommendation:** If you want to audit “what did the Bard/Scribe/Accountant do?” after play, add a lightweight persistent log (e.g. one document per turn with node → tool names / result summary) written before cleanup runs, and leave cleanup only clearing the high-volume `agent_activity` stream.

---

## 6. Summary table

| Agent      | Intended role              | Status   | Notes                                                                 |
|-----------|----------------------------|----------|-----------------------------------------------------------------------|
| **Scribe**   | Events + chronicles        | ✅ Good  | New events and chronicles; latest chronicle matches recent story.    |
| **Bard**     | Lore + new NPCs/locations  | ⚠️ Partial | NPCs/locations from earlier pass present; no new lore from recent play. |
| **Accountant** | HP, status, items, etc.  | ⚠️ Partial | HP updated (e.g. 2 for both PCs); Thorne missing HP max.              |
| **Agent logs** | Visibility into agent actions | ❌ None | agent_activity cleared each turn; no persistent audit trail.         |

---

## 7. Optional next steps

1. **Bard:** Tighten prompt/examples so “significant lore reveals” in dialogue (e.g. Kess’s backstory, Rift nature) are explicitly turned into `set_lore` when they’re new.
2. **Accountant:** Always set `max` for HP when setting HP if the value is known (e.g. from world state or narrative).
3. **Observability:** Introduce a small persistent “turn summary” (e.g. in `agent_turn_log`) so you can later see which agents ran and which tools they called, without keeping full `agent_activity` forever.
4. **World game_time:** ✅ Done. Optionally have Scribe (or a post-Scribe step) update `world.game_time` to the latest event’s `game_time` so the world document stays in sync with events for API/UI.
