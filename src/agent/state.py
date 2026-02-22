"""Shared state schema for the GM agent pipeline."""

import operator
from typing import Annotated
from langchain_core.messages import BaseMessage
from typing import TypedDict


class GMAgentState(TypedDict, total=False):
    """State for the GM agent pipeline.

    Uses TypedDict instead of Pydantic BaseModel to ensure proper
    message handling by LangGraph. The 'messages' field uses the
    add_messages reducer to properly accumulate messages.
    """
    # Core identifiers
    world_id: str

    # Messages (conversation history, read-only) - uses operator.add to accumulate messages
    messages: Annotated[list[BaseMessage], operator.add]

    # Per-agent isolated message lists for ReAct loops and context isolation
    historian_messages: Annotated[list[BaseMessage], operator.add]
    barrister_messages: Annotated[list[BaseMessage], operator.add]
    gm_messages: Annotated[list[BaseMessage], operator.add]
    bard_messages: Annotated[list[BaseMessage], operator.add]
    accountant_messages: Annotated[list[BaseMessage], operator.add]
    scribe_messages: Annotated[list[BaseMessage], operator.add]
    world_creator_messages: Annotated[list[BaseMessage], operator.add]
    char_creator_messages: Annotated[list[BaseMessage], operator.add]

    # How many messages in the initial state were history (before this turn)
    # Used by scribe to identify which messages are "this turn"
    history_message_count: int

    # Loaded context (populated by load_context node)
    world_context: str   # JSON string of world data
    events_context: str  # JSON string of events since last chronicle
    enriched_context: str  # JSON string from Historian (additional findings)
    mechanics_context: str  # mechanics brief from Barrister, injected into GM as [MECHANICS BRIEF]
    last_chronicle_id: str  # ID of most recent chronicle
    first_event_id: str  # First event ID since last chronicle (for linking)
    last_event_id: str   # Last event ID since last chronicle (for linking)
    current_game_time: int  # Game time in seconds from last event (for scribe to track time)

    # GM's final response (captured before scribe runs)
    # This is what gets persisted to the database
    gm_final_response: str

    # GM's tool calls (captured for Accountant to avoid duplicates)
    # Contains list of tool call dicts with 'name' and 'args'
    gm_tool_calls: list[dict]

    # World creation phase: when True, route to world_creator agent
    creation_in_progress: bool

    # Character creation phase: when True, route to char_creator agent
    needs_character_creation: bool
