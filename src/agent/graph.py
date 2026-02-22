"""Graph assembly — builds and wires the full GM agent LangGraph pipeline.

Architecture:
1. load_context       - Deterministic: fetch world state, events, messages
2. historian          - Enrich context with read-only search (lore, chars, locations, events)
3. compile_historian  - Capture historian output as enriched_context
4. barrister          - Produce a mechanics brief (no tools)
5. compile_barrister  - Capture barrister output as mechanics_context
6. gm_agent           - Main GM (narration, dice, combat)
7. capture_response   - Capture GM final response + tool calls

After capture_response, bard / accountant / scribe run in PARALLEL:
  - bard       - Record new entities (NPCs, locations, lore, factions)
  - accountant - Sync state changes (damage, healing, status, items)
  - scribe     - Record events and chronicles

Creation paths (bypass normal gameplay):
  - world_creator  - Conversational world-building
  - char_creator   - Conversational character creation
"""

import logging
import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from .llm import get_llm
from .mcp_tools import get_mcp_tools
from .state import GMAgentState

from .nodes.context_loader import create_load_context_node
from .nodes.historian import (
    HISTORIAN_TOOL_NAMES,
    compile_historian_context_node,
    create_historian_agent_node,
    historian_init_node,
    should_continue_historian,
)
from .nodes.barrister import (
    barrister_init_node,
    compile_barrister_context_node,
    create_barrister_agent_node,
)
from .nodes.gm import (
    GM_TOOL_NAMES,
    capture_gm_response_node,
    create_gm_agent_node,
    gm_init_node,
    should_continue_gm,
)
from .nodes.bard import (
    BARD_TOOL_NAMES,
    bard_init_node,
    create_bard_agent_node,
    should_continue_bard,
)
from .nodes.accountant import (
    ACCOUNTANT_TOOL_NAMES,
    accountant_init_node,
    create_accountant_agent_node,
    should_continue_accountant,
)
from .nodes.scribe import (
    SCRIBE_TOOL_NAMES,
    create_scribe_agent_node,
    scribe_init_node,
    should_continue_scribe,
)
from .nodes.world_creator import (
    WORLD_CREATOR_TOOL_NAMES,
    create_world_creator_agent_node,
    should_continue_world_creator,
    world_creator_init_node,
)
from .nodes.char_creator import (
    CHAR_CREATOR_TOOL_NAMES,
    char_creator_init_node,
    create_char_creator_agent_node,
    persist_response_node,
    should_continue_char_creator,
)
from .nodes.tool_node import create_logging_tool_node

# Debug mode — controls verbose state logging and debug-level logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

logger = logging.getLogger(__name__)

if DEBUG:
    logger.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    if root_logger.level == logging.NOTSET or root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)


# ============================================================================
# Entry Routing
# ============================================================================

def route_entry(state: GMAgentState) -> Literal["world_creator", "char_creator", "historian"]:
    """Route from load_context to the appropriate agent based on creation flags."""
    if state.get("creation_in_progress", False):
        logger.info("[route_entry] creation_in_progress=True -> world_creator")
        return "world_creator"
    if state.get("needs_character_creation", False):
        logger.info("[route_entry] needs_character_creation=True -> char_creator")
        return "char_creator"
    logger.debug("[route_entry] -> historian (normal gameplay)")
    return "historian"


# ============================================================================
# Graph Factory
# ============================================================================

async def create_gm_agent(
    db,
    mcp_url: str = "http://localhost:8080",
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.7,
) -> tuple[StateGraph, dict]:
    """Create the GM agent graph with separated concerns.

    Flow (normal gameplay):
      load_context
        -> historian <-> historian_tools -> compile_historian
        -> barrister -> compile_barrister
        -> gm_agent <-> gm_tools -> capture_response
        -> [parallel] bard <-> bard_tools -> END
                      accountant -> accountant_tools -> END
                      scribe -> scribe_tools -> END

    Creation paths:
      load_context -> world_creator_init -> world_creator <-> world_creator_tools -> persist_response -> END
      load_context -> char_creator_init  -> char_creator  <-> char_creator_tools  -> persist_response -> END

    Args:
        db: MongoDB database connection
        mcp_url: URL of the rpg-mcp server
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (uses provider default if not specified)
        temperature: LLM temperature

    Returns:
        Tuple of (compiled_graph, config_dict)
    """
    gm_llm = get_llm(provider, model, temperature)
    scribe_llm = get_llm(provider, model, temperature=0.3)

    all_tools = await get_mcp_tools(mcp_url)

    # Slice the full tool list into per-agent subsets
    historian_tools = [t for t in all_tools if t.name in HISTORIAN_TOOL_NAMES]
    bard_tools = [t for t in all_tools if t.name in BARD_TOOL_NAMES]
    world_creator_tools = [t for t in all_tools if t.name in WORLD_CREATOR_TOOL_NAMES]
    char_creator_tools = [t for t in all_tools if t.name in CHAR_CREATOR_TOOL_NAMES]
    scribe_tools = [t for t in all_tools if t.name in SCRIBE_TOOL_NAMES]
    accountant_tools = [t for t in all_tools if t.name in ACCOUNTANT_TOOL_NAMES]
    gm_tools = [t for t in all_tools if t.name in GM_TOOL_NAMES]

    logger.info(
        f"Tool separation: {len(historian_tools)} Historian, {len(bard_tools)} Bard, "
        f"{len(gm_tools)} GM, {len(accountant_tools)} Accountant, {len(scribe_tools)} Scribe, "
        f"{len(world_creator_tools)} WorldCreator, {len(char_creator_tools)} CharCreator"
    )

    # Per-agent LLMs (lower temperature for deterministic post-processing agents)
    historian_llm = get_llm(provider, model, temperature=0.3)
    barrister_llm = get_llm(provider, model, temperature=0.3)
    bard_llm = get_llm(provider, model, temperature=0.3)
    accountant_llm = get_llm(provider, model, temperature=0.3)
    world_creator_llm = get_llm(provider, model, temperature=0.7)
    char_creator_llm = get_llm(provider, model, temperature=0.7)

    # Bind tools to LLMs
    historian_llm_with_tools = historian_llm.bind_tools(historian_tools)
    bard_llm_with_tools = bard_llm.bind_tools(bard_tools)
    gm_llm_with_tools = gm_llm.bind_tools(gm_tools)
    accountant_llm_with_tools = accountant_llm.bind_tools(accountant_tools)
    scribe_llm_with_tools = scribe_llm.bind_tools(scribe_tools)
    world_creator_llm_with_tools = world_creator_llm.bind_tools(world_creator_tools)
    char_creator_llm_with_tools = char_creator_llm.bind_tools(char_creator_tools)

    # Instantiate node callables
    load_context_node = await create_load_context_node(db)
    historian_agent_node = create_historian_agent_node(historian_llm_with_tools, db)
    historian_tools_node = create_logging_tool_node(historian_tools, db, "Historian", "historian_messages")
    barrister_agent_node = create_barrister_agent_node(barrister_llm)
    gm_agent_node = create_gm_agent_node(gm_llm_with_tools, db)
    gm_tools_node = create_logging_tool_node(gm_tools, db, "GM", "gm_messages")
    accountant_agent_node = create_accountant_agent_node(accountant_llm_with_tools, db)
    accountant_tools_node = create_logging_tool_node(accountant_tools, db, "Accountant", "accountant_messages")
    bard_agent_node = create_bard_agent_node(bard_llm_with_tools, db)
    bard_tools_node = create_logging_tool_node(bard_tools, db, "Bard", "bard_messages")
    scribe_agent_node = create_scribe_agent_node(scribe_llm_with_tools, db)
    scribe_tools_node = create_logging_tool_node(scribe_tools, db, "Scribe", "scribe_messages")
    world_creator_agent_node = create_world_creator_agent_node(world_creator_llm_with_tools, db)
    world_creator_tools_node = create_logging_tool_node(world_creator_tools, db, "WorldCreator", "world_creator_messages")
    char_creator_agent_node = create_char_creator_agent_node(char_creator_llm_with_tools, db)
    char_creator_tools_node = create_logging_tool_node(char_creator_tools, db, "CharCreator", "char_creator_messages")

    # -------------------------------------------------------------------------
    # Build the graph
    # -------------------------------------------------------------------------
    workflow = StateGraph(GMAgentState)

    # Register nodes
    workflow.add_node("load_context", load_context_node)

    workflow.add_node("historian_init", historian_init_node)
    workflow.add_node("historian", historian_agent_node)
    workflow.add_node("historian_tools", historian_tools_node)
    workflow.add_node("compile_historian", compile_historian_context_node)

    workflow.add_node("barrister_init", barrister_init_node)
    workflow.add_node("barrister", barrister_agent_node)
    workflow.add_node("compile_barrister", compile_barrister_context_node)

    workflow.add_node("gm_init", gm_init_node)
    workflow.add_node("gm_agent", gm_agent_node)
    workflow.add_node("gm_tools", gm_tools_node)
    workflow.add_node("capture_response", capture_gm_response_node)

    # Parallel post-processing nodes
    workflow.add_node("bard_init", bard_init_node)
    workflow.add_node("bard", bard_agent_node)
    workflow.add_node("bard_tools", bard_tools_node)

    workflow.add_node("accountant_init", accountant_init_node)
    workflow.add_node("accountant", accountant_agent_node)
    workflow.add_node("accountant_tools", accountant_tools_node)

    workflow.add_node("scribe_init", scribe_init_node)
    workflow.add_node("scribe", scribe_agent_node)
    workflow.add_node("scribe_tools", scribe_tools_node)

    # Creation path nodes
    workflow.add_node("world_creator_init", world_creator_init_node)
    workflow.add_node("world_creator", world_creator_agent_node)
    workflow.add_node("world_creator_tools", world_creator_tools_node)

    workflow.add_node("char_creator_init", char_creator_init_node)
    workflow.add_node("char_creator", char_creator_agent_node)
    workflow.add_node("char_creator_tools", char_creator_tools_node)

    workflow.add_node("persist_response", persist_response_node)

    # -------------------------------------------------------------------------
    # Wire edges
    # -------------------------------------------------------------------------
    workflow.set_entry_point("load_context")

    # Entry routing: creation paths vs normal gameplay
    workflow.add_conditional_edges(
        "load_context",
        route_entry,
        {
            "world_creator": "world_creator_init",
            "char_creator": "char_creator_init",
            "historian": "historian_init",
        }
    )

    # World creator path
    workflow.add_edge("world_creator_init", "world_creator")
    workflow.add_conditional_edges(
        "world_creator",
        should_continue_world_creator,
        {
            "world_creator_tools": "world_creator_tools",
            "persist_response": "persist_response",
        }
    )
    workflow.add_edge("world_creator_tools", "world_creator")

    # Char creator path
    workflow.add_edge("char_creator_init", "char_creator")
    workflow.add_conditional_edges(
        "char_creator",
        should_continue_char_creator,
        {
            "char_creator_tools": "char_creator_tools",
            "persist_response": "persist_response",
        }
    )
    workflow.add_edge("char_creator_tools", "char_creator")

    # Both creation paths end here
    workflow.add_edge("persist_response", END)

    # Normal gameplay: historian -> barrister -> GM
    workflow.add_edge("historian_init", "historian")
    workflow.add_conditional_edges(
        "historian",
        should_continue_historian,
        {
            "historian_tools": "historian_tools",
            "compile_historian": "compile_historian",
        }
    )
    workflow.add_edge("historian_tools", "historian")
    workflow.add_edge("compile_historian", "barrister_init")
    workflow.add_edge("barrister_init", "barrister")
    workflow.add_edge("barrister", "compile_barrister")
    workflow.add_edge("compile_barrister", "gm_init")

    # GM loop
    workflow.add_edge("gm_init", "gm_agent")
    workflow.add_conditional_edges(
        "gm_agent",
        should_continue_gm,
        {
            "gm_tools": "gm_tools",
            "capture_response": "capture_response",
        }
    )
    workflow.add_edge("gm_tools", "gm_agent")

    # Fan-out: capture_response -> bard, accountant, scribe (all three in parallel)
    workflow.add_edge("capture_response", "bard_init")
    workflow.add_edge("capture_response", "accountant_init")
    workflow.add_edge("capture_response", "scribe_init")

    # Bard branch (ReAct loop allowed; bard searches before creating)
    workflow.add_edge("bard_init", "bard")
    workflow.add_conditional_edges(
        "bard",
        should_continue_bard,
        {
            "bard_tools": "bard_tools",
            END: END,
        }
    )
    workflow.add_edge("bard_tools", "bard")

    # Accountant branch (single-batch; no loop back to prevent duplicate syncs)
    workflow.add_edge("accountant_init", "accountant")
    workflow.add_conditional_edges(
        "accountant",
        should_continue_accountant,
        {
            "accountant_tools": "accountant_tools",
            END: END,
        }
    )
    workflow.add_edge("accountant_tools", END)

    # Scribe branch (single-batch; no loop back to prevent duplicate events)
    workflow.add_edge("scribe_init", "scribe")
    workflow.add_conditional_edges(
        "scribe",
        should_continue_scribe,
        {
            "scribe_tools": "scribe_tools",
            END: END,
        }
    )
    workflow.add_edge("scribe_tools", END)

    graph = workflow.compile()
    logger.info(
        "Compiled GM agent graph: load_context -> historian -> barrister -> gm_agent "
        "-> [parallel] bard | accountant | scribe -> END"
    )

    config = {
        "provider": provider,
        "model": model or ("claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o"),
        "mcp_url": mcp_url,
        "gm_tools_count": len(gm_tools),
        "accountant_tools_count": len(accountant_tools),
        "scribe_tools_count": len(scribe_tools),
    }

    return graph, config


# ============================================================================
# High-Level Wrapper
# ============================================================================

class GMAgent:
    """High-level wrapper for the GM agent."""

    def __init__(
        self,
        mcp_url: str = "http://localhost:8080",
        provider: str = "anthropic",
        model: str | None = None,
    ):
        self.mcp_url = mcp_url
        self.provider = provider
        self.model = model
        self._graph = None
        self._config = None
        self._db = None

    async def initialize(self, db) -> None:
        """Initialize the agent with database connection.

        Args:
            db: MongoDB database connection for context loading
        """
        self._db = db
        self._graph, self._config = await create_gm_agent(
            db=db,
            mcp_url=self.mcp_url,
            provider=self.provider,
            model=self.model,
        )

    @property
    def is_initialized(self) -> bool:
        return self._graph is not None

    async def chat(
        self,
        message: str,
        thread_id: str | None = None,
        world_id: str | None = None,
    ) -> str:
        """Simple chat method for backwards compatibility.

        Note: This method works best with a world_id. Without it,
        context loading will be limited.

        Args:
            message: Player's message
            thread_id: Ignored (for backwards compatibility)
            world_id: World ID for context loading

        Returns:
            GM's response text
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize(db) first.")

        response_text = ""
        async for event in self.stream_chat(
            message=message,
            world_id=world_id or "",
            history=None,
        ):
            messages = event.get("messages", [])
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        content = msg.content
                        if content and not content.lower().startswith("events recorded"):
                            response_text = content

        return response_text

    async def stream_chat(
        self,
        message: str,
        world_id: str,
        history: list[dict] | None = None,
        needs_character_creation: bool = False,
    ):
        """Stream responses from the GM.

        Context is loaded deterministically:
        1. World state (characters, quests, encounters)
        2. All events since last chronicle
        3. Recent message history (passed in)

        Args:
            message: Current user message
            world_id: World ID for context loading
            history: Recent conversation history as list of dicts:
                     [{"role": "player"|"gm", "content": "...", "character_name": "..."}]
            needs_character_creation: If True, route to character creator agent

        Yields:
            Events as they occur (tool calls, responses, etc.)
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize(db) first.")

        messages = []

        if history:
            for msg in history:
                if msg["role"] == "player":
                    char_name = msg.get("character_name", "Player")
                    content = f"**Submitted by {char_name}**\n\n{msg['content']}"
                    messages.append(HumanMessage(content=content))
                elif msg["role"] == "gm":
                    messages.append(AIMessage(content=msg["content"]))

        history_message_count = len(messages)
        messages.append(HumanMessage(content=message))

        input_state: GMAgentState = {
            "world_id": world_id,
            "messages": messages,
            "history_message_count": history_message_count,
            "needs_character_creation": needs_character_creation,
        }

        logger.info(f"Running agent with {len(messages)} messages (history={len(history) if history else 0})")

        async for event in self._graph.astream(input_state, stream_mode="values"):
            yield event
