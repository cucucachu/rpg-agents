"""Shared logging tool node factory for all sub-agents."""

import logging
import time
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from ..state import GMAgentState

logger = logging.getLogger(__name__)


def create_logging_tool_node(tools, db, node_name: str = "tools", messages_key: str = "messages"):
    """Create a tool node with logging that reads/writes to a specific message stream.

    Args:
        tools: List of tools available to this node
        db: Database connection for activity logging
        node_name: Display name for logging
        messages_key: State key for the message list (e.g. "messages", "bard_messages")
    """
    base_tool_node = ToolNode(tools)

    async def logging_tool_node(state: GMAgentState) -> dict[str, Any]:
        """Wrapper that logs tool execution timing."""
        messages = state.get(messages_key, [])
        last_message = messages[-1] if messages else None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls
            tool_names = [tc["name"] for tc in tool_calls]

            start_time = time.time()
            logger.info(f"[{node_name}] Executing {len(tool_calls)} tool(s): {tool_names}")
            logger.debug(f"[{node_name}] tool_calls args (truncated): {[(tc.get('name'), str(tc.get('args', {}))[:80]) for tc in tool_calls]}")

            # Execute via the base ToolNode (always uses "messages" key)
            result = await base_tool_node.ainvoke({"messages": messages})

            elapsed = time.time() - start_time
            logger.info(f"[{node_name}] Tool execution completed in {elapsed:.2f}s")

            # Log individual results
            if "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage):
                        content_preview = str(msg.content)[:50] if len(str(msg.content)) > 50 else str(msg.content)
                        logger.info(f"  [{node_name}] Tool result [{msg.name}]: {content_preview}")

            # Re-map "messages" -> messages_key for correct state field
            if "messages" in result:
                return {messages_key: result["messages"]}
            return result

        # No tool calls - invoke base node
        result = await base_tool_node.ainvoke({"messages": messages})
        if "messages" in result:
            return {messages_key: result["messages"]}
        return result

    return logging_tool_node
