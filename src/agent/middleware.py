"""Middleware for the GM agent."""

from typing import Any, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


def create_summarization_middleware(
    max_messages: int = 20,
    summarize_threshold: int = 15,
    model: Any = None
) -> Callable:
    """
    Create middleware that summarizes conversation history when it gets too long.
    
    Args:
        max_messages: Maximum messages to keep in full
        summarize_threshold: When to trigger summarization
        model: LLM to use for summarization (optional, uses same model if not provided)
    """
    
    async def summarize_messages(messages: list[BaseMessage], llm: Any) -> str:
        """Summarize a list of messages into a concise summary."""
        # Build context for summarization
        conversation_text = []
        for msg in messages:
            role = "Player" if isinstance(msg, HumanMessage) else "GM"
            conversation_text.append(f"{role}: {msg.content[:500]}")  # Truncate long messages
        
        summary_prompt = f"""Summarize this RPG session conversation concisely. Focus on:
- Key events and decisions
- Current situation and location
- Important NPCs encountered
- Quest progress
- Any unresolved tensions or hooks

Conversation:
{chr(10).join(conversation_text)}

Summary (2-3 paragraphs):"""
        
        response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
        return response.content
    
    async def middleware(
        state: dict[str, Any],
        config: dict[str, Any],
        llm: Any
    ) -> dict[str, Any]:
        """Process state before model call, potentially summarizing history."""
        messages = state.get("messages", [])
        
        # Check if we need to summarize
        if len(messages) <= summarize_threshold:
            return state
        
        # Find system message if present
        system_msg = None
        other_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_msg = msg
            else:
                other_messages.append(msg)
        
        # Keep recent messages, summarize older ones
        recent_messages = other_messages[-max_messages:]
        older_messages = other_messages[:-max_messages]
        
        if not older_messages:
            return state
        
        # Summarize older messages
        summary_llm = model or llm
        summary = await summarize_messages(older_messages, summary_llm)
        
        # Create new message list with summary
        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        
        # Add summary as a system-like context message
        summary_msg = SystemMessage(content=f"[Previous session summary: {summary}]")
        new_messages.append(summary_msg)
        new_messages.extend(recent_messages)
        
        return {**state, "messages": new_messages}
    
    return middleware


class ConversationSummarizer:
    """Manages conversation summarization for long-running sessions."""
    
    def __init__(
        self,
        llm: Any,
        max_tokens_estimate: int = 8000,
        chars_per_token: int = 4
    ):
        self.llm = llm
        self.max_chars = max_tokens_estimate * chars_per_token
        self._summary: str | None = None
    
    def estimate_chars(self, messages: list[BaseMessage]) -> int:
        """Estimate character count of messages."""
        return sum(len(str(msg.content)) for msg in messages)
    
    async def maybe_summarize(
        self,
        messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], str | None]:
        """
        Check if messages need summarization and return processed list.
        
        Returns:
            Tuple of (processed_messages, summary_if_created)
        """
        total_chars = self.estimate_chars(messages)
        
        if total_chars <= self.max_chars:
            return messages, None
        
        # Find cutoff point - keep ~60% of max in recent messages
        target_recent = int(self.max_chars * 0.6)
        
        # Walk backwards to find cutoff
        recent_chars = 0
        cutoff_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg_chars = len(str(messages[i].content))
            if recent_chars + msg_chars > target_recent:
                cutoff_idx = i + 1
                break
            recent_chars += msg_chars
        
        if cutoff_idx <= 1:
            # Can't summarize meaningfully
            return messages, None
        
        # Separate messages
        older = messages[:cutoff_idx]
        recent = messages[cutoff_idx:]
        
        # Summarize older messages
        summary = await self._create_summary(older)
        self._summary = summary
        
        # Return recent messages (summary stored separately)
        return recent, summary
    
    async def _create_summary(self, messages: list[BaseMessage]) -> str:
        """Create a summary of the given messages."""
        # Extract key information
        events = []
        npcs = []
        locations = []
        
        conversation_text = []
        for msg in messages:
            content = str(msg.content)[:1000]  # Truncate
            role = "Player" if isinstance(msg, HumanMessage) else "GM"
            conversation_text.append(f"{role}: {content}")
        
        prompt = f"""Summarize this RPG session segment. Extract:
1. Key events and player decisions
2. NPCs encountered and their attitudes
3. Locations visited
4. Quest/plot progress
5. Current situation at end

Keep it factual and concise (3-4 paragraphs max).

Conversation:
{chr(10).join(conversation_text[-50:])}  # Last 50 exchanges max

Summary:"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
    
    def get_context_message(self) -> SystemMessage | None:
        """Get a system message with the current summary context."""
        if not self._summary:
            return None
        
        return SystemMessage(
            content=f"[PRIOR SESSION CONTEXT]\n{self._summary}\n[END PRIOR CONTEXT]"
        )
