"""FastAPI server for the GM agent."""

import os
import asyncio
import json
import logging
import traceback
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agent import create_gm_agent, GM_SYSTEM_PROMPT
from ..agent.graph import GMAgent
from ..db import connect_db, close_db
from . import auth, worlds, websocket as ws_module, messages as messages_module, events as events_module, updates as updates_module, bugs as bugs_module
from .cors_config import CORS_CONFIG, print_cors_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance
_agent: GMAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage agent and database lifecycle."""
    global _agent
    
    # Connect to database
    logger.info("Connecting to MongoDB...")
    db = await connect_db()
    logger.info("MongoDB connected")
    
    # Initialize agent on startup
    mcp_url = os.getenv("MCP_URL", "http://localhost:8080")
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL")
    
    logger.info(f"Initializing GM Agent with MCP_URL={mcp_url}, provider={provider}, model={model}")
    _agent = GMAgent(mcp_url=mcp_url, provider=provider, model=model)
    
    # Set agent references BEFORE initialization so retry logic works
    # if initialization fails (messages.py checks is_initialized and retries)
    ws_module.set_gm_agent(_agent)
    logger.info("GM Agent set for WebSocket handler")
    messages_module.set_gm_agent(_agent)
    logger.info("GM Agent set for messages module")
    
    try:
        # Pass database connection for context loading
        await _agent.initialize(db)
        logger.info(f"GM Agent initialized with {_agent._config['gm_tools_count']} GM tools, {_agent._config['scribe_tools_count']} scribe tools")
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Agent will retry on first request")
    
    yield
    
    # Cleanup
    _agent = None
    await close_db()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="RPG GM Agent",
        description="AI Game Master powered by LangGraph and MCP",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware with production-safe configuration
    logger.info("Configuring CORS middleware...")
    print_cors_config()
    
    app.add_middleware(
        CORSMiddleware,
        **CORS_CONFIG
    )
    
    # Mount routers
    app.include_router(auth.router)
    app.include_router(worlds.router)
    app.include_router(ws_module.router)  # Legacy WebSocket (will be deprecated)
    app.include_router(messages_module.router)  # New message endpoints
    app.include_router(events_module.router)  # Events endpoint
    app.include_router(updates_module.router)  # Multi-user update notifications
    app.include_router(bugs_module.router)  # Bug reports
    
    return app


app = create_app()


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="Player's message to the GM")
    thread_id: str = Field(default="default", description="Conversation thread ID")
    world_id: str | None = Field(default=None, description="World ID for context")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="GM's response")
    thread_id: str = Field(..., description="Thread ID used")


class StreamEvent(BaseModel):
    """Event in a streaming response."""
    type: str  # "token", "tool_call", "tool_result", "done"
    content: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    agent_initialized: bool
    tools_count: int | None


class AgentConfig(BaseModel):
    """Agent configuration."""
    provider: str
    model: str
    mcp_url: str
    tools_count: int


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    global _agent
    
    tools_count = None
    if _agent and _agent._config:
        gm_tools = _agent._config.get("gm_tools_count", 0)
        scribe_tools = _agent._config.get("scribe_tools_count", 0)
        tools_count = gm_tools + scribe_tools
    
    return HealthResponse(
        status="healthy",
        agent_initialized=_agent is not None and _agent.is_initialized,
        tools_count=tools_count,
    )


@app.get("/config", response_model=AgentConfig)
async def get_config():
    """Get current agent configuration."""
    global _agent
    
    if not _agent or not _agent._config:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    config = _agent._config
    return AgentConfig(
        provider=config.get("provider", "unknown"),
        model=config.get("model", "unknown"),
        mcp_url=config.get("mcp_url", "unknown"),
        tools_count=config.get("gm_tools_count", 0) + config.get("scribe_tools_count", 0),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the GM and get a response (legacy endpoint).
    
    This is a synchronous endpoint that waits for the full response.
    For streaming, use /chat/stream instead.
    
    Note: Requires world_id for full functionality with the new architecture.
    """
    global _agent
    
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        if not _agent.is_initialized:
            from ..db import get_db
            db = await get_db()
            await _agent.initialize(db)
        
        response = await _agent.chat(
            message=request.message,
            thread_id=request.thread_id,
            world_id=request.world_id,
        )
        
        return ChatResponse(
            response=response,
            thread_id=request.thread_id,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream responses from the GM (legacy endpoint).
    
    Returns a server-sent events stream with:
    - type: "tool_call" - when a tool is being called
    - type: "tool_result" - tool execution result
    - type: "token" - response text chunk
    - type: "done" - stream complete
    
    Note: Requires world_id for full functionality with the new architecture.
    """
    global _agent
    
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            if not _agent.is_initialized:
                from ..db import get_db
                db = await get_db()
                await _agent.initialize(db)
            
            async for event in _agent.stream_chat(
                message=request.message,
                world_id=request.world_id or "",
                history=None,
            ):
                messages = event.get("messages", [])
                for msg in messages:
                    # Check for tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield f"data: {json.dumps({'type': 'tool_call', 'tool_name': tc['name'], 'tool_args': tc.get('args', {})})}\n\n"
                    
                    # Check for tool results
                    if hasattr(msg, "type") and msg.type == "tool":
                        yield f"data: {json.dumps({'type': 'tool_result', 'tool_name': msg.name, 'content': str(msg.content)[:500]})}\n\n"
                    
                    # Check for AI response content (skip scribe confirmations)
                    if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                        content = msg.content
                        if content and not content.lower().startswith("events recorded"):
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
    )


@app.get("/prompt")
async def get_system_prompt():
    """Get the current system prompt."""
    return {"prompt": GM_SYSTEM_PROMPT}


# CLI entry point
def main():
    """Run the server."""
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8081"))
    
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
