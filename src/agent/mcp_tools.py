"""MCP client for connecting to rpg-mcp server and exposing tools to LangChain."""

import json
import asyncio
import logging
from typing import Any
import httpx
from httpx_sse import aconnect_sse
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for connecting to an MCP server via SSE transport."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self._tools_cache: dict[str, dict] = {}
        self._message_endpoint: str | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._pending_responses: dict[int, asyncio.Future] = {}
        self._message_id = 0
        self._sse_task: asyncio.Task | None = None
    
    async def connect(self) -> None:
        """Connect to the MCP server and fetch available tools via SSE."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._do_connect()
                return
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)
    
    async def _do_connect(self) -> None:
        """Perform the actual connection with SSE for bidirectional communication."""
        self._http_client = httpx.AsyncClient(timeout=60.0)
        
        logger.info(f"Connecting to SSE at {self.base_url}/sse")
        
        # Create an event to wait for endpoint
        endpoint_event = asyncio.Event()
        
        # Start SSE listener in background
        async def sse_listener():
            try:
                async with aconnect_sse(self._http_client, "GET", f"{self.base_url}/sse") as event_source:
                    async for sse in event_source.aiter_sse():
                        logger.info(f"SSE event: {sse.event}")
                        
                        if sse.event == "endpoint":
                            # Data is the endpoint URL
                            self._message_endpoint = sse.data.strip()
                            if self._message_endpoint.startswith("/"):
                                self._message_endpoint = f"{self.base_url}{self._message_endpoint}"
                            logger.info(f"Got message endpoint: {self._message_endpoint}")
                            endpoint_event.set()
                        
                        elif sse.event == "message":
                            # This is a response to one of our requests
                            try:
                                data = json.loads(sse.data)
                                msg_id = data.get("id")
                                if msg_id and msg_id in self._pending_responses:
                                    self._pending_responses[msg_id].set_result(data)
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse SSE message: {e}")
            except Exception as e:
                logger.error(f"SSE listener error: {e}")
        
        # Start the SSE listener task
        self._sse_task = asyncio.create_task(sse_listener())
        
        # Wait for endpoint with timeout
        try:
            await asyncio.wait_for(endpoint_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            raise ValueError("Timeout waiting for endpoint from MCP server")
        
        # Now perform initialization
        logger.info("Sending initialize request...")
        init_response = await self._send_message_and_wait({
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "rpg-agents", "version": "1.0.0"}
            }
        })
        logger.info(f"Initialize response: {init_response}")
        
        # Send initialized notification (no response expected)
        await self._send_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })
        
        # List tools
        logger.info("Listing tools...")
        tools_response = await self._send_message_and_wait({
            "method": "tools/list",
            "params": {}
        })
        
        if tools_response and "result" in tools_response:
            tools = tools_response["result"].get("tools", [])
            for tool in tools:
                self._tools_cache[tool["name"]] = tool
            logger.info(f"Loaded {len(self._tools_cache)} tools")
        else:
            logger.warning(f"Unexpected tools response: {tools_response}")
    
    async def _send_message(self, message: dict) -> None:
        """Send a JSON-RPC message without waiting for response."""
        if not self._message_endpoint or not self._http_client:
            raise ValueError("Not connected")
        
        await self._http_client.post(
            self._message_endpoint,
            json=message,
            headers={"Content-Type": "application/json"}
        )
    
    async def _send_message_and_wait(self, message: dict, timeout: float = 30.0) -> dict:
        """Send a JSON-RPC message and wait for response via SSE."""
        if not self._message_endpoint or not self._http_client:
            raise ValueError("Not connected")
        
        self._message_id += 1
        msg_id = self._message_id
        
        # Prepare the full message
        full_message = {
            "jsonrpc": "2.0",
            "id": msg_id,
            **message
        }
        
        # Create a future for the response
        response_future = asyncio.get_event_loop().create_future()
        self._pending_responses[msg_id] = response_future
        
        try:
            # Send the message
            await self._http_client.post(
                self._message_endpoint,
                json=full_message,
                headers={"Content-Type": "application/json"}
            )
            
            # Wait for response via SSE
            return await asyncio.wait_for(response_future, timeout=timeout)
        finally:
            self._pending_responses.pop(msg_id, None)
    
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        response = await self._send_message_and_wait({
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        })
        
        if response:
            if "result" in response:
                content = response["result"].get("content", [])
                if content and len(content) > 0:
                    return content[0].get("text", str(content))
                return str(response["result"])
            elif "error" in response:
                return f"Error: {response['error']}"
        
        return "No response from server"
    
    async def close(self):
        """Close the connection."""
        if self._sse_task:
            self._sse_task.cancel()
        if self._http_client:
            await self._http_client.aclose()
    
    def get_langchain_tools(self) -> list[StructuredTool]:
        """Convert MCP tools to LangChain StructuredTools."""
        tools = []
        
        for name, tool_def in self._tools_cache.items():
            tool = self._create_langchain_tool(name, tool_def)
            if tool:
                tools.append(tool)
        
        return tools
    
    def _create_langchain_tool(self, name: str, tool_def: dict) -> StructuredTool | None:
        """Create a LangChain tool from an MCP tool definition."""
        description = tool_def.get("description", f"Tool: {name}")
        input_schema = tool_def.get("inputSchema", {})
        
        # Build Pydantic model from JSON schema
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        # Create field definitions for the Pydantic model
        fields = {}
        for prop_name, prop_def in properties.items():
            field_type = self._json_type_to_python(prop_def.get("type", "string"))
            field_desc = prop_def.get("description", "")
            default = prop_def.get("default", ...)
            
            if prop_name not in required and default is ...:
                default = None
                field_type = field_type | None
            
            fields[prop_name] = (field_type, Field(default=default, description=field_desc))
        
        # Create dynamic Pydantic model
        if fields:
            ArgsModel = create_model(f"{name}Args", **fields)
        else:
            ArgsModel = create_model(f"{name}Args")
        
        # Create the async function that calls the MCP tool
        async def tool_func(**kwargs) -> str:
            # Filter out None values
            args = {k: v for k, v in kwargs.items() if v is not None}
            result = await self.call_tool(name, args)
            return str(result)
        
        # Create synchronous wrapper for LangChain
        def sync_tool_func(**kwargs) -> str:
            return asyncio.get_event_loop().run_until_complete(tool_func(**kwargs))
        
        return StructuredTool(
            name=name,
            description=description,
            args_schema=ArgsModel,
            func=sync_tool_func,
            coroutine=tool_func,
        )
    
    def _json_type_to_python(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, str)


# Global client instance
_mcp_client: MCPClient | None = None


async def get_mcp_client(base_url: str = "http://localhost:8080") -> MCPClient:
    """Get or create the MCP client singleton."""
    global _mcp_client
    
    if _mcp_client is None:
        _mcp_client = MCPClient(base_url)
        await _mcp_client.connect()
    
    return _mcp_client


async def get_mcp_tools(base_url: str = "http://localhost:8080") -> list[StructuredTool]:
    """Get LangChain tools from the MCP server."""
    client = await get_mcp_client(base_url)
    return client.get_langchain_tools()
