"""MCP client for connecting to rpg-mcp server and exposing tools to LangChain."""

import asyncio
import logging
from typing import Any
import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for connecting to an MCP server via Streamable HTTP transport."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self._mcp_endpoint = f"{base_url}/mcp"
        self._tools_cache: dict[str, dict] = {}
        self._http_client: httpx.AsyncClient | None = None
        self._message_id = 0
    
    async def connect(self) -> None:
        """Connect to the MCP server and fetch available tools."""
        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                await self._do_connect()
                return
            except Exception as e:
                delay = base_delay * (2 ** attempt)  # exponential backoff: 2, 4, 8, 16, 32s
                logger.error(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} connection attempts to {self.base_url} failed")
                    raise
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
    
    async def _do_connect(self) -> None:
        """Perform the actual connection via Streamable HTTP."""
        # Clean up any previous client
        if self._http_client:
            await self._http_client.aclose()
        
        self._http_client = httpx.AsyncClient(timeout=60.0)
        
        logger.info(f"Connecting to MCP server at {self.base_url}")
        
        # Health check to verify the server is reachable
        try:
            health_resp = await self._http_client.get(f"{self.base_url}/health", timeout=15.0)
            logger.info(f"MCP health check: status={health_resp.status_code}, body={health_resp.text[:200]}")
            if health_resp.status_code != 200:
                raise ValueError(f"MCP health check failed: HTTP {health_resp.status_code} - {health_resp.text[:200]}")
        except httpx.ConnectError as e:
            raise ValueError(f"Cannot reach MCP server at {self.base_url}: {e}")
        except httpx.TimeoutException as e:
            raise ValueError(f"MCP health check timed out at {self.base_url}: {e}")
        
        # Initialize
        logger.info("Sending initialize request...")
        init_response = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "rpg-agents", "version": "1.0.0"}
            }
        )
        logger.info(f"Initialize response: {init_response}")
        
        # Send initialized notification (no response expected)
        await self._send_notification("notifications/initialized")
        
        # List tools
        logger.info("Listing tools...")
        tools_response = await self._send_request("tools/list", {})
        
        if tools_response and "result" in tools_response:
            tools = tools_response["result"].get("tools", [])
            for tool in tools:
                self._tools_cache[tool["name"]] = tool
            logger.info(f"Loaded {len(self._tools_cache)} tools")
        else:
            logger.warning(f"Unexpected tools response: {tools_response}")
    
    async def _send_request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        """Send JSON-RPC request, get JSON response."""
        if not self._http_client:
            raise ValueError("Not connected")
        
        self._message_id += 1
        resp = await self._http_client.post(
            self._mcp_endpoint,
            json={"jsonrpc": "2.0", "id": self._message_id, "method": method, "params": params},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=timeout,
        )
        return resp.json()
    
    async def _send_notification(self, method: str, params: dict | None = None) -> None:
        """Send JSON-RPC notification (no id, no response expected)."""
        if not self._http_client:
            raise ValueError("Not connected")
        
        message = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        
        await self._http_client.post(
            self._mcp_endpoint,
            json=message,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
    
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        response = await self._send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments
            }
        )
        
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
    """Get or create the MCP client singleton.
    
    If the previous connection failed or the client has no tools cached,
    reset and reconnect to avoid returning a stale/broken client.
    """
    global _mcp_client
    
    # Reset stale client: if it exists but has no tools, something went wrong
    if _mcp_client is not None and len(_mcp_client._tools_cache) == 0:
        logger.warning("MCP client exists but has 0 cached tools — resetting for fresh connection")
        await _mcp_client.close()
        _mcp_client = None
    
    if _mcp_client is None:
        client = MCPClient(base_url)
        try:
            await client.connect()
        except Exception:
            # Don't leave a broken client as the singleton
            await client.close()
            raise
        _mcp_client = client
    
    return _mcp_client


async def get_mcp_tools(base_url: str = "http://localhost:8080") -> list[StructuredTool]:
    """Get LangChain tools from the MCP server."""
    client = await get_mcp_client(base_url)
    return client.get_langchain_tools()
