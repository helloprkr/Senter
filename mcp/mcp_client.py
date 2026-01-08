#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client for Senter (CG-010)

Provides tool discovery and execution via MCP servers.
Supports both stdio (subprocess) and HTTP-based MCP servers.

MCP Protocol Reference:
- tools/list: Discover available tools
- tools/call: Execute a tool with arguments
"""

import json
import logging
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict, List, Callable
from queue import Queue, Empty

logger = logging.getLogger("senter.mcp")

# Try to import requests for HTTP transport
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class MCPTool:
    """Represents an MCP tool discovered from a server."""
    name: str
    description: str
    server_name: str
    input_schema: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "server_name": self.server_name,
            "input_schema": self.input_schema
        }


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport: str  # "stdio" or "http"
    command: Optional[str] = None  # For stdio: command to run
    args: List[str] = field(default_factory=list)  # For stdio: command args
    url: Optional[str] = None  # For http: server URL
    env: Dict[str, str] = field(default_factory=dict)  # Environment variables
    enabled: bool = True

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "MCPServerConfig":
        return cls(
            name=name,
            transport=data.get("transport", "stdio"),
            command=data.get("command"),
            args=data.get("args", []),
            url=data.get("url"),
            env=data.get("env", {}),
            enabled=data.get("enabled", True)
        )


class MCPStdioTransport:
    """MCP transport over stdio (subprocess)."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._response_queue: Queue = Queue()
        self._request_id = 0
        self._pending_requests: Dict[int, Queue] = {}
        self._running = False

    def connect(self) -> bool:
        """Start the MCP server subprocess."""
        if not self.config.command:
            logger.error(f"No command specified for server {self.config.name}")
            return False

        try:
            cmd = [self.config.command] + self.config.args
            env = {**dict(__import__('os').environ), **self.config.env}

            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )

            self._running = True
            self._reader_thread = threading.Thread(target=self._read_responses, daemon=True)
            self._reader_thread.start()

            logger.info(f"Connected to MCP server: {self.config.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {self.config.name}: {e}")
            return False

    def disconnect(self):
        """Stop the MCP server subprocess."""
        self._running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        logger.info(f"Disconnected from MCP server: {self.config.name}")

    def _read_responses(self):
        """Read responses from the subprocess stdout."""
        while self._running and self.process:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                response = json.loads(line)
                request_id = response.get("id")
                if request_id in self._pending_requests:
                    self._pending_requests[request_id].put(response)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.debug(f"Reader error: {e}")

    def send_request(self, method: str, params: dict = None, timeout: float = 30) -> Optional[dict]:
        """Send a JSON-RPC request and wait for response."""
        if not self.process:
            return None

        self._request_id += 1
        request_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }

        # Set up response queue
        response_queue = Queue()
        self._pending_requests[request_id] = response_queue

        try:
            # Send request
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

            # Wait for response
            response = response_queue.get(timeout=timeout)
            return response

        except Empty:
            logger.warning(f"Request timed out: {method}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
        finally:
            del self._pending_requests[request_id]


class MCPHttpTransport:
    """MCP transport over HTTP."""

    def __init__(self, config: MCPServerConfig):
        self.config = config

    def connect(self) -> bool:
        """Verify HTTP server is reachable."""
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available for HTTP transport")
            return False

        if not self.config.url:
            logger.error(f"No URL specified for server {self.config.name}")
            return False

        try:
            response = requests.get(self.config.url, timeout=5)
            logger.info(f"Connected to MCP server: {self.config.name}")
            return True
        except Exception as e:
            logger.warning(f"HTTP server not available: {self.config.name}: {e}")
            return False

    def disconnect(self):
        """No-op for HTTP transport."""
        pass

    def send_request(self, method: str, params: dict = None, timeout: float = 30) -> Optional[dict]:
        """Send a JSON-RPC request over HTTP."""
        if not self.config.url:
            return None

        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {}
        }

        try:
            response = requests.post(
                self.config.url,
                json=request,
                timeout=timeout
            )
            return response.json()
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return None


class MCPClient:
    """
    Client for interacting with MCP servers.

    Features:
    - Multi-server support
    - Tool discovery and registration
    - Standardized tool calling interface
    """

    def __init__(self, senter_root: Path = None):
        self.senter_root = Path(senter_root) if senter_root else Path(".")
        self.config_path = self.senter_root / "config" / "mcp_servers.json"

        # Server connections
        self.servers: Dict[str, Any] = {}  # name -> transport

        # Global tool registry
        self.tools: Dict[str, MCPTool] = {}  # tool_name -> MCPTool

        # Load configuration and connect
        self.server_configs: Dict[str, MCPServerConfig] = {}
        self._load_config()

    def _load_config(self):
        """Load MCP server configuration."""
        if not self.config_path.exists():
            logger.info("No MCP server configuration found")
            return

        try:
            config = json.loads(self.config_path.read_text())
            for name, server_config in config.get("servers", {}).items():
                self.server_configs[name] = MCPServerConfig.from_dict(name, server_config)
            logger.info(f"Loaded {len(self.server_configs)} MCP server configs")
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")

    def connect_all(self):
        """Connect to all enabled MCP servers."""
        for name, config in self.server_configs.items():
            if not config.enabled:
                continue

            try:
                self.connect_server(name)
            except Exception as e:
                logger.warning(f"Failed to connect to {name}: {e}")

    def connect_server(self, name: str) -> bool:
        """Connect to a specific MCP server."""
        config = self.server_configs.get(name)
        if not config:
            logger.error(f"Unknown server: {name}")
            return False

        # Create transport
        if config.transport == "stdio":
            transport = MCPStdioTransport(config)
        elif config.transport == "http":
            transport = MCPHttpTransport(config)
        else:
            logger.error(f"Unknown transport: {config.transport}")
            return False

        # Connect
        if transport.connect():
            self.servers[name] = transport
            # Discover tools
            self._discover_tools(name, transport)
            return True

        return False

    def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for name, transport in self.servers.items():
            try:
                transport.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting {name}: {e}")
        self.servers.clear()
        self.tools.clear()

    def _discover_tools(self, server_name: str, transport: Any):
        """Discover tools from an MCP server."""
        response = transport.send_request("tools/list")

        if not response:
            logger.warning(f"No response from {server_name} for tools/list")
            return

        if "error" in response:
            logger.error(f"Error discovering tools from {server_name}: {response['error']}")
            return

        tools = response.get("result", {}).get("tools", [])
        for tool_data in tools:
            tool = MCPTool(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                server_name=server_name,
                input_schema=tool_data.get("inputSchema", {})
            )
            self.tools[tool.name] = tool
            logger.info(f"Discovered tool: {tool.name} from {server_name}")

        logger.info(f"Discovered {len(tools)} tools from {server_name}")

    def call_tool(self, tool_name: str, arguments: dict = None) -> dict:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result dict with "content" or "error" key
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}

        transport = self.servers.get(tool.server_name)
        if not transport:
            return {"error": f"Server not connected: {tool.server_name}"}

        response = transport.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {}
        })

        if not response:
            return {"error": "No response from server"}

        if "error" in response:
            return {"error": response["error"]}

        return response.get("result", {})

    def list_tools(self) -> List[MCPTool]:
        """Get list of all discovered tools."""
        return list(self.tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        return self.tools.get(name)

    def refresh_tools(self):
        """Re-discover tools from all connected servers."""
        self.tools.clear()
        for name, transport in self.servers.items():
            self._discover_tools(name, transport)


class MCPToolRegistry:
    """
    Global registry for MCP tools (CG-010).

    Provides standardized interface for task engine to discover and call tools.
    """

    _instance: Optional['MCPToolRegistry'] = None

    def __init__(self, senter_root: Path = None):
        self.client = MCPClient(senter_root)
        self._initialized = False

    @classmethod
    def get_instance(cls, senter_root: Path = None) -> 'MCPToolRegistry':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(senter_root)
        return cls._instance

    def initialize(self):
        """Initialize and connect to MCP servers."""
        if self._initialized:
            return

        self.client.connect_all()
        self._initialized = True
        logger.info(f"MCP Tool Registry initialized with {len(self.client.tools)} tools")

    def shutdown(self):
        """Disconnect from all servers."""
        self.client.disconnect_all()
        self._initialized = False

    def get_all_tools(self) -> List[MCPTool]:
        """Get all available tools."""
        return self.client.list_tools()

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get tool by name."""
        return self.client.get_tool(name)

    def call(self, tool_name: str, arguments: dict = None) -> dict:
        """Call a tool by name."""
        return self.client.call_tool(tool_name, arguments)

    def is_available(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        return tool_name in self.client.tools


# Convenience functions
def get_mcp_registry(senter_root: Path = None) -> MCPToolRegistry:
    """Get the global MCP tool registry."""
    return MCPToolRegistry.get_instance(senter_root)


def discover_tools(senter_root: Path = None) -> List[MCPTool]:
    """Discover all available MCP tools."""
    registry = get_mcp_registry(senter_root)
    registry.initialize()
    return registry.get_all_tools()


def call_tool(tool_name: str, arguments: dict = None, senter_root: Path = None) -> dict:
    """Call an MCP tool."""
    registry = get_mcp_registry(senter_root)
    return registry.call(tool_name, arguments)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Senter MCP Client")
    parser.add_argument("--list", "-l", action="store_true", help="List available tools")
    parser.add_argument("--call", "-c", help="Call a tool by name")
    parser.add_argument("--args", "-a", help="Tool arguments (JSON string)")
    parser.add_argument("--servers", "-s", action="store_true", help="List configured servers")
    parser.add_argument("--connect", help="Connect to a specific server")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    client = MCPClient(Path("."))

    if args.servers:
        print(f"\nConfigured MCP Servers ({len(client.server_configs)}):")
        for name, config in client.server_configs.items():
            status = "enabled" if config.enabled else "disabled"
            print(f"  [{status}] {name} ({config.transport})")
        print()

    elif args.list:
        client.connect_all()
        tools = client.list_tools()
        print(f"\nAvailable Tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
            print(f"    Server: {tool.server_name}")
        print()
        client.disconnect_all()

    elif args.connect:
        if client.connect_server(args.connect):
            print(f"Connected to {args.connect}")
            tools = [t for t in client.list_tools() if t.server_name == args.connect]
            print(f"Tools: {len(tools)}")
        client.disconnect_all()

    elif args.call:
        client.connect_all()
        tool_args = json.loads(args.args) if args.args else {}
        result = client.call_tool(args.call, tool_args)
        print(f"\nResult: {json.dumps(result, indent=2)}")
        client.disconnect_all()

    else:
        parser.print_help()
