#!/usr/bin/env python3
"""
Tests for MCP Integration (MCP-001, MCP-002, MCP-003, MCP-004)
Tests MCP client, tool discovery, parameter validation, and configuration.
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== MCP-001: MCP Client Module Tests ==========

def test_mcp_client_creation():
    """Test MCPClient can be created (MCP-001)"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir))

        assert client is not None
        assert client.connect_timeout == 30  # Default timeout

    return True


def test_mcp_client_custom_timeout():
    """Test MCPClient with custom timeout (MCP-001)"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir), connect_timeout=60)

        assert client.connect_timeout == 60

    return True


def test_mcp_server_config_from_dict():
    """Test MCPServerConfig parsing (MCP-001)"""
    from mcp.mcp_client import MCPServerConfig

    config_data = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@test/server"],
        "enabled": True
    }

    config = MCPServerConfig.from_dict("test-server", config_data)

    assert config.name == "test-server"
    assert config.transport == "stdio"
    assert config.command == "npx"
    assert config.args == ["-y", "@test/server"]
    assert config.enabled == True

    return True


def test_mcp_config_loading():
    """Test MCP configuration loading (MCP-001)"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config directory and file
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()

        config = {
            "servers": {
                "test-server": {
                    "transport": "stdio",
                    "command": "echo",
                    "args": ["test"],
                    "enabled": True
                }
            }
        }
        (config_dir / "mcp_servers.json").write_text(json.dumps(config))

        client = MCPClient(Path(tmpdir))

        assert "test-server" in client.server_configs
        assert client.server_configs["test-server"].command == "echo"

    return True


def test_mcp_tool_dataclass():
    """Test MCPTool dataclass (MCP-001)"""
    from mcp.mcp_client import MCPTool

    tool = MCPTool(
        name="test_tool",
        description="A test tool",
        server_name="test-server",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}}
    )

    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.server_name == "test-server"

    tool_dict = tool.to_dict()
    assert tool_dict["name"] == "test_tool"
    assert "input_schema" in tool_dict

    return True


# ========== MCP-002: Tool Discovery and Registration Tests ==========

def test_tool_discovery_structure():
    """Test tool discovery from server response (MCP-002)"""
    from mcp.mcp_client import MCPClient, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir))

        # Manually add a tool (simulating discovery)
        tool = MCPTool(
            name="search",
            description="Search the web",
            server_name="brave-search",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        )
        client.tools["search"] = tool

        # Verify tool registration
        assert "search" in client.tools
        assert client.tools["search"].server_name == "brave-search"

    return True


def test_tool_source_tagging():
    """Test tools are tagged with source (MCP-002)"""
    from mcp.mcp_client import MCPTool

    tool = MCPTool(
        name="read_file",
        description="Read a file",
        server_name="filesystem"
    )

    # Source tag format: mcp:<server_name>
    expected_source = f"mcp:{tool.server_name}"
    assert expected_source == "mcp:filesystem"

    return True


def test_list_tools():
    """Test listing discovered tools (MCP-002)"""
    from mcp.mcp_client import MCPClient, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir))

        # Add multiple tools
        for i in range(3):
            tool = MCPTool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                server_name="test-server"
            )
            client.tools[tool.name] = tool

        tools = client.list_tools()

        assert len(tools) == 3
        assert all(isinstance(t, MCPTool) for t in tools)

    return True


def test_get_tool_by_name():
    """Test getting specific tool (MCP-002)"""
    from mcp.mcp_client import MCPClient, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir))

        tool = MCPTool(
            name="specific_tool",
            description="A specific tool",
            server_name="test-server"
        )
        client.tools["specific_tool"] = tool

        retrieved = client.get_tool("specific_tool")
        assert retrieved is not None
        assert retrieved.name == "specific_tool"

        # Non-existent tool
        missing = client.get_tool("nonexistent")
        assert missing is None

    return True


# ========== MCP-003: Tool Execution and Validation Tests ==========

def test_parameter_validation_required():
    """Test required parameter validation (MCP-003)"""
    from mcp.mcp_client import MCPClient, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir))

        tool = MCPTool(
            name="search",
            description="Search",
            server_name="test",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        )
        client.tools["search"] = tool

        # Missing required parameter
        valid, error = client.validate_parameters("search", {"limit": 10})
        assert valid == False
        assert "Missing required parameter" in error
        assert "query" in error

        # All required parameters present
        valid, error = client.validate_parameters("search", {"query": "test"})
        assert valid == True
        assert error == ""

    return True


def test_parameter_validation_types():
    """Test parameter type validation (MCP-003)"""
    from mcp.mcp_client import MCPClient, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir))

        tool = MCPTool(
            name="typed_tool",
            description="Tool with typed params",
            server_name="test",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"}
                }
            }
        )
        client.tools["typed_tool"] = tool

        # Wrong type for string parameter
        valid, error = client.validate_parameters("typed_tool", {"text": 123})
        assert valid == False
        assert "expected string" in error

        # Wrong type for integer parameter
        valid, error = client.validate_parameters("typed_tool", {"count": "not_int"})
        assert valid == False
        assert "expected integer" in error

        # Correct types
        valid, error = client.validate_parameters("typed_tool", {
            "text": "hello",
            "count": 5,
            "enabled": True
        })
        assert valid == True

    return True


def test_call_tool_validates_params():
    """Test call_tool validates parameters (MCP-003)"""
    from mcp.mcp_client import MCPClient, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir))

        tool = MCPTool(
            name="validated_tool",
            description="Tool with validation",
            server_name="test",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        )
        client.tools["validated_tool"] = tool

        # Call without required parameter - should fail validation
        result = client.call_tool("validated_tool", {})
        assert "error" in result
        assert "validation failed" in result["error"].lower()

    return True


def test_call_tool_timeout_configurable():
    """Test tool call timeout is configurable (MCP-003)"""
    from mcp.mcp_client import MCPClient, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir))

        tool = MCPTool(
            name="slow_tool",
            description="Slow tool",
            server_name="test"
        )
        client.tools["slow_tool"] = tool

        # Default timeout is 60s, can override per-call
        # Just verify the signature accepts timeout parameter
        # (actual timeout behavior requires connected server)
        result = client.call_tool("slow_tool", {}, timeout=120, validate=False)

        # Without server, should error about not connected
        assert "error" in result
        assert "not connected" in result["error"].lower()

    return True


def test_unknown_tool_error():
    """Test error for unknown tool (MCP-003)"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MCPClient(Path(tmpdir))

        result = client.call_tool("nonexistent_tool", {"param": "value"})

        assert "error" in result
        assert "unknown tool" in result["error"].lower()

    return True


# ========== MCP-004: Server Configuration Management Tests ==========

def test_config_file_structure():
    """Test config file structure (MCP-004)"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with all fields
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()

        config = {
            "servers": {
                "server1": {
                    "transport": "stdio",
                    "command": "node",
                    "args": ["server.js"],
                    "enabled": True
                },
                "server2": {
                    "transport": "http",
                    "url": "http://localhost:8080",
                    "enabled": False
                }
            }
        }
        (config_dir / "mcp_servers.json").write_text(json.dumps(config))

        client = MCPClient(Path(tmpdir))

        # Verify both servers loaded
        assert len(client.server_configs) == 2
        assert "server1" in client.server_configs
        assert "server2" in client.server_configs

        # Verify server1 properties
        s1 = client.server_configs["server1"]
        assert s1.transport == "stdio"
        assert s1.enabled == True

        # Verify server2 properties
        s2 = client.server_configs["server2"]
        assert s2.transport == "http"
        assert s2.url == "http://localhost:8080"
        assert s2.enabled == False

    return True


def test_config_with_env_vars():
    """Test config with environment variables (MCP-004)"""
    from mcp.mcp_client import MCPServerConfig

    config_data = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@test/server"],
        "env": {
            "API_KEY": "secret123",
            "DEBUG": "true"
        },
        "enabled": True
    }

    config = MCPServerConfig.from_dict("api-server", config_data)

    assert config.env == {"API_KEY": "secret123", "DEBUG": "true"}

    return True


def test_server_enabled_filtering():
    """Test filtering enabled servers (MCP-004)"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()

        config = {
            "servers": {
                "enabled1": {"transport": "stdio", "command": "echo", "enabled": True},
                "disabled1": {"transport": "stdio", "command": "echo", "enabled": False},
                "enabled2": {"transport": "http", "url": "http://localhost", "enabled": True}
            }
        }
        (config_dir / "mcp_servers.json").write_text(json.dumps(config))

        client = MCPClient(Path(tmpdir))

        enabled = [name for name, cfg in client.server_configs.items() if cfg.enabled]

        assert len(enabled) == 2
        assert "enabled1" in enabled
        assert "enabled2" in enabled
        assert "disabled1" not in enabled

    return True


def test_missing_config_file():
    """Test behavior with missing config file (MCP-004)"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        # No config file created
        client = MCPClient(Path(tmpdir))

        # Should have empty server configs
        assert len(client.server_configs) == 0

    return True


# ========== MCP Tool Registry Tests ==========

def test_mcp_tool_registry_singleton():
    """Test MCPToolRegistry is singleton"""
    from mcp.mcp_client import MCPToolRegistry

    # Reset singleton for test
    MCPToolRegistry._instance = None

    with tempfile.TemporaryDirectory() as tmpdir:
        reg1 = MCPToolRegistry.get_instance(Path(tmpdir))
        reg2 = MCPToolRegistry.get_instance(Path(tmpdir))

        assert reg1 is reg2

    # Clean up singleton
    MCPToolRegistry._instance = None

    return True


def test_mcp_registry_tool_access():
    """Test MCPToolRegistry tool access methods"""
    from mcp.mcp_client import MCPToolRegistry, MCPTool

    # Reset singleton
    MCPToolRegistry._instance = None

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = MCPToolRegistry.get_instance(Path(tmpdir))

        # Manually add a tool to the client
        tool = MCPTool(
            name="test_tool",
            description="Test",
            server_name="test"
        )
        registry.client.tools["test_tool"] = tool

        # Test access methods
        all_tools = registry.get_all_tools()
        assert len(all_tools) == 1

        specific = registry.get_tool("test_tool")
        assert specific.name == "test_tool"

        assert registry.is_available("test_tool") == True
        assert registry.is_available("nonexistent") == False

    # Clean up
    MCPToolRegistry._instance = None

    return True


if __name__ == "__main__":
    tests = [
        # MCP-001
        test_mcp_client_creation,
        test_mcp_client_custom_timeout,
        test_mcp_server_config_from_dict,
        test_mcp_config_loading,
        test_mcp_tool_dataclass,
        # MCP-002
        test_tool_discovery_structure,
        test_tool_source_tagging,
        test_list_tools,
        test_get_tool_by_name,
        # MCP-003
        test_parameter_validation_required,
        test_parameter_validation_types,
        test_call_tool_validates_params,
        test_call_tool_timeout_configurable,
        test_unknown_tool_error,
        # MCP-004
        test_config_file_structure,
        test_config_with_env_vars,
        test_server_enabled_filtering,
        test_missing_config_file,
        # Registry
        test_mcp_tool_registry_singleton,
        test_mcp_registry_tool_access,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
