#!/usr/bin/env python3
"""
Tests for MCP Client (CG-010)
Tests MCP tool discovery and calling interface.
"""

import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mcp_tool_dataclass():
    """Test MCPTool dataclass"""
    from mcp.mcp_client import MCPTool

    tool = MCPTool(
        name="test_tool",
        description="A test tool",
        server_name="test_server",
        input_schema={"type": "object", "properties": {"arg1": {"type": "string"}}}
    )

    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.server_name == "test_server"

    # Test to_dict
    d = tool.to_dict()
    assert d["name"] == "test_tool"
    assert d["description"] == "A test tool"
    assert "input_schema" in d

    return True


def test_mcp_server_config():
    """Test MCPServerConfig loading"""
    from mcp.mcp_client import MCPServerConfig

    # Test stdio config
    stdio_data = {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "test_server"],
        "enabled": True
    }
    config = MCPServerConfig.from_dict("test", stdio_data)

    assert config.name == "test"
    assert config.transport == "stdio"
    assert config.command == "python"
    assert config.args == ["-m", "test_server"]
    assert config.enabled is True

    # Test http config
    http_data = {
        "transport": "http",
        "url": "http://localhost:8080",
        "enabled": False
    }
    config = MCPServerConfig.from_dict("http_test", http_data)

    assert config.name == "http_test"
    assert config.transport == "http"
    assert config.url == "http://localhost:8080"
    assert config.enabled is False

    return True


def test_mcp_client_config_loading():
    """Test that MCPClient loads configuration"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        config_dir = senter_root / "config"
        config_dir.mkdir(parents=True)

        # Create test config
        config = {
            "servers": {
                "test_server": {
                    "transport": "stdio",
                    "command": "echo",
                    "args": ["hello"],
                    "enabled": True
                },
                "disabled_server": {
                    "transport": "http",
                    "url": "http://localhost:9999",
                    "enabled": False
                }
            }
        }
        (config_dir / "mcp_servers.json").write_text(json.dumps(config))

        # Create client
        client = MCPClient(senter_root)

        # Should have loaded configs
        assert len(client.server_configs) == 2
        assert "test_server" in client.server_configs
        assert "disabled_server" in client.server_configs

        # Check enabled status
        assert client.server_configs["test_server"].enabled is True
        assert client.server_configs["disabled_server"].enabled is False

        return True


def test_mcp_tool_discovery():
    """Test tool discovery from server response"""
    from mcp.mcp_client import MCPClient, MCPStdioTransport, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        (senter_root / "config").mkdir(parents=True)
        (senter_root / "config" / "mcp_servers.json").write_text("{\"servers\": {}}")

        client = MCPClient(senter_root)

        # Create mock transport
        mock_transport = MagicMock()
        mock_transport.send_request.return_value = {
            "result": {
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file from disk",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            }
                        }
                    },
                    {
                        "name": "write_file",
                        "description": "Write to a file",
                        "inputSchema": {}
                    }
                ]
            }
        }

        # Manually add server and discover
        client.servers["mock"] = mock_transport
        client._discover_tools("mock", mock_transport)

        # Should have discovered 2 tools
        assert len(client.tools) == 2
        assert "read_file" in client.tools
        assert "write_file" in client.tools

        # Check tool details
        tool = client.tools["read_file"]
        assert tool.description == "Read a file from disk"
        assert tool.server_name == "mock"

        return True


def test_mcp_tool_call():
    """Test calling an MCP tool"""
    from mcp.mcp_client import MCPClient, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        (senter_root / "config").mkdir(parents=True)
        (senter_root / "config" / "mcp_servers.json").write_text("{\"servers\": {}}")

        client = MCPClient(senter_root)

        # Create mock transport
        mock_transport = MagicMock()
        mock_transport.send_request.return_value = {
            "result": {
                "content": [{"type": "text", "text": "File contents here"}]
            }
        }

        # Add tool and server
        client.tools["read_file"] = MCPTool(
            name="read_file",
            description="Read a file",
            server_name="mock"
        )
        client.servers["mock"] = mock_transport

        # Call tool
        result = client.call_tool("read_file", {"path": "/test/file.txt"})

        # Check result
        assert "content" in result
        mock_transport.send_request.assert_called_with("tools/call", {
            "name": "read_file",
            "arguments": {"path": "/test/file.txt"}
        })

        return True


def test_mcp_tool_call_unknown_tool():
    """Test calling unknown tool returns error"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        (senter_root / "config").mkdir(parents=True)
        (senter_root / "config" / "mcp_servers.json").write_text("{\"servers\": {}}")

        client = MCPClient(senter_root)

        result = client.call_tool("nonexistent_tool", {})

        assert "error" in result
        assert "Unknown tool" in result["error"]

        return True


def test_mcp_tool_registry():
    """Test MCPToolRegistry singleton"""
    from mcp.mcp_client import MCPToolRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        (senter_root / "config").mkdir(parents=True)
        (senter_root / "config" / "mcp_servers.json").write_text("{\"servers\": {}}")

        # Reset singleton for testing
        MCPToolRegistry._instance = None

        registry1 = MCPToolRegistry.get_instance(senter_root)
        registry2 = MCPToolRegistry.get_instance(senter_root)

        # Should be same instance
        assert registry1 is registry2

        # Clean up
        MCPToolRegistry._instance = None

        return True


def test_mcp_graceful_no_config():
    """Test graceful handling when no config exists"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        # No config file created

        # Should not raise error
        client = MCPClient(senter_root)

        assert len(client.server_configs) == 0
        assert len(client.tools) == 0

        return True


def test_mcp_graceful_server_unavailable():
    """Test graceful handling when server unavailable"""
    from mcp.mcp_client import MCPClient

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        config_dir = senter_root / "config"
        config_dir.mkdir(parents=True)

        # Create config with invalid server
        config = {
            "servers": {
                "invalid": {
                    "transport": "http",
                    "url": "http://localhost:99999",  # Invalid port
                    "enabled": True
                }
            }
        }
        (config_dir / "mcp_servers.json").write_text(json.dumps(config))

        client = MCPClient(senter_root)

        # Connect should fail gracefully
        result = client.connect_server("invalid")

        # Should return False, not raise exception
        assert result is False
        assert "invalid" not in client.servers

        return True


def test_mcp_list_tools():
    """Test listing all tools"""
    from mcp.mcp_client import MCPClient, MCPTool

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        (senter_root / "config").mkdir(parents=True)
        (senter_root / "config" / "mcp_servers.json").write_text("{\"servers\": {}}")

        client = MCPClient(senter_root)

        # Add some tools
        client.tools["tool1"] = MCPTool("tool1", "First tool", "server1")
        client.tools["tool2"] = MCPTool("tool2", "Second tool", "server2")

        tools = client.list_tools()

        assert len(tools) == 2
        assert any(t.name == "tool1" for t in tools)
        assert any(t.name == "tool2" for t in tools)

        return True


if __name__ == "__main__":
    tests = [
        test_mcp_tool_dataclass,
        test_mcp_server_config,
        test_mcp_client_config_loading,
        test_mcp_tool_discovery,
        test_mcp_tool_call,
        test_mcp_tool_call_unknown_tool,
        test_mcp_tool_registry,
        test_mcp_graceful_no_config,
        test_mcp_graceful_server_unavailable,
        test_mcp_list_tools,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
