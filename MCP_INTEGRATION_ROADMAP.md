# MCP (Model Context Protocol) Integration Roadmap

## Overview

MCP is becoming the industry standard for connecting AI systems to external tools and data sources. Senter is designed to be model-agnostic and tool-agnostic, making MCP integration a natural evolution of the architecture.

## Current State

Senter currently uses:
- **Direct Python tool discovery** - Scans Functions/ directory
- **Focus-specific tool wrappers** - Each Focus can have tool metadata in SENTER.md
- **SENTER_Md_Writer** - Auto-creates configurations for new tools
- **Manual configuration** - Users must add tools manually

## MCP Integration Vision

### Why MCP Fits Senter

1. **Model-Agnostic Philosophy**: Senter already works with any model - MCP extends this to work with any tool
2. **Universal Configuration**: SENTER.md format is perfect for declaring MCP tools
3. **Tool Discovery**: Auto-discovery of MCP servers would be natural extension
4. **Router Enhancement**: Router could query MCP servers for available tools
5. **Standard Protocol**: MCP is becoming standard - Senter will be future-proof

## Proposed SENTER.md Extensions

### New Section: `mcp_tools`

```yaml
mcp_tools:
  - name: tool_name
    server: mcp_server_id
    type: tool_type  # read/write/execute
    description: What this tool does
    auto_discovery: true  # Auto-discover from MCP server
    
  # Example: File system tool
  - name: read_file
    server: filesystem-server
    type: read
    description: Read files from local filesystem
    auto_discovery: true
    
  # Example: Web search tool
  - name: web_search
    server: search-server
    type: read
    description: Search web for current information
    auto_discovery: true
```

### New Section: `mcp_servers`

```yaml
mcp_servers:
  - id: server_name
    endpoint: url_or_path  # http://localhost:port or /path/to/socket
    type: stdio  # Standard MCP transport
    tools: [list of available tool names]
    capabilities: [list of server capabilities]
    authentication: none|api_key|oauth
    
  # Example: Local filesystem server
  - id: local-filesystem
    endpoint: /tmp/mcp-fs.sock
    type: stdio
    tools: [read_file, write_file, list_directory, stat_file]
    capabilities: []
    authentication: none
```

### New Section: `mcp_config`

```yaml
mcp_config:
  enabled: true  # Enable/disable MCP integration
  auto_discover: true  # Automatically discover MCP servers on startup
  tool_registration: true  # Register tools automatically
  timeout: 30  # Timeout for MCP requests (seconds)
  max_connections: 10  # Maximum concurrent MCP connections
```

## Router Agent Enhancements

### MCP-Aware Routing

Router agent should query MCP servers when:
1. User query matches available MCP tool keywords
2. Context suggests MCP tool would be helpful
3. Multiple tools available - Router considers all options (Focus + MCP)

### New Routing Decision Logic

```yaml
## Enhanced Routing Process
For each query:
  1. Analyze query for keywords
  2. Check Focus SENTER.md tool metadata
  3. Check MCP server tools (if enabled)
  4. Evaluate all options (Focus tools, MCP tools, web search)
  5. Select best match based on:
     - Semantic similarity
     - User preferences
     - Context relevance
     - Tool capability match
  6. Return best option with reasoning
```

## Tool_Discovery Agent Enhancements

### MCP Server Discovery

Tool_Discovery should:
1. Scan for MCP servers in local network
2. Query mcp_config for server list
3. Auto-register tools from discovered servers
4. Create Focus SENTER.md entries for MCP tools

### Auto-Registration Flow

```python
# Tool_Discovery discovers MCP server
mcp_server = discover_mcp_server()
tools = mcp_server.list_tools()

# For each tool, create/update SENTER.md
for tool in tools:
    focus_name = tool.name
    if not focus_exists(focus_name):
        create_focus_with_mcp_tool(tool)
    else:
        update_focus_with_mcp_tool(focus_name, tool)
```

## Implementation Strategy

### Phase 1: Foundation (Current - Q1 2026)
- [ ] Create MCP client module (Functions/mcp_client.py)
- [ ] Add MCP discovery to Tool_Discovery agent
- [ ] Update Router to consider MCP tools
- [ ] Add MCP tools to SENTER.md format
- [ ] Add mcp_config section to user_profile.json

### Phase 2: Integration (Q2 2026)
- [ ] Auto-discovery of MCP servers on startup
- [ ] Tool registration from MCP servers
- [ ] Router uses MCP tools alongside Focus tools
- [ ] Web search integrated with MCP tools
- [ ] Unified tool calling interface (Focus + MCP)

### Phase 3: Advanced (Q3 2026)
- [ ] Bi-directional communication (MCP servers can call back to Senter)
- [ ] Streaming tool responses from MCP
- [ ] Tool chaining across Focus and MCP boundaries
- [ ] MCP tool marketplace/discovery
- [ ] Advanced tool selection and ranking

## Benefits of MCP Integration

1. **Standard Protocol**: Aligns with industry standard for tool connectivity
2. **Extensibility**: Access to growing MCP ecosystem of tools
3. **Interoperability**: Work with other MCP-compliant systems
4. **No Lock-in**: Users can choose any MCP servers
5. **Security**: Proper authentication and authorization
6. **Performance**: Standard MCP protocol optimized for low latency

## Backward Compatibility

**Critical**: All current functionality must remain working:
- Direct Python tool discovery (Functions/) continues to work
- Focus-based tools (from SENTER.md) continue to work
- Manual tool registration still supported
- MCP is optional layer - Senter works with or without it

## Technical Considerations

### MCP Client Implementation

```python
# Functions/mcp_client.py
class MCPServer:
    def __init__(self, config: Dict):
        self.endpoint = config["endpoint"]
        self.transport = config.get("type", "stdio")  # or "sse"
        self.timeout = config.get("timeout", 30)
    
    async def list_tools(self) -> List[Dict]:
        # Query MCP server for available tools
        pass
    
    async def call_tool(self, tool_name: str, params: Dict) -> Any:
        # Call tool on MCP server
        pass
    
    async def connect(self):
        # Establish connection to MCP server
        pass
    
    async def disconnect(self):
        # Close connection gracefully
        pass
```

### Router Integration

Router agent needs additional context:
```python
# Router should query MCP servers
async def router_with_mcp(query: str, mcp_servers: List[MCPServer]) -> Dict:
    focus_options = parser.list_all_focuses()
    mcp_tools = await gather_mcp_tools(mcp_servers)
    
    all_options = focus_options + mcp_tools
    best_match = select_best_option(query, all_options)
    
    return {
        "focus": best_match.get("name"),
        "source": best_match.get("source"),  # "focus" or "mcp"
        "reasoning": best_match.get("reasoning"),
        "confidence": best_match.get("confidence")
    }
```

## Testing Strategy

1. **Unit Tests**: MCP client module
2. **Integration Tests**: Router with mock MCP servers
3. **End-to-End Tests**: Full query flow with real MCP servers
4. **Performance Tests**: Latency, throughput, concurrent connections

## Documentation Requirements

1. **SENTER.md Format Specification Update**: Add MCP sections
2. **MCP Integration Guide**: How to add MCP tools
3. **Architecture Update**: Add MCP components to diagrams
4. **User Guide**: Using MCP tools in Senter
5. **Migration Guide**: Moving from pure Focus tools to Focus + MCP

## Timeline

- **Q1 2026**: Foundation - MCP client module
- **Q2 2026**: Integration - Router aware of MCP tools
- **Q3 2026**: Advanced - Full bi-directional MCP support

## Conclusion

MCP integration aligns perfectly with Senter's philosophy:
- **Model-agnostic**: Works with any tool/protocol
- **Universal Configuration**: SENTER.md format handles MCP naturally
- **Auto-discovery**: Tool_Discovery can find MCP servers automatically
- **Self-organizing**: Tools register themselves via MCP protocol
- **Backward Compatible**: Current Focus system continues to work

This positions Senter as a modern, standards-compliant AI personal assistant that can leverage the entire ecosystem of MCP-compliant tools and services.
