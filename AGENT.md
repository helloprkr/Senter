# Senter AI Personal Assistant - Agent Guidelines

## Core Philosophy
Senter is a universal AI personal assistant built on an agent-first architecture. Everything is an agent - all processes, tools, and capabilities are defined as JSON agents following the JSON Agents specification.

## Agent Architecture
- **JSON-Native**: All agents defined using JSON Agents Portable Agent Manifest (PAM)
- **Self-Learning**: Each agent maintains its own Topic/SENTER.md for evolution
- **Interoperable**: Agents communicate via standardized JSON protocols
- **Parallel Execution**: Multiple agents can run simultaneously without blocking user interaction

## Agent Categories

### 1. Core Processing Agents
- **Analyzer**: Content understanding, classification, sentiment analysis
- **Summarizer**: Content condensation and synthesis
- **Creative Writer**: Content generation and creative tasks

### 2. System Management Agents
- **Topic Manager**: Creates/manages Topics and SENTER.md files
- **Agent Generator**: Auto-creates JSON agents from functions
- **Model Manager**: Handles model serving, switching, and assignment
- **User Profiler**: Analyzes conversations for preferences and goals

### 3. Interaction Agents
- **Chat Router**: Routes user queries to appropriate agents
- **Goal Tracker**: Monitors progress and suggests goal completion steps
- **Feedback Analyzer**: Analyzes user responses to suggestions

## Agent Development Guidelines

### JSON Agent Structure
All agents must follow the JSON Agents specification with these profiles:
- **core**: Identity, capabilities, tools, context (required)
- **exec**: Runtime and environment metadata
- **gov**: Security, policies, observability
- **graph**: Multi-agent orchestration

### Agent Capabilities
- Define specific, actionable capabilities
- Use clear, descriptive names
- Include input/output schemas
- Specify tool functions with parameters

### Self-Learning System
- Each agent has its own Topic/SENTER.md for evolution notes
- Track performance metrics and improvement opportunities
- Automatically expand capabilities based on usage patterns
- Collaborate with other agents for shared learning

## Topic-Based Organization

### Topic Structure
```
Senter/Topics/
├── user_personal/
│   ├── SENTER.md (user preferences, habits)
│   ├── goals.json (tracked objectives)
│   └── history/ (conversation logs)
├── coding/
│   ├── SENTER.md (coding preferences, frameworks)
│   └── projects/ (code-related content)
├── creative/
│   ├── SENTER.md (creative preferences)
│   ├── music/ (generated audio)
│   └── images/ (generated images)
└── agents/
    ├── analyzer/
    │   └── SENTER.md (agent evolution notes)
    └── [each agent has its own topic]
```

### SENTER.md Evolution
- **Context Learning**: Continuously updated with user preferences and patterns
- **Pattern Recognition**: Identify user preferences, recurring topics
- **Content Updates**: Append new insights to SENTER.md
- **Cross-Reference**: Link related topics and preferences

## User Profiling & Goal Tracking

### User Profiler Agent
- **Conversation Analysis**: Monitor all Senter interactions
- **Preference Detection**: Identify communication style, likes/dislikes
- **Goal Inference**: Detect potential objectives from context
- **Pattern Recognition**: Track recurring themes and interests

### Goal Management
- **Detection**: Flag potential goals from user interactions
- **Verification**: Ask clarifying questions in chat
- **Tracking**: Monitor progress and suggest next steps
- **Completion**: Celebrate and suggest related goals

## Parallel Processing Framework

### Execution Architecture
- **Main Thread**: User interaction and immediate responses
- **Background Threads**:
  - Context analysis and SENTER.md updates
  - User profiling and goal detection
  - Agent evolution and optimization
  - Model health monitoring

### Agent Orchestration
- **Query Analysis**: Understand user intent
- **Agent Selection**: Choose appropriate agent(s) for task
- **Parallel Execution**: Run multiple agents simultaneously when beneficial
- **Result Synthesis**: Combine outputs from parallel agents

## OpenAI-Compatible Model Serving

### Model Management
- **vLLM Integration**: Primary server for Qwen2.5-Omni-3B
- **Role-Based Assignment**: Separate models for text, vision, audio
- **Provider Flexibility**: Easy switching between model providers
- **Dynamic Scaling**: Spin up/down servers based on usage

## User Interface

### Textual-Based TUI
- **Main Chat**: Green matrix theme with proper interaction
- **Sidebar**: Current topic, goals, tasks, model status
- **Agent Activity**: Background processes and status indicators

## Agent Evolution & Learning

### Self-Improvement
- **Performance Tracking**: Monitor success rates and bottlenecks
- **Capability Expansion**: Add tools based on usage patterns
- **Prompt Optimization**: Refine system prompts from results
- **Collaborative Learning**: Share insights across agent ecosystem

### External Integration
- **Prompt Analysis**: Evaluate new system prompts from external apps
- **Capability Assessment**: Determine if new functionality is represented
- **Agent Creation**: Generate JSON agents for novel capabilities
- **Registry Integration**: Add to agent discovery system

## Development Workflow

### Agent Creation Process
1. **Identify Need**: Determine required functionality
2. **Check Existing**: Use Agent Generator to check for duplicates
3. **Define JSON**: Create agent manifest following specification
4. **Implement Functions**: Write supporting Python functions
5. **Test Integration**: Verify agent orchestration
6. **Add to Registry**: Include in agent discovery system

### Quality Standards
- **Validation**: All agents validated against JSON Agents schema
- **Documentation**: Clear capability descriptions and usage examples
- **Error Handling**: Robust error handling and fallback mechanisms
- **Performance**: Efficient execution without blocking user interaction

## Resources & Dependencies

### Core Resources (in ai-toolbox/Resources/)
- **Standard**: JSON Agents specification and validation
- **Qwen3-Omni**: Multimodal model capabilities
- **ACE-Step**: Music generation framework
- **vllm**: OpenAI-compatible model serving
- **llama-cpp-python**: GGUF model inference
- **textual**: Terminal user interface framework

### Model Requirements
- **Primary**: Qwen2.5-Omni-3B (text, vision, audio)
- **Music**: ACE-Step models
- **Vision**: Specialized vision models when needed
- **Fallback**: Compatible models from other providers

## Security & Safety

### Input Validation
- Sanitize all user inputs
- Validate agent communications
- Filter model outputs for safety

### Sandboxing
- Isolated agent execution
- Resource limits and monitoring
- Graceful failure handling

### Privacy
- Local processing by default
- User data stored securely
- Clear data retention policies

## Future Expansion

### Ecosystem Growth
- Third-party agent marketplace
- Plugin system for custom agents
- API for external integrations
- Multi-user support

### Advanced Features
- Agent collaboration networks
- Predictive goal suggestions
- Automated workflow creation
- Performance optimization