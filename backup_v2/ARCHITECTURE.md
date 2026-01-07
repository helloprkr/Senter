# Senter OmniAgent Chain Architecture

## Core Principle

**Everything is an omniagent instance with a unique SENTER.md configuration**

Every capability, feature, and focus in Senter is just an omniagent with:
- A model config (from SENTER.md or user_profile.json)
- A system prompt (from SENTER.md)
- Specific purpose and context (from SENTER.md)

No special scripts, no complex infrastructure - just omniagent + SENTER.md.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│              Senter OmniAgent Chain                     │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                                │
         ┌────▼────┐                   ┌────▼────┐
         │  Router  │                   │  Chat     │
         │  Agent   │                   │  Agent    │
         └────┬────┘                   └────┬────┘
              │                                │
    ┌─────────┼────────────────┴────────────┐
    │         │                           │       │
┌───▼───┐┌───▼───┐┌───▼───┐┌───▼───┐┌───▼───┐
│ Goal_  ││ Tool_  ││Context_││Plan-   ││Profil- │
│Detec- ││Discov-││Gather-││ner    ││er     │
│tor     ││ery    ││er     ││Agent   ││Agent  │
└───┬───┘└───┬───┘└───┬───┘└───┬───┘└───┬───┘
   │         │         │         │         │         │
   │         │         │         │         │         │
   └────┬────┴─────────┴─────────┴─────────┴─────────┘
        │
        │ Updates SENTER.md files
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│         User Focuses (omniagents)           │
│  coding, research, creative, etc.           │
│  Each with SENTER.md config               │
└─────────────────────────────────────────────────────────┘
```

---

## Internal Focus Agents

### 1. Router Agent
**Location**: `Focuses/internal/Router/SENTER.md`

**Purpose**: Route user queries to the best matching Focus

**Input**: User query + list of available Focuses

**Output**: JSON:
```json
{
  "focus": "focus_name",
  "reasoning": "why this Focus was selected",
  "confidence": "high/medium/low"
}
```

**Process**:
1. Analyze query intent
2. Match against available Focuses
3. Consider previous interactions
4. Select best Focus
5. Default to 'general' if no clear match

---

### 2. Goal_Detector Agent
**Location**: `Focuses/internal/Goal_Detector/SENTER.md`

**Purpose**: Extract goals from user queries and conversations

**Input**: Query or conversation context

**Output**: JSON:
```json
{
  "goals": [
    {
      "text": "goal text",
      "focus": "target_focus",
      "priority": "high/medium/low",
      "detected_from": "user query/context"
    }
  ]
}
```

**Process**:
1. Scan for goal indicators ("I want to", "I need to", etc.)
2. Extract goal text
3. Associate with specific Focus
4. Estimate priority
5. **NO CAP** on number of goals
6. Goals are Focus-specific (stored in each Focus's SENTER.md)

---

### 3. Tool_Discovery Agent
**Location**: `Focuses/internal/Tool_Discovery/SENTER.md`

**Purpose**: Discover Python tools in Functions/ directory

**Input**: Functions/ directory scan

**Output**: Creates Focuses for each tool

**Process**:
1. Scan Functions/ for .py files
2. Extract function signatures and docstrings
3. Create Focuses/<tool_name>/SENTER.md
4. Generate appropriate system prompt
5. Update config/focus_agent_map.json if needed

**Key Principle**: Every Python function becomes its own Focus

---

### 4. Context_Gatherer Agent
**Location**: `Focuses/internal/Context_Gatherer/SENTER.md`

**Purpose**: Update SENTER.md files with conversation context

**Input**: Conversation history

**Output**: Direct file updates (no JSON output)

**Process**:
1. Read conversation history (last 20 messages)
2. Identify discussed Focuses
3. Update SENTER.md sections:
   - Detected Goals
   - Context
   - Patterns Observed
   - Evolution Notes
4. Keep summaries, truncate if too long

---

### 5. Planner Agent
**Location**: `Focuses/internal/Planner/SENTER.md`

**Purpose**: Break down goals into actionable tasks

**Input**: Goals from Focus SENTER.md files

**Output**: JSON:
```json
{
  "plans": [
    {
      "goal": "the goal text",
      "focus": "focus_name",
      "tasks": [
        {
          "step": 1,
          "task": "specific actionable task",
          "priority": "high/medium/low",
          "complexity": "simple/medium/complex",
          "dependencies": []
        }
      ]
    }
  ]
}
```

**Process**:
1. Read "Detected Goals" from Focus SENTER.md
2. For each goal, create step-by-step plan
3. Ensure tasks are SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
4. Update "Goals & Objectives" section with tasks

---

### 6. Profiler Agent
**Location**: `Focuses/internal/Profiler/SENTER.md`

**Purpose**: Analyze user patterns per Focus

**Input**: Conversation history per Focus

**Output**: JSON:
```json
{
  "preferences_updated": [
    {
      "type": "preference_type",
      "value": "preference_value",
      "confidence": "high/medium/low",
      "evidence": "what led to this conclusion"
    }
  ],
  "patterns_detected": [...],
  "notes": "additional observations"
}
```

**Process**:
1. Read Focus SENTER.md for existing preferences
2. Analyze conversation history
3. Extract:
   - User Preferences (style, format, detail level)
   - Patterns Observed (query types, timing, frequency)
   - Update SENTER.md sections
4. Build profiles incrementally

**Important**: One profiler instance per user Focus (not a single profiler for all)

---

### 7. Chat Agent
**Location**: `Focuses/internal/Chat/SENTER.md`

**Purpose**: Main conversational AI assistance

**Input**: Query + full context

**Output**: Natural conversational responses

**Process**:
1. Read Focus SENTER.md for context
2. Use "Detected Goals" to prioritize responses
3. Use "User Preferences" to match style
4. Maintain conversation continuity
5. Provide helpful, engaging responses

**Response Style**:
- Context-aware
- Goal-aware
- Adapts to user preferences
- Natural, conversational

---

## User Focus Agents

### Coding Focus
**Location**: `Focuses/coding/SENTER.md`

**Purpose**: Programming, debugging, code review

**System Prompt**:
- Focus on technical accuracy
- Provide code examples
- Explain errors clearly
- Suggest best practices

### Research Focus
**Location**: `Focuses/research/SENTER.md`

**Purpose**: Information gathering, learning, fact-checking

**System Prompt**:
- Comprehensive coverage
- Cite sources when possible
- Verify facts
- Present balanced view

### Creative Focus
**Location**: `Focuses/creative/SENTER.md`

**Purpose**: Art, music, writing, creative projects

**System Prompt**:
- Encourage creativity
- Provide varied options
- Offer constructive feedback
- Inspire experimentation

### User_Personal Focus
**Location**: `Focuses/user_personal/SENTER.md`

**Purpose**: Scheduling, goals, preferences

**System Prompt**:
- Respect user's time and priorities
- Help with organization
- Maintain privacy
- Be encouraging and supportive

### General Focus
**Location**: `Focuses/general/SENTER.md`

**Purpose**: Catch-all for other topics

**System Prompt**:
- Versatile and adaptable
- Seek clarification when needed
- Provide balanced responses
- Know when to delegate to specialized Focuses

---

## Query Processing Flow

```
User Input
    │
    ▼
┌───────────────┐
│ OmniAgentChain │
└───────┬───────┘
        │
        │ 1. Route query
        ▼
   ┌────────┐
   │ Router │ → Target Focus
   └────────┘
        │
        ├────────────────────────┐
        │                    │
        │ 2. Gather Context  │ 3. Extract Goals
        ▼                    ▼
   ┌─────────────┐     ┌───────────────┐
   │Context_     │     │Goal_Detector │
   │Gatherer    │     │              │
   └─────────────┘     └──────┬───────┘
                              │
                              │ Updates SENTER.md
                              │
                              ▼
                    ┌───────────────────────────────┐
                    │                            │
                    │ 4. Read Focus SENTER.md      │
                    │    - Goals                   │
                    │    - Context                 │
                    │    - User Preferences         │
                    │                            │
                    └─────────────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────────┐
                │                            │
                │ 5. Build final prompt        │
                │    - Query                    │
                │    - Focus Context            │
                │    - Goals                    │
                │    - User Preferences         │
                │                            │
                └────────────────────────────────┘
                              │
                              ▼
              ┌─────────────┐
              │ Target Focus│
              │  omniagent  │
              └──────┬─────┘
                     │
                     ▼
            Response
```

---

## Adding New Capabilities

To add ANY new capability to Senter:

1. **Create a new Focus directory**:
   ```bash
   mkdir -p Focuses/MyNewCapability
   ```

2. **Create SENTER.md** with:
   ```yaml
   ---
   model:
     type: gguf  # Inherit from user_profile.json
   
   system_prompt: |
     You are the MyNewCapability Agent.
     Your job: [describe what it does]
   focus:
     type: conversational  # or functional
   ---
   
   ## User Preferences
   
   ## Patterns Observed
   
   ## Goals & Objectives
   
   ## Evolution Notes
   ```

3. **Restart Senter** - it loads automatically!

That's it. No code changes needed.

---

## Key Differences from Old Architecture

### Old (Script-Based)
- ❌ 20+ specialized scripts (~3500 lines)
- ❌ Complex function_agent_generator.py (296 lines)
- ❌ Naive goal detection with pattern matching
- ❌ Goals capped at 3
- ❌ No async/await support
- ❌ Tool discovery via AST parsing
- ❌ Hard to add new capabilities

### New (OmniAgent Chain)
- ✅ ~500 lines total (omniagent_async.py + omniagent_chain.py)
- ✅ Everything is omniagent + SENTER.md
- ✅ LLM-based goal detection
- ✅ Unlimited goals per Focus
- ✅ Full async/await support
- ✅ Tool discovery = omniagent
- ✅ Easy: Create Focus directory + SENTER.md

**Reduction**: 85% less code, 10x more flexible

---

## SENTER.md File Structure

### YAML Frontmatter
```yaml
---
model:
  type: gguf  # or openai, or vllm
  # Optional overrides
  
focus:
  type: internal  # or conversational, or functional
  id: ajson://senter/focuses/...
  name: "Focus Name"
  created: "2026-01-03T00:00:00Z"

system_prompt: |
  You are the Focus Name Agent.
  Your job: [describe purpose]
  [additional instructions]

context:
  type: internal_instructions  # or wiki
  content: |
    [context for this Focus]
---
```

### Markdown Sections

```markdown
## User Preferences
[Preferences detected by Profiler agent]

## Patterns Observed
[Interaction patterns detected by Profiler agent]

## Goals & Objectives
[Goals extracted by Goal_Detector, planned by Planner]

## Evolution Notes
[Timestamped notes about changes]
```

---

## Configuration Files

### user_profile.json
```json
{
  "central_model": {
    "type": "gguf",
    "path": "/path/to/model.gguf",
    "is_vlm": false,
    "settings": {
      "max_tokens": 512,
      "temperature": 0.7
    }
  }
}
```

### senter_config.json
```json
{
  "infrastructure_models": {
    "multimodal_decoder": {...},
    "embedding_model": {...}
  },
  "recommended_models": {...},
  "focus_creation": {...},
  "review_process": {...},
  "learning": {...}
}
```

---

## Async Performance

### Parallel Agent Calls
Multiple omniagents can run simultaneously:
- Router + Goal_Detector (parallel)
- Context_Gatherer + Planner (parallel)
- All Profiler agents (parallel)

### Non-Blocking I/O
- Thread pool for blocking operations
- Async/await for coroutine coordination
- Responsive UI even during heavy processing

### Resource Efficiency
- Lazy model loading
- Configurable thread pool size
- Clean shutdown procedures

---

## Evolution

### Goals
1. Extracted by Goal_Detector
2. Associated with specific Focus
3. Planned by Planner
4. Updated in Focus's SENTER.md

### Context
1. Gathered by Context_Gatherer
2. Summarized and stored in SENTER.md
3. Used by Chat agent for responses

### Preferences
1. Analyzed by Profiler
2. Stored in SENTER.md
3. Adapted responses automatically

---

## Commands

### Interactive
- `/list` - List all Focuses
- `/focus <name>` - Set current Focus
- `/goals` - Show goals for current Focus
- `/discover` - Run tool discovery
- `/exit` - Exit

### Programmatic
```python
from Functions.omniagent_chain import OmniAgentChain
import asyncio

async def example():
    chain = OmniAgentChain(senter_root)
    await chain.initialize()
    
    # Process query
    response = await chain.process_query("Your query here")
    print(response)
    
    # List Focuses
    print(chain.list_user_focuses())
    
    # Close
    await chain.close()

asyncio.run(example())
```

---

## Benefits of This Architecture

1. **Radical Simplicity**: 85% less code
2. **True "Everything is Omniagent"**: Every capability is just an omniagent with SENTER.md
3. **Async Performance**: Parallel agent calls, non-blocking I/O
4. **Unlimited Goals**: No caps, Focus-specific storage
5. **Self-Contained**: All config in SENTER.md
6. **Extensibility**: Add capability = Create Focus + SENTER.md
7. **Intelligent**: LLM-based instead of pattern matching
8. **Maintainable**: Clear structure, easy to debug
9. **Universal**: Same pattern for internal and user Focuses
10. **Dynamic**: Auto-discovery, auto-loading, auto-updating

---

## Files

### Core
- `Functions/omniagent_async.py` - Async wrapper
- `Functions/omniagent_chain.py` - Chain orchestrator
- `Focuses/senter_md_parser.py` - Parser

### Internal Focuses
- `Focuses/internal/Router/SENTER.md`
- `Focuses/internal/Goal_Detector/SENTER.md`
- `Focuses/internal/Tool_Discovery/SENTER.md`
- `Focuses/internal/Context_Gatherer/SENTER.md`
- `Focuses/internal/Planner/SENTER.md`
- `Focuses/internal/Profiler/SENTER.md`
- `Focuses/internal/Chat/SENTER.md`

### User Focuses
- `Focuses/coding/SENTER.md`
- `Focuses/research/SENTER.md`
- `Focuses/creative/SENTER.md`
- `Focuses/user_personal/SENTER.md`
- `Focuses/general/SENTER.md`

### Entry Points
- `scripts/senter.py` - CLI interface
- `scripts/senter_app.py` - TUI interface (needs async update)

### Configuration
- `config/user_profile.json`
- `config/senter_config.json`

---

## Migration Guide

### From Old Senter
1. Backup existing configurations
2. Run setup to create new Focuses
3. Tools in Functions/ become Focuses automatically
4. Goals migrate to per-Focus storage
5. Profile data rebuilds over time

### Adding New Focuses
1. Create directory: `Focuses/NewName/`
2. Create `SENTER.md` with proper structure
3. Add system prompt describing purpose
4. Restart Senter

---

**Created**: January 3, 2026
**Version**: 2.0
**Status**: Implementation Complete, Testing In Progress
