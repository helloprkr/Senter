# SENTER.md Universal Format Specification

## Overview

Every agent in Senter is defined by a **single SENTER.md file** with a universal format. This enables:
- Self-documenting system
- Automatic agent creation
- Easy extensibility
- Consistent behavior across all Focuses

---

## Format Structure

### Part 1: YAML Frontmatter

Bounded by `---` delimiters, contains configuration data.

```yaml
---
model:
  type: gguf|openai|vllm
  path: /path/to/model.gguf              # GGUF only
  endpoint: http://localhost:8000         # OpenAI/vLLM only
  model_name: model-name                  # OpenAI/vLLM only
  n_gpu_layers: -1                       # GGUF only
  n_ctx: 8192                           # Context window
  max_tokens: 512
  temperature: 0.7
  is_vlm: false                           # Vision capability

focus:
  type: internal|conversational|functional
  id: ajson://senter/focuses/<name>
  name: Human-Readable Name
  created: 2026-01-04T00:00:00Z
  version: 1.0

system_prompt: |
  Multi-line system prompt defining agent behavior
---
```

**Required Fields:**
- `model.type`: Model backend (gguf/openai/vllm)
- `focus.type`: Agent type (internal/conversational/functional)
- `focus.id`: Unique identifier
- `focus.name`: Display name
- `system_prompt`: Agent's personality and behavior

### Part 2: Markdown Context Sections

Optional sections that other agents read and update.

```markdown
## User Preferences
[To be populated by Profiler]

## Patterns Observed
[To be populated by Profiler]

## Goals & Objectives
[To be populated by Goal_Detector]

## Evolution Notes
[To be populated by Profiler]

## Function Metadata (functional Focuses only)
functions:
  - name: function_name
    description: What it does
    parameters: [list]
    returns: type

## Tool Information (tool Focuses only)
tool_name: <name>
tool_path: /path/to/tool
usage_examples:
  - Example usage
```

---

## Focus Types

### 1. Internal Agents
**Purpose**: Senter orchestration and management

**Examples:**
- Router: Routes queries to appropriate Focus
- Goal_Detector: Extracts goals from conversations
- Context_Gatherer: Updates Focus context
- Tool_Discovery: Finds and configures new tools
- Profiler: Analyzes user patterns
- Planner: Breaks down goals into tasks
- Chat: Main conversational orchestrator

**System Prompt Requirements:**
- Clear input/output format
- Specific logic instructions
- JSON output format where applicable

### 2. Conversational Focuses
**Purpose**: User-facing specialized assistance

**Examples:**
- general: Catch-all queries
- coding: Programming and development
- research: Information gathering
- creative: Artistic and creative work
- user_personal: Personal organization

**System Prompt Requirements:**
- Friendly and helpful tone
- Clear expertise boundaries
- When to redirect to other Focuses
- Response style guidelines

### 3. Functional Focuses
**Purpose**: Single-purpose tool integration

**Examples:**
- image_generator: Generate images
- encrypt_file: Encrypt files
- compose_music: Create music

**System Prompt Requirements:**
- Tool usage instructions
- Parameter descriptions
- Usage examples
- Error handling

---

## Parser Behavior

The `senter_md_parser.py` handles:

1. **YAML Frontmatter Parsing**
   - Extracts configuration data
   - Validates required fields
   - Provides defaults for missing values
   - Resolves model config chain (SENTER.md → user_profile.json → senter_config.json)

2. **System Prompt Extraction**
   - Returns full system prompt string
   - Used by omniagent instances
   - Defines agent behavior

3. **Markdown Section Updates**
   - `update_markdown_section()`: Update specific sections
   - `update_wiki()`: Append to wiki.md
   - Preserve YAML frontmatter
   - Maintain markdown formatting

4. **Context Retrieval**
   - `get_focus_context()`: Get context configuration
   - `get_focus_functions()`: Get function metadata
   - Used by other internal agents

---

## SENTER.md Writer Agent

**Location**: `Focuses/internal/SENTER_Md_Writer/SENTER.md`

**Purpose**: Generate valid SENTER.md files automatically

**When Called:**
1. **Tool Discovery**: New Python function found in Functions/
   - Analyze function signature
   - Determine Focus type
   - Generate appropriate SENTER.md
   - Create Focus directory

2. **User Request**: "Create a Focus for X"
   - Ask clarifying questions
   - Generate SENTER.md with proper config
   - Create directory structure

3. **Context Updates**: Focus needs SENTER.md update
   - Read existing file
   - Update relevant sections
   - Write updated SENTER.md

4. **Migration**: Converting legacy formats
   - Parse old agent.json
   - Generate new SENTER.md
   - Maintain backward compatibility

---

## Future: SENTER.md Specialized Model

**Vision**: Train a specialized model using Unsloth to generate perfect SENTER.md files.

**Training Data:**
- All existing SENTER.md files (internal, conversational, functional)
- High-quality examples across all Focus types
- Standardized format examples

**Benefits:**
- Faster SENTER.md generation
- Higher quality configurations
- Consistent format adherence
- Self-improving agent ecosystem

**Integration:**
- Replace SENTER_Md_Writer's generic model
- Fine-tune on SENTER.md corpus
- Continuous improvement as Senter grows

---

## Example: Complete SENTER.md File

```yaml
---
model:
  type: gguf
  
focus:
  type: conversational
  id: ajson://senter/focuses/coding
  name: coding
  created: 2026-01-04T00:00:00Z

system_prompt: |
  You are Coding Focus Agent for Senter, specializing in programming and software development.
  
  Your mission: Provide expert assistance with all aspects of coding and development.
  
  ## Your Expertise
  - Programming in any language
  - Debugging and troubleshooting
  - Code review and optimization
  - Software architecture and design patterns
  
  ## Capabilities
  - Write clean, well-commented code
  - Debug and fix bugs
  - Refactor and optimize existing code
  - Explain complex code and concepts
  
  ## Response Style
  - Provide clear, working code examples
  - Explain "why" behind solutions
  - Use proper formatting for code blocks
  - Ask clarifying questions when requirements are vague
---

# Coding Context

## User Preferences
[Programming languages, coding style, etc.]

## Patterns Observed
[Common languages, project types, etc.]

## Goals & Objectives
[Learning goals, project deadlines, etc.]

## Evolution Notes
[To be updated by Profiler agent over time]
```

---

## Benefits of Universal Format

1. **Self-Organizing**: New agents can be created automatically
2. **Self-Documenting**: Every agent documents its own configuration
3. **Easy Extensibility**: Add new Focus by creating SENTER.md
4. **Model-Agnostic**: Works with any model backend
5. **Consistent Behavior**: Same parsing for all Focuses
6. **Automated Maintenance**: Agents can update each other's configs

---

## Implementation Checklist

When implementing SENTER.md support:

- [x] Create `senter_md_parser.py` with YAML frontmatter parsing
- [x] Support model config resolution chain
- [x] Implement markdown section updates
- [x] Create SENTER_Md_Writer agent
- [ ] Update Tool_Discovery to use SENTER_Md_Writer
- [ ] Update Context_Gatherer to use SENTER_Md_Writer
- [ ] Create test suite for SENTER.md generation
- [ ] Train specialized model with Unsloth (future)

---

This universal format enables Senter to be a **truly self-maintaining AI system** where every agent can create, configure, and maintain other agents automatically.
