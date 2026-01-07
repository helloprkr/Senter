# Senter - AI Personal Assistant

**Version:** 2.0.0  
**Last Updated:** January 2, 2026  
**Status:** Development - Focus System Architecture Complete

---

## Table of Contents

1. [Purpose](#purpose)
2. [Architecture Overview](#architecture-overview)
3. [System Flow](#system-flow)
4. [Implementation Details](#implementation-details)
5. [Current State](#current-state)
6. [Key Components](#key-components)
7. [Configuration](#configuration)
8. [Testing](#testing)
9. [Roadmap](#roadmap)

---

## Purpose

Senter is a **universal AI personal assistant** built with a **Focus-based agent architecture**. Unlike traditional AI assistants with fixed categories, Senter uses **dynamic Focus creation** - automatically creating and organizing Focuses based on user interactions. It combines multimodal AI capabilities (text, vision, audio, video) with intelligent agent orchestration through a modern terminal interface.

### Core Philosophy

- **Focus-First Architecture**: Everything is a Focus - defined in SENTER.md with mixed YAML + Markdown format
- **Dynamic Focus Creation**: No fixed categories - Focuses are created on-demand based on user interests
- **Model-Agnostic**: User brings their own model - Senter straps multimodality and intelligent search onto it
- **Self-Learning**: Each Focus evolves through SENTER.md files with user preferences, goals, and context
- **Privacy-First**: All processing happens locally - no data leaves your machine
- **Extensible**: Easy to add new agents, functions, and capabilities

### Key Capabilities

| Capability | Status | Description |
|------------|---------|-------------|
| 🤖 Text Generation | ✅ Complete | Conversational AI with user's chosen model |
| 🖼️ Image Understanding | ✅ Complete | Describe and analyze images |
| 🎵 Audio Processing | ✅ Complete | Speech recognition, music analysis |
| 📺 Video Processing | ✅ Complete | YouTube downloads and analysis |
| 🎨 Image Generation | ✅ Complete | Text-to-image via Qwen Image GGUF |
| 🎵 Music Generation | ✅ Complete | Text-to-music via ACE-Step |
| 🎯 Topic Routing | ✅ Complete | Intelligent Focus/agent selection |
| 🧠 Embedding Search | ✅ Complete | Nomic embed similarity filtering |
| 🖥️ TUI Interface | ✅ Complete | Beautiful terminal UI with Textual |
| 📚 Self-Learning | ✅ Complete | Continuous evolution through SENTER.md |
| 🔧 Self-Healing | ✅ Complete | Error detection and automatic fixing |
| 🧩 User Profiling | ✅ Complete | Psychology-based personality and goal detection |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Senter System 2.0.0                    │
└─────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                      │
     ┌────▼────┐        │      ┌────▼──────┐    │
     │   TUI    │        │      │  Backend      │    │
     │  Layer   │        │      │  Layer        │    │
     └────┬────┘        │      │      └────┬──────┘    │
          │                    │                      │                 │
          │         ┌──────────────┴──────────────────────┐
          │         │                                 Storage & Context │
          │         │                     ┌──────────────────┴────────┐
          │         │                     │                     │          │
          │         │   Focuses/     │      Config/    │          │
          │         │     Agents/      │      │          │
          │         └──────────────────────────────────────────┘
```

### Layer Descriptions

#### 1. **TUI Layer** (scripts/senter_app.py, senter_widgets.py)
- **Purpose**: Modern terminal interface with Textual framework
- **Components**:
  - Chat panel with message history
  - Modular sidebar (Focuses, Goals, Tasks, Calendar)
  - Searchable Focus explorer (replaces Topic explorer)
  - Inline editing for all items
  - Modal dialogs for adding/editing Focuses
  - Wiki display for conversational Focuses

#### 2. **Backend Layer** (scripts/senter.py, senter_selector.py)
- **Purpose**: Core orchestration and intelligent routing
- **Components**:
  - Senter main orchestrator
  - Dynamic Focus selection with creation
  - Embedding-based filtering (top 4 Focuses)
  - LLM-powered final selection with CREATE_NEW option
  - Focus-to-agent routing

#### 3. **Functions Layer** (Functions/)
- **Purpose**: Reusable AI pipelines
- **Modules**:
  - `omniagent.py` - Universal multimodal processing
  - `senter_md_parser.py` - SENTER.md parsing (YAML + Markdown)
  - `focus_factory.py` - Dynamic Focus creation
  - `review_chain.py` - Background Focus review system
  - `self_healing_chain.py` - Error detection and fix
  - `embedding_utils.py` - Vector search utilities
  - `compose_music.py` - Music generation
  - `qwen_image_gguf_generator.py` - Image generation

#### 4. **Storage & Context Layer** (Focuses/, Agents/, config/)
- **Focuses/**: Dynamic Focus directories with SENTER.md
  - **internal/**: Senter's internal Focuses (Reviewer, Planner, Coder, Profiler, etc.)
- **Agents/**: Legacy agent JSON manifests (being phased out in favor of Focuses)
- **config/**: System configuration files

---

## System Flow

### 1. User Interaction Flow

```
User Input
    │
    ├─────────────────────────────────────────────────────────┐
    │                                             │
    ▼
[Route to Focus]
    │
    │
    ┌────────────────────┼────────────────────┐
    │                    │                      │
 ▼
[Load Focus Config]
    │
    │                      │
    ▼
[Get Agent Response]
    │
    │                      │
    ▼
[Update Focus Context]
    │
    │
    │
    │         ┌────────────┴──────────────────────┐
    │
    │
    └─────────────┬───────────────────────────┘
                 │
                 ▼
           User Response
```

### 2. Dynamic Focus Creation Flow

```
User: "Tell me about Solana price"
    │
    ▼
[Intelligent Selection]
    │
    ├────────────────────┼────────────────────┐
    │                    │                      │
 ▼
[Embed Filter]
    ├────────────────────┴──────────────────────┐
│  Load all Focuses with SENTER.md
│  Create embeddings of Focus descriptions
│  Vector search: Query → Top 4 Focuses
    └─────────────────────────────────────────┘
    │
    ▼
[Top 4 Candidates + Scores]
│  - Solana (0.89)
│  - Bitcoin (0.75)
│  - Research (0.68)
│  - General (0.62)
    └───────────────────────────────────────────────────┘
    │
    ▼
[LLM Selection with CREATE_NEW Option]
    ├───────────────────────────────────────────────────────┐
    │  Prompt: "Select best Focus. If ALL have low confidence
    │           (<0.5), you may CREATE_NEW:Focus_name"
    │
    │  Response: "Low confidence in all. CREATE_NEW:Solana"
    │  └───────────────────────────────────────────────────────┘
    │
    ▼
[FocusFactory: Create New Focus]
│  Create Focuses/Solana/
│  Generate SENTER.md with user's default model
│  Create wiki.md (conversational Focus)
    └──────────────────────────────────────────────────────┘
    │
    ▼
"Created new Focus: Solana"
```

### 3. Focus Review Flow (Background)

```
┌─────────────────────────────────────────────────────────┐
│       Background Review Process (always running)        │
└─────────────────────────────────────────────────────────┘
                    │
          ┌────────────┴──────────────────────┐
          │   New Data from:
          │   - Web search
          │   - Chat interactions
          │   - Function outputs
          └─────────────────────────────────────────┘
                    │
                    ▼
         [Focus_Reviewer omniagent]
                    │
                    │
          Analyze: "Should this Focus be updated, merged, split?"
                    │
                    │
          Response: {action: "update", reasoning: "...", confidence: 0.7}
                    │
                    │
          └──────────────────────────────────────────────┘
                    │
                    ▼
         [Update Focus SENTER.md]
                    │
                    └──────────────────────────────────────────────┘
```

### 4. Self-Healing Chain Flow

```
[Function Error Detected]
    │
    ▼
[Self-Inference: Analyze Problem]
    │
    ├───────────────────────────────────────────────────────┐
    │  "Error: WiFi light script failed at line 45"
    │  └───────────────────────────────────────────────────────┘
    │
    ▼
[Planner: Create Fix Plan]
    │
    "Step 1: Update error handling in script"
    "Step 2: Add retry logic with backoff"
    │
    │
    └──────────────────────────────────────────────────────┘
    │
    ▼
[Coder: Write Fix Code]
    │
    "```python
    def connect_lights():
    # Fixed version with proper error handling
    ```
    │
    │
    └───────────────────────────────────────────────────────┘
    │
    ▼
[Update Focus SENTER.md with Fix]
    └──────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Project Structure

```
Senter/
├── Focuses/                    # Dynamic Focus system (replaces Topics/)
│   ├── internal/              # Senter's internal Focuses
│   │   ├── Focus_Reviewer/SENTER.md
│   │   ├── Focus_Merger/SENTER.md
│   │   ├── Focus_Splitter/SENTER.md
│   │   ├── Planner_Agent/SENTER.md
│   │   ├── Coder_Agent/SENTER.md
│   │   ├── User_Profiler/SENTER.md
│   │   ├── Diagnostic_Agent/SENTER.md
│   │   └── Chat_Agent/SENTER.md
│   ├── senter_md_parser.py   # YAML + Markdown parser
│   ├── focus_factory.py        # Dynamic Focus creation
│   ├── review_chain.py          # Background review system
│   └── self_healing_chain.py   # Error detection & fix
├── config/                    # System configuration
│   ├── senter_config.json     # Infrastructure models + recommended models
│   └── user_profile.json        # User's model & preferences
├── scripts/                    # Application code
│   ├── senter_app.py           # TUI application
│   ├── senter_selector.py      # Intelligent Focus selection
│   ├── senter.py              # Main orchestrator (updated with Focuses)
│   ├── senter_widgets.py      # UI components
│   ├── setup_internal_focuses.py # Internal Focus initialization
│   └── setup_senter.py          # Main setup script
├── Agents/                     # Legacy agent manifests
├── Functions/                  # Reusable pipelines
└── Models/                     # Downloaded AI models
```

### Focus System (Key Innovation)

The **Focus system** replaces both fixed Topics and monolithic Functions:

| Type | Description | Example | Has Wiki? |
|------|-------------|---------|----------|
| **Conversational Focus** | Interest/topic with research and discussion | Bitcoin, AI, Coding | ✅ Yes |
| **Functional Focus** | Single-purpose task execution | WiFi Lights, Calendar, Todo | ❌ No |
| **Internal Focus** | Senter's own operation | Planner, Coder, Reviewer | ❌ No |

---

## Current State

### ✅ Completed Features

#### Core System
- ✅ **Focus System Architecture** - Dynamic Focus creation with SENTER.md format
- ✅ **SENTER.md Parser** - Mixed YAML + Markdown parsing for all configurations
- ✅ **Focus Factory** - Automatic Focus creation with user's model
- ✅ **Model-Agnostic Config** - User brings their own model, infrastructure models fixed
- ✅ **Review Chain** - Background Focus review with omniagent instances
- ✅ **Self-Healing Chain** - Error detection → Planner → Coder → Update
- ✅ **Internal Focuses** - 8 internal Focuses created (Reviewer, Merger, Splitter, Planner, Coder, Profiler, Diagnostic, Chat)

#### Configuration System
- ✅ **senter_config.json** - Infrastructure models + recommended models (NOT hardcoded)
- ✅ **user_profile.json** - User configuration template with central model setup
- ✅ **setup_senter.py** - Main setup script with optional model downloads

#### File Structure Migration
- ✅ **Topics → Focuses** - All existing Topics migrated to Focuses/ directory
- ✅ **Scripts Updated** - senter.py and senter_selector.py updated for Focus system

### 🚧 Remaining Work

| Component | Status | Description |
|-----------|--------|-------------|
| **omniagent.py Refactor** | 🚧 Pending | Model-agnostic loading with VLM bypass |
| **UI Updates** | 🚧 Pending | Update senter_app.py for Focus display |
| **Widget Updates** | 🚧 Pending | Update senter_widgets.py for Focus system |

---

## Key Components

### 1. SenterOmniAgent (Functions/omniagent.py)

**Purpose**: Universal orchestrator - works with ANY user-provided model

**Architecture**:
```
User Input → [Stage 1: Omni 3B (Multimodal Decoder)] → [Stage 2: User's Model (Reasoning)]
                    ↓
                 [Optional VLM Bypass: If user's model has vision, send image directly]
```

**Key Changes Needed**:
- Model-agnostic loading: Support GGUF, OpenAI API, vLLM
- VLM detection: Auto-detect if user's model supports vision
- Bypass Omni 3B for images when user's model is VLM

### 2. Focus System (Focuses/)

**Purpose**: Dynamic, self-organizing Focuses based on user interactions

**Focus Types**:
- **Conversational Focuses** (with wiki.md):
  - User interests (Bitcoin, AI, Coding)
  - Research topics with evolving knowledge base
  - Example: `Focuses/Bitcoin/SENTER.md`, `Focuses/Bitcoin/wiki.md`

- **Functional Focuses** (no wiki):
  - Single-purpose tasks
  - Direct function execution
  - Example: `Focuses/Wifi_Lights/SENTER.md`

**Dynamic Creation Flow**:
```
User Query
    ↓
[Intelligent Selection]
    ↓
[Embed Filter: Nomic Embed → Top 4 Focuses]
    ↓
[LLM Selection: With CREATE_NEW option if all top 4 have low confidence (<0.5)]
    ↓
[Decision: Select existing Focus OR CREATE_NEW:Focus_Name]
    ↓
[FocusFactory: Create new Focus]
```

### 3. SENTER.md Format

**Mixed YAML + Markdown Structure**:

```yaml
---
manifest_version: "1.0"
focus:
  name: "Focus Name"
  id: "ajson://senter/focuses/focus_name"
  type: "conversational" | "functional"
  created: "2025-01-02T00:00:00Z"

model:
  type: "gguf" | "openai" | "vllm" | null
  # If null, use user_profile.json central_model
  path: "/path/to/model.gguf"
  endpoint: "https://api.openai.com/v1"
  is_vlm: true | false
  context_window: 8192
  max_tokens: 512
  temperature: 0.7

settings:
  max_tokens: 512
  temperature: 0.7

system_prompt: |
  You are Senter's agent for this Focus.
  Your purpose is to assist the user with anything related to Focus Name.
  Use the provided context and wiki to give helpful, accurate responses.

functions:
  - name: "function_name"
    script: "path/to/script.py"
    description: "Function description"

ui_config:
  show_wiki: true | false
  widgets:
    - type: "calendar"
    - type: "todo_list"

context:
  type: "internal_instructions" | "wiki" | "research"
  content: |
    Internal instructions or wiki content

---

# Markdown Sections (Human-Editable)

## Detected Goals
(List of proposed/confirmed goals)

## Explorative Follow-Up Questions
(List of questions to validate goals)

## Wiki Content (for conversational Focuses)
(User-facing content that updates live)
```

### 4. Intelligent Selection (senter_selector.py)

**Two-Stage Selection Process**:

1. **Embed Filtering (Stage 1)**:
   - Uses Nomic Embed model (infrastructure)
   - Filters all Focuses down to top 4 most similar
   - Returns with similarity scores

2. **LLM Selection (Stage 2)**:
   - Uses user's model (NOT Omni 3B)
   - Enhanced prompt with CREATE_NEW option
   - Creates new Focus if top 4 all have low confidence (<0.5)

**CREATE_NEW Trigger**:
```
"IMPORTANT: Create new Focus if ALL of the top 4 Focuses have low confidence (<0.5).
New Focus name should be descriptive and concise.
```

### 5. Review Chain (Focuses/review_chain.py)

**Background "Awareness" System**

Components:
- **Focus_Reviewer**: Analyzes Focuses and determines update/merge/split needed
- **Focus_Merger**: Combines overlapping Focuses
- **Focus_Splitter**: Splits overly diverse Focuses
- **Planner_Agent**: Creates step-by-step plans for goals
- **Coder_Agent**: Writes fixes for function errors
- **User_Profiler**: Psychology-based personality and goal detection
- **Diagnostic_Agent**: Error analysis and classification

**All use omniagent instances** configured by SENTER.md

### 6. Self-Healing Chain (Focuses/self_healing_chain.py)

**Automatic Error Detection and Fixing**:

Flow:
```
Function Error
    ↓
[Self-Inference: Diagnostic_Agent]
    ↓
[Planner: Create Fix Plan]
    ↓
[Coder: Write Fix Code]
    ↓
[Update Focus SENTER.md with Fix]
```

### 7. Internal Focuses (8 Created)

| Focus | Purpose | System Prompt |
|--------|---------|----------------|
| **Focus_Reviewer** | Review Focuses for updates/merges/splits | Analyze Focus and determine action |
| **Focus_Merger** | Merge overlapping Focuses while preserving important information | Combine two Focuses into unified version |
| **Focus_Splitter** | Split diverse Focuses into more focused sub-Focuses | Identify when a Focus has grown too diverse, suggest sub-Focuses |
| **Planner_Agent** | Break down user goals into actionable steps | Create step-by-step plans for goals |
| **Coder_Agent** | Write and fix code for Senter's functions | Analyze errors and produce fix code |
| **User_Profiler** | Psychology-based personality and goal detection | Analyze user interactions for goals, personality, humor |
| **Diagnostic_Agent** | Analyze function errors and determine severity | Classify errors and recommend fixes |
| **Chat_Agent** | Final response agent with personality injection | Always-follows agent that provides final unified response to user |

### 8. Configuration

#### senter_config.json (Model-Agnostic)

```json
{
  "name": "Senter",
  "version": "2.0.0",
  "focuses_dir": "Focuses",
  "outputs_dir": "outputs",

  "infrastructure_models": {
    "multimodal_decoder": {
      "path": "/path/to/Qwen2.5-Omni-3B-Q4_K_M.gguf",
      "mmproj": "/path/to/mmproj-Q8_0.gguf",
      "description": "Fixed infrastructure - Omni 3B for multimodal decoding ONLY (not for reasoning)"
    },
    "embedding_model": {
      "path": "/path/to/nomic-embed-text-v1.5-GGUF",
      "description": "Fixed infrastructure - Embeddings for intelligent search"
    }
  },

  "recommended_models": {
    "hermes_3b": {
      "name": "Hermes 3 Llama 3.2 3B",
      "url": "https://huggingface.co/nousresearch/Hermes-3-Llama-3.2-3B/resolve/main/Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
      "description": "Fast, efficient text model",
      "size_gb": 2.1,
      "is_vlm": false
    },
    "qwen_vl_8b": {
      "name": "Qwen VL 8B",
      "url": "https://huggingface.co/Qwen/Qwen2-VL/resolve/main/Qwen2-VL-8B-Q4_K_M.gguf",
      "description": "Vision + text model",
      "size_gb": 5.4,
      "is_vlm": true
    }
  },

  "focus_creation": {
    "embed_filter_threshold": 4,
    "low_confidence_threshold": 0.5,
    "allow_dynamic_creation": true
  },

  "review_process": {
    "focus_review_interval": 60,
    "user_profile_interval": 60,
    "self_heal_interval": 30,
    "merge_confidence_threshold": 0.7
  },

  "ui": {
    "theme": "matrix_green",
    "show_internal_processes": false
  },

  "system": {
    "max_parallel_processes": 2,
    "context_window": 32768
  }
}
```

**Key Principle**: **User brings their own model** - Senter is model-agnostic except for:
- ✅ Omni 3B (multimodal decoder) - **FIXED infrastructure**
- ✅ Nomic Embed (intelligent search) - **FIXED infrastructure**
- ❌ Hermes 3B and Qwen VL 8B are **OFFERED** (not hardcoded in code)

---

## Configuration

### user_profile.json (User Configuration Template)

```json
{
  "user_id": "default_user",
  "name": "Default User",

  "central_model": {
    "type": "gguf",
    "path": "/path/to/user/model.gguf",
    "is_vlm": false,
    "settings": {
      "max_tokens": 512,
      "temperature": 0.7,
      "context_window": 8192
    }
  },

  "preferences": {
    "response_style": "balanced",
    "detail_level": "moderate",
    "creativity_level": 0.7,
    "technical_level": "intermediate",
    "language": "en",
    "timezone": "UTC"
  },

  "agent_preferences": {
    "default_agent": "senter",
    "favorite_focuses": [],
    "blocked_focuses": [],
    "custom_settings": {}
  },

  "learning_profile": {
    "topics_of_interest": ["programming", "ai", "music", "art"],
    "skill_level": {
      "technical": "intermediate",
      "creative": "advanced",
      "analytical": "advanced"
    },
    "learning_goals": [
      "Improve programming skills",
      "Learn AI development",
      "Enhance creative writing"
    ]
  },

  "context_settings": {
    "chat_history_length": 10,
    "topic_memory_days": 30,
    "max_context_per_topic": 5000
  },

  "privacy_settings": {
    "data_retention": true,
    "analytics_enabled": false,
    "share_learnings": false
  },

  "review_settings": {
    "confidence_threshold": 0.5,
    "merge_threshold": 0.7
  },

  "theme": "matrix_green",
  "setup_complete": false
}
```

**Setup Process**:
```bash
cd /home/sovthpaw/ai-toolbox/Senter
python3 setup_senter.py
```

This guides users through:
1. Download infrastructure models (Omni 3B + Nomic Embed)
2. Configure central model:
   - Option A: Download recommended model (Hermes 3B or Qwen VL 8B)
   - Option B: Use existing local model
   - Option C: Use OpenAI-compatible API endpoint
3. Verify setup
4. Run Senter

---

## Testing

### Component Tests

```bash
cd /home/sovthpaw/ai-toolbox/Senter/Focuses/internal
python3 setup_internal_focuses.py

cd /home/sovthpaw/ai-toolbox/Senter/scripts
python3 setup_senter.py
```

### Manual Testing

```bash
# Test dynamic Focus creation
python3 -c "
from senter_selector import select_focus_and_agent
focus, agent, reasoning = select_focus_and_agent(
    'Tell me about blockchain technology',
    ['general', 'coding', 'creative', 'Bitcoin', 'research'],
    confidence_threshold=0.5
)
print(f'Selected: {focus}, agent: {agent}')
"

# Test Focus creation
python3 -c "
from senter_factory import FocusFactory
factory = FocusFactory('/home/sovthpaw/ai-toolbox/Senter')
factory.create_focus('Blockchain Technology', 'User wants to understand blockchain concepts')
print('Focus created')
"
```

---

## Roadmap

### Phase 1: Core Enhancements (Current)
- [ ] Complete omniagent.py model-agnostic refactor
- [ ] Update senter_app.py UI for Focus system
- [ ] Update senter_widgets.py for Focus display
- [ ] Implement VLM detection and bypass
- [ ] Test all components together

### Phase 2: Advanced Features (Future)
- [ ] Goal-aware workflow with Planner agent
- [ ] Enhanced User Profiler with personality mirroring
- [ ] Advanced self-healing with predictive error prevention
- [ ] Multi-user collaboration
- [ ] Plugin system for custom integrations
- [ ] API server mode for external apps
- [ ] Browser-based UI option

### Phase 3: Intelligence (Research)
- [ ] Advanced goal inference with psychological modeling
- [ ] Multi-step reasoning chains
- [ ] Long-term memory system
- [ ] Predictive task suggestions
- [ ] Enhanced Focus merging/splitting logic

---

## Migration Guide

### For Users Upgrading from v1.0

If upgrading from an older version of Senter:

1. **Backup your data**:
   ```bash
   cp -r Topics/ Topics_backup/
   cp -r config/ config_backup/
   cp -r user_profile.json user_profile_backup.json
   ```

2. **Run setup script**:
   ```bash
   cd Senter
   python3 setup_senter.py
   ```

3. **Topic → Focus migration** (automatic):
   - Old `Topics/` directory structure is automatically migrated to `Focuses/`
   - SENTER.md files are created/updated automatically
   - Wiki files are created for conversational Focuses

4. **Verify your configuration**:
   ```bash
   # Check Focuses were created
   ls Focuses/
   
   # Check config was updated
   cat config/user_profile.json
   ```

---

## Development Guidelines

### Code Style

- **Language**: Python 3.10+ with strict type hints
- **Formatting**: PEP 8 standards, 4-space indentation
- **Documentation**: Mandatory docstrings for all modules, classes, and public methods
- **Imports**: Grouped: Standard Library → Third Party → Local
- **Naming**: `PascalCase` for classes, `snake_case` for functions/variables
- **Error Handling**: Use specific exceptions (e.g., `ImportError`, `ValueError`)
- **File Paths**: Use `pathlib.Path` over `os.path` where possible

### Adding New Focuses

1. **Create Focus via setup or CLI**:
   ```bash
   python3 -c "
from senter import Senter
senter = Senter()
success = senter.create_focus('MyTopic', 'Initial context')
print(f'Focus created: {success}')
"
   ```

2. **Manually create SENTER.md**:
   ```bash
   mkdir -p Focuses/MyTopic
   cat > Focuses/MyTopic/SENTER.md << 'EOF'
---
manifest_version: "1.0"
focus:
  name: "MyTopic"
  type: "conversational"

model:
  type: null  # Uses user's default model

system_prompt: |
  You are Senter's agent for MyTopic.
  Assist the user with anything related to MyTopic.

ui_config:
  show_wiki: true

context:
  type: "wiki"
  content: |
    Initial context for this Focus

## Detected Goals
*None yet*

## Explorative Follow-Up Questions
*None yet*

## Wiki Content
# MyTopic

Initial context for this Focus.
EOF
   ```

3. **Internal Focuses** - Automatically created by setup script:
   - Focus_Reviewer: Reviews Focuses for updates/merges/splits
   - Focus_Merger: Combines overlapping Focuses
   - Focus_Splitter: Splits diverse Focuses
   - Planner_Agent: Creates step-by-step plans for goals
   - Coder_Agent: Writes fixes for function errors
   - User_Profiler: Psychology-based personality and goal detection
   - Diagnostic_Agent: Analyzes function errors
   - Chat_Agent: Final response agent with personality injection

### Model Configuration

**User's Model** (configurable in `user_profile.json`):
- Hermes 3 Llama 3.2 3B (2.1GB, fast, efficient)
- Qwen VL 8B (5.4GB, vision + text)
- Or any custom GGUF model
- Or OpenAI-compatible API endpoint (OpenAI, OpenRouter, local vLLM)

**Infrastructure Models** (Fixed, in `senter_config.json`):
- **Omni 3B** (Qwen2.5-Omni-3B-Q4_K_M): Multimodal decoder ONLY
  - Purpose: Decode text, images, audio, video into descriptions
  - **Does NOT do reasoning** - user's model handles all thinking
- **Nomic Embed** (nomic-embed-text-v1.5-GGUF): Intelligent similarity search
  - Used for Focus selection and content filtering

**VLM Bypass**:
If user's model has vision capabilities (like Qwen VL 8B):
- Images go directly to user's model (bypass Omni 3B)
- Omni 3B only decodes non-visual modalities (audio-only video, etc.)

---

## Troubleshooting

### Common Issues

**Issue**: "No Focuses found" after setup
- **Solution**: Run `python3 scripts/setup_internal_focuses.py` to create internal Focuses

**Issue**: "Model not configured" error
- **Solution**: Run `python3 setup_senter.py` to configure your model

**Issue**: "Dynamic Focus creation not working"
- **Solution**: Ensure `user_profile.json` has valid `central_model` configuration

**Issue**: "Focus SENTER.md parsing errors"
- **Solution**: Ensure YAML frontmatter is properly formatted with `---` delimiters

**Issue**: "Internal agents not responding"
- **Solution**: Check that internal Focuses were created and SENTER.md files are valid

---

## Key Differences from v1.0

| Feature | v1.0 | v2.0 |
|---------|-------|--------|
| **Topic System** | Fixed categories in Topics/ | **Dynamic Focuses** in Focuses/ |
| **Model Support** | Hardcoded Hermes/Qwen in code | **User brings their own model** |
| **SENTER.md Format** | JSON-only or Markdown | **Mixed YAML + Markdown** |
| **Focus Creation** | Manual only | **Dynamic with intelligent selection** |
| **Internal Agents** | None | **8 internal Focuses with omniagent** |
| **Review System** | None | **Background review chain** |
| **Self-Healing** | None | **Error detection → Planner → Coder** |

---

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-change`
3. **Make your changes**
4. **Follow code style guidelines**
5. **Add tests for new features**
6. **Update this documentation**
7. **Submit a pull request**

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

Built with amazing open-source projects:
- **Qwen Models** (Alibaba Cloud) - Multimodal LLM (Omni 3B for decoding)
- **Hermes 3** (Nous Research) - Text model (optional)
- **nomic-ai** - Text embeddings for intelligent search
- **Textual** - Terminal user interface framework
- **llama-cpp-python** - GGUF model inference
- **ACE-Step** - Music generation model (optional)
- **JSON Agents Specification** - Agent manifest standard

---

**Senter Version 2.0.0** - Universal AI Personal Assistant with Focus-Based Self-Learning

*For questions or feedback, visit the repository or open an issue.*
