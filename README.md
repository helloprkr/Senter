# Senter 3.0

Unified Personal AI with Cognitive Coupling, Evolution, and 24/7 Operation.

## Quick Start

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run Senter (interactive CLI)
python senter.py

# Or use the run script
./run.sh
```

## Features

### Core Capabilities
- **Cognitive Coupling**: Bidirectional modeling of human and AI state
- **Living Memory**: 4-layer memory system (semantic, episodic, procedural, affective)
- **Real Evolution**: Self-improvement through mutations to genome.yaml
- **Trust Tracking**: Persistent trust level affects proactive behavior

### New in 3.0
- **Daemon Mode**: 24/7 background service with task queue
- **Parallel Inference**: Foreground and background model processing
- **Voice Input**: Whisper-based speech-to-text
- **Gaze Detection**: Wake-word-free activation via camera
- **Goal Detection**: Automatic detection of user goals from conversations
- **Proactive Suggestions**: Context-aware suggestions based on goals and patterns

## Usage Modes

### Interactive CLI (Default)
```bash
python senter.py
```

### Daemon Mode (24/7 Operation)
```bash
# Start daemon
python senter.py --daemon

# Connect to daemon from another terminal
python senter.py --connect

# Check daemon status
python senter.py --status
```

### Voice + Gaze Mode
```bash
# Requires: pip install openai-whisper sounddevice mediapipe opencv-python
python senter.py --voice
```

## Commands

Inside Senter:
- `/status` - Show system status
- `/memory` - Show memory statistics
- `/trust` - Show trust level
- `/capabilities` - List available capabilities
- `/goals` - Show detected user goals
- `/evolution` - Show mutation history
- `/suggest` - Get proactive suggestions
- `/help` - Show help
- `quit` or `exit` - Exit Senter

## Requirements

- Python 3.10+
- Ollama running locally with `llama3.2` model (or OpenAI API key)

### Optional Dependencies
```bash
# For voice input
pip install openai-whisper sounddevice

# For gaze detection
pip install mediapipe opencv-python
```

## Configuration

Edit `genome.yaml` to configure:
- **Models**: Switch between Ollama, OpenAI, or local GGUF
- **Coupling**: Adjust trust thresholds and protocols
- **Memory**: Configure retention and decay
- **Evolution**: Enable/disable self-improvement

## Architecture

```
Senter 3.0/
├── senter.py          # Main entry point
├── genome.yaml        # All configuration (the DNA)
├── core/              # Configuration engine
├── coupling/          # Human-AI coupling layer
├── knowledge/         # Knowledge graph
├── memory/            # Living memory system
├── evolution/         # Self-improvement
├── models/            # LLM backends
├── daemon/            # 24/7 daemon mode
├── interface/         # Voice/gaze input
├── intelligence/      # Goal detection, proactive suggestions
├── tools/             # Web search, file ops
└── tests/             # Test suite
```

## Switching Models

### Use Ollama (default)
```yaml
models:
  primary:
    type: ollama
    model: "llama3.2"
    base_url: "http://localhost:11434"
```

### Use OpenAI
```yaml
models:
  primary:
    type: openai
    model: "gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"
```

### Use Local GGUF
```yaml
models:
  primary:
    type: gguf
    path: "/path/to/model.gguf"
```

## How Evolution Works

Senter 3.0 can improve itself over time:

1. **Fitness Tracking**: Each interaction is scored based on user satisfaction signals
2. **Mutation Proposals**: Low fitness triggers mutation proposals
3. **Experiments**: Mutations are tested over ~10 interactions
4. **Selection**: Successful mutations are persisted to genome.yaml
5. **Rollback**: Failed mutations are automatically rolled back

View evolution status with `/evolution`.

## How Goals Work

Senter automatically detects your goals:

1. **Explicit**: "I want to learn Python" → detected immediately
2. **Implicit**: Repeated topics → detected after 3+ mentions
3. **Frustrations**: "I'm stuck on..." → detected as blocked goals

View detected goals with `/goals`.

## Daemon Mode

The daemon allows Senter to work even when you're away:

```bash
# Start daemon
python senter.py --daemon &

# Add background task
/task research Latest AI developments

# Check what was completed
python senter.py --connect
# Shows "While you were away..." summary
```

Task types:
- `research` - Research a topic using web search
- `remind` - Set a reminder
- `summarize` - Summarize recent interactions
- `organize` - Organize files (coming soon)
