# Senter

**An ambient AI companion that responds when you look at it and speak.**

<p align="center">
  <img src="https://img.shields.io/badge/version-2.1.0-00ffaa?style=for-the-badge" alt="Version 2.1.0">
  <img src="https://img.shields.io/badge/python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/badge/status-fully%20functional-success?style=for-the-badge" alt="Status">
</p>

---

## The Promise: Just Look and Talk

Senter runs quietly in the background. When you look at your camera, it notices. When you speak, it listens, understands, and responds - out loud.

No wake words. No buttons. No friction.

```
You look at your screen
    ↓
Gaze detector: "Attention gained"
    ↓
Audio pipeline activates
    ↓
You speak: "What's on my calendar today?"
    ↓
Whisper transcribes → LLM responds → TTS speaks the answer
```

---

## Quick Start

```bash
# Clone and enter
cd "/path/to/Senter ⎊"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install sounddevice openai-whisper opencv-python mediapipe

# Start Senter
python3 scripts/senter_ctl.py start

# Check status
python3 scripts/senter_ctl.py status

# Send a query
python3 scripts/senter_ctl.py query "Hello, what can you do?"

# Interactive shell
python3 scripts/senter_ctl.py shell

# Stop
python3 scripts/senter_ctl.py stop
```

**Requirements:**
- Python 3.10+
- [Ollama](https://ollama.ai) running locally with `llama3.2` (or configure another model)
- Camera and microphone access (grant permissions when prompted)

---

## Architecture

Senter is a **multiprocess daemon** with 8 independent components communicating via message queues:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SENTER DAEMON                                 │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │    Gaze     │  │    Audio    │  │   Model     │  │   Model     │   │
│  │  Detector   │─▶│  Pipeline   │─▶│  Primary    │  │  Research   │   │
│  │  (camera)   │  │ (mic/speak) │  │  (queries)  │  │ (background)│   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│         │                │                │                │           │
│         └────────────────┴────────────────┴────────────────┘           │
│                              Message Bus                                │
│         ┌────────────────┬────────────────┬────────────────┐           │
│         │                │                │                │           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │    Task     │  │  Scheduler  │  │  Reporter   │  │  Learning   │   │
│  │   Engine    │  │   (cron)    │  │  (activity) │  │  (behavior) │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                         │
│  ┌─────────────┐                                                       │
│  │ IPC Server  │◀──── Unix Socket (/tmp/senter.sock) ◀──── CLI        │
│  └─────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Gaze Detector** | Watches camera for user attention | MediaPipe Face Mesh / Haar cascade fallback, attention scoring, timeout handling |
| **Audio Pipeline** | Voice capture, transcription, TTS | Whisper STT, energy-based VAD, system TTS (macOS `say`) |
| **Model Primary** | Handles user queries | Ollama integration, context-aware responses |
| **Model Research** | Background research tasks | Lower temperature, longer output, stores results |
| **Task Engine** | Plans and executes complex tasks | Goal decomposition, result storage |
| **Scheduler** | Cron-like job execution | Hourly research triggers, daily digests |
| **Reporter** | Activity logging | Aggregates what Senter did |
| **Learning** | Behavioral database | User events, pattern detection, topic analysis |

---

## Features

### Attention-Activated Voice Interface
- **Gaze Detection**: Camera tracks your face; activates when you look at the screen
- **Voice Activity Detection**: Distinguishes speech from background noise
- **Speech-to-Text**: Whisper transcribes your words
- **Text-to-Speech**: Responses spoken aloud

### Background Intelligence
- **Autonomous Research**: Scheduler triggers research based on your recent queries
- **Pattern Detection**: Learns your peak usage hours and preferred topics
- **Activity Reports**: Ask "what did you do today?" for a summary

### Developer-Friendly
- **IPC Interface**: Unix socket for programmatic control
- **CLI Tools**: Complete control via `senter_ctl.py`
- **Modular Processes**: Each component runs independently
- **Comprehensive Logging**: All activity logged to `data/daemon.log`

---

## CLI Reference

```bash
# Daemon Control
senter_ctl.py start              # Start the daemon
senter_ctl.py stop               # Stop the daemon
senter_ctl.py restart            # Restart the daemon
senter_ctl.py status             # Show component health

# Interaction
senter_ctl.py query "..."        # Send a query
senter_ctl.py shell              # Interactive chat mode

# Monitoring
senter_ctl.py logs               # View daemon logs
senter_ctl.py report             # Activity report (what did Senter do)
senter_ctl.py report -H 24       # Last 24 hours
senter_ctl.py events             # User interaction history

# Tasks
senter_ctl.py goal "..."         # Create a new goal
senter_ctl.py goals              # List active goals

# Configuration
senter_ctl.py config             # Edit configuration
```

---

## Configuration

Edit `config/daemon_config.json`:

```json
{
  "components": {
    "model_workers": {
      "enabled": true,
      "models": {
        "primary": "llama3.2",
        "research": "llama3.2"
      }
    },
    "audio_pipeline": {
      "enabled": true,
      "stt_model": "whisper-small",
      "vad_threshold": 0.5
    },
    "gaze_detection": {
      "enabled": true,
      "camera_id": 0,
      "attention_threshold": 0.7
    }
  }
}
```

### Model Options
- Change `primary` and `research` to any Ollama model
- Adjust `attention_threshold` (0.0-1.0) for gaze sensitivity
- Modify `vad_threshold` for voice detection sensitivity

---

## Data Storage

```
data/
├── daemon.log              # All component logs
├── daemon.pid              # Process ID file
├── senter.pid              # Daemon state
├── state/                  # Crash recovery state
├── learning/
│   ├── behavior.db         # User behavior database
│   ├── events.db           # Interaction events
│   └── patterns.json       # Detected patterns
├── tasks/
│   └── results/            # Completed task results
├── research/
│   └── results/            # Background research output
├── progress/
│   └── activity/           # Activity logs
└── scheduler/
    └── jobs.json           # Scheduled jobs
```

---

## How It Works

### The Attention Flow

```
1. GAZE DETECTOR (every 66ms)
   ├── Captures camera frame
   ├── Detects face (MediaPipe or Haar cascade)
   ├── Calculates attention score:
   │   └── face_center (30%) + eye_openness (30%) + gaze_direction (40%)
   ├── If score >= 0.7 for multiple frames:
   │   └── Sends ATTENTION_GAINED → Audio Pipeline
   └── If score < 0.7 for 2 seconds:
       └── Sends ATTENTION_LOST → Audio Pipeline

2. AUDIO PIPELINE (when attention gained)
   ├── Starts listening to microphone
   ├── Runs VAD on audio buffer
   ├── When speech detected:
   │   ├── Buffers audio until silence
   │   ├── Sends to Whisper for transcription
   │   └── Sends USER_VOICE → Model Primary
   └── When MODEL_RESPONSE received:
       └── Speaks response via TTS

3. MODEL PRIMARY
   ├── Receives USER_VOICE or USER_QUERY
   ├── Sends to Ollama with system prompt
   └── Returns MODEL_RESPONSE → Audio Pipeline (for TTS)
```

### Message Types

| Message | From | To | Purpose |
|---------|------|-----|---------|
| `attention_gained` | Gaze | Audio | Activate voice listening |
| `attention_lost` | Gaze | Audio | Deactivate voice listening |
| `user_voice` | Audio | Model | Transcribed speech |
| `user_query` | IPC | Model | CLI/API queries |
| `model_response` | Model | Audio, Reporter | LLM output |
| `speak` | Any | Audio | TTS request |

---

## Troubleshooting

### Microphone Not Working
```bash
# Check permissions
System Settings → Privacy & Security → Microphone → Enable Terminal/IDE

# Test directly
python3 -c "
import sounddevice as sd
import numpy as np
r = sd.rec(48000, samplerate=16000, channels=1, dtype=np.float32)
sd.wait()
print(f'Max level: {np.max(np.abs(r)):.4f} (need > 0.01)')
"
```

### Camera Using Wrong Device
```bash
# List cameras
python3 -c "
import cv2
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
"

# Update config/daemon_config.json with correct camera_id
```

### Gaze Not Detecting
- Ensure good lighting on your face
- Face the camera directly
- Lower `attention_threshold` in config (e.g., 0.5)

### Logs
```bash
# Live logs
tail -f data/daemon.log

# Filter for specific component
tail -f data/daemon.log | grep -E "(Audio|Gaze|Attention)"
```

---

## Development

### Project Structure

```
Senter ⎊/
├── daemon/
│   ├── senter_daemon.py    # Main daemon + all process functions
│   ├── ipc_server.py       # Unix socket server
│   ├── ipc_client.py       # Client library
│   ├── message_bus.py      # Pub/sub messaging
│   └── health_monitor.py   # Component health tracking
├── audio/
│   └── audio_pipeline.py   # VAD, STT, TTS classes
├── vision/
│   └── gaze_detector.py    # Face tracking, attention scoring
├── engine/
│   ├── task_engine.py      # Task planning and execution
│   └── task_results.py     # Result storage
├── scheduler/
│   ├── action_scheduler.py # Cron-like job scheduling
│   └── research_trigger.py # Topic extraction, task generation
├── learning/
│   ├── learning_db.py      # Behavioral database
│   ├── events_db.py        # User interaction events
│   └── pattern_detector.py # Time-based pattern analysis
├── reporter/
│   └── progress_reporter.py # Activity logging
├── scripts/
│   └── senter_ctl.py       # CLI control script
├── config/
│   └── daemon_config.json  # Configuration
├── tests/
│   └── test_*.py           # Test suite
└── docs/
    └── AUDIO_GAZE_SETUP.md # Hardware setup guide
```

### Running Tests

```bash
source .venv/bin/activate

# All tests
python3 -m pytest tests/

# Specific test
python3 tests/test_audio_enabled.py
python3 tests/test_gaze_enabled.py
python3 tests/test_attention_voice_flow.py
```

### Adding a New Component

1. Create process function in `daemon/senter_daemon.py`:
```python
def my_component_process(input_queue, output_queue, shutdown_event, config):
    while not shutdown_event.is_set():
        # Process messages from input_queue
        # Send results to output_queue
        pass
```

2. Add start method to `SenterDaemon` class:
```python
def _start_my_component(self):
    q, out = Queue(), Queue()
    self.queues["my_component"] = q
    self.output_queues["my_component"] = out
    p = Process(target=my_component_process, args=(q, out, self.shutdown_event, self.config))
    p.start()
    self.processes["my_component"] = p
```

3. Update message routing in `_route_message()`:
```python
routing = {
    "my_message_type": ["my_component"],
    ...
}
```

---

## Philosophy

Senter embodies a simple idea: **AI should be ambient, not intrusive**.

- No wake words that interrupt your flow
- No apps to switch to
- No buttons to press

Just look up. Speak. Get an answer.

The goal is symbiotic computing - where the boundary between thinking and asking becomes imperceptible.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Ollama** - Local LLM inference
- **OpenAI Whisper** - Speech-to-text
- **MediaPipe** - Face mesh and gaze tracking
- **OpenCV** - Computer vision
- **macOS TTS** - Text-to-speech

---

<p align="center">
  <strong>Built for a future where AI just... works.</strong>
</p>
