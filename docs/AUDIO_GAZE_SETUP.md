# Audio & Gaze Detection Setup

This document covers the installation and configuration for audio pipeline and gaze detection features (US-010, US-011, US-012).

## Overview

These features enable the "just look and talk" promise:
1. **Gaze Detection (US-011)**: Camera watches for user attention
2. **Audio Pipeline (US-010)**: Microphone listens when user is looking
3. **Attention-Voice Integration (US-012)**: Wires them together

## Required Dependencies

### Audio Pipeline (US-010)

```bash
# Voice Activity Detection & Audio Capture
pip install sounddevice

# Speech-to-Text (Whisper)
pip install openai-whisper
# OR for faster inference:
pip install faster-whisper
```

**Hardware Requirements:**
- Microphone (built-in or external)
- macOS: Grant microphone permission to Terminal/IDE

### Gaze Detection (US-011)

```bash
# Computer Vision
pip install opencv-python

# Face/Eye Tracking
pip install mediapipe
```

**Hardware Requirements:**
- Webcam (built-in or external)
- macOS: Grant camera permission to Terminal/IDE

## Installation Script

Run this to install all dependencies:

```bash
#!/bin/bash
pip install sounddevice openai-whisper opencv-python mediapipe
```

Or with conda:
```bash
conda install -c conda-forge sounddevice
pip install openai-whisper opencv-python mediapipe
```

## Configuration

After installing dependencies, enable features in `config/daemon_config.json`:

```json
{
  "components": {
    "audio_pipeline": {
      "enabled": true,
      "vad_sensitivity": 0.5,
      "whisper_model": "base",
      "sample_rate": 16000
    },
    "gaze_detection": {
      "enabled": true,
      "attention_threshold": 0.7,
      "camera_index": 0,
      "check_interval_ms": 100
    }
  }
}
```

## Verification

After installation, verify with:

```bash
# Test audio
python3 -c "import sounddevice; print('sounddevice OK')"
python3 -c "import whisper; print('whisper OK')"

# Test gaze
python3 -c "import cv2; print('opencv OK')"
python3 -c "import mediapipe; print('mediapipe OK')"
```

## Troubleshooting

### macOS Permissions
- System Preferences > Security & Privacy > Privacy
- Enable Camera and Microphone for Terminal/your IDE

### M1/M2 Mac Issues
```bash
# If opencv fails on Apple Silicon:
pip install opencv-python-headless

# If mediapipe fails:
pip install mediapipe-silicon
```

### Audio Device Not Found
```python
import sounddevice as sd
print(sd.query_devices())  # List available devices
```

## Implementation Status

| Story | Feature | Status |
|-------|---------|--------|
| US-010 | Audio Pipeline | BLOCKED - waiting for deps |
| US-011 | Gaze Detection | BLOCKED - waiting for deps |
| US-012 | Attention-Voice Wire | BLOCKED - depends on US-010, US-011 |

## Next Steps After Installation

1. Run `pip install sounddevice openai-whisper opencv-python mediapipe`
2. Update `config/daemon_config.json` to enable features
3. Run tests: `python3 tests/test_audio_enabled.py`
4. Start daemon: `python3 scripts/senter_ctl.py start`
5. Verify in status: `python3 scripts/senter_ctl.py status`
