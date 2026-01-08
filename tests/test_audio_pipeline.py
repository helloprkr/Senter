#!/usr/bin/env python3
"""
Tests for Audio Pipeline (AP-001, AP-002, AP-003, AP-004)
Tests audio pipeline features including safety checks, VAD, STT, and device support.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== AP-001: Audio Pipeline Safety Tests ==========

def test_audio_disabled_env_var():
    """Test SENTER_DISABLE_AUDIO environment variable (AP-001)"""
    from audio.audio_pipeline import is_audio_disabled

    # Save original value
    original = os.environ.get("SENTER_DISABLE_AUDIO")

    try:
        # Test disabled states
        for val in ["1", "true", "yes", "TRUE", "Yes"]:
            os.environ["SENTER_DISABLE_AUDIO"] = val
            assert is_audio_disabled(), f"Should be disabled with {val}"

        # Test enabled states
        for val in ["0", "false", "no", ""]:
            os.environ["SENTER_DISABLE_AUDIO"] = val
            assert not is_audio_disabled(), f"Should be enabled with {val}"

        # Test unset
        if "SENTER_DISABLE_AUDIO" in os.environ:
            del os.environ["SENTER_DISABLE_AUDIO"]
        assert not is_audio_disabled(), "Should be enabled when unset"

    finally:
        # Restore original value
        if original is not None:
            os.environ["SENTER_DISABLE_AUDIO"] = original
        elif "SENTER_DISABLE_AUDIO" in os.environ:
            del os.environ["SENTER_DISABLE_AUDIO"]

    return True


def test_audio_pipeline_get_status():
    """Test AudioPipeline.get_status() returns all fields (AP-001)"""
    from audio.audio_pipeline import AudioPipeline, NUMPY_AVAILABLE
    from multiprocessing import Event

    if not NUMPY_AVAILABLE:
        print("  (skipped - numpy not available)")
        return True

    # Create mock message bus
    mock_bus = MagicMock()
    mock_bus.register.return_value = MagicMock()

    pipeline = AudioPipeline(
        stt_model="small",
        tts_model="system",
        vad_threshold=0.5,
        message_bus=mock_bus,
        shutdown_event=Event(),
        audio_device=None
    )

    status = pipeline.get_status()

    # Check all required fields
    required_fields = [
        "enabled", "microphone_available", "is_listening",
        "has_attention", "stt_model", "tts_model", "vad_threshold",
        "audio_device", "device_id"
    ]
    for field in required_fields:
        assert field in status, f"Missing field: {field}"

    return True


def test_audio_enabled_by_default():
    """Test that audio_pipeline.enabled is true in default config (AP-001)"""
    import json

    config_path = Path(__file__).parent.parent / "config" / "daemon_config.json"
    if not config_path.exists():
        print("  (skipped - config not found)")
        return True

    with open(config_path) as f:
        config = json.load(f)

    assert config["components"]["audio_pipeline"]["enabled"] is True

    return True


# ========== AP-002: Silero VAD Tests ==========

def test_vad_energy_fallback():
    """Test energy-based VAD fallback (AP-002)"""
    from audio.audio_pipeline import VoiceActivityDetector, NUMPY_AVAILABLE

    if not NUMPY_AVAILABLE:
        print("  (skipped - numpy not available)")
        return True

    import numpy as np

    # Create VAD without Silero (mock failure)
    with patch.dict('sys.modules', {'torch': None}):
        vad = VoiceActivityDetector(threshold=0.5, energy_threshold=0.01)

    # Test energy-based detection
    # Low energy - should not be speech
    silence = np.zeros(1600, dtype=np.float32)
    assert not vad.is_speech_energy(silence)

    # High energy - should be speech
    loud = np.ones(1600, dtype=np.float32) * 0.5
    assert vad.is_speech_energy(loud)

    return True


def test_vad_speech_probability():
    """Test VAD get_speech_probability returns 0-1 (AP-002)"""
    from audio.audio_pipeline import VoiceActivityDetector, NUMPY_AVAILABLE

    if not NUMPY_AVAILABLE:
        print("  (skipped - numpy not available)")
        return True

    import numpy as np

    vad = VoiceActivityDetector(threshold=0.5)

    # Test with various audio
    silence = np.zeros(1600, dtype=np.float32)
    prob = vad.get_speech_probability(silence)
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    loud = np.ones(1600, dtype=np.float32) * 0.1
    prob = vad.get_speech_probability(loud)
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    return True


def test_vad_threshold_configurable():
    """Test VAD threshold is configurable (AP-002)"""
    from audio.audio_pipeline import VoiceActivityDetector, NUMPY_AVAILABLE

    if not NUMPY_AVAILABLE:
        print("  (skipped - numpy not available)")
        return True

    vad = VoiceActivityDetector(threshold=0.3)
    assert vad.threshold == 0.3

    vad.set_threshold(0.7)
    assert vad.threshold == 0.7

    # Test clamping
    vad.set_threshold(1.5)
    assert vad.threshold == 1.0

    vad.set_threshold(-0.5)
    assert vad.threshold == 0.0

    return True


def test_vad_speech_events():
    """Test VAD emits speech start/end events (AP-002)"""
    from audio.audio_pipeline import VoiceActivityDetector, NUMPY_AVAILABLE

    if not NUMPY_AVAILABLE:
        print("  (skipped - numpy not available)")
        return True

    import numpy as np

    # Create mock message bus
    mock_bus = MagicMock()

    vad = VoiceActivityDetector(
        threshold=0.01,  # Low threshold for testing
        energy_threshold=0.005,
        message_bus=mock_bus
    )

    # Simulate speech start
    speech = np.ones(1600, dtype=np.float32) * 0.1
    vad.is_speech(speech)

    # Check speech_start was sent
    calls = mock_bus.send.call_args_list
    if calls:
        # Verify event structure
        _, kwargs = calls[0]
        assert kwargs.get("source") == "vad"

    # Simulate speech end
    silence = np.zeros(1600, dtype=np.float32)
    vad.is_speech(silence)

    return True


# ========== AP-003: STT Model Selection Tests ==========

def test_stt_model_normalization():
    """Test STT model name normalization (AP-003)"""
    from audio.audio_pipeline import STTEngine

    stt = STTEngine("whisper-small")
    assert stt.model_name == "small"

    stt = STTEngine("WHISPER-MEDIUM")
    assert stt.model_name == "medium"

    stt = STTEngine("tiny")
    assert stt.model_name == "tiny"

    stt = STTEngine("invalid_model")
    assert stt.model_name == "small"  # Should default to small

    return True


def test_stt_supported_models():
    """Test STT returns list of supported models (AP-003)"""
    from audio.audio_pipeline import STTEngine

    models = STTEngine.get_available_models()

    assert "tiny" in models
    assert "base" in models
    assert "small" in models
    assert "medium" in models
    assert "large" in models

    return True


def test_stt_model_change():
    """Test STT model can be changed (AP-003)"""
    from audio.audio_pipeline import STTEngine

    stt = STTEngine("small")
    assert stt.model_name == "small"

    changed = stt.set_model("medium")
    assert changed
    assert stt.model_name == "medium"

    # No change if same model
    changed = stt.set_model("medium")
    assert not changed

    return True


def test_transcription_result_structure():
    """Test TranscriptionResult has all fields (AP-003)"""
    from audio.audio_pipeline import TranscriptionResult

    result = TranscriptionResult(
        text="Hello world",
        confidence=0.95,
        language="en",
        duration=1.5
    )

    d = result.to_dict()

    assert d["text"] == "Hello world"
    assert d["confidence"] == 0.95
    assert d["language"] == "en"
    assert d["duration"] == 1.5

    return True


# ========== AP-004: External Microphone Tests ==========

def test_audio_device_dataclass():
    """Test AudioDevice dataclass (AP-004)"""
    from audio.audio_pipeline import AudioDevice

    device = AudioDevice(
        id=1,
        name="External Microphone",
        channels=2,
        sample_rate=44100.0,
        is_default=False
    )

    d = device.to_dict()

    assert d["id"] == 1
    assert d["name"] == "External Microphone"
    assert d["channels"] == 2
    assert d["sample_rate"] == 44100.0
    assert d["is_default"] is False

    return True


def test_list_audio_devices():
    """Test list_audio_devices returns list (AP-004)"""
    from audio.audio_pipeline import list_audio_devices, SOUNDDEVICE_AVAILABLE

    devices = list_audio_devices()

    # Should return a list (may be empty if sounddevice not available)
    assert isinstance(devices, list)

    if SOUNDDEVICE_AVAILABLE and devices:
        # If devices found, check structure
        dev = devices[0]
        assert hasattr(dev, 'id')
        assert hasattr(dev, 'name')
        assert hasattr(dev, 'channels')
        assert hasattr(dev, 'sample_rate')
        assert hasattr(dev, 'is_default')

    return True


def test_device_by_name_or_id():
    """Test get_device_by_name_or_id (AP-004)"""
    from audio.audio_pipeline import get_device_by_name_or_id

    # Test with None/empty
    assert get_device_by_name_or_id(None) is None
    assert get_device_by_name_or_id("") is None

    # Test with invalid device (should return None)
    result = get_device_by_name_or_id("nonexistent_device_xyz_123")
    assert result is None

    return True


def test_audio_pipeline_device_selection():
    """Test AudioPipeline accepts device parameter (AP-004)"""
    from audio.audio_pipeline import AudioPipeline, NUMPY_AVAILABLE
    from multiprocessing import Event

    if not NUMPY_AVAILABLE:
        print("  (skipped - numpy not available)")
        return True

    mock_bus = MagicMock()
    mock_bus.register.return_value = MagicMock()

    # Create pipeline with device specification
    pipeline = AudioPipeline(
        stt_model="small",
        tts_model="system",
        vad_threshold=0.5,
        message_bus=mock_bus,
        shutdown_event=Event(),
        audio_device="External Mic"  # AP-004
    )

    assert pipeline.audio_device == "External Mic"

    return True


def test_microphone_availability_check():
    """Test check_microphone_available function (AP-001)"""
    from audio.audio_pipeline import check_microphone_available, SOUNDDEVICE_AVAILABLE

    # Function should return bool
    result = check_microphone_available()
    assert isinstance(result, bool)

    # If sounddevice not available, should return False
    if not SOUNDDEVICE_AVAILABLE:
        assert result is False

    return True


if __name__ == "__main__":
    tests = [
        # AP-001
        test_audio_disabled_env_var,
        test_audio_pipeline_get_status,
        test_audio_enabled_by_default,
        # AP-002
        test_vad_energy_fallback,
        test_vad_speech_probability,
        test_vad_threshold_configurable,
        test_vad_speech_events,
        # AP-003
        test_stt_model_normalization,
        test_stt_supported_models,
        test_stt_model_change,
        test_transcription_result_structure,
        # AP-004
        test_audio_device_dataclass,
        test_list_audio_devices,
        test_device_by_name_or_id,
        test_audio_pipeline_device_selection,
        test_microphone_availability_check,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
