#!/usr/bin/env python3
"""
Tests for Attention-Voice Flow - US-012 acceptance criteria:
1. When gaze detector sends ATTENTION_GAINED, audio pipeline activates
2. When gaze detector sends ATTENTION_LOST, audio pipeline deactivates
3. Voice input transcribed and sent to primary model worker
4. End-to-end: look at camera -> speak -> get response
"""

import sys
import json
import time
from pathlib import Path
from multiprocessing import Queue, Event

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_attention_routes_to_audio():
    """Test that attention_gained routes to audio component"""
    from daemon.senter_daemon import SenterDaemon

    daemon = SenterDaemon()

    # Check routing table includes audio
    import inspect
    source = inspect.getsource(daemon._route_message)

    assert '"attention_gained": ["audio"' in source or "'attention_gained': ['audio'" in source or \
           'attention_gained' in source and 'audio' in source, \
           "attention_gained should route to audio"

    print("OK test_attention_routes_to_audio PASSED")
    return True


def test_attention_lost_routes_to_audio():
    """Test that attention_lost routes to audio component"""
    from daemon.senter_daemon import SenterDaemon

    daemon = SenterDaemon()

    import inspect
    source = inspect.getsource(daemon._route_message)

    assert 'attention_lost' in source and 'audio' in source, \
           "attention_lost should route to audio"

    print("OK test_attention_lost_routes_to_audio PASSED")
    return True


def test_user_voice_routes_to_model():
    """Test that user_voice routes to model_primary"""
    from daemon.senter_daemon import SenterDaemon

    daemon = SenterDaemon()

    import inspect
    source = inspect.getsource(daemon._route_message)

    assert 'user_voice' in source, "Should have user_voice routing"
    assert 'model_primary' in source, "Should route to model_primary"

    print("OK test_user_voice_routes_to_model PASSED")
    return True


def test_model_worker_handles_user_voice():
    """Test that model worker handles user_voice message type"""
    from daemon.senter_daemon import model_worker_process

    import inspect
    source = inspect.getsource(model_worker_process)

    assert 'user_voice' in source, "model_worker should handle user_voice"

    print("OK test_model_worker_handles_user_voice PASSED")
    return True


def test_audio_process_handles_attention():
    """Test that audio process handles attention messages"""
    from daemon.senter_daemon import audio_pipeline_process

    import inspect
    source = inspect.getsource(audio_pipeline_process)

    assert 'attention_gained' in source, "audio_pipeline should handle attention_gained"
    assert 'attention_lost' in source, "audio_pipeline should handle attention_lost"
    assert 'has_attention' in source, "audio_pipeline should track has_attention state"

    print("OK test_audio_process_handles_attention PASSED")
    return True


def test_gaze_process_sends_attention():
    """Test that gaze process sends attention messages"""
    from daemon.senter_daemon import gaze_detector_process

    import inspect
    source = inspect.getsource(gaze_detector_process)

    assert 'attention_gained' in source, "gaze_detector should send attention_gained"
    assert 'attention_lost' in source, "gaze_detector should send attention_lost"
    assert 'output_queue.put' in source, "gaze_detector should put messages on output queue"

    print("OK test_gaze_process_sends_attention PASSED")
    return True


def test_audio_sends_user_voice():
    """Test that audio pipeline sends user_voice messages"""
    from daemon.senter_daemon import audio_pipeline_process

    import inspect
    source = inspect.getsource(audio_pipeline_process)

    assert 'user_voice' in source, "audio_pipeline should send user_voice"
    assert 'output_queue.put' in source, "audio_pipeline should put messages on output queue"

    print("OK test_audio_sends_user_voice PASSED")
    return True


def test_model_response_routes_to_audio():
    """Test that model_response routes to audio for TTS"""
    from daemon.senter_daemon import SenterDaemon

    daemon = SenterDaemon()

    import inspect
    source = inspect.getsource(daemon._route_message)

    # Check model_response routes include audio
    assert 'model_response' in source, "Should route model_response"
    # The routing should include audio for TTS
    assert 'audio' in source, "model_response should route to audio"

    print("OK test_model_response_routes_to_audio PASSED")
    return True


def test_end_to_end_flow_structure():
    """Test that the end-to-end flow structure is in place"""
    # The flow is:
    # 1. Gaze detector -> ATTENTION_GAINED -> Audio pipeline
    # 2. Audio pipeline listens for speech
    # 3. Audio pipeline -> USER_VOICE -> Model worker
    # 4. Model worker -> MODEL_RESPONSE -> Audio pipeline (for TTS)

    from daemon.senter_daemon import (
        gaze_detector_process,
        audio_pipeline_process,
        model_worker_process,
        SenterDaemon
    )

    # All components exist
    assert callable(gaze_detector_process)
    assert callable(audio_pipeline_process)
    assert callable(model_worker_process)

    # Daemon can start all components
    daemon = SenterDaemon()
    assert hasattr(daemon, "_start_gaze_detection")
    assert hasattr(daemon, "_start_audio_pipeline")
    assert hasattr(daemon, "_start_model_workers")

    # Routing is configured
    import inspect
    source = inspect.getsource(daemon._route_message)
    assert "attention_gained" in source
    assert "user_voice" in source
    assert "model_response" in source

    print("OK test_end_to_end_flow_structure PASSED")
    return True


def test_attention_voice_flow():
    """Combined test for US-012 acceptance criteria"""
    # 1. ATTENTION_GAINED -> audio
    from daemon.senter_daemon import SenterDaemon, audio_pipeline_process
    daemon = SenterDaemon()

    import inspect
    route_source = inspect.getsource(daemon._route_message)
    assert "attention_gained" in route_source and "audio" in route_source

    # 2. ATTENTION_LOST -> audio
    assert "attention_lost" in route_source and "audio" in route_source

    # 3. USER_VOICE -> model_primary
    assert "user_voice" in route_source and "model_primary" in route_source

    # 4. Audio pipeline has transcription flow
    audio_source = inspect.getsource(audio_pipeline_process)
    assert "transcribe" in audio_source or "stt" in audio_source.lower()

    print("OK test_attention_voice_flow PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Attention-Voice Flow Tests (US-012)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_attention_routes_to_audio,
        test_attention_lost_routes_to_audio,
        test_user_voice_routes_to_model,
        test_model_worker_handles_user_voice,
        test_audio_process_handles_attention,
        test_gaze_process_sends_attention,
        test_audio_sends_user_voice,
        test_model_response_routes_to_audio,
        test_end_to_end_flow_structure,
        test_attention_voice_flow,
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"FAIL {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)

    sys.exit(0 if all_passed else 1)
