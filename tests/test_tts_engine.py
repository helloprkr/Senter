#!/usr/bin/env python3
"""
Tests for TTSEngine (CG-005)
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tts_sentence_chunking():
    """Test that TTS chunks text into sentences"""
    from audio.audio_pipeline import TTSEngine

    tts = TTSEngine()

    # Test basic sentence splitting
    text = "Hello world. This is a test. How are you?"
    sentences = tts._chunk_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "This is a test."
    assert sentences[2] == "How are you?"

    # Test with abbreviations (should not split)
    text = "Dr. Smith went to the store. Mr. Jones followed."
    sentences = tts._chunk_into_sentences(text)
    assert len(sentences) == 2
    assert "Dr." in sentences[0]
    assert "Mr." in sentences[1]

    # Test with question marks and exclamation points
    text = "What is this? I love it! Amazing."
    sentences = tts._chunk_into_sentences(text)
    assert len(sentences) == 3

    # Test empty input
    assert tts._chunk_into_sentences("") == []
    assert tts._chunk_into_sentences("   ") == []

    return True


def test_tts_async_execution():
    """Test that TTS runs asynchronously"""
    from audio.audio_pipeline import TTSEngine

    tts = TTSEngine()

    # Measure time - async should return immediately
    start = time.time()
    tts.speak_async("This is a test")
    elapsed = time.time() - start

    # Async call should return very quickly (< 100ms)
    assert elapsed < 0.5, f"Async call took too long: {elapsed}s"

    # Thread should be running (unless speech completed very fast)
    # Give it a moment to start
    time.sleep(0.1)

    # Stop speaking
    tts.stop_speaking()

    return True


def test_tts_voice_configuration():
    """Test that voice can be configured"""
    from audio.audio_pipeline import TTSEngine

    # Test with default voice
    tts = TTSEngine()
    assert tts.voice is None

    # Test with custom voice
    tts = TTSEngine(voice="Samantha")
    assert tts.voice == "Samantha"

    # Test set_voice method
    tts.set_voice("Alex")
    assert tts.voice == "Alex"

    return True


def test_tts_error_handling():
    """Test TTS error handling"""
    from audio.audio_pipeline import TTSEngine

    tts = TTSEngine()

    # Empty text should not crash
    try:
        tts._speak_sync("")
    except Exception as e:
        assert False, f"Empty text caused exception: {e}"

    # None-like text should not crash
    try:
        tts._speak_sync("   ")
    except Exception as e:
        assert False, f"Whitespace text caused exception: {e}"

    return True


def test_tts_stop_speaking():
    """Test that speaking can be stopped"""
    from audio.audio_pipeline import TTSEngine

    tts = TTSEngine()

    # Start speaking
    long_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
    tts.speak_async(long_text)

    # Stop immediately
    tts.stop_speaking()

    # Should not raise any errors
    # Wait a moment to ensure cleanup
    time.sleep(0.2)

    return True


def test_tts_list_voices():
    """Test listing available voices (macOS only)"""
    from audio.audio_pipeline import TTSEngine
    import platform

    voices = TTSEngine.list_voices()

    if platform.system() == "Darwin":
        # On macOS, should have some voices
        assert isinstance(voices, list)
        # Usually there are at least a few built-in voices
        if len(voices) > 0:
            assert all(isinstance(v, str) for v in voices)
    else:
        # On non-macOS, should return empty list
        assert voices == []

    return True


if __name__ == "__main__":
    tests = [
        test_tts_sentence_chunking,
        test_tts_async_execution,
        test_tts_voice_configuration,
        test_tts_error_handling,
        test_tts_stop_speaking,
        test_tts_list_voices,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
