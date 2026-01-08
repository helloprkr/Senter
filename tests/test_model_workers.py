#!/usr/bin/env python3
"""
Tests for Model Workers (MW-001, MW-002, MW-003)
Tests GGUF loading, streaming, and hot-swapping.
"""

import sys
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
from multiprocessing import Event

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_gguf_loader_initialization():
    """Test GGUFModelLoader initialization (MW-001)"""
    from workers.model_worker import GGUFModelLoader

    loader = GGUFModelLoader(
        model_path="/path/to/model.gguf",
        n_gpu_layers=-1,
        n_ctx=4096
    )

    assert loader.model_path == Path("/path/to/model.gguf")
    assert loader.n_gpu_layers == -1
    assert loader.n_ctx == 4096
    assert loader._llm is None
    assert not loader.is_loaded

    return True


def test_gguf_loader_unload():
    """Test GGUFModelLoader unload (MW-001)"""
    from workers.model_worker import GGUFModelLoader

    loader = GGUFModelLoader(model_path="/path/to/model.gguf")

    # Mock a loaded model
    loader._llm = MagicMock()
    assert loader.is_loaded

    # Unload
    loader.unload()
    assert not loader.is_loaded
    assert loader._llm is None

    return True


def test_gguf_fallback_to_ollama():
    """Test that GGUF failure falls back to Ollama (MW-001)"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown,
            model_type="gguf",
            model_path="/nonexistent/model.gguf"
        )

        # Try to load non-existent GGUF
        result = worker.load_gguf_model()

        # Should have fallen back to Ollama
        assert result is False
        assert worker.model_type == "ollama"

        return True


def test_worker_model_info():
    """Test get_model_info method (MW-001)"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown,
            model_type="ollama"
        )

        info = worker.get_model_info()

        assert info["name"] == "test"
        assert info["model"] == "llama3.2"
        assert info["type"] == "ollama"
        assert "loaded" in info
        assert "in_flight" in info

        return True


def test_streaming_enabled_by_default():
    """Test that streaming is enabled by default (MW-002)"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown
        )

        assert worker.stream_default is True

        return True


def test_streaming_can_be_disabled():
    """Test that streaming can be disabled (MW-002)"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown,
            stream_default=False
        )

        assert worker.stream_default is False

        return True


def test_sentence_boundary_detection():
    """Test sentence boundary chunking in streaming (MW-002)"""
    import re

    # The pattern used in _stream_response
    sentence_endings = re.compile(r'([.!?]+[\s]+|[\n]+)')

    # Test various sentence endings
    test_cases = [
        ("Hello world. ", True),  # Period followed by space
        ("What? ", True),  # Question mark followed by space
        ("Great! ", True),  # Exclamation followed by space
        ("Hello\n", True),  # Newline
        ("Hello world", False),  # No ending
        ("Mr.", False),  # Period with no space
        ("Hello...", False),  # Ellipsis with no space after
    ]

    for text, should_match in test_cases:
        match = sentence_endings.search(text)
        assert (match is not None) == should_match, f"Failed for: '{text}'"

    return True


def test_hot_swap_initialization():
    """Test hot-swap related attributes are initialized (MW-003)"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown
        )

        assert worker._swap_in_progress is False
        assert worker._swap_lock is not None
        assert worker._in_flight_count == 0
        assert worker._in_flight_lock is not None

        return True


def test_in_flight_tracking():
    """Test in-flight request tracking (MW-003)"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown
        )

        # Simulate incrementing in-flight count
        with worker._in_flight_lock:
            worker._in_flight_count += 1
        assert worker._in_flight_count == 1

        with worker._in_flight_lock:
            worker._in_flight_count += 1
        assert worker._in_flight_count == 2

        with worker._in_flight_lock:
            worker._in_flight_count -= 1
        assert worker._in_flight_count == 1

        return True


def test_swap_model_blocks_during_swap():
    """Test that concurrent swaps are blocked (MW-003)"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown
        )

        # Simulate swap in progress
        with worker._swap_lock:
            worker._swap_in_progress = True

        # Try another swap - should fail
        result = worker.swap_model("new-model")
        assert result is False

        # Cleanup
        with worker._swap_lock:
            worker._swap_in_progress = False

        return True


def test_swap_model_waits_for_in_flight():
    """Test that swap waits for in-flight requests (MW-003)"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown
        )

        # Simulate in-flight request
        with worker._in_flight_lock:
            worker._in_flight_count = 1

        # Start swap in background - it should wait
        swap_started = threading.Event()
        swap_completed = threading.Event()
        swap_result = [None]

        def do_swap():
            swap_started.set()
            swap_result[0] = worker.swap_model("new-model")
            swap_completed.set()

        swap_thread = threading.Thread(target=do_swap)
        swap_thread.start()

        # Wait for swap to start
        swap_started.wait(timeout=1)
        time.sleep(0.2)

        # Swap should not be complete yet
        assert not swap_completed.is_set()

        # Clear in-flight
        with worker._in_flight_lock:
            worker._in_flight_count = 0

        # Wait for swap to complete
        swap_completed.wait(timeout=2)
        swap_thread.join(timeout=1)

        # Swap should succeed
        assert swap_result[0] is True
        assert worker.model_name == "new-model"

        return True


def test_swap_model_updates_name():
    """Test that swap_model updates the model name (MW-003)"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown
        )

        old_model = worker.model_name

        # Swap to new model
        result = worker.swap_model("llama3.3")

        assert result is True
        assert worker.model_name == "llama3.3"
        assert worker.model_name != old_model

        return True


def test_worker_stats():
    """Test get_stats method"""
    from workers.model_worker import ModelWorker
    from daemon.message_bus import MessageBus

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        shutdown = Event()

        worker = ModelWorker(
            name="test",
            model="llama3.2",
            message_bus=bus,
            shutdown_event=shutdown
        )

        # Set some stats
        worker.requests_processed = 10
        worker.total_latency = 5.0
        worker.total_tokens = 1000

        stats = worker.get_stats()

        assert stats["name"] == "test"
        assert stats["model"] == "llama3.2"
        assert stats["requests_processed"] == 10
        assert stats["avg_latency_ms"] == 500  # 5.0 / 10 * 1000
        assert stats["total_tokens"] == 1000

        return True


if __name__ == "__main__":
    tests = [
        test_gguf_loader_initialization,
        test_gguf_loader_unload,
        test_gguf_fallback_to_ollama,
        test_worker_model_info,
        test_streaming_enabled_by_default,
        test_streaming_can_be_disabled,
        test_sentence_boundary_detection,
        test_hot_swap_initialization,
        test_in_flight_tracking,
        test_swap_model_blocks_during_swap,
        test_swap_model_waits_for_in_flight,
        test_swap_model_updates_name,
        test_worker_stats,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
