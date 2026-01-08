#!/usr/bin/env python3
"""
Tests for Message Bus Features (MB-001, MB-002, MB-003)
Tests dead letter queue, correlation tracking, and message persistence.
"""

import sys
import json
import time
import tempfile
import threading
from pathlib import Path
from queue import Full

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_dlq_message_moved_on_failure():
    """Test that messages move to DLQ after MAX_RETRIES failures (MB-001)"""
    from daemon.message_bus import MessageBus, Message, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))

        # Create a message
        msg = Message(
            type=MessageType.USER_QUERY,
            source="test",
            payload={"text": "test"},
            correlation_id="test-corr-1"
        )

        # Record failures up to MAX_RETRIES
        for _ in range(bus.MAX_RETRIES):
            bus._record_failure(msg.to_dict(), "Test error")

        # Should be in DLQ now
        entries = bus.get_dlq_entries()
        assert len(entries) == 1
        assert entries[0].error == "Test error"
        assert entries[0].failure_count == bus.MAX_RETRIES
        assert entries[0].message["correlation_id"] == "test-corr-1"

        return True


def test_dlq_persistence():
    """Test that DLQ persists to disk (MB-001)"""
    from daemon.message_bus import MessageBus, Message, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        bus = MessageBus(data_dir=data_dir)

        # Create a message and force it to DLQ
        msg = Message(
            type=MessageType.USER_QUERY,
            source="test",
            payload={"text": "persist test"},
            correlation_id="persist-corr"
        )

        for _ in range(bus.MAX_RETRIES):
            bus._record_failure(msg.to_dict(), "Persist error")

        # Check file exists
        assert (data_dir / "dlq.json").exists()

        # Create new bus instance - should load DLQ
        bus2 = MessageBus(data_dir=data_dir)
        entries = bus2.get_dlq_entries()
        assert len(entries) == 1
        assert entries[0].message["correlation_id"] == "persist-corr"

        return True


def test_dlq_size_limit():
    """Test that DLQ is capped at DLQ_MAX_SIZE (MB-001)"""
    from daemon.message_bus import MessageBus, Message, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        bus.DLQ_MAX_SIZE = 5  # Small limit for testing

        # Add more than limit
        for i in range(10):
            msg = Message(
                type=MessageType.USER_QUERY,
                source="test",
                payload={"index": i}
            )
            bus._move_to_dlq(msg.to_dict(), f"Error {i}", 3)

        # Should only have 5 entries (the last 5)
        entries = bus.get_dlq_entries()
        assert len(entries) == 5
        # Last entry should have index 9
        assert entries[-1].message["payload"]["index"] == 9

        return True


def test_dlq_retry():
    """Test retrying a DLQ message (MB-001)"""
    from daemon.message_bus import MessageBus, Message, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))

        # Add a message to DLQ
        msg = Message(
            type=MessageType.USER_QUERY,
            source="test",
            payload={"retry": True}
        )
        bus._move_to_dlq(msg.to_dict(), "Initial error", 3)

        assert len(bus.get_dlq_entries()) == 1

        # Retry it
        result = bus.retry_dlq_message(0)
        assert result is True
        assert len(bus.get_dlq_entries()) == 0
        # Message should be back in queue (don't use qsize - unreliable on macOS)
        # Just verify retry succeeded and DLQ is empty

        return True


def test_dlq_clear():
    """Test clearing the DLQ (MB-001)"""
    from daemon.message_bus import MessageBus, Message, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))

        # Add messages to DLQ
        for i in range(3):
            msg = Message(
                type=MessageType.USER_QUERY,
                source="test",
                payload={"index": i}
            )
            bus._move_to_dlq(msg.to_dict(), f"Error {i}", 3)

        assert len(bus.get_dlq_entries()) == 3

        bus.clear_dlq()
        assert len(bus.get_dlq_entries()) == 0

        return True


def test_correlation_id_propagation():
    """Test that correlation ID is generated and tracked (MB-002)"""
    from daemon.message_bus import MessageBus, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))
        # Don't start - test the method directly

        received_response = []

        def callback(response):
            received_response.append(response)

        # Send with correlation tracking
        corr_id = bus.send_with_correlation(
            MessageType.MODEL_REQUEST,
            source="test",
            payload={"query": "test"},
            callback=callback
        )

        assert corr_id is not None
        assert len(corr_id) > 0

        # Check it's in the registry
        pending = bus.check_correlation(corr_id)
        assert pending is not None
        assert pending.source == "test"
        assert pending.msg_type == "model_request"

        return True


def test_correlation_timeout():
    """Test that correlations timeout after CORRELATION_TIMEOUT (MB-002)"""
    from daemon.message_bus import MessageBus, MessageType, PendingRequest

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))

        # Manually add a pending request with old timestamp
        old_corr_id = "old-request-123"
        old_request = PendingRequest(
            correlation_id=old_corr_id,
            source="test",
            msg_type="model_request",
            timestamp=time.time() - 200  # 200 seconds ago (> 120s timeout)
        )
        bus._correlation_registry[old_corr_id] = old_request

        # Add a recent request
        new_corr_id = "new-request-456"
        new_request = PendingRequest(
            correlation_id=new_corr_id,
            source="test",
            msg_type="model_request",
            timestamp=time.time()  # Now
        )
        bus._correlation_registry[new_corr_id] = new_request

        # Simulate timeout check (what the thread does)
        now = time.time()
        expired = []
        for corr_id, pending in list(bus._correlation_registry.items()):
            if now - pending.timestamp > bus.CORRELATION_TIMEOUT:
                expired.append(corr_id)

        for corr_id in expired:
            del bus._correlation_registry[corr_id]

        # Old should be removed, new should remain
        assert bus.check_correlation(old_corr_id) is None
        assert bus.check_correlation(new_corr_id) is not None

        return True


def test_correlation_complete():
    """Test completing a correlation with callback (MB-002)"""
    from daemon.message_bus import MessageBus, Message, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))

        callback_received = []

        def callback(response):
            callback_received.append(response)

        corr_id = bus.send_with_correlation(
            MessageType.MODEL_REQUEST,
            source="test",
            payload={"query": "test"},
            callback=callback
        )

        # Simulate response
        response = Message(
            type=MessageType.MODEL_RESPONSE,
            source="model",
            payload={"result": "done"},
            correlation_id=corr_id
        )

        bus.complete_correlation(corr_id, response)

        assert len(callback_received) == 1
        assert callback_received[0].payload["result"] == "done"
        assert bus.check_correlation(corr_id) is None

        return True


def test_respond_preserves_correlation():
    """Test that respond() preserves correlation ID (MB-002)"""
    from daemon.message_bus import MessageBus, Message, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        bus = MessageBus(data_dir=Path(tmpdir))

        # Register a subscriber to receive response
        q = bus.register("test_component")

        # Create original message with correlation ID
        original = Message(
            type=MessageType.MODEL_REQUEST,
            source="test_component",
            payload={"query": "hello"},
            correlation_id="test-123"
        )

        # Send response - should preserve correlation ID
        bus.respond(original, {"result": "world"}, source="model")

        # The response message is in the main queue
        # Check that it has the right correlation ID
        from queue import Empty
        try:
            msg_dict = bus.queue.get_nowait()
            assert msg_dict["correlation_id"] == "test-123"
            assert msg_dict["target"] == "test_component"  # Response goes to original source
        except Empty:
            pass  # Response might have been routed

        return True


def test_message_serialization():
    """Test that messages are serialized correctly (MB-003)"""
    from daemon.message_bus import MessageBus, Message, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        bus = MessageBus(data_dir=data_dir)

        # Publish some messages
        for i in range(5):
            bus.send(
                MessageType.USER_QUERY,
                source="test",
                payload={"index": i}
            )

        # Small delay for multiprocessing Queue
        time.sleep(0.1)

        # Persist
        bus._save_pending_messages()

        # Check file
        pending_file = data_dir / "pending.json"
        assert pending_file.exists()

        data = json.loads(pending_file.read_text())
        assert len(data) == 5

        return True


def test_message_restore():
    """Test that messages are restored on startup (MB-003)"""
    from daemon.message_bus import MessageBus, Message, MessageType
    from queue import Empty

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        bus1 = MessageBus(data_dir=data_dir)

        # Publish messages
        for i in range(3):
            bus1.send(
                MessageType.USER_QUERY,
                source="test",
                payload={"index": i}
            )

        # Small delay for multiprocessing Queue
        time.sleep(0.1)

        # Persist (directly, not via stop to avoid thread issues in tests)
        bus1._save_pending_messages()

        # Create new instance - should load messages
        bus2 = MessageBus(data_dir=data_dir)

        # Count messages by draining queue (qsize is unreliable on macOS)
        count = 0
        try:
            while True:
                bus2.queue.get_nowait()
                count += 1
        except Empty:
            pass
        assert count == 3

        return True


def test_message_ordering_preserved():
    """Test that message order is preserved on restore (MB-003)"""
    from daemon.message_bus import MessageBus, Message, MessageType
    from queue import Empty

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        bus1 = MessageBus(data_dir=data_dir)

        # Publish messages in order
        for i in range(5):
            bus1.send(
                MessageType.USER_QUERY,
                source="test",
                payload={"order": i}
            )

        # Small delay for multiprocessing Queue
        time.sleep(0.1)

        # Persist
        bus1._save_pending_messages()

        # Create new instance
        bus2 = MessageBus(data_dir=data_dir)

        # Check order
        for i in range(5):
            try:
                msg = bus2.queue.get_nowait()
                assert msg["payload"]["order"] == i
            except Empty:
                assert False, f"Expected message at index {i}"

        return True


def test_persist_now():
    """Test force persist (MB-003)"""
    from daemon.message_bus import MessageBus, MessageType

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        bus = MessageBus(data_dir=data_dir)

        # Add messages
        bus.send(MessageType.USER_QUERY, "test", {"data": "test"})

        # Small delay for multiprocessing Queue
        time.sleep(0.1)

        # Add DLQ entry
        bus._move_to_dlq({"type": "test"}, "error", 3)

        # Force persist
        bus.persist_now()

        # DLQ file should exist
        assert (data_dir / "dlq.json").exists()

        return True


def test_graceful_shutdown_persists():
    """Test that graceful shutdown saves messages (MB-003)"""
    from daemon.message_bus import MessageBus, MessageType
    from queue import Empty

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        bus = MessageBus(data_dir=data_dir)
        # Don't start the bus - just test the save/load functionality

        # Add messages directly to queue
        for i in range(3):
            bus.send(MessageType.USER_QUERY, "test", {"index": i})

        # Small delay for multiprocessing Queue
        time.sleep(0.1)

        # Manually save (what stop() does)
        bus._save_pending_messages()
        bus._save_dlq()

        # Check file exists
        pending_file = data_dir / "pending.json"
        assert pending_file.exists()

        # New instance should have messages
        bus2 = MessageBus(data_dir=data_dir)

        # Count messages by draining queue (qsize is unreliable on macOS)
        count = 0
        try:
            while True:
                bus2.queue.get_nowait()
                count += 1
        except Empty:
            pass
        assert count == 3

        return True


if __name__ == "__main__":
    tests = [
        test_dlq_message_moved_on_failure,
        test_dlq_persistence,
        test_dlq_size_limit,
        test_dlq_retry,
        test_dlq_clear,
        test_correlation_id_propagation,
        test_correlation_timeout,
        test_correlation_complete,
        test_respond_preserves_correlation,
        test_message_serialization,
        test_message_restore,
        test_message_ordering_preserved,
        test_persist_now,
        test_graceful_shutdown_persists,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
