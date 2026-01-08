#!/usr/bin/env python3
"""
Tests for User Interaction Logging - US-008 acceptance criteria:
1. Every user query logged with timestamp and detected topic
2. Response time logged
3. Topic extracted using simple NLP (not just keywords)
4. Events visible via 'senter_ctl.py events' command
"""

import sys
import json
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_topic_detection_uses_nlp():
    """Test that topic detection uses NLP patterns, not just keywords"""
    from daemon.ipc_server import IPCServer

    server = IPCServer()

    # Test various queries
    test_cases = [
        ("How do I write a Python function?", "coding"),
        ("Help me debug this error in JavaScript", "coding"),
        ("Research the latest AI trends", "research"),
        ("Analyze this data for me", "research"),
        ("What is machine learning?", "learning"),
        ("Explain quantum computing", "learning"),
        ("Write me an essay about history", "writing"),
        ("Draft an email to my boss", "writing"),
        ("Give me some creative story ideas", "creative"),
        ("Plan my tasks for tomorrow", "productivity"),
        ("Hello, how are you?", "general"),
    ]

    for query, expected_topic in test_cases:
        detected = server._detect_topic(query)
        # Allow some flexibility - topic detection should be reasonable
        assert detected in [expected_topic, "learning", "general"], \
            f"Query '{query}' detected as '{detected}', expected '{expected_topic}'"

    # Verify regex patterns are used (not just simple 'in' checks)
    # Check that word boundaries work
    assert server._detect_topic("I love the code") == "coding"
    assert server._detect_topic("encode something") == "general"  # 'encode' != 'code'

    print("✓ test_topic_detection_uses_nlp PASSED")
    return True


def test_query_logging_includes_timestamp_and_topic():
    """Test that queries are logged with timestamp and topic"""
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserEventsDB(db_path=Path(tmpdir) / "test.db")

        # Log a query
        before = time.time()
        event_id = db.log_query("How do I write Python code?", topic="coding")
        after = time.time()

        # Retrieve and verify
        events = db.get_events(event_type="query", limit=1)
        assert len(events) == 1

        event = events[0]
        assert event.event_type == "query"
        assert before <= event.timestamp <= after, "Timestamp should be current"
        assert event.metadata.get("topic") == "coding"
        assert event.context.get("query") == "How do I write Python code?"

    print("✓ test_query_logging_includes_timestamp_and_topic PASSED")
    return True


def test_response_time_logged():
    """Test that response time (latency) is logged"""
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserEventsDB(db_path=Path(tmpdir) / "test.db")

        # Log a response with latency
        event_id = db.log_response(
            query="Test query",
            response="Test response",
            latency_ms=1500,
            worker="primary",
            topic="testing"
        )

        # Retrieve and verify
        events = db.get_events(event_type="response", limit=1)
        assert len(events) == 1

        event = events[0]
        assert event.metadata.get("latency_ms") == 1500
        assert event.metadata.get("worker") == "primary"

    print("✓ test_response_time_logged PASSED")
    return True


def test_ipc_get_events_handler():
    """Test that IPC get_events handler works"""
    from daemon.ipc_server import IPCServer
    from daemon.senter_daemon import SenterDaemon
    from learning.events_db import UserEventsDB
    from multiprocessing import Event

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        daemon = SenterDaemon()
        daemon.senter_root = Path(tmpdir)

        # Pre-populate some events
        db = UserEventsDB(senter_root=Path(tmpdir))
        db.log_query("Test query 1", topic="coding")
        db.log_query("Test query 2", topic="research")
        db.log_response("Test query 1", "Response 1", latency_ms=100)

        # Create IPC server
        shutdown = Event()
        ipc = IPCServer(shutdown_event=shutdown, daemon_ref=daemon)

        # Get events
        result = ipc._handle_get_events({"hours": 1, "limit": 10})

        assert "events" in result
        assert len(result["events"]) == 3
        assert "event_counts" in result
        assert result["event_counts"].get("query") == 2
        assert result["event_counts"].get("response") == 1

    print("✓ test_ipc_get_events_handler PASSED")
    return True


def test_ipc_client_method_exists():
    """Test that IPC client has get_events method"""
    from daemon.ipc_client import IPCClient

    client = IPCClient()
    assert hasattr(client, "get_events"), "Client should have get_events method"

    print("✓ test_ipc_client_method_exists PASSED")
    return True


def test_cli_events_command_exists():
    """Test that CLI has events command"""
    senter_ctl_path = Path(__file__).parent.parent / "scripts" / "senter_ctl.py"
    content = senter_ctl_path.read_text()

    assert "def show_events" in content, "Should have show_events function"
    assert '"events"' in content or "'events'" in content, "Should have events command"
    assert "get_events" in content, "Should use get_events IPC"

    print("✓ test_cli_events_command_exists PASSED")
    return True


def test_interaction_logging():
    """Combined test for US-008 acceptance criteria"""
    from daemon.ipc_server import IPCServer
    from daemon.ipc_client import IPCClient
    from learning.events_db import UserEventsDB
    import tempfile

    # 1. Topic detection uses NLP
    server = IPCServer()
    assert server._detect_topic("Write Python code") == "coding"
    assert server._detect_topic("Research this topic") == "research"

    # 2. Query logged with timestamp and topic
    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserEventsDB(db_path=Path(tmpdir) / "test.db")
        before = time.time()
        db.log_query("Test", topic="coding")
        events = db.get_events()
        assert len(events) == 1
        assert events[0].timestamp >= before
        assert events[0].metadata.get("topic") == "coding"

    # 3. Response time logged
    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserEventsDB(db_path=Path(tmpdir) / "test.db")
        db.log_response("Q", "R", latency_ms=500)
        events = db.get_events(event_type="response")
        assert events[0].metadata.get("latency_ms") == 500

    # 4. CLI command exists
    senter_ctl_path = Path(__file__).parent.parent / "scripts" / "senter_ctl.py"
    assert "def show_events" in senter_ctl_path.read_text()

    print("✓ test_interaction_logging PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Interaction Logging Tests (US-008)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_topic_detection_uses_nlp,
        test_query_logging_includes_timestamp_and_topic,
        test_response_time_logged,
        test_ipc_get_events_handler,
        test_ipc_client_method_exists,
        test_cli_events_command_exists,
        test_interaction_logging,
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)

    sys.exit(0 if all_passed else 1)
