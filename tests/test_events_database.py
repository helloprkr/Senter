#!/usr/bin/env python3
"""
Tests for Behavioral Event Database - US-007 acceptance criteria:
1. SQLite table 'user_events' with: timestamp, event_type, context, metadata
2. Events logged: query, response, topic, time_of_day
3. Database file at data/learning/events.db
4. Query to retrieve events by time range
"""

import sys
import json
import time
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_user_events_table_schema():
    """Test that user_events table has correct schema"""
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserEventsDB(db_path=Path(tmpdir) / "test.db")

        # Connect directly to verify schema
        conn = sqlite3.connect(str(db.db_path))
        cursor = conn.execute("PRAGMA table_info(user_events)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        # Check required columns
        assert "timestamp" in columns, "Should have timestamp column"
        assert "event_type" in columns, "Should have event_type column"
        assert "context" in columns, "Should have context column"
        assert "metadata" in columns, "Should have metadata column"

        # Check types
        assert columns["timestamp"] == "REAL", "timestamp should be REAL"
        assert columns["event_type"] == "TEXT", "event_type should be TEXT"
        assert columns["context"] == "TEXT", "context should be TEXT (JSON)"
        assert columns["metadata"] == "TEXT", "metadata should be TEXT (JSON)"

    print("✓ test_user_events_table_schema PASSED")
    return True


def test_events_logged_with_required_fields():
    """Test that events are logged with query, response, topic, time_of_day"""
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserEventsDB(db_path=Path(tmpdir) / "test.db")

        # Log a query event
        event_id = db.log_query("Test query", topic="testing")
        assert event_id > 0, "Should return event ID"

        # Retrieve and verify
        events = db.get_events(event_type="query", limit=1)
        assert len(events) == 1, "Should retrieve logged query"

        event = events[0]
        assert event.event_type == "query"
        assert event.context.get("query") == "Test query"
        assert event.metadata.get("topic") == "testing"
        assert "time_of_day" in event.metadata, "Should have time_of_day"
        assert event.metadata["time_of_day"] in ["morning", "afternoon", "evening", "night"]

        # Log a response event
        event_id = db.log_response(
            query="Test query",
            response="Test response",
            latency_ms=100,
            worker="primary",
            topic="testing"
        )

        events = db.get_events(event_type="response", limit=1)
        assert len(events) == 1, "Should retrieve logged response"

        event = events[0]
        assert event.event_type == "response"
        assert "response_preview" in event.context
        assert event.metadata.get("latency_ms") == 100
        assert event.metadata.get("topic") == "testing"
        assert "time_of_day" in event.metadata

    print("✓ test_events_logged_with_required_fields PASSED")
    return True


def test_database_file_location():
    """Test that database is created at data/learning/events.db"""
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)

        # Use senter_root parameter
        db = UserEventsDB(senter_root=senter_root)

        # Verify path
        expected_path = senter_root / "data" / "learning" / "events.db"
        assert db.db_path == expected_path, \
            f"Database should be at {expected_path}, got {db.db_path}"

        # Verify file exists after logging something
        db.log_query("test")
        assert db.db_path.exists(), "Database file should exist"

    print("✓ test_database_file_location PASSED")
    return True


def test_query_events_by_time_range():
    """Test that events can be queried by time range"""
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserEventsDB(db_path=Path(tmpdir) / "test.db")

        now = time.time()

        # Log events at different times (simulate)
        for i in range(5):
            # Log events with timestamps in the past
            from learning.events_db import UserEvent
            event = UserEvent(
                event_type="query",
                timestamp=now - (i * 3600),  # i hours ago
                context={"query": f"Query {i}"},
                metadata={"hour": i}
            )
            db.log_event(event)

        # Query last 2 hours
        events = db.get_events_by_time_range(hours=2)
        assert len(events) >= 2, "Should get events from last 2 hours"

        # Query with since/until
        events = db.get_events(since=now - 7200, until=now - 3600)
        assert len(events) >= 1, "Should get events in time window"

        # Query all
        events = db.get_events_by_time_range(hours=24)
        assert len(events) == 5, "Should get all events"

    print("✓ test_query_events_by_time_range PASSED")
    return True


def test_event_counts_and_stats():
    """Test event counting and statistics functions"""
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserEventsDB(db_path=Path(tmpdir) / "test.db")

        # Log various events
        db.log_query("Query 1", topic="coding")
        db.log_query("Query 2", topic="research")
        db.log_response("Query 1", "Response 1", latency_ms=100)
        db.log_response("Query 2", "Response 2", latency_ms=200)
        db.log_response("Query 3", "Response 3", latency_ms=150)

        # Get counts
        counts = db.get_event_counts(hours=1)
        assert counts.get("query") == 2, "Should have 2 queries"
        assert counts.get("response") == 3, "Should have 3 responses"

        # Get stats
        stats = db.get_stats()
        assert stats["total_events"] == 5, "Should have 5 total events"
        assert "query" in stats["events_by_type"]
        assert "response" in stats["events_by_type"]
        assert stats["events_by_type"]["query"] == 2
        assert stats["events_by_type"]["response"] == 3

    print("✓ test_event_counts_and_stats PASSED")
    return True


def test_user_event_dataclass():
    """Test UserEvent dataclass serialization"""
    from learning.events_db import UserEvent

    event = UserEvent(
        event_type="query",
        timestamp=time.time(),
        context={"query": "test", "session": "abc"},
        metadata={"topic": "testing", "time_of_day": "morning"}
    )

    # Test to_dict
    d = event.to_dict()
    assert d["event_type"] == "query"
    assert d["context"]["query"] == "test"
    assert d["metadata"]["topic"] == "testing"
    assert "datetime" in d, "Should include human-readable datetime"

    # Test from_dict
    event2 = UserEvent.from_dict(d)
    assert event2.event_type == event.event_type
    assert event2.context == event.context
    assert event2.metadata == event.metadata

    print("✓ test_user_event_dataclass PASSED")
    return True


def test_events_database():
    """Combined test for US-007 acceptance criteria"""
    from learning.events_db import UserEventsDB, UserEvent
    import sqlite3

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Database at data/learning/events.db
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)
        expected_path = senter_root / "data" / "learning" / "events.db"
        assert db.db_path == expected_path

        # 2. Table has correct schema (timestamp, event_type, context, metadata)
        conn = sqlite3.connect(str(db.db_path))
        cursor = conn.execute("PRAGMA table_info(user_events)")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        assert "timestamp" in columns
        assert "event_type" in columns
        assert "context" in columns
        assert "metadata" in columns

        # 3. Events logged with query, response, topic, time_of_day
        db.log_query("Test query", topic="coding")
        db.log_response("Test query", "Test response", topic="coding")

        events = db.get_events()
        assert len(events) == 2
        for event in events:
            assert "time_of_day" in event.metadata

        # 4. Query by time range
        recent = db.get_events_by_time_range(hours=1)
        assert len(recent) == 2

    print("✓ test_events_database PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Events Database Tests (US-007)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_user_events_table_schema,
        test_events_logged_with_required_fields,
        test_database_file_location,
        test_query_events_by_time_range,
        test_event_counts_and_stats,
        test_user_event_dataclass,
        test_events_database,
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
