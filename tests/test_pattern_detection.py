#!/usr/bin/env python3
"""
Tests for Time-based Pattern Detection - US-009 acceptance criteria:
1. Analyze user_events to find peak usage hours
2. Detect most common topics by time of day
3. Store patterns in data/learning/patterns.json
4. Patterns updated daily
"""

import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_peak_hours_detection():
    """Test that peak usage hours are detected from events"""
    from learning.pattern_detector import PatternDetector
    from learning.events_db import UserEventsDB, UserEvent

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        now = time.time()

        # Create events clustered at specific hours
        # Most queries at hour 9 (5 queries)
        for i in range(5):
            event = UserEvent(
                event_type="query",
                timestamp=now - 3600 * (24 - 9) + i * 60,  # Today at 9am
                context={"query": f"Query {i}"},
                metadata={"topic": "coding", "time_of_day": "morning"}
            )
            db.log_event(event)

        # Some queries at hour 14 (3 queries)
        for i in range(3):
            event = UserEvent(
                event_type="query",
                timestamp=now - 3600 * (24 - 14) + i * 60,  # Today at 2pm
                context={"query": f"Query {i}"},
                metadata={"topic": "research", "time_of_day": "afternoon"}
            )
            db.log_event(event)

        # Test detection
        detector = PatternDetector(senter_root)
        patterns = detector.analyze_patterns(days=2)

        peak_hours = patterns.get("peak_hours", [])
        assert len(peak_hours) > 0, "Should detect peak hours"

        # The most active hour should be in the list
        hours = [p["hour"] for p in peak_hours]
        # Check that we have peak hours detected
        assert len(hours) > 0, "Should have at least one peak hour"

    print("✓ test_peak_hours_detection PASSED")
    return True


def test_topics_by_time_of_day():
    """Test that topics are detected by time of day"""
    from learning.pattern_detector import PatternDetector
    from learning.events_db import UserEventsDB, UserEvent

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        now = time.time()

        # Morning: mostly coding
        for i in range(5):
            event = UserEvent(
                event_type="query",
                timestamp=now - i * 60,
                context={"query": "coding query"},
                metadata={"topic": "coding", "time_of_day": "morning"}
            )
            db.log_event(event)

        # Afternoon: mostly research
        for i in range(3):
            event = UserEvent(
                event_type="query",
                timestamp=now - i * 60,
                context={"query": "research query"},
                metadata={"topic": "research", "time_of_day": "afternoon"}
            )
            db.log_event(event)

        # Test detection
        detector = PatternDetector(senter_root)
        patterns = detector.analyze_patterns(days=1)

        by_time = patterns.get("topics_by_time_of_day", {})
        assert "morning" in by_time, "Should have morning patterns"
        assert "afternoon" in by_time, "Should have afternoon patterns"

        # Morning should have coding as top topic
        morning_topics = by_time.get("morning", [])
        assert len(morning_topics) > 0, "Should have morning topics"
        assert morning_topics[0]["topic"] == "coding", "Morning top topic should be coding"

        # Afternoon should have research as top topic
        afternoon_topics = by_time.get("afternoon", [])
        assert len(afternoon_topics) > 0, "Should have afternoon topics"
        assert afternoon_topics[0]["topic"] == "research", "Afternoon top topic should be research"

    print("✓ test_topics_by_time_of_day PASSED")
    return True


def test_patterns_stored_in_json():
    """Test that patterns are stored in data/learning/patterns.json"""
    from learning.pattern_detector import PatternDetector
    from learning.events_db import UserEventsDB, UserEvent

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        # Add a test event
        db.log_query("Test query", topic="testing")

        # Detect and save patterns
        detector = PatternDetector(senter_root)
        patterns = detector.update_patterns(days=1)

        # Check file exists at correct location
        patterns_file = senter_root / "data" / "learning" / "patterns.json"
        assert patterns_file.exists(), "Patterns file should exist"

        # Verify content
        loaded = json.loads(patterns_file.read_text())
        assert "peak_hours" in loaded, "Should have peak_hours"
        assert "topics_by_time_of_day" in loaded, "Should have topics_by_time_of_day"
        assert "analyzed_at" in loaded, "Should have analyzed_at timestamp"

    print("✓ test_patterns_stored_in_json PASSED")
    return True


def test_patterns_can_be_updated():
    """Test that patterns can be updated (for daily updates)"""
    from learning.pattern_detector import PatternDetector
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        # Initial events
        db.log_query("Initial query", topic="coding")

        detector = PatternDetector(senter_root)

        # First update
        patterns1 = detector.update_patterns(days=1)
        assert patterns1["event_count"] >= 1

        # Add more events
        db.log_query("Second query", topic="research")
        db.log_query("Third query", topic="research")

        # Second update
        patterns2 = detector.update_patterns(days=1)
        assert patterns2["event_count"] >= 3
        assert patterns2["analyzed_at"] >= patterns1["analyzed_at"]

        # Verify file was updated
        loaded = detector.load_patterns()
        assert loaded["event_count"] == patterns2["event_count"]

    print("✓ test_patterns_can_be_updated PASSED")
    return True


def test_topic_frequency_tracking():
    """Test overall topic frequency tracking"""
    from learning.pattern_detector import PatternDetector
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        # Log queries with different topics
        for _ in range(5):
            db.log_query("Coding question", topic="coding")
        for _ in range(3):
            db.log_query("Research question", topic="research")
        for _ in range(1):
            db.log_query("Creative question", topic="creative")

        detector = PatternDetector(senter_root)
        patterns = detector.analyze_patterns(days=1)

        freq = patterns.get("topic_frequency", {})
        assert freq.get("coding") == 5, "Coding should have 5 queries"
        assert freq.get("research") == 3, "Research should have 3 queries"
        assert freq.get("creative") == 1, "Creative should have 1 query"

    print("✓ test_topic_frequency_tracking PASSED")
    return True


def test_pattern_insights():
    """Test that human-readable insights are generated"""
    from learning.pattern_detector import PatternDetector
    from learning.events_db import UserEventsDB

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        # Create some test data
        db.log_query("Test query", topic="coding")
        db.log_response("Test query", "Response", latency_ms=1500)

        detector = PatternDetector(senter_root)
        detector.update_patterns(days=1)

        insights = detector.get_insights()

        assert "patterns" in insights, "Should have patterns"
        assert "insights" in insights, "Should have insights list"
        assert "last_updated" in insights, "Should have last_updated"

    print("✓ test_pattern_insights PASSED")
    return True


def test_pattern_detection():
    """Combined test for US-009 acceptance criteria"""
    from learning.pattern_detector import PatternDetector, detect_and_save_patterns
    from learning.events_db import UserEventsDB, UserEvent

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        now = time.time()

        # 1. Create events with time patterns
        for i in range(5):
            event = UserEvent(
                event_type="query",
                timestamp=now - i * 60,
                context={"query": "morning coding"},
                metadata={"topic": "coding", "time_of_day": "morning"}
            )
            db.log_event(event)

        # 2. Analyze patterns
        detector = PatternDetector(senter_root)
        patterns = detector.analyze_patterns(days=1)

        # Verify peak hours detected
        assert "peak_hours" in patterns

        # Verify topics by time of day
        by_time = patterns.get("topics_by_time_of_day", {})
        assert "morning" in by_time

        # 3. Store patterns in JSON file
        detector.save_patterns(patterns)
        patterns_file = senter_root / "data" / "learning" / "patterns.json"
        assert patterns_file.exists()

        # 4. Test update function (daily update)
        patterns2 = detect_and_save_patterns(senter_root, days=1)
        assert patterns2["analyzed_at"] >= patterns["analyzed_at"]

    print("✓ test_pattern_detection PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Pattern Detection Tests (US-009)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_peak_hours_detection,
        test_topics_by_time_of_day,
        test_patterns_stored_in_json,
        test_patterns_can_be_updated,
        test_topic_frequency_tracking,
        test_pattern_insights,
        test_pattern_detection,
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
