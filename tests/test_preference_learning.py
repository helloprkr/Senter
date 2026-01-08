#!/usr/bin/env python3
"""
Tests for PreferenceLearner (CG-004)
"""

import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_preference_learner_clustering():
    """Test that PreferenceLearner clusters queries using TF-IDF"""
    from learning.pattern_detector import PreferenceLearner
    from learning.events_db import UserEventsDB, UserEvent

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        # Create 15 test queries
        now = time.time()
        test_queries = [
            # Coding queries (cluster 1)
            "How do I write a Python function?",
            "What is the best way to debug Python code?",
            "Can you explain Python decorators?",
            "How do I use async await in Python?",
            "What are Python type hints?",
            # Research queries (cluster 2)
            "Tell me about machine learning",
            "What is deep learning?",
            "Explain neural networks",
            "How does AI work?",
            "What are transformers in AI?",
            # General queries (cluster 3)
            "What's the weather like?",
            "Tell me a joke",
            "What time is it?",
            "Who won the game?",
            "What's for dinner?",
        ]

        for i, query in enumerate(test_queries):
            event = UserEvent(
                event_type="query",
                timestamp=now - i * 3600,
                context={"query": query},
                metadata={"topic": "general", "time_of_day": "afternoon"}
            )
            db.log_event(event)

        # Test preference learning
        learner = PreferenceLearner(senter_root)
        preferences = learner.analyze_preferences(min_queries=10)

        # Should have preferences
        assert preferences["query_count"] == 15
        assert preferences["method"] == "tfidf"  # No embedding model available
        assert len(preferences["topic_preferences"]) > 0
        assert preferences["confidence"] > 0

        # Should have peak hours
        assert "peak_hours" in preferences

        # Should detect response style
        assert "response_style" in preferences
        assert "preferred_length" in preferences["response_style"]

        return True


def test_preference_confidence_scoring():
    """Test that preferences have confidence scores based on frequency"""
    from learning.pattern_detector import PreferenceLearner
    from learning.events_db import UserEventsDB, UserEvent

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        now = time.time()
        # Create 20 queries - mostly about Python
        for i in range(15):
            event = UserEvent(
                event_type="query",
                timestamp=now - i * 3600,
                context={"query": f"Python question {i}"},
                metadata={"topic": "coding"}
            )
            db.log_event(event)

        for i in range(5):
            event = UserEvent(
                event_type="query",
                timestamp=now - (15 + i) * 3600,
                context={"query": f"Other question {i}"},
                metadata={"topic": "general"}
            )
            db.log_event(event)

        learner = PreferenceLearner(senter_root)
        preferences = learner.analyze_preferences(min_queries=10)

        # Should have topic preferences with confidence scores
        topic_prefs = preferences["topic_preferences"]
        assert len(topic_prefs) > 0

        # All preferences should have confidence between 0 and 1
        for pref in topic_prefs:
            assert "confidence" in pref
            assert 0 <= pref["confidence"] <= 1
            assert "query_count" in pref

        return True


def test_peak_hours_detection():
    """Test that peak usage hours are detected from timestamps"""
    from learning.pattern_detector import PreferenceLearner
    from learning.events_db import UserEventsDB, UserEvent
    from datetime import datetime

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        # Create queries at specific hours (9 AM, 2 PM, 8 PM)
        base_time = time.time()

        # 10 queries at 9 AM
        for i in range(10):
            event = UserEvent(
                event_type="query",
                timestamp=base_time - (i * 86400) + 9 * 3600,
                context={"query": f"Morning query {i}"},
                metadata={"topic": "general"}
            )
            db.log_event(event)

        # 5 queries at 2 PM
        for i in range(5):
            event = UserEvent(
                event_type="query",
                timestamp=base_time - (i * 86400) + 14 * 3600,
                context={"query": f"Afternoon query {i}"},
                metadata={"topic": "general"}
            )
            db.log_event(event)

        learner = PreferenceLearner(senter_root)
        preferences = learner.analyze_preferences(min_queries=10)

        # Should detect peak hours
        peak_hours = preferences.get("peak_hours", [])
        assert len(peak_hours) > 0

        # Peak hours should have hour, count, percentage
        for ph in peak_hours:
            assert "hour" in ph
            assert "count" in ph
            assert "percentage" in ph
            assert 0 <= ph["hour"] <= 23

        return True


def test_preference_persistence():
    """Test that preferences persist to file"""
    from learning.pattern_detector import PreferenceLearner
    from learning.events_db import UserEventsDB, UserEvent

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db = UserEventsDB(senter_root=senter_root)

        now = time.time()
        for i in range(12):
            event = UserEvent(
                event_type="query",
                timestamp=now - i * 3600,
                context={"query": f"Test query {i}"},
                metadata={"topic": "general"}
            )
            db.log_event(event)

        # Save preferences
        learner = PreferenceLearner(senter_root)
        preferences = learner.update_preferences(min_queries=10)
        learner.save_preferences(preferences)

        # Load preferences
        loaded = learner.load_preferences()
        assert loaded is not None
        assert loaded["query_count"] == preferences["query_count"]
        assert "topic_preferences" in loaded

        return True


def test_system_prompt_additions():
    """Test that preferences generate system prompt additions"""
    from learning.pattern_detector import PreferenceLearner

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)

        # Create preferences manually for testing
        learner = PreferenceLearner(senter_root)
        preferences = {
            "analyzed_at": "2026-01-07T12:00:00",
            "query_count": 50,
            "method": "tfidf",
            "confidence": 1.0,
            "topic_preferences": [
                {"topic": "coding", "confidence": 0.5, "query_count": 25},
                {"topic": "research", "confidence": 0.3, "query_count": 15},
            ],
            "response_style": {
                "preferred_length": "detailed",
                "formality": "formal",
                "detail_level": "high"
            },
            "peak_hours": []
        }
        learner.save_preferences(preferences)

        # Get system prompt additions
        additions = learner.get_system_prompt_additions()

        # Should include response style
        assert "detailed" in additions or "thorough" in additions
        assert "formal" in additions or "professional" in additions

        # Should include topic interests
        assert "coding" in additions or "research" in additions

        return True


if __name__ == "__main__":
    tests = [
        test_preference_learner_clustering,
        test_preference_confidence_scoring,
        test_peak_hours_detection,
        test_preference_persistence,
        test_system_prompt_additions,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
