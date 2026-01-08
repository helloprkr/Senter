#!/usr/bin/env python3
"""
Tests for UI-003: Feedback That Improves Future Research

Tests feedback analysis and learning system.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.feedback import FeedbackAnalyzer, FeedbackStats, TopicPattern
from research.research_store import ResearchStore
from research.synthesizer import SynthesizedResearch


# ========== Unit Tests ==========

def test_feedback_stats_dataclass():
    """Test FeedbackStats creation"""
    stats = FeedbackStats(
        total_ratings=10,
        average_rating=4.2,
        rating_distribution={1: 0, 2: 1, 3: 2, 4: 4, 5: 3},
        top_rated_topics=["topic1", "topic2"],
        low_rated_topics=["topic3"]
    )

    assert stats.total_ratings == 10
    assert stats.average_rating == 4.2
    assert stats.rating_distribution[4] == 4

    return True


def test_topic_pattern_dataclass():
    """Test TopicPattern creation"""
    from datetime import datetime

    pattern = TopicPattern(
        pattern="kubernetes",
        avg_rating=4.5,
        count=5,
        last_seen=datetime.now()
    )

    assert pattern.pattern == "kubernetes"
    assert pattern.avg_rating == 4.5
    assert pattern.count == 5

    return True


def test_analyzer_creation():
    """Test FeedbackAnalyzer can be created"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Create store first to initialize DB
        store = ResearchStore(db_path)

        analyzer = FeedbackAnalyzer(db_path)
        assert analyzer is not None

    return True


def test_empty_stats():
    """Test stats with no data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)
        analyzer = FeedbackAnalyzer(db_path)

        stats = analyzer.get_stats()

        assert stats.total_ratings == 0
        assert stats.average_rating == 0.0

    return True


def test_stats_with_feedback():
    """Test stats with actual feedback"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Add research with ratings
        for i, (topic, rating) in enumerate([
            ("python async", 5),
            ("kubernetes networking", 4),
            ("rust ownership", 4),
            ("bad topic", 2)
        ]):
            research = SynthesizedResearch(
                topic=topic,
                summary=f"Summary about {topic}",
                key_insights=[],
                sources_used=["https://example.com"],
                confidence=0.8
            )
            rid = store.store(research)
            store.set_feedback(rid, rating)

        analyzer = FeedbackAnalyzer(db_path)
        stats = analyzer.get_stats()

        assert stats.total_ratings == 4
        assert 3.5 <= stats.average_rating <= 4.0
        assert stats.rating_distribution[5] == 1
        assert stats.rating_distribution[4] == 2
        assert stats.rating_distribution[2] == 1

    return True


def test_topic_patterns():
    """Test topic pattern extraction"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Add research with same keyword
        for i in range(3):
            research = SynthesizedResearch(
                topic=f"kubernetes topic {i}",
                summary="Summary",
                key_insights=[],
                sources_used=[],
                confidence=0.8
            )
            rid = store.store(research)
            store.set_feedback(rid, 4)  # All rated 4

        analyzer = FeedbackAnalyzer(db_path)
        patterns = analyzer.get_topic_patterns(min_count=2)

        # Should find "kubernetes" pattern
        kubernetes_patterns = [p for p in patterns if p.pattern == "kubernetes"]
        assert len(kubernetes_patterns) == 1
        assert kubernetes_patterns[0].avg_rating == 4.0
        assert kubernetes_patterns[0].count == 3

    return True


def test_should_research_topic():
    """Test topic recommendation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Good topic history
        for i in range(3):
            research = SynthesizedResearch(
                topic=f"python advanced {i}",
                summary="Summary",
                key_insights=[],
                sources_used=[],
                confidence=0.8
            )
            rid = store.store(research)
            store.set_feedback(rid, 5)

        analyzer = FeedbackAnalyzer(db_path)

        # Should recommend similar topic
        advice = analyzer.should_research_topic("python basics")
        assert advice["recommend"] is True

    return True


def test_improvement_suggestions():
    """Test improvement suggestions"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Add low-rated research (need 5+ for quality analysis)
        for i in range(6):
            research = SynthesizedResearch(
                topic=f"bad topic {i}",
                summary="Summary",
                key_insights=[],
                sources_used=["https://bad-source.com"],
                confidence=0.5
            )
            rid = store.store(research)
            store.set_feedback(rid, 2)

        analyzer = FeedbackAnalyzer(db_path)
        suggestions = analyzer.get_improvement_suggestions()

        assert len(suggestions) > 0
        # Should mention quality issue or source issue
        has_quality_mention = any(
            "quality" in s.lower() or "below" in s.lower() or "source" in s.lower()
            for s in suggestions
        )
        assert has_quality_mention

    return True


def test_source_quality():
    """Test source quality analysis"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Add research with sources
        research = SynthesizedResearch(
            topic="test topic",
            summary="Summary",
            key_insights=[],
            sources_used=[
                "https://good-source.com/article1",
                "https://good-source.com/article2"
            ],
            confidence=0.8
        )
        rid = store.store(research)
        store.set_feedback(rid, 5)

        analyzer = FeedbackAnalyzer(db_path)
        quality = analyzer.get_source_quality()

        assert "good-source.com" in quality
        assert quality["good-source.com"] == 5.0

    return True


if __name__ == "__main__":
    tests = [
        test_feedback_stats_dataclass,
        test_topic_pattern_dataclass,
        test_analyzer_creation,
        test_empty_stats,
        test_stats_with_feedback,
        test_topic_patterns,
        test_should_research_topic,
        test_improvement_suggestions,
        test_source_quality,
    ]

    print("=" * 60)
    print("UI-003: Feedback System Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                print(f"✓ {test.__name__}")
                passed += 1
            else:
                print(f"✗ {test.__name__}: returned False")
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
