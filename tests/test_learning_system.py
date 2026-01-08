#!/usr/bin/env python3
"""
Tests for Learning System (LS-001, LS-002, LS-003, LS-004)
Tests temporal patterns, preference prediction, feedback, and expertise tracking.
"""

import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== LS-001: Temporal Pattern Analysis Tests ==========

def test_temporal_analyzer_creation():
    """Test TemporalPatternAnalyzer can be created (LS-001)"""
    from learning.pattern_detector import TemporalPatternAnalyzer

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = TemporalPatternAnalyzer(Path(tmpdir))

        assert analyzer is not None
        assert analyzer.MIN_DAYS_FOR_ANALYSIS == 14

    return True


def test_temporal_empty_patterns():
    """Test empty patterns when no data (LS-001)"""
    from learning.pattern_detector import TemporalPatternAnalyzer

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = TemporalPatternAnalyzer(Path(tmpdir))
        patterns = analyzer._empty_temporal_patterns()

        assert patterns["days_analyzed"] == 0
        assert patterns["query_count"] == 0
        assert patterns["peak_hours"] == []
        assert patterns["confidence"] == 0.0

    return True


def test_hour_to_label():
    """Test hour to label conversion (LS-001)"""
    from learning.pattern_detector import TemporalPatternAnalyzer

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = TemporalPatternAnalyzer(Path(tmpdir))

        assert analyzer._hour_to_label(8) == "morning"
        assert analyzer._hour_to_label(14) == "afternoon"
        assert analyzer._hour_to_label(19) == "evening"
        assert analyzer._hour_to_label(23) == "night"
        assert analyzer._hour_to_label(2) == "night"

    return True


def test_temporal_save_load():
    """Test saving and loading temporal patterns (LS-001)"""
    from learning.pattern_detector import TemporalPatternAnalyzer

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = TemporalPatternAnalyzer(Path(tmpdir))

        # Create sample patterns
        patterns = {
            "analyzed_at": datetime.now().isoformat(),
            "days_analyzed": 14,
            "query_count": 100,
            "peak_hours": [{"hour": 10, "count": 25}],
            "confidence": 1.0
        }

        analyzer.save_patterns(patterns)

        loaded = analyzer.load_patterns()
        assert loaded is not None
        assert loaded["days_analyzed"] == 14
        assert loaded["query_count"] == 100

    return True


def test_detect_activity_streaks():
    """Test activity streak detection (LS-001)"""
    from learning.pattern_detector import TemporalPatternAnalyzer
    from dataclasses import dataclass

    @dataclass
    class MockEvent:
        timestamp: float

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = TemporalPatternAnalyzer(Path(tmpdir))

        # Create events over 5 consecutive days
        now = time.time()
        events = []
        for i in range(5):
            events.append(MockEvent(timestamp=now - i * 24 * 3600))

        streaks = analyzer._detect_activity_streaks(events)

        assert len(streaks) >= 1
        assert streaks[0]["length"] >= 3

    return True


# ========== LS-002: Preference Prediction Model Tests ==========

def test_prediction_model_creation():
    """Test PreferencePredictionModel can be created (LS-002)"""
    from learning.pattern_detector import PreferencePredictionModel

    with tempfile.TemporaryDirectory() as tmpdir:
        model = PreferencePredictionModel(Path(tmpdir))

        assert model is not None
        assert model.predictions_file.parent.exists()

    return True


def test_empty_predictions():
    """Test empty predictions structure (LS-002)"""
    from learning.pattern_detector import PreferencePredictionModel

    with tempfile.TemporaryDirectory() as tmpdir:
        model = PreferencePredictionModel(Path(tmpdir))
        predictions = model._empty_predictions()

        assert predictions["sample_size"] == 0
        assert predictions["response_length"]["prediction"] == "medium"
        assert predictions["formality"]["prediction"] == "casual"
        assert predictions["detail_level"]["prediction"] == "moderate"
        assert predictions["overall_confidence"] == 0.0

    return True


def test_predict_response_length():
    """Test response length prediction (LS-002)"""
    from learning.pattern_detector import PreferencePredictionModel

    with tempfile.TemporaryDirectory() as tmpdir:
        model = PreferencePredictionModel(Path(tmpdir))

        # Test with brief keywords
        brief_queries = [
            {"text": "give me a quick answer"},
            {"text": "brief summary please"},
            {"text": "tldr"},
            {"text": "short version"},
            {"text": "just tell me"},
        ]
        result = model._predict_response_length(brief_queries)
        assert result["brief_signal_count"] >= 4

        # Test with detailed keywords
        detailed_queries = [
            {"text": "explain in detail"},
            {"text": "comprehensive overview"},
            {"text": "elaborate on this"},
        ]
        result = model._predict_response_length(detailed_queries)
        assert result["detailed_signal_count"] >= 2

    return True


def test_predict_formality():
    """Test formality prediction (LS-002)"""
    from learning.pattern_detector import PreferencePredictionModel

    with tempfile.TemporaryDirectory() as tmpdir:
        model = PreferencePredictionModel(Path(tmpdir))

        # Test formal indicators
        formal_queries = [
            {"text": "could you please help me"},
            {"text": "i would appreciate"},
            {"text": "kindly assist"},
        ]
        result = model._predict_formality(formal_queries)
        assert result["formal_signal_count"] >= 2

        # Test casual indicators
        casual_queries = [
            {"text": "hey what's up"},
            {"text": "yo gimme the answer"},
            {"text": "gonna need help lol"},
        ]
        result = model._predict_formality(casual_queries)
        assert result["casual_signal_count"] >= 2

    return True


def test_prediction_save_load():
    """Test saving and loading predictions (LS-002)"""
    from learning.pattern_detector import PreferencePredictionModel

    with tempfile.TemporaryDirectory() as tmpdir:
        model = PreferencePredictionModel(Path(tmpdir))

        predictions = {
            "predicted_at": datetime.now().isoformat(),
            "sample_size": 50,
            "response_length": {"prediction": "detailed", "confidence": 0.8},
            "overall_confidence": 0.75
        }

        model.save_predictions(predictions)

        loaded = model.load_predictions()
        assert loaded is not None
        assert loaded["sample_size"] == 50
        assert loaded["response_length"]["prediction"] == "detailed"

    return True


# ========== LS-003: Feedback Collection Tests ==========

def test_feedback_collector_creation():
    """Test FeedbackCollector can be created (LS-003)"""
    from learning.pattern_detector import FeedbackCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FeedbackCollector(Path(tmpdir))

        assert collector is not None
        assert len(collector.POSITIVE_KEYWORDS) > 0
        assert len(collector.NEGATIVE_KEYWORDS) > 0
        assert len(collector.CORRECTION_KEYWORDS) > 0

    return True


def test_detect_positive_feedback():
    """Test positive feedback detection (LS-003)"""
    from learning.pattern_detector import FeedbackCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FeedbackCollector(Path(tmpdir))

        feedback = collector.detect_feedback("thanks, that was really helpful!")
        assert feedback is not None
        assert feedback.feedback_type == "positive"

        feedback = collector.detect_feedback("great answer!")
        assert feedback is not None
        assert feedback.feedback_type == "positive"

    return True


def test_detect_negative_feedback():
    """Test negative feedback detection (LS-003)"""
    from learning.pattern_detector import FeedbackCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FeedbackCollector(Path(tmpdir))

        feedback = collector.detect_feedback("that's wrong")
        assert feedback is not None
        assert feedback.feedback_type == "negative"

        feedback = collector.detect_feedback("no, that's incorrect")
        assert feedback is not None
        assert feedback.feedback_type == "negative"

    return True


def test_detect_correction_feedback():
    """Test correction feedback detection (LS-003)"""
    from learning.pattern_detector import FeedbackCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FeedbackCollector(Path(tmpdir))

        feedback = collector.detect_feedback("actually, the correct answer is 42")
        assert feedback is not None
        assert feedback.feedback_type == "correction"

        feedback = collector.detect_feedback("correction: it should be X not Y")
        assert feedback is not None
        assert feedback.feedback_type == "correction"

    return True


def test_record_and_summarize_feedback():
    """Test recording and summarizing feedback (LS-003)"""
    from learning.pattern_detector import FeedbackCollector, FeedbackEntry

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FeedbackCollector(Path(tmpdir))

        # Record some feedback
        collector.record_feedback(FeedbackEntry(
            feedback_type="positive",
            message_id="1",
            timestamp=time.time(),
            original_response="test"
        ))
        collector.record_feedback(FeedbackEntry(
            feedback_type="negative",
            message_id="2",
            timestamp=time.time(),
            original_response="test"
        ))
        collector.record_feedback(FeedbackEntry(
            feedback_type="positive",
            message_id="3",
            timestamp=time.time(),
            original_response="test"
        ))

        summary = collector.get_feedback_summary()
        assert summary["total_feedback"] == 3
        assert summary["positive"] == 2
        assert summary["negative"] == 1
        assert summary["sentiment_ratio"] > 0.5

    return True


def test_should_clarify_on_negative():
    """Test clarification trigger on negative feedback (LS-003)"""
    from learning.pattern_detector import FeedbackCollector, FeedbackEntry

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FeedbackCollector(Path(tmpdir))

        negative = FeedbackEntry(
            feedback_type="negative",
            message_id="1",
            timestamp=time.time(),
            original_response="test"
        )
        assert collector.should_clarify(negative)

        positive = FeedbackEntry(
            feedback_type="positive",
            message_id="2",
            timestamp=time.time(),
            original_response="test"
        )
        assert not collector.should_clarify(positive)

    return True


def test_clarification_prompt():
    """Test clarification prompt generation (LS-003)"""
    from learning.pattern_detector import FeedbackCollector, FeedbackEntry

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FeedbackCollector(Path(tmpdir))

        feedback = FeedbackEntry(
            feedback_type="negative",
            message_id="1",
            timestamp=time.time(),
            original_response="test"
        )

        prompt = collector.get_clarification_prompt(feedback)
        assert "apologize" in prompt.lower() or "sorry" in prompt.lower()
        assert "?" in prompt

    return True


# ========== LS-004: Topic Expertise Modeling Tests ==========

def test_expertise_tracker_creation():
    """Test TopicExpertiseTracker can be created (LS-004)"""
    from learning.pattern_detector import TopicExpertiseTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicExpertiseTracker(Path(tmpdir))

        assert tracker is not None
        assert len(tracker.NOVICE_INDICATORS) > 0
        assert len(tracker.INTERMEDIATE_INDICATORS) > 0
        assert len(tracker.EXPERT_INDICATORS) > 0

    return True


def test_analyze_novice_query():
    """Test novice query analysis (LS-004)"""
    from learning.pattern_detector import TopicExpertiseTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicExpertiseTracker(Path(tmpdir))

        level = tracker.analyze_query_expertise("what is python?", "python")
        assert level == "novice"

        level = tracker.analyze_query_expertise("explain how loops work", "python")
        assert level == "novice"

        level = tracker.analyze_query_expertise("I'm new to programming, how do I start?", "programming")
        assert level == "novice"

    return True


def test_analyze_expert_query():
    """Test expert query analysis (LS-004)"""
    from learning.pattern_detector import TopicExpertiseTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicExpertiseTracker(Path(tmpdir))

        level = tracker.analyze_query_expertise("what's the performance impact of this implementation?", "python")
        assert level == "expert"

        level = tracker.analyze_query_expertise("explain the architecture internals", "system")
        assert level == "expert"

    return True


def test_record_query_updates_expertise():
    """Test query recording updates expertise (LS-004)"""
    from learning.pattern_detector import TopicExpertiseTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicExpertiseTracker(Path(tmpdir))

        # Record novice queries
        tracker.record_query("what is python?", "python")
        tracker.record_query("explain variables to me", "python")
        tracker.record_query("how do I start with python?", "python")

        expertise = tracker.get_all_expertise()
        assert "python" in expertise
        assert expertise["python"]["novice_count"] >= 3
        assert expertise["python"]["total_queries"] >= 3

    return True


def test_expertise_level_calculation():
    """Test expertise level calculation from history (LS-004)"""
    from learning.pattern_detector import TopicExpertiseTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicExpertiseTracker(Path(tmpdir))

        # Record mostly expert queries
        for _ in range(5):
            tracker.record_query("what's the performance architecture?", "systems")

        # Should be expert
        level = tracker.get_expertise_level("systems")
        assert level == "expert"

        # Record mostly novice queries for another topic
        for _ in range(5):
            tracker.record_query("what is machine learning?", "ml")

        level = tracker.get_expertise_level("ml")
        assert level == "novice"

    return True


def test_complexity_adjustment():
    """Test complexity adjustment based on expertise (LS-004)"""
    from learning.pattern_detector import TopicExpertiseTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicExpertiseTracker(Path(tmpdir))

        # Record expert queries
        for _ in range(5):
            tracker.record_query("architecture implementation", "systems")

        adjustment = tracker.get_complexity_adjustment("systems")
        assert adjustment["expertise_level"] == "expert"
        assert adjustment["adjustments"]["use_technical_terms"] == True
        assert adjustment["adjustments"]["include_examples"] == False

    return True


def test_system_prompt_for_topic():
    """Test system prompt generation for topic (LS-004)"""
    from learning.pattern_detector import TopicExpertiseTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicExpertiseTracker(Path(tmpdir))

        # Record novice queries
        for _ in range(5):
            tracker.record_query("what is databases?", "databases")

        prompt = tracker.get_system_prompt_for_topic("databases")
        assert "beginner" in prompt.lower()
        assert "explain" in prompt.lower()

        # Unknown topic should get intermediate prompt
        prompt = tracker.get_system_prompt_for_topic("unknown_topic")
        assert "intermediate" in prompt.lower()

    return True


def test_expertise_persistence():
    """Test expertise data persistence (LS-004)"""
    from learning.pattern_detector import TopicExpertiseTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tracker and record data
        tracker1 = TopicExpertiseTracker(Path(tmpdir))
        tracker1.record_query("what is python?", "python")
        tracker1.record_query("explain basics", "python")

        # Create new tracker and verify data loaded
        tracker2 = TopicExpertiseTracker(Path(tmpdir))
        expertise = tracker2.get_all_expertise()

        assert "python" in expertise
        assert expertise["python"]["total_queries"] >= 2

    return True


if __name__ == "__main__":
    tests = [
        # LS-001
        test_temporal_analyzer_creation,
        test_temporal_empty_patterns,
        test_hour_to_label,
        test_temporal_save_load,
        test_detect_activity_streaks,
        # LS-002
        test_prediction_model_creation,
        test_empty_predictions,
        test_predict_response_length,
        test_predict_formality,
        test_prediction_save_load,
        # LS-003
        test_feedback_collector_creation,
        test_detect_positive_feedback,
        test_detect_negative_feedback,
        test_detect_correction_feedback,
        test_record_and_summarize_feedback,
        test_should_clarify_on_negative,
        test_clarification_prompt,
        # LS-004
        test_expertise_tracker_creation,
        test_analyze_novice_query,
        test_analyze_expert_query,
        test_record_query_updates_expertise,
        test_expertise_level_calculation,
        test_complexity_adjustment,
        test_system_prompt_for_topic,
        test_expertise_persistence,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
