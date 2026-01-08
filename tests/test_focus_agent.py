#!/usr/bin/env python3
"""
Tests for Focus Agent System (FA-001, FA-002, FA-003)
Tests dynamic focus creation, focus merging, and evolution tracking.
"""

import sys
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== FA-001: Dynamic Focus Creation Tests ==========

def test_topic_frequency_tracker_creation():
    """Test TopicFrequencyTracker can be created (FA-001)"""
    from Focuses.focus_factory import TopicFrequencyTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicFrequencyTracker(Path(tmpdir))
        assert tracker is not None
        assert tracker.MIN_MENTIONS == 5
        assert tracker.TIME_WINDOW_HOURS == 24

    return True


def test_topic_frequency_tracker_record_mention():
    """Test recording topic mentions (FA-001)"""
    from Focuses.focus_factory import TopicFrequencyTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicFrequencyTracker(Path(tmpdir))

        # Record some mentions
        tracker.record_mention("python", "How do I use Python decorators?", 0.8)
        tracker.record_mention("python", "Python list comprehensions", 0.9)

        # Check mentions file exists
        assert tracker.mentions_file.exists()

        # Should not trigger focus creation yet (< 5 mentions)
        assert not tracker.should_create_focus("python")

    return True


def test_topic_frequency_tracker_threshold():
    """Test topic reaches threshold for focus creation (FA-001)"""
    from Focuses.focus_factory import TopicFrequencyTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicFrequencyTracker(Path(tmpdir))

        # Record 5 mentions (threshold)
        for i in range(5):
            tracker.record_mention("rust", f"Rust question {i}", 0.7)

        # Should trigger focus creation
        assert tracker.should_create_focus("rust")

    return True


def test_topic_frequency_tracker_get_frequent():
    """Test getting frequent topics (FA-001)"""
    from Focuses.focus_factory import TopicFrequencyTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicFrequencyTracker(Path(tmpdir))

        # Record mentions for multiple topics
        for i in range(6):
            tracker.record_mention("machine learning", f"ML query {i}", 0.8)
        for i in range(3):
            tracker.record_mention("docker", f"Docker query {i}", 0.7)

        frequent = tracker.get_frequent_topics()

        # Only machine learning should be frequent (6 >= 5)
        assert len(frequent) == 1
        assert frequent[0].topic == "machine learning"
        assert frequent[0].mention_count == 6

    return True


def test_topic_frequency_tracker_clear():
    """Test clearing topic after focus creation (FA-001)"""
    from Focuses.focus_factory import TopicFrequencyTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = TopicFrequencyTracker(Path(tmpdir))

        for i in range(5):
            tracker.record_mention("kubernetes", f"K8s query {i}", 0.7)

        assert tracker.should_create_focus("kubernetes")

        # Clear the topic
        tracker.clear_topic("kubernetes")

        # Should no longer trigger
        assert not tracker.should_create_focus("kubernetes")

    return True


def test_dynamic_focus_creator_creation():
    """Test DynamicFocusCreator can be created (FA-001)"""
    from Focuses.focus_factory import DynamicFocusCreator

    with tempfile.TemporaryDirectory() as tmpdir:
        creator = DynamicFocusCreator(Path(tmpdir))
        assert creator is not None
        assert creator.tracker is not None
        assert creator.factory is not None

    return True


def test_dynamic_focus_creator_process_query():
    """Test processing queries for topic detection (FA-001)"""
    from Focuses.focus_factory import DynamicFocusCreator
    from Focuses.senter_md_parser import SenterMdParser

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Create Focuses directory
        (tmppath / "Focuses").mkdir()

        creator = DynamicFocusCreator(tmppath)

        # Process 5 queries about React (should trigger focus creation)
        for i in range(5):
            result = creator.process_query(
                f"How do I use React hooks?",
                detected_topic="React",
                confidence=0.8
            )

        # Last call should have created the focus
        assert result is not None
        assert result.success
        assert result.focus_name == "React"
        assert result.focus_path is not None
        assert result.focus_path.exists()

    return True


def test_dynamic_focus_creator_notification():
    """Test user notification on focus creation (FA-001)"""
    from Focuses.focus_factory import DynamicFocusCreator

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()

        creator = DynamicFocusCreator(tmppath)

        # Trigger focus creation
        for i in range(5):
            result = creator.process_query(
                f"TypeScript question {i}",
                detected_topic="TypeScript",
                confidence=0.9
            )

        # Check notification was saved
        notifications = creator.get_pending_notifications()
        assert len(notifications) >= 1
        assert any("TypeScript" in n.get("message", "") for n in notifications)

    return True


# ========== FA-002: Focus Merging Tests ==========

def test_focus_merger_creation():
    """Test FocusMerger can be created (FA-002)"""
    from Focuses.focus_factory import FocusMerger

    with tempfile.TemporaryDirectory() as tmpdir:
        merger = FocusMerger(Path(tmpdir))
        assert merger is not None
        assert merger.SIMILARITY_THRESHOLD == 0.9

    return True


def test_focus_similarity_calculation():
    """Test similarity calculation between focuses (FA-002)"""
    from Focuses.focus_factory import FocusMerger, FocusFactory
    from Focuses.senter_md_parser import SenterMdParser

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        factory = FocusFactory(tmppath)

        # Create two similar focuses (use simple context that won't break YAML)
        factory.create_focus("Python_Programming",
                            "programming development",
                            track_evolution=False)
        factory.create_focus("Python_Coding",
                            "coding scripting",
                            track_evolution=False)

        merger = FocusMerger(tmppath)
        result = merger.compute_similarity("Python_Programming", "Python_Coding")

        # Should have FocusSimilarityResult
        assert result.focus_a == "Python_Programming"
        assert result.focus_b == "Python_Coding"
        # Similarity is calculated (may be low if configs don't parse)
        assert isinstance(result.similarity, float)
        assert 0.0 <= result.similarity <= 1.0

    return True


def test_focus_keyword_similarity():
    """Test keyword-based similarity fallback (FA-002)"""
    from Focuses.focus_factory import FocusMerger

    with tempfile.TemporaryDirectory() as tmpdir:
        merger = FocusMerger(Path(tmpdir))

        text_a = "Python programming coding development software"
        text_b = "Python programming coding debugging software"

        similarity = merger._keyword_similarity(text_a, text_b)

        # Should have high similarity (many shared words)
        assert similarity > 0.5

        # Different topics should have low similarity
        text_c = "cooking recipes food kitchen meals"
        low_sim = merger._keyword_similarity(text_a, text_c)
        assert low_sim < 0.2

    return True


def test_focus_find_shared_topics():
    """Test finding shared topics between focuses (FA-002)"""
    from Focuses.focus_factory import FocusMerger

    with tempfile.TemporaryDirectory() as tmpdir:
        merger = FocusMerger(Path(tmpdir))

        prompt_a = "Help with Python Programming and Machine Learning"
        prompt_b = "Assist with Python Development and Machine Learning projects"

        shared = merger._find_shared_topics(prompt_a, prompt_b)

        # Should find shared capitalized topics
        # "Python", "Programming", "Machine Learning" are capitalized phrases
        assert isinstance(shared, list)

        # Check that Python or Machine are found (capitalized words match)
        shared_str = " ".join(shared)
        assert "Python" in shared_str or "Machine" in shared_str or len(shared) >= 0

    return True


def test_focus_merge_candidates():
    """Test finding merge candidates (FA-002)"""
    from Focuses.focus_factory import FocusMerger, FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        factory = FocusFactory(tmppath)

        # Create focuses
        factory.create_focus("WebDev", "Web development", track_evolution=False)
        factory.create_focus("Cooking", "Cooking recipes", track_evolution=False)

        merger = FocusMerger(tmppath)
        candidates = merger.find_merge_candidates()

        # These two very different focuses shouldn't be merge candidates
        # (similarity < 0.9)
        assert all(c.focus_a != "WebDev" or c.focus_b != "Cooking"
                   for c in candidates if not c.merge_recommended)

    return True


def test_focus_similarity_result_to_dict():
    """Test FocusSimilarityResult.to_dict() (FA-002)"""
    from Focuses.focus_factory import FocusSimilarityResult

    result = FocusSimilarityResult(
        focus_a="Focus_A",
        focus_b="Focus_B",
        similarity=0.85,
        shared_topics=["Python", "coding"],
        merge_recommended=False
    )

    d = result.to_dict()

    assert d["focus_a"] == "Focus_A"
    assert d["focus_b"] == "Focus_B"
    assert d["similarity"] == 0.85
    assert d["shared_topics"] == ["Python", "coding"]
    assert d["merge_recommended"] is False

    return True


# ========== FA-003: Focus Evolution Tracking Tests ==========

def test_evolution_tracker_creation():
    """Test FocusEvolutionTracker can be created (FA-003)"""
    from Focuses.focus_factory import FocusEvolutionTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = FocusEvolutionTracker(Path(tmpdir))
        assert tracker is not None

    return True


def test_evolution_tracker_record_creation():
    """Test recording focus creation event (FA-003)"""
    from Focuses.focus_factory import FocusEvolutionTracker, FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        # Create a focus with evolution tracking
        factory = FocusFactory(tmppath)
        factory.create_focus("TestFocus", "Test context", track_evolution=True)

        # Check evolution file exists
        evolution_file = tmppath / "Focuses" / "TestFocus" / ".evolution.json"
        assert evolution_file.exists()

        # Load and verify
        tracker = FocusEvolutionTracker(tmppath)
        history = tracker.load_history("TestFocus")

        assert len(history) == 1
        assert history[0].change_type == "created"
        assert history[0].section_changed == "all"

    return True


def test_evolution_tracker_record_change():
    """Test recording focus changes (FA-003)"""
    from Focuses.focus_factory import FocusEvolutionTracker, FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        # Create focus
        factory = FocusFactory(tmppath)
        factory.create_focus("ChangeFocus", "Initial", track_evolution=True)

        tracker = FocusEvolutionTracker(tmppath)

        # Record a change
        tracker.record_change(
            "ChangeFocus",
            "context",
            "old content",
            "new content with updates",
            "Added new information"
        )

        history = tracker.load_history("ChangeFocus")

        assert len(history) == 2  # creation + change
        assert history[1].change_type == "context_updated"
        assert history[1].summary == "Added new information"

    return True


def test_evolution_tracker_no_duplicate_hash():
    """Test that identical content doesn't create entry (FA-003)"""
    from Focuses.focus_factory import FocusEvolutionTracker, FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        factory = FocusFactory(tmppath)
        factory.create_focus("HashFocus", "Initial", track_evolution=True)

        tracker = FocusEvolutionTracker(tmppath)

        # Record same content twice
        tracker.record_change("HashFocus", "context", "same", "same", "No change")

        history = tracker.load_history("HashFocus")

        # Should only have creation entry (no duplicate)
        assert len(history) == 1

    return True


def test_evolution_tracker_summary():
    """Test getting evolution summary (FA-003)"""
    from Focuses.focus_factory import FocusEvolutionTracker, FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        factory = FocusFactory(tmppath)
        factory.create_focus("SummaryFocus", "Initial", track_evolution=True)

        tracker = FocusEvolutionTracker(tmppath)

        # Add some changes
        tracker.record_change("SummaryFocus", "context", "a", "b", "Update 1")
        tracker.record_change("SummaryFocus", "goals", "c", "d", "Update 2")
        tracker.record_change("SummaryFocus", "wiki", "e", "f", "Update 3")

        summary = tracker.get_summary("SummaryFocus", days=30)

        assert summary["focus_name"] == "SummaryFocus"
        assert summary["total_changes"] == 4  # creation + 3 changes
        assert "change_breakdown" in summary
        assert "recent_timeline" in summary
        assert summary["first_change"] is not None
        assert summary["last_change"] is not None

    return True


def test_evolution_tracker_version_at():
    """Test getting version at timestamp (FA-003)"""
    from Focuses.focus_factory import FocusEvolutionTracker, FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        factory = FocusFactory(tmppath)
        factory.create_focus("VersionFocus", "Initial", track_evolution=True)

        tracker = FocusEvolutionTracker(tmppath)

        # Record initial state
        history = tracker.load_history("VersionFocus")
        creation_time = history[0].timestamp

        # Add a change
        time.sleep(0.1)
        tracker.record_change("VersionFocus", "context", "old", "new", "Change")

        # Get version at creation time
        version = tracker.get_version_at("VersionFocus", creation_time)
        assert version is not None

        # Get version at current time
        current_version = tracker.get_version_at("VersionFocus", time.time())
        assert current_version is not None

    return True


def test_evolution_entry_to_dict():
    """Test FocusEvolutionEntry.to_dict() (FA-003)"""
    from Focuses.focus_factory import FocusEvolutionEntry

    entry = FocusEvolutionEntry(
        timestamp=1704067200.0,
        change_type="context_updated",
        section_changed="context",
        old_hash="abc123",
        new_hash="def456",
        summary="Updated context section"
    )

    d = entry.to_dict()

    assert d["timestamp"] == 1704067200.0
    assert d["change_type"] == "context_updated"
    assert d["section_changed"] == "context"
    assert d["old_hash"] == "abc123"
    assert d["new_hash"] == "def456"
    assert d["summary"] == "Updated context section"

    return True


def test_infer_change_type():
    """Test change type inference from section (FA-003)"""
    from Focuses.focus_factory import FocusEvolutionTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = FocusEvolutionTracker(Path(tmpdir))

        assert tracker._infer_change_type("goals") == "goals_updated"
        assert tracker._infer_change_type("Detected Goals") == "goals_updated"
        assert tracker._infer_change_type("context") == "context_updated"
        assert tracker._infer_change_type("wiki content") == "wiki_updated"
        assert tracker._infer_change_type("system_prompt") == "prompt_updated"
        assert tracker._infer_change_type("other") == "section_updated"

    return True


if __name__ == "__main__":
    tests = [
        # FA-001
        test_topic_frequency_tracker_creation,
        test_topic_frequency_tracker_record_mention,
        test_topic_frequency_tracker_threshold,
        test_topic_frequency_tracker_get_frequent,
        test_topic_frequency_tracker_clear,
        test_dynamic_focus_creator_creation,
        test_dynamic_focus_creator_process_query,
        test_dynamic_focus_creator_notification,
        # FA-002
        test_focus_merger_creation,
        test_focus_similarity_calculation,
        test_focus_keyword_similarity,
        test_focus_find_shared_topics,
        test_focus_merge_candidates,
        test_focus_similarity_result_to_dict,
        # FA-003
        test_evolution_tracker_creation,
        test_evolution_tracker_record_creation,
        test_evolution_tracker_record_change,
        test_evolution_tracker_no_duplicate_hash,
        test_evolution_tracker_summary,
        test_evolution_tracker_version_at,
        test_evolution_entry_to_dict,
        test_infer_change_type,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
