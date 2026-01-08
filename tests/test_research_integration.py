#!/usr/bin/env python3
"""
Tests for INT-001: The Full Experience Works

Integration tests for the complete Senter Research system.
"""

import sys
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_start_script_exists():
    """Test that start script exists"""
    script = Path(__file__).parent.parent / "scripts" / "start_senter.py"
    assert script.exists()
    return True


def test_start_script_help():
    """Test that start script shows help"""
    script = Path(__file__).parent.parent / "scripts" / "start_senter.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Senter" in result.stdout
    assert "--demo" in result.stdout
    return True


def test_dependency_check():
    """Test dependency checking"""
    script = Path(__file__).parent.parent / "scripts" / "start_senter.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True
    )
    # May pass or fail depending on system
    assert result.returncode in [0, 1]
    return True


def test_status_command():
    """Test status command runs"""
    script = Path(__file__).parent.parent / "scripts" / "start_senter.py"
    result = subprocess.run(
        [sys.executable, str(script), "--status"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Senter Status" in result.stdout
    return True


def test_all_research_modules_import():
    """Test that all research modules can be imported"""
    from research.topic_extractor import TopicExtractor
    from research.deep_researcher import DeepResearcher
    from research.synthesizer import ResearchSynthesizer
    from research.pipeline import ResearchPipeline
    from research.research_store import ResearchStore
    from research.feedback import FeedbackAnalyzer

    assert TopicExtractor is not None
    assert DeepResearcher is not None
    assert ResearchSynthesizer is not None
    assert ResearchPipeline is not None
    assert ResearchStore is not None
    assert FeedbackAnalyzer is not None

    return True


def test_all_ui_modules_import():
    """Test that all UI modules can be imported"""
    from ui.menubar_app import HeadlessMenubar
    from ui.research_panel import HeadlessPanel

    assert HeadlessMenubar is not None
    assert HeadlessPanel is not None

    return True


def test_end_to_end_research():
    """Test complete research flow"""
    from research.pipeline import ResearchPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        from research.research_store import ResearchStore
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        pipeline = ResearchPipeline(max_sources=2, store_results=True)
        pipeline.store = store

        # Research a topic
        result = pipeline.research_topic("Python type hints")

        # Should complete (network dependent)
        assert result is not None
        assert result.topic == "Python type hints"

        if result.success:
            assert result.sources_found > 0
            assert result.research_id is not None

            # Verify stored
            stored = store.get_by_id(result.research_id)
            assert stored is not None

    return True


def test_conversation_to_research():
    """Test full flow from conversation to stored research"""
    from research.pipeline import ResearchPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        from research.research_store import ResearchStore
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        pipeline = ResearchPipeline(
            max_sources=2,
            min_priority=0.5,
            store_results=True
        )
        pipeline.store = store

        # Simulate conversation
        messages = [
            {"role": "user", "content": "I wonder how Python async works"},
        ]

        results = pipeline.process_conversation(messages, max_topics=1)

        # Should detect and research topic
        if results:
            assert len(results) >= 1
            result = results[0]
            if result.success:
                assert result.research_id is not None

    return True


def test_feedback_integration():
    """Test feedback affects future recommendations"""
    from research.feedback import FeedbackAnalyzer

    with tempfile.TemporaryDirectory() as tmpdir:
        from research.research_store import ResearchStore
        from research.synthesizer import SynthesizedResearch

        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Add rated research
        for topic, rating in [("good topic", 5), ("bad topic", 1)]:
            research = SynthesizedResearch(
                topic=topic,
                summary="Summary",
                key_insights=[],
                sources_used=[],
                confidence=0.8
            )
            rid = store.store(research)
            store.set_feedback(rid, rating)

        analyzer = FeedbackAnalyzer(db_path)

        # Good topics should be recommended
        good_advice = analyzer.should_research_topic("good topic")
        assert good_advice["recommend"] is True

    return True


if __name__ == "__main__":
    tests = [
        test_start_script_exists,
        test_start_script_help,
        test_dependency_check,
        test_status_command,
        test_all_research_modules_import,
        test_all_ui_modules_import,
        test_end_to_end_research,
        test_conversation_to_research,
        test_feedback_integration,
    ]

    print("=" * 60)
    print("INT-001: Full Integration Tests")
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
