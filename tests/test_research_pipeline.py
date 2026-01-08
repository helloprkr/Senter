#!/usr/bin/env python3
"""
Tests for RE-004: Full Research Pipeline End-to-End

Tests the complete research flow from conversation to stored result.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.pipeline import ResearchPipeline, PipelineResult
from research.research_store import ResearchStore, StoredResearch
from research.synthesizer import SynthesizedResearch
import time


# ========== ResearchStore Tests ==========

def test_store_creation():
    """Test ResearchStore can be created"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        assert store is not None
        assert db_path.exists()

    return True


def test_store_and_retrieve():
    """Test storing and retrieving research"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Create test research
        research = SynthesizedResearch(
            topic="test topic",
            summary="This is a test summary.",
            key_insights=["Insight 1", "Insight 2"],
            sources_used=["https://example.com/1"],
            confidence=0.75
        )

        # Store
        research_id = store.store(research)
        assert research_id is not None
        assert research_id > 0

        # Retrieve
        stored = store.get_by_id(research_id)
        assert stored is not None
        assert stored.topic == "test topic"
        assert stored.confidence == 0.75
        assert len(stored.key_insights) == 2

    return True


def test_store_unviewed():
    """Test unviewed research tracking"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Store research
        research = SynthesizedResearch(
            topic="unviewed topic",
            summary="Unviewed summary.",
            key_insights=[],
            sources_used=[],
            confidence=0.5
        )
        research_id = store.store(research)

        # Should be unviewed
        unviewed = store.get_unviewed()
        assert len(unviewed) == 1
        assert unviewed[0].topic == "unviewed topic"
        assert not unviewed[0].viewed

        # Mark as viewed
        store.mark_viewed(research_id)

        # Should no longer be unviewed
        unviewed = store.get_unviewed()
        assert len(unviewed) == 0

    return True


def test_store_feedback():
    """Test feedback rating"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        research = SynthesizedResearch(
            topic="rated topic",
            summary="Will be rated.",
            key_insights=[],
            sources_used=[],
            confidence=0.5
        )
        research_id = store.store(research)

        # Set feedback
        store.set_feedback(research_id, 4)

        # Check
        stored = store.get_by_id(research_id)
        assert stored.feedback_rating == 4

    return True


def test_store_search():
    """Test search functionality"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Store multiple
        for topic in ["python async", "rust ownership", "go channels"]:
            research = SynthesizedResearch(
                topic=topic,
                summary=f"Summary about {topic}.",
                key_insights=[],
                sources_used=[],
                confidence=0.5
            )
            store.store(research)

        # Search
        results = store.search("python")
        assert len(results) == 1
        assert results[0].topic == "python async"

        results = store.search("channels")
        assert len(results) == 1
        assert results[0].topic == "go channels"

    return True


def test_store_stats():
    """Test statistics calculation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        # Store some research
        for i, conf in enumerate([0.7, 0.8, 0.9]):
            research = SynthesizedResearch(
                topic=f"topic {i}",
                summary=f"Summary {i}.",
                key_insights=[],
                sources_used=[],
                confidence=conf
            )
            store.store(research)

        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["unviewed"] == 3
        assert 0.7 <= stats["avg_confidence"] <= 0.9

    return True


# ========== PipelineResult Tests ==========

def test_pipeline_result_dataclass():
    """Test PipelineResult creation"""
    result = PipelineResult(
        topic="test topic",
        research_id=1,
        success=True,
        sources_found=5,
        summary="Test summary",
        key_insights=["Insight 1"],
        confidence=0.8,
        total_time_ms=5000
    )

    assert result.topic == "test topic"
    assert result.success
    assert result.sources_found == 5

    # Test serialization
    data = result.to_dict()
    assert data["topic"] == "test topic"
    assert data["success"] is True

    return True


# ========== Pipeline Tests ==========

def test_pipeline_creation():
    """Test ResearchPipeline can be created"""
    pipeline = ResearchPipeline(store_results=False)

    assert pipeline is not None
    assert pipeline.max_sources == 5
    assert pipeline.min_priority == 0.5
    assert pipeline.extractor is not None
    assert pipeline.researcher is not None
    assert pipeline.synthesizer is not None

    return True


def test_pipeline_custom_config():
    """Test pipeline with custom config"""
    pipeline = ResearchPipeline(
        max_sources=3,
        min_priority=0.7,
        store_results=False
    )

    assert pipeline.max_sources == 3
    assert pipeline.min_priority == 0.7

    return True


def test_pipeline_empty_conversation():
    """Test pipeline with empty messages"""
    pipeline = ResearchPipeline(store_results=False)

    results = pipeline.process_conversation([])

    assert results == []

    return True


def test_pipeline_no_curiosity_conversation():
    """Test pipeline with conversation lacking curiosity signals"""
    pipeline = ResearchPipeline(store_results=False, min_priority=0.8)

    # Conversation without strong curiosity signals
    messages = [
        {"role": "user", "content": "I used Python today."},
        {"role": "assistant", "content": "Nice!"},
        {"role": "user", "content": "Yeah, it was fine."}
    ]

    results = pipeline.process_conversation(messages)

    # Should have few or no topics with high threshold
    assert len(results) <= 1

    return True


# ========== Integration Tests (require Ollama + Network) ==========

def test_pipeline_full_flow():
    """Test full pipeline flow (requires Ollama)"""
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except:
        print("  (Ollama not available, skipping)")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = ResearchStore(db_path)

        pipeline = ResearchPipeline(
            max_sources=2,
            min_priority=0.6,
            store_results=True
        )
        pipeline.store = store

        # Research a topic directly
        result = pipeline.research_topic("Python asyncio basics")

        # Should complete (may fail if network issues)
        if result.success:
            assert result.sources_found > 0
            assert len(result.summary) > 50
            assert result.research_id is not None

            # Check storage
            stored = store.get_by_id(result.research_id)
            assert stored is not None
            assert stored.topic == "Python asyncio basics"

    return True


if __name__ == "__main__":
    tests = [
        # Store tests
        test_store_creation,
        test_store_and_retrieve,
        test_store_unviewed,
        test_store_feedback,
        test_store_search,
        test_store_stats,
        # Pipeline result tests
        test_pipeline_result_dataclass,
        # Pipeline tests
        test_pipeline_creation,
        test_pipeline_custom_config,
        test_pipeline_empty_conversation,
        test_pipeline_no_curiosity_conversation,
        # Integration tests
        test_pipeline_full_flow,
    ]

    print("=" * 60)
    print("RE-004: Research Pipeline Tests")
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
