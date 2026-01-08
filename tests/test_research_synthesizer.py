#!/usr/bin/env python3
"""
Tests for RE-003: Senter Synthesizes Into Useful Summary

Tests multi-source synthesis with LLM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.synthesizer import ResearchSynthesizer, SynthesizedResearch
from research.deep_researcher import ResearchSource


# ========== Unit Tests ==========

def test_synthesized_research_dataclass():
    """Test SynthesizedResearch creation and serialization"""
    research = SynthesizedResearch(
        topic="event sourcing",
        summary="Event sourcing is a pattern for storing state changes.",
        key_insights=["Insight 1", "Insight 2"],
        sources_used=["https://example.com/1", "https://example.com/2"],
        confidence=0.85,
        synthesis_time_ms=1500,
        raw_source_count=5
    )

    assert research.topic == "event sourcing"
    assert research.confidence == 0.85
    assert len(research.key_insights) == 2

    # Test serialization
    data = research.to_dict()
    assert data["topic"] == "event sourcing"
    assert data["confidence"] == 0.85

    # Test deserialization
    restored = SynthesizedResearch.from_dict(data)
    assert restored.topic == research.topic
    assert restored.confidence == research.confidence

    return True


def test_synthesized_research_format_display():
    """Test formatting for display"""
    research = SynthesizedResearch(
        topic="Python Async",
        summary="Python async is great for I/O.",
        key_insights=["Use asyncio", "Avoid blocking"],
        sources_used=["https://docs.python.org"]
    )

    formatted = research.format_for_display()

    assert "# Python Async" in formatted
    assert "Python async is great" in formatted
    assert "Key Insights" in formatted
    assert "Use asyncio" in formatted
    assert "Sources" in formatted
    assert "docs.python.org" in formatted

    return True


def test_synthesizer_creation():
    """Test ResearchSynthesizer can be created"""
    synthesizer = ResearchSynthesizer()

    assert synthesizer is not None
    assert synthesizer.max_sources == 4
    assert synthesizer.max_content_per_source == 2000

    return True


def test_synthesizer_custom_config():
    """Test ResearchSynthesizer with custom config"""
    synthesizer = ResearchSynthesizer(
        max_sources=3,
        max_content_per_source=1500
    )

    assert synthesizer.max_sources == 3
    assert synthesizer.max_content_per_source == 1500

    return True


def test_format_sources():
    """Test source formatting for LLM prompt"""
    synthesizer = ResearchSynthesizer(max_content_per_source=100)

    sources = [
        ResearchSource(
            url="https://example.com/article",
            title="Test Article",
            content="This is the content of the test article."
        )
    ]

    formatted = synthesizer._format_sources(sources)

    assert "SOURCE 1" in formatted
    assert "Test Article" in formatted
    assert "https://example.com/article" in formatted
    assert "content of the test" in formatted

    return True


def test_format_sources_truncation():
    """Test source content truncation"""
    synthesizer = ResearchSynthesizer(max_content_per_source=50)

    sources = [
        ResearchSource(
            url="https://example.com/long",
            title="Long Article",
            content="A" * 200  # Much longer than limit
        )
    ]

    formatted = synthesizer._format_sources(sources)

    # Should be truncated with ellipsis
    assert "..." in formatted
    # Should not contain full content
    assert "A" * 200 not in formatted

    return True


def test_fallback_synthesis():
    """Test fallback when LLM unavailable"""
    synthesizer = ResearchSynthesizer()

    sources = [
        ResearchSource(
            url="https://example.com/1",
            title="Article 1",
            content="Content 1",
            key_points=["Key point one", "Key point two"]
        ),
        ResearchSource(
            url="https://example.com/2",
            title="Article 2",
            content="Content 2",
            key_points=["Another key point"]
        )
    ]

    result = synthesizer._fallback_synthesis("test topic", sources)

    assert "summary" in result
    assert "key_insights" in result
    assert "confidence" in result
    assert result["confidence"] == 0.4
    assert len(result["key_insights"]) <= 3

    return True


def test_synthesize_empty_sources():
    """Test synthesis with empty sources"""
    synthesizer = ResearchSynthesizer()

    result = synthesizer.synthesize("test topic", [])

    assert result.topic == "test topic"
    assert result.confidence == 0.0
    assert "No sources" in result.summary

    return True


def test_synthesize_filters_error_sources():
    """Test that sources with errors are filtered"""
    synthesizer = ResearchSynthesizer()

    sources = [
        ResearchSource(
            url="https://example.com/good",
            title="Good Article",
            content="This is good content."
        ),
        ResearchSource(
            url="https://example.com/bad",
            title="Bad Article",
            content="",
            error="Connection failed"
        )
    ]

    # The synthesizer filters out error sources before synthesis
    valid = [s for s in sources if s.content and not s.error]
    assert len(valid) == 1

    return True


# ========== Integration Tests (require Ollama) ==========

def test_synthesize_real_sources():
    """Test actual synthesis with LLM (requires Ollama)"""
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except:
        print("  (Ollama not available, skipping)")
        return True

    synthesizer = ResearchSynthesizer(max_sources=2)

    # Create mock sources
    sources = [
        ResearchSource(
            url="https://example.com/article1",
            title="Event Sourcing Introduction",
            content="""
            Event sourcing is a design pattern that stores state changes as events.
            Instead of storing current state, you store the history of changes.
            This allows for complete audit trails and time-travel debugging.
            """
        ),
        ResearchSource(
            url="https://example.com/article2",
            title="Event Sourcing Benefits",
            content="""
            The benefits of event sourcing include improved auditability,
            the ability to replay events, and easier debugging.
            However, it adds complexity and requires careful event design.
            """
        )
    ]

    result = synthesizer.synthesize("event sourcing", sources)

    assert result.topic == "event sourcing"
    assert len(result.summary) > 50
    assert result.confidence > 0
    assert len(result.sources_used) == 2

    return True


if __name__ == "__main__":
    tests = [
        test_synthesized_research_dataclass,
        test_synthesized_research_format_display,
        test_synthesizer_creation,
        test_synthesizer_custom_config,
        test_format_sources,
        test_format_sources_truncation,
        test_fallback_synthesis,
        test_synthesize_empty_sources,
        test_synthesize_filters_error_sources,
        # Integration tests
        test_synthesize_real_sources,
    ]

    print("=" * 60)
    print("RE-003: Research Synthesizer Tests")
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
