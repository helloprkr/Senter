#!/usr/bin/env python3
"""
Tests for RE-001: Senter Notices What You're Curious About

Tests semantic topic extraction with curiosity detection.
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.topic_extractor import TopicExtractor, ExtractedTopic, CURIOSITY_SIGNALS
import re


# ========== Unit Tests ==========

def test_extracted_topic_dataclass():
    """Test ExtractedTopic creation and serialization"""
    topic = ExtractedTopic(
        topic="event sourcing",
        priority=0.8,
        reason="User wondered about it",
        source_messages=["I wonder about event sourcing"]
    )

    assert topic.topic == "event sourcing"
    assert topic.priority == 0.8

    # Test serialization
    data = topic.to_dict()
    assert data["topic"] == "event sourcing"

    # Test deserialization
    restored = ExtractedTopic.from_dict(data)
    assert restored.topic == topic.topic
    assert restored.priority == topic.priority

    return True


def test_extractor_creation():
    """Test TopicExtractor can be created"""
    extractor = TopicExtractor()

    assert extractor is not None
    assert extractor.model == "llama3.2"
    assert extractor.dedup_days == 7

    return True


def test_curiosity_signal_patterns():
    """Test curiosity signal detection patterns"""
    test_cases = [
        ("I wonder if this would work", True),
        ("What is event sourcing?", True),
        ("How does kubernetes networking work?", True),
        ("I'm curious about rust async", True),
        ("I'd like to learn about machine learning", True),
        ("Should I use TypeScript?", True),
        ("Just mentioning python", False),
        ("The weather is nice", False),
    ]

    for text, should_match in test_cases:
        text_lower = text.lower()
        matched = any(re.search(pattern, text_lower) for pattern in CURIOSITY_SIGNALS)
        assert matched == should_match, f"Failed for: {text}"

    return True


def test_format_conversation():
    """Test conversation formatting"""
    extractor = TopicExtractor()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "What is X?"},
    ]

    formatted = extractor._format_conversation(messages)

    assert "USER: Hello" in formatted
    assert "ASSISTANT: Hi there" in formatted
    assert "USER: What is X?" in formatted

    return True


def test_detect_curiosity_signals():
    """Test curiosity signal counting"""
    extractor = TopicExtractor()

    # Multiple signals
    text1 = "I wonder what event sourcing is and how does it work?"
    count1 = extractor._detect_curiosity_signals(text1)
    assert count1 >= 2, f"Expected >= 2 signals, got {count1}"

    # No signals
    text2 = "Just a regular statement about code."
    count2 = extractor._detect_curiosity_signals(text2)
    assert count2 == 0, f"Expected 0 signals, got {count2}"

    return True


def test_word_overlap():
    """Test word overlap calculation"""
    extractor = TopicExtractor()

    # High overlap
    overlap1 = extractor._word_overlap("event sourcing patterns", "event sourcing")
    assert overlap1 >= 0.5

    # No overlap
    overlap2 = extractor._word_overlap("python programming", "rust async")
    assert overlap2 == 0.0

    return True


def test_recently_researched_tracking():
    """Test deduplication of recently researched topics"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "researched.json"
        extractor = TopicExtractor(recently_researched_file=cache_file)

        # Mark as researched
        extractor.mark_as_researched("event sourcing")

        # Should be detected as recently researched
        assert extractor._is_recently_researched("event sourcing")
        assert extractor._is_recently_researched("Event Sourcing")  # Case insensitive
        assert extractor._is_recently_researched("event sourcing patterns")  # Substring

        # Should not match unrelated
        assert not extractor._is_recently_researched("kubernetes networking")

        # Cache file should exist
        assert cache_file.exists()

    return True


def test_find_source_messages():
    """Test finding source messages for a topic"""
    extractor = TopicExtractor()

    messages = [
        {"role": "user", "content": "Working on data pipeline"},
        {"role": "assistant", "content": "Interesting"},
        {"role": "user", "content": "I wonder about event sourcing for this"},
        {"role": "user", "content": "Also curious about CQRS"},
    ]

    sources = extractor._find_source_messages("event sourcing", messages)

    assert len(sources) >= 1
    assert any("event sourcing" in s.lower() for s in sources)

    return True


def test_fallback_extraction():
    """Test pattern-based fallback when LLM unavailable"""
    extractor = TopicExtractor()

    messages = [
        {"role": "user", "content": "I wonder about kubernetes networking"},
        {"role": "user", "content": "What is service mesh?"},
    ]

    topics = extractor._fallback_extraction(messages)

    assert len(topics) >= 1
    # Should extract at least one topic
    topic_texts = [t["topic"] for t in topics]
    assert any("kubernetes" in t or "service mesh" in t for t in topic_texts)

    return True


def test_empty_messages():
    """Test handling of empty messages"""
    extractor = TopicExtractor()

    topics = extractor.extract_topics([])
    assert topics == []

    return True


def test_min_priority_filtering():
    """Test minimum priority threshold"""
    extractor = TopicExtractor()

    # With fallback extraction (no LLM needed)
    messages = [{"role": "user", "content": "just mentioning python briefly"}]

    # High threshold should filter out weak signals
    topics = extractor.extract_topics(messages, min_priority=0.9)
    # Should have few or no topics with such high threshold
    assert all(t.priority >= 0.9 for t in topics)

    return True


# ========== Integration Tests (require Ollama) ==========

def test_llm_extraction_with_clear_curiosity():
    """Test LLM extraction with clear curiosity signals (requires Ollama)"""
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except:
        print("  (Ollama not available, skipping)")
        return True

    extractor = TopicExtractor()

    messages = [
        {"role": "user", "content": "I'm building a web app"},
        {"role": "user", "content": "I wonder if I should use event sourcing for the audit log requirements"},
    ]

    topics = extractor.extract_topics(messages)

    # Should find event sourcing with high priority
    assert len(topics) >= 1
    assert any("event" in t.topic.lower() or "sourcing" in t.topic.lower() for t in topics)
    assert topics[0].priority >= 0.7

    return True


def test_llm_extraction_ignores_casual_mentions():
    """Test that casual mentions don't get high priority (requires Ollama)"""
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except:
        print("  (Ollama not available, skipping)")
        return True

    extractor = TopicExtractor()

    messages = [
        {"role": "user", "content": "I used Python for this project"},
        {"role": "assistant", "content": "Nice, Python is great"},
        {"role": "user", "content": "Yeah, the code is working now"},
    ]

    topics = extractor.extract_topics(messages, min_priority=0.6)

    # Casual mention of Python shouldn't be extracted with high priority
    # (might be empty or have low priority)
    high_priority = [t for t in topics if t.priority >= 0.7]
    assert len(high_priority) == 0 or not any("python" in t.topic.lower() for t in high_priority)

    return True


if __name__ == "__main__":
    tests = [
        test_extracted_topic_dataclass,
        test_extractor_creation,
        test_curiosity_signal_patterns,
        test_format_conversation,
        test_detect_curiosity_signals,
        test_word_overlap,
        test_recently_researched_tracking,
        test_find_source_messages,
        test_fallback_extraction,
        test_empty_messages,
        test_min_priority_filtering,
        # Integration tests
        test_llm_extraction_with_clear_curiosity,
        test_llm_extraction_ignores_casual_mentions,
    ]

    print("=" * 60)
    print("RE-001: Topic Extractor Tests")
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
