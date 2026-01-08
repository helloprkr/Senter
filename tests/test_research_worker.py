#!/usr/bin/env python3
"""
Tests for Research Worker (RW-001, RW-002, RW-003)
Tests result summarization, topic selection, and result presentation.
"""

import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== RW-001: Research Result Summarization Tests ==========

def test_summarizer_creation():
    """Test ResearchResultSummarizer can be created (RW-001)"""
    from scheduler.research_trigger import ResearchResultSummarizer

    with tempfile.TemporaryDirectory() as tmpdir:
        summarizer = ResearchResultSummarizer(Path(tmpdir))

        assert summarizer is not None
        assert summarizer.DEFAULT_SUMMARY_SENTENCES == 3
        assert summarizer.results_dir.exists()

    return True


def test_summarize_short_content():
    """Test summarization of short content (RW-001)"""
    from scheduler.research_trigger import ResearchResultSummarizer

    with tempfile.TemporaryDirectory() as tmpdir:
        summarizer = ResearchResultSummarizer(Path(tmpdir))

        content = "This is a short piece of content."
        summary = summarizer.summarize(content)

        # Short content should return as-is
        assert summary == content

    return True


def test_summarize_long_content():
    """Test summarization of long content (RW-001)"""
    from scheduler.research_trigger import ResearchResultSummarizer

    with tempfile.TemporaryDirectory() as tmpdir:
        summarizer = ResearchResultSummarizer(Path(tmpdir))

        content = """
        Machine learning is transforming industries. This research found that deep learning
        models have significantly improved accuracy. The results indicate a 40% improvement.
        Natural language processing has advanced rapidly. Computer vision is another area
        of significant progress. The technology shows that applications are becoming more
        practical. Healthcare has seen major benefits from these developments. Finance is
        also adopting machine learning widely. The future looks promising for AI research.
        """

        summary = summarizer.summarize(content.strip(), max_sentences=3)

        # Summary should be shorter than original
        assert len(summary) < len(content)
        # Should contain some sentences
        assert "." in summary

    return True


def test_extract_key_findings():
    """Test key findings extraction (RW-001)"""
    from scheduler.research_trigger import ResearchResultSummarizer

    with tempfile.TemporaryDirectory() as tmpdir:
        summarizer = ResearchResultSummarizer(Path(tmpdir))

        content = """
        The study found that regular exercise improves cognitive function.
        Results indicate a 25% improvement in memory.
        This shows that physical activity has significant benefits.
        The conclusion is clear: exercise is important.
        """

        findings = summarizer.extract_key_findings(content)

        assert len(findings) > 0
        # Should contain finding indicators
        assert any("found" in f.lower() or "indicate" in f.lower() or "shows" in f.lower()
                   for f in findings)

    return True


def test_process_result():
    """Test processing a research result (RW-001)"""
    from scheduler.research_trigger import ResearchResultSummarizer

    with tempfile.TemporaryDirectory() as tmpdir:
        summarizer = ResearchResultSummarizer(Path(tmpdir))

        result = summarizer.process_result(
            task_id="test_001",
            topic="Python programming",
            content="Python is a versatile language. The research found that it is widely used in data science. Results indicate strong community support.",
            sources=["python.org"],
            confidence=0.8
        )

        assert result.task_id == "test_001"
        assert result.topic == "Python programming"
        assert result.summary != ""
        assert len(result.sources) == 1
        assert result.confidence == 0.8

    return True


def test_save_load_result():
    """Test saving and loading result (RW-001)"""
    from scheduler.research_trigger import ResearchResultSummarizer, ResearchResult

    with tempfile.TemporaryDirectory() as tmpdir:
        summarizer = ResearchResultSummarizer(Path(tmpdir))

        result = ResearchResult(
            task_id="save_test",
            topic="Testing",
            content="Test content",
            summary="Test summary",
            key_findings=["Finding 1"]
        )

        summarizer.save_result(result)

        loaded = summarizer.load_result("save_test")
        assert loaded is not None
        assert loaded.task_id == "save_test"
        assert loaded.topic == "Testing"

    return True


# ========== RW-002: Autonomous Research Topic Selection Tests ==========

def test_selector_creation():
    """Test AutonomousResearchSelector can be created (RW-002)"""
    from scheduler.research_trigger import AutonomousResearchSelector

    with tempfile.TemporaryDirectory() as tmpdir:
        selector = AutonomousResearchSelector(Path(tmpdir))

        assert selector is not None
        assert selector.is_autonomous_enabled()

    return True


def test_disable_autonomous():
    """Test disabling autonomous research (RW-002)"""
    from scheduler.research_trigger import AutonomousResearchSelector

    with tempfile.TemporaryDirectory() as tmpdir:
        selector = AutonomousResearchSelector(Path(tmpdir))

        selector.set_autonomous_enabled(False)
        assert not selector.is_autonomous_enabled()

        selector.set_autonomous_enabled(True)
        assert selector.is_autonomous_enabled()

    return True


def test_select_topics_with_min_mentions():
    """Test topic selection respects min mentions (RW-002)"""
    from scheduler.research_trigger import AutonomousResearchSelector

    with tempfile.TemporaryDirectory() as tmpdir:
        selector = AutonomousResearchSelector(Path(tmpdir))

        topics = [
            {"topic": "python", "query_count": 10},
            {"topic": "rust", "query_count": 2},  # Below threshold
            {"topic": "javascript", "query_count": 5}
        ]

        selected = selector.select_topics(topics)

        # Should not include rust (below min mentions)
        topic_names = [t["topic"] for t in selected]
        assert "python" in topic_names
        assert "rust" not in topic_names

    return True


def test_select_topics_respects_exclusions():
    """Test topic selection respects exclusions (RW-002)"""
    from scheduler.research_trigger import AutonomousResearchSelector

    with tempfile.TemporaryDirectory() as tmpdir:
        selector = AutonomousResearchSelector(Path(tmpdir))
        selector.add_excluded_topic("python")

        topics = [
            {"topic": "python", "query_count": 10},
            {"topic": "javascript", "query_count": 5}
        ]

        selected = selector.select_topics(topics)

        topic_names = [t["topic"] for t in selected]
        assert "python" not in topic_names
        assert "javascript" in topic_names

    return True


def test_priority_topics():
    """Test priority topics are selected first (RW-002)"""
    from scheduler.research_trigger import AutonomousResearchSelector

    with tempfile.TemporaryDirectory() as tmpdir:
        selector = AutonomousResearchSelector(Path(tmpdir))
        selector.add_priority_topic("ai")

        topics = [
            {"topic": "python", "query_count": 10},
            {"topic": "ai", "query_count": 3},  # Low count but priority
            {"topic": "javascript", "query_count": 8}
        ]

        selected = selector.select_topics(topics)

        # AI should be first despite lower count
        assert selected[0]["topic"] == "ai"

    return True


def test_select_topics_disabled():
    """Test topic selection returns empty when disabled (RW-002)"""
    from scheduler.research_trigger import AutonomousResearchSelector

    with tempfile.TemporaryDirectory() as tmpdir:
        selector = AutonomousResearchSelector(Path(tmpdir))
        selector.set_autonomous_enabled(False)

        topics = [
            {"topic": "python", "query_count": 10}
        ]

        selected = selector.select_topics(topics)
        assert selected == []

    return True


# ========== RW-003: Research Result Presentation Tests ==========

def test_presenter_creation():
    """Test ResearchResultPresenter can be created (RW-003)"""
    from scheduler.research_trigger import ResearchResultPresenter

    with tempfile.TemporaryDirectory() as tmpdir:
        presenter = ResearchResultPresenter(Path(tmpdir))

        assert presenter is not None
        assert "researched" in presenter.OFFER_TEMPLATE.lower()

    return True


def test_queue_result():
    """Test queuing a research result (RW-003)"""
    from scheduler.research_trigger import ResearchResultPresenter, ResearchResult

    with tempfile.TemporaryDirectory() as tmpdir:
        presenter = ResearchResultPresenter(Path(tmpdir))

        result = ResearchResult(
            task_id="queue_test",
            topic="Python",
            content="Test content",
            summary="Test summary"
        )

        offer = presenter.queue_result(result)

        assert offer.result_id == "queue_test"
        assert offer.topic == "Python"
        assert offer.status == "pending"

    return True


def test_get_pending_offers():
    """Test getting pending offers (RW-003)"""
    from scheduler.research_trigger import ResearchResultPresenter, ResearchResult

    with tempfile.TemporaryDirectory() as tmpdir:
        presenter = ResearchResultPresenter(Path(tmpdir))

        # Queue two results
        for topic in ["Python", "JavaScript"]:
            result = ResearchResult(
                task_id=f"test_{topic}",
                topic=topic,
                content="Content",
                summary="Summary"
            )
            presenter.queue_result(result)

        pending = presenter.get_pending_offers()
        assert len(pending) == 2

    return True


def test_generate_offer_message():
    """Test offer message generation (RW-003)"""
    from scheduler.research_trigger import ResearchResultPresenter, ResearchOffer

    with tempfile.TemporaryDirectory() as tmpdir:
        presenter = ResearchResultPresenter(Path(tmpdir))

        offer = ResearchOffer(
            result_id="test",
            topic="Machine Learning",
            summary="Summary"
        )

        message = presenter.generate_offer_message(offer)

        assert "Machine Learning" in message
        assert "?" in message  # Should be a question

    return True


def test_on_attention_gained():
    """Test attention gained trigger (RW-003)"""
    from scheduler.research_trigger import ResearchResultPresenter, ResearchResult

    with tempfile.TemporaryDirectory() as tmpdir:
        presenter = ResearchResultPresenter(Path(tmpdir))

        # Should return None with no pending offers
        message = presenter.on_attention_gained()
        assert message is None

        # Add a result
        result = ResearchResult(
            task_id="attention_test",
            topic="AI Research",
            content="Content",
            summary="Summary"
        )
        presenter.queue_result(result)

        # Should now return an offer
        message = presenter.on_attention_gained()
        assert message is not None
        assert "AI Research" in message

    return True


def test_accept_offer():
    """Test accepting an offer (RW-003)"""
    from scheduler.research_trigger import (
        ResearchResultPresenter, ResearchResultSummarizer, ResearchResult
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # First save a result
        summarizer = ResearchResultSummarizer(Path(tmpdir))
        result = ResearchResult(
            task_id="accept_test",
            topic="Testing",
            content="Test content",
            summary="Test summary",
            key_findings=["Finding 1"]
        )
        summarizer.save_result(result)

        # Then queue and accept
        presenter = ResearchResultPresenter(Path(tmpdir))
        presenter.queue_result(result)

        accepted = presenter.accept_offer("accept_test")
        assert accepted is not None
        assert accepted.task_id == "accept_test"

        # Check status changed
        pending = presenter.get_pending_offers()
        assert len(pending) == 0

    return True


def test_defer_offer():
    """Test deferring an offer (RW-003)"""
    from scheduler.research_trigger import ResearchResultPresenter, ResearchResult

    with tempfile.TemporaryDirectory() as tmpdir:
        presenter = ResearchResultPresenter(Path(tmpdir))

        result = ResearchResult(
            task_id="defer_test",
            topic="Deferred Topic",
            content="Content",
            summary="Summary"
        )
        presenter.queue_result(result)

        presenter.defer_offer("defer_test")

        # Should still be pending but with later timestamp
        pending = presenter.get_pending_offers()
        assert len(pending) == 1
        assert pending[0].result_id == "defer_test"

    return True


def test_dismiss_offer():
    """Test dismissing an offer (RW-003)"""
    from scheduler.research_trigger import ResearchResultPresenter, ResearchResult

    with tempfile.TemporaryDirectory() as tmpdir:
        presenter = ResearchResultPresenter(Path(tmpdir))

        result = ResearchResult(
            task_id="dismiss_test",
            topic="Dismissed Topic",
            content="Content",
            summary="Summary"
        )
        presenter.queue_result(result)

        presenter.dismiss_offer("dismiss_test")

        # Should no longer be pending
        pending = presenter.get_pending_offers()
        assert len(pending) == 0

    return True


def test_handle_user_response_accept():
    """Test handling user accept response (RW-003)"""
    from scheduler.research_trigger import (
        ResearchResultPresenter, ResearchResultSummarizer, ResearchResult
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save result first
        summarizer = ResearchResultSummarizer(Path(tmpdir))
        result = ResearchResult(
            task_id="response_test",
            topic="Response Topic",
            content="Content",
            summary="Test summary",
            key_findings=["Finding 1", "Finding 2"]
        )
        summarizer.save_result(result)

        presenter = ResearchResultPresenter(Path(tmpdir))
        presenter.queue_result(result)

        response = presenter.handle_user_response("response_test", "yes")

        assert response is not None
        assert "Response Topic" in response
        assert "Finding" in response

    return True


def test_handle_user_response_defer():
    """Test handling user defer response (RW-003)"""
    from scheduler.research_trigger import ResearchResultPresenter, ResearchResult

    with tempfile.TemporaryDirectory() as tmpdir:
        presenter = ResearchResultPresenter(Path(tmpdir))

        result = ResearchResult(
            task_id="defer_response",
            topic="Topic",
            content="Content",
            summary="Summary"
        )
        presenter.queue_result(result)

        response = presenter.handle_user_response("defer_response", "later")

        assert response is not None
        assert "later" in response.lower()

    return True


def test_handle_user_response_dismiss():
    """Test handling user dismiss response (RW-003)"""
    from scheduler.research_trigger import ResearchResultPresenter, ResearchResult

    with tempfile.TemporaryDirectory() as tmpdir:
        presenter = ResearchResultPresenter(Path(tmpdir))

        result = ResearchResult(
            task_id="dismiss_response",
            topic="Topic",
            content="Content",
            summary="Summary"
        )
        presenter.queue_result(result)

        response = presenter.handle_user_response("dismiss_response", "no thanks")

        assert response is not None
        assert "dismissed" in response.lower()

    return True


if __name__ == "__main__":
    tests = [
        # RW-001
        test_summarizer_creation,
        test_summarize_short_content,
        test_summarize_long_content,
        test_extract_key_findings,
        test_process_result,
        test_save_load_result,
        # RW-002
        test_selector_creation,
        test_disable_autonomous,
        test_select_topics_with_min_mentions,
        test_select_topics_respects_exclusions,
        test_priority_topics,
        test_select_topics_disabled,
        # RW-003
        test_presenter_creation,
        test_queue_result,
        test_get_pending_offers,
        test_generate_offer_message,
        test_on_attention_gained,
        test_accept_offer,
        test_defer_offer,
        test_dismiss_offer,
        test_handle_user_response_accept,
        test_handle_user_response_defer,
        test_handle_user_response_dismiss,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
