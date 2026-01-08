#!/usr/bin/env python3
"""
RE-004: Full Research Pipeline End-to-End

Orchestrates the complete research flow:
1. Topic Extraction - Detect what user is curious about
2. Deep Research - Search and fetch web content
3. Synthesis - Create actionable summary
4. Storage - Persist for later access

VALUE: User mentions something interesting → Senter automatically
researches it and has a summary ready when they next check in.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger("senter.research.pipeline")

# Import all components
try:
    from .topic_extractor import TopicExtractor, ExtractedTopic
    from .deep_researcher import DeepResearcher, ResearchSource
    from .synthesizer import ResearchSynthesizer, SynthesizedResearch
    from .research_store import ResearchStore
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from topic_extractor import TopicExtractor, ExtractedTopic
    from deep_researcher import DeepResearcher, ResearchSource
    from synthesizer import ResearchSynthesizer, SynthesizedResearch
    from research_store import ResearchStore


@dataclass
class PipelineResult:
    """Result from a complete pipeline run."""
    topic: str
    research_id: Optional[int] = None  # ID in storage
    success: bool = False
    sources_found: int = 0
    summary: str = ""
    key_insights: List[str] = field(default_factory=list)
    confidence: float = 0.0
    total_time_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "research_id": self.research_id,
            "success": self.success,
            "sources_found": self.sources_found,
            "summary": self.summary,
            "key_insights": self.key_insights,
            "confidence": self.confidence,
            "total_time_ms": self.total_time_ms,
            "error": self.error
        }


class ResearchPipeline:
    """
    End-to-end research pipeline.

    Usage:
        pipeline = ResearchPipeline()

        # From conversation messages
        results = pipeline.process_conversation(messages)

        # For a specific topic
        result = pipeline.research_topic("event sourcing")
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        max_sources: int = 5,
        min_priority: float = 0.5,
        store_results: bool = True
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.max_sources = max_sources
        self.min_priority = min_priority
        self.store_results = store_results

        # Initialize components
        self.extractor = TopicExtractor(
            ollama_url=ollama_url,
            model=model
        )
        self.researcher = DeepResearcher(
            ollama_url=ollama_url,
            model=model,
            max_sources=max_sources
        )
        self.synthesizer = ResearchSynthesizer(
            ollama_url=ollama_url,
            model=model,
            max_sources=min(4, max_sources)  # Synthesis uses fewer sources
        )
        self.store = ResearchStore() if store_results else None

    def process_conversation(
        self,
        messages: List[Dict[str, str]],
        max_topics: int = 2
    ) -> List[PipelineResult]:
        """
        Process a conversation and research detected topics.

        Args:
            messages: Conversation messages [{role, content}, ...]
            max_topics: Maximum topics to research (default 2)

        Returns:
            List of PipelineResult for each researched topic
        """
        logger.info(f"Processing conversation with {len(messages)} messages")

        # Step 1: Extract topics
        topics = self.extractor.extract_topics(
            messages,
            min_priority=self.min_priority
        )

        if not topics:
            logger.info("No research-worthy topics detected")
            return []

        logger.info(f"Found {len(topics)} topics above priority {self.min_priority}")

        # Research top N topics
        results = []
        for topic in topics[:max_topics]:
            logger.info(f"Researching: {topic.topic} (priority={topic.priority})")
            result = self.research_topic(topic.topic)
            results.append(result)

            # Mark as researched to prevent duplicates
            self.extractor.mark_as_researched(topic.topic)

        return results

    def research_topic(self, topic: str) -> PipelineResult:
        """
        Research a single topic through the full pipeline.

        Args:
            topic: Topic string to research

        Returns:
            PipelineResult with summary and metadata
        """
        start_time = time.time()
        result = PipelineResult(topic=topic)

        try:
            # Step 1: Deep research
            logger.info(f"[1/3] Researching: {topic}")
            sources = self.researcher.research(topic)
            result.sources_found = len(sources)

            if not sources:
                result.error = "No sources found"
                result.total_time_ms = int((time.time() - start_time) * 1000)
                return result

            logger.info(f"[1/3] Found {len(sources)} sources")

            # Step 2: Synthesize
            logger.info(f"[2/3] Synthesizing...")
            synthesis = self.synthesizer.synthesize(topic, sources)
            result.summary = synthesis.summary
            result.key_insights = synthesis.key_insights
            result.confidence = synthesis.confidence

            logger.info(f"[2/3] Synthesis complete (confidence={synthesis.confidence})")

            # Step 3: Store
            if self.store and synthesis.summary:
                logger.info(f"[3/3] Storing result...")
                result.research_id = self.store.store(synthesis)
                logger.info(f"[3/3] Stored as ID {result.research_id}")

            result.success = True

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.error = str(e)

        result.total_time_ms = int((time.time() - start_time) * 1000)
        return result

    def get_pending_research(self, limit: int = 5):
        """Get unviewed research from storage."""
        if not self.store:
            return []
        return self.store.get_unviewed(limit)

    def get_research_by_id(self, research_id: int):
        """Get specific research by ID."""
        if not self.store:
            return None
        return self.store.get_by_id(research_id)

    def mark_viewed(self, research_id: int):
        """Mark research as viewed."""
        if self.store:
            self.store.mark_viewed(research_id)

    def give_feedback(self, research_id: int, rating: int):
        """Give feedback on research quality (1-5 stars)."""
        if self.store:
            self.store.set_feedback(research_id, rating)


# CLI for testing
if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Test Research Pipeline")
    parser.add_argument("topic", nargs="?", help="Topic to research")
    parser.add_argument("--conversation", "-c", action="store_true",
                        help="Interactive conversation mode")
    parser.add_argument("--demo", "-d", action="store_true",
                        help="Run demo conversation")
    parser.add_argument("--pending", "-p", action="store_true",
                        help="Show pending research")

    args = parser.parse_args()

    pipeline = ResearchPipeline(max_sources=3)

    if args.pending:
        print("\n=== Pending Research ===")
        pending = pipeline.get_pending_research()
        if pending:
            for r in pending:
                print(f"\n{r.id}. {r.topic}")
                print(f"   {r.summary[:150]}...")
                print(f"   Confidence: {r.confidence}")
        else:
            print("No pending research")

    elif args.demo:
        print("\n=== Demo: Processing Conversation ===")
        demo_messages = [
            {"role": "user", "content": "I'm building a data pipeline project."},
            {"role": "assistant", "content": "Interesting! What kind of data?"},
            {"role": "user", "content": "Mostly event data. I wonder if event sourcing would help with the audit requirements we have."},
            {"role": "assistant", "content": "Event sourcing could definitely help with audit trails."},
            {"role": "user", "content": "Yeah, I'm curious about how it compares to traditional CRUD."}
        ]

        print("\nConversation:")
        for msg in demo_messages:
            print(f"  {msg['role'].upper()}: {msg['content']}")

        print("\n--- Processing ---")
        results = pipeline.process_conversation(demo_messages, max_topics=1)

        if results:
            for r in results:
                print(f"\n=== Result: {r.topic} ===")
                print(f"Success: {r.success}")
                print(f"Sources: {r.sources_found}")
                print(f"Confidence: {r.confidence}")
                print(f"Time: {r.total_time_ms}ms")
                if r.research_id:
                    print(f"Stored as ID: {r.research_id}")
                print(f"\nSummary:\n{r.summary}")
                if r.key_insights:
                    print("\nKey Insights:")
                    for insight in r.key_insights:
                        print(f"  • {insight}")
        else:
            print("\nNo topics to research")

    elif args.conversation:
        print("\n=== Interactive Conversation Mode ===")
        print("Enter conversation messages (empty line when done):")

        messages = []
        while True:
            user_input = input("USER: ")
            if not user_input:
                break
            messages.append({"role": "user", "content": user_input})

            assistant = input("ASSISTANT (or empty): ")
            if assistant:
                messages.append({"role": "assistant", "content": assistant})

        if messages:
            print("\n--- Processing ---")
            results = pipeline.process_conversation(messages)
            for r in results:
                print(f"\n{r.topic}: {r.success}")
                if r.summary:
                    print(f"  {r.summary[:200]}...")

    elif args.topic:
        print(f"\n=== Researching: {args.topic} ===")
        result = pipeline.research_topic(args.topic)

        print(f"\nSuccess: {result.success}")
        print(f"Sources: {result.sources_found}")
        print(f"Confidence: {result.confidence}")
        print(f"Time: {result.total_time_ms}ms")

        if result.success:
            print(f"\n--- Summary ---\n{result.summary}")
            if result.key_insights:
                print("\n--- Key Insights ---")
                for insight in result.key_insights:
                    print(f"  • {insight}")

    else:
        parser.print_help()
