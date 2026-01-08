#!/usr/bin/env python3
"""
Research Trigger Module (US-005)

Extracts research topics from recent user queries and generates
research tasks for background processing.
"""

import json
import time
import logging
import uuid
from pathlib import Path
from collections import Counter
from typing import Optional

logger = logging.getLogger('senter.research_trigger')


class ResearchTopicExtractor:
    """
    Extracts research topics from user queries stored in learning database.
    """

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.db_path = self.senter_root / "data" / "learning" / "behavior.db"

    def get_recent_topics(self, hours: int = 24, limit: int = 5) -> list[dict]:
        """
        Get research topics from recent user queries.

        Returns list of dicts with 'topic', 'query_count', 'sample_queries'.
        """
        import sqlite3

        if not self.db_path.exists():
            logger.warning(f"Learning database not found: {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            # Get recent query events
            since = time.time() - (hours * 3600)
            cursor = conn.execute(
                """SELECT data FROM events
                   WHERE event_type = 'query' AND timestamp >= ?
                   ORDER BY timestamp DESC LIMIT 100""",
                (since,)
            )

            # Extract topics and queries
            topic_queries = {}  # topic -> list of queries
            for row in cursor.fetchall():
                try:
                    data = json.loads(row["data"])
                    query = data.get("query", "")
                    topics = data.get("topics", [])

                    for topic in topics:
                        if topic not in topic_queries:
                            topic_queries[topic] = []
                        if query and query not in topic_queries[topic]:
                            topic_queries[topic].append(query)
                except:
                    continue

            conn.close()

            # Sort by query count and format result
            results = []
            for topic, queries in sorted(topic_queries.items(),
                                         key=lambda x: len(x[1]), reverse=True)[:limit]:
                results.append({
                    "topic": topic,
                    "query_count": len(queries),
                    "sample_queries": queries[:3]  # Up to 3 samples
                })

            return results

        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []

    def get_top_patterns(self, limit: int = 5) -> list[dict]:
        """Get top topic patterns from the patterns table."""
        import sqlite3

        if not self.db_path.exists():
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """SELECT pattern_key, count, last_seen FROM patterns
                   WHERE pattern_type = 'topic'
                   ORDER BY count DESC LIMIT ?""",
                (limit,)
            )

            results = []
            for row in cursor.fetchall():
                results.append({
                    "topic": row["pattern_key"],
                    "count": row["count"],
                    "last_seen": row["last_seen"]
                })

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Error getting patterns: {e}")
            return []


class ResearchTaskGenerator:
    """
    Generates research tasks from topics.
    """

    # Research prompts for different topics
    RESEARCH_PROMPTS = {
        "coding": "Research current best practices and emerging patterns in {topic}. Focus on practical techniques and tools.",
        "research": "Conduct a comprehensive literature review on {topic}. Summarize key findings and identify gaps.",
        "writing": "Research effective techniques and strategies for {topic}. Include examples and templates.",
        "creative": "Explore creative approaches and inspiration for {topic}. Include diverse perspectives.",
        "productivity": "Research productivity methods and tools related to {topic}. Focus on evidence-based approaches.",
        "learning": "Create a comprehensive learning guide for {topic}. Include key concepts, resources, and practice exercises.",
    }

    DEFAULT_PROMPT = "Research the following topic thoroughly and provide comprehensive insights: {topic}"

    def generate_task(self, topic: str, sample_queries: list[str] = None) -> dict:
        """Generate a research task for a topic."""
        # Get appropriate prompt template
        prompt_template = self.RESEARCH_PROMPTS.get(topic, self.DEFAULT_PROMPT)

        # Build context from sample queries
        context = ""
        if sample_queries:
            context = f"\n\nRecent related questions from user:\n" + \
                      "\n".join(f"- {q}" for q in sample_queries[:3])

        # Generate description
        description = prompt_template.format(topic=topic)
        if context:
            description += context

        return {
            "id": f"research_{int(time.time())}_{uuid.uuid4().hex[:6]}",
            "description": description,
            "topic": topic,
            "source": "background_research",
            "priority": "low",
            "created_at": time.time()
        }

    def generate_tasks_from_topics(self, topics: list[dict], max_tasks: int = 3) -> list[dict]:
        """Generate research tasks from a list of topics."""
        tasks = []
        for topic_info in topics[:max_tasks]:
            topic = topic_info["topic"]
            sample_queries = topic_info.get("sample_queries", [])
            task = self.generate_task(topic, sample_queries)
            tasks.append(task)
        return tasks


def trigger_background_research(senter_root: Path) -> list[dict]:
    """
    Main function to trigger background research.

    Called by the scheduler or IPC handler.
    Returns list of generated research tasks.
    """
    extractor = ResearchTopicExtractor(senter_root)
    generator = ResearchTaskGenerator()

    # Get recent topics from user queries
    topics = extractor.get_recent_topics(hours=24, limit=3)

    if not topics:
        # Fall back to top patterns if no recent queries
        patterns = extractor.get_top_patterns(limit=3)
        if patterns:
            topics = [{"topic": p["topic"], "query_count": p["count"], "sample_queries": []}
                      for p in patterns]

    if not topics:
        logger.info("No topics found for background research")
        return []

    # Generate tasks
    tasks = generator.generate_tasks_from_topics(topics, max_tasks=2)

    logger.info(f"Generated {len(tasks)} background research tasks from topics: "
                f"{[t['topic'] for t in topics[:2]]}")

    return tasks


# ========== RW-001: Research Result Summarization ==========

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ResearchResult:
    """A completed research result"""
    task_id: str
    topic: str
    content: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.5
    completed_at: float = field(default_factory=time.time)
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "content": self.content[:500],  # Truncate for storage
            "sources": self.sources,
            "confidence": self.confidence,
            "completed_at": self.completed_at,
            "summary": self.summary,
            "key_findings": self.key_findings
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ResearchResult":
        return cls(**data)


class ResearchResultSummarizer:
    """
    Research Result Summarization (RW-001)

    Generates summaries for completed research results:
    - Key findings extraction
    - Source attribution
    - Confidence level assessment
    """

    DEFAULT_SUMMARY_SENTENCES = 3

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.results_dir = self.senter_root / "data" / "research" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def summarize(self, content: str, max_sentences: int = None) -> str:
        """
        Generate a summary of research content.

        Args:
            content: Full research content
            max_sentences: Max sentences in summary (default 3)

        Returns:
            Summary string
        """
        if max_sentences is None:
            max_sentences = self.DEFAULT_SUMMARY_SENTENCES

        if not content:
            return "No content available for summarization."

        # Simple extractive summarization
        sentences = self._split_sentences(content)

        if len(sentences) <= max_sentences:
            return content

        # Score sentences by position and keyword density
        scored = []
        keywords = self._extract_keywords(content)

        for i, sentence in enumerate(sentences):
            score = 0
            # First sentences get higher scores
            if i < 3:
                score += (3 - i) * 2
            # Sentences with keywords get higher scores
            sentence_lower = sentence.lower()
            for keyword in keywords:
                if keyword in sentence_lower:
                    score += 1
            # Longer sentences get slightly higher scores (more information)
            if len(sentence) > 50:
                score += 1

            scored.append((score, i, sentence))

        # Sort by score and take top sentences
        scored.sort(reverse=True)
        top_sentences = sorted(scored[:max_sentences], key=lambda x: x[1])

        return " ".join(s[2] for s in top_sentences)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords from text"""
        import re
        # Simple keyword extraction
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        # Filter common stopwords
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they',
                     'their', 'about', 'which', 'when', 'where', 'what', 'there',
                     'would', 'could', 'should', 'will', 'more', 'some', 'also'}
        words = [w for w in words if w not in stopwords]

        word_counts = Counter(words)
        return [w for w, _ in word_counts.most_common(top_n)]

    def extract_key_findings(self, content: str, max_findings: int = 5) -> List[str]:
        """
        Extract key findings from research content.

        Returns list of key finding strings.
        """
        if not content:
            return []

        findings = []
        sentences = self._split_sentences(content)

        # Look for sentences with finding indicators
        finding_indicators = [
            "found that", "shows that", "indicates", "reveals",
            "key insight", "important", "significant", "conclusion",
            "main point", "discovered", "determined", "established"
        ]

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for indicator in finding_indicators:
                if indicator in sentence_lower:
                    findings.append(sentence)
                    break

        # If no explicit findings, take first few sentences
        if not findings and sentences:
            findings = sentences[:max_findings]

        return findings[:max_findings]

    def process_result(self, task_id: str, topic: str, content: str,
                      sources: List[str] = None, confidence: float = 0.5) -> ResearchResult:
        """
        Process a research result and generate summary.

        Returns complete ResearchResult with summary and key findings.
        """
        summary = self.summarize(content)
        key_findings = self.extract_key_findings(content)

        result = ResearchResult(
            task_id=task_id,
            topic=topic,
            content=content,
            sources=sources or [],
            confidence=confidence,
            summary=summary,
            key_findings=key_findings
        )

        # Save result
        self.save_result(result)

        return result

    def save_result(self, result: ResearchResult):
        """Save research result to file"""
        result_file = self.results_dir / f"{result.task_id}.json"
        result_file.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info(f"Research result saved: {result.task_id}")

    def load_result(self, task_id: str) -> Optional[ResearchResult]:
        """Load a research result by task ID"""
        result_file = self.results_dir / f"{task_id}.json"
        if not result_file.exists():
            return None
        try:
            data = json.loads(result_file.read_text())
            return ResearchResult.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading result {task_id}: {e}")
            return None

    def get_recent_results(self, limit: int = 10) -> List[ResearchResult]:
        """Get recent research results"""
        results = []
        for result_file in sorted(self.results_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(result_file.read_text())
                results.append(ResearchResult.from_dict(data))
            except:
                continue
        return results


# ========== RW-002: Autonomous Research Topic Selection ==========

class AutonomousResearchSelector:
    """
    Autonomous Research Topic Selection (RW-002)

    Selects research topics based on user interests:
    - Analyzes frequent discussion topics
    - Uses pattern detector insights
    - Respects user preferences (can be disabled)
    """

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.config_file = self.senter_root / "config" / "research_config.json"
        self._config = self._load_config()

    def _load_config(self) -> Dict:
        """Load research configuration"""
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text())
            except:
                pass
        return {
            "autonomous_enabled": True,
            "max_topics_per_day": 3,
            "min_topic_mentions": 3,
            "excluded_topics": [],
            "priority_topics": []
        }

    def save_config(self):
        """Save configuration"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(self._config, indent=2))

    def is_autonomous_enabled(self) -> bool:
        """Check if autonomous research is enabled"""
        return self._config.get("autonomous_enabled", True)

    def set_autonomous_enabled(self, enabled: bool):
        """Enable or disable autonomous research"""
        self._config["autonomous_enabled"] = enabled
        self.save_config()
        logger.info(f"Autonomous research {'enabled' if enabled else 'disabled'}")

    def select_topics(self, available_topics: List[Dict]) -> List[Dict]:
        """
        Select topics for autonomous research.

        Args:
            available_topics: List of topic dicts with 'topic', 'query_count'

        Returns:
            Selected topics for research
        """
        if not self.is_autonomous_enabled():
            return []

        max_topics = self._config.get("max_topics_per_day", 3)
        min_mentions = self._config.get("min_topic_mentions", 3)
        excluded = set(self._config.get("excluded_topics", []))
        priority = self._config.get("priority_topics", [])

        selected = []

        # First add priority topics if they appear
        for topic_info in available_topics:
            topic = topic_info.get("topic", "")
            if topic in priority and topic not in excluded:
                selected.append(topic_info)

        # Then add frequent topics
        for topic_info in available_topics:
            topic = topic_info.get("topic", "")
            count = topic_info.get("query_count", 0)

            if topic in excluded:
                continue
            if topic_info in selected:
                continue
            if count < min_mentions:
                continue

            selected.append(topic_info)

            if len(selected) >= max_topics:
                break

        return selected

    def get_topic_from_patterns(self) -> Optional[Dict]:
        """
        Get a research topic from pattern detector insights.

        Integrates with learning system to find interesting topics.
        """
        try:
            from learning.pattern_detector import PatternDetector

            detector = PatternDetector(self.senter_root)
            patterns = detector.load_patterns()

            if patterns:
                topic_freq = patterns.get("topic_frequency", {})
                for topic, count in list(topic_freq.items())[:1]:
                    return {"topic": topic, "query_count": count, "source": "patterns"}
        except Exception as e:
            logger.debug(f"Could not get patterns: {e}")

        return None

    def add_excluded_topic(self, topic: str):
        """Exclude a topic from autonomous research"""
        excluded = self._config.get("excluded_topics", [])
        if topic not in excluded:
            excluded.append(topic)
            self._config["excluded_topics"] = excluded
            self.save_config()

    def add_priority_topic(self, topic: str):
        """Add a priority topic for autonomous research"""
        priority = self._config.get("priority_topics", [])
        if topic not in priority:
            priority.append(topic)
            self._config["priority_topics"] = priority
            self.save_config()


# ========== RW-003: Research Result Presentation ==========

@dataclass
class ResearchOffer:
    """An offer to share research results"""
    result_id: str
    topic: str
    summary: str
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, accepted, deferred, dismissed

    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "topic": self.topic,
            "summary": self.summary,
            "created_at": self.created_at,
            "status": self.status
        }


class ResearchResultPresenter:
    """
    Research Result Presentation (RW-003)

    Presents research results to user when they become active:
    - Generates offer messages
    - Tracks offer status (accept/defer/dismiss)
    - Queues results for presentation
    """

    OFFER_TEMPLATE = "I researched {topic} while you were away. Want a summary?"

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.offers_file = self.senter_root / "data" / "research" / "pending_offers.json"
        self.offers_file.parent.mkdir(parents=True, exist_ok=True)
        self._pending_offers: List[ResearchOffer] = []
        self._load_offers()

    def _load_offers(self):
        """Load pending offers from file"""
        if self.offers_file.exists():
            try:
                data = json.loads(self.offers_file.read_text())
                self._pending_offers = [
                    ResearchOffer(**o) for o in data.get("offers", [])
                ]
            except:
                self._pending_offers = []

    def _save_offers(self):
        """Save pending offers to file"""
        data = {
            "updated_at": datetime.now().isoformat(),
            "offers": [o.to_dict() for o in self._pending_offers]
        }
        self.offers_file.write_text(json.dumps(data, indent=2))

    def queue_result(self, result: ResearchResult) -> ResearchOffer:
        """
        Queue a research result for presentation.

        Returns the created offer.
        """
        offer = ResearchOffer(
            result_id=result.task_id,
            topic=result.topic,
            summary=result.summary[:200]  # Truncate for offer
        )

        self._pending_offers.append(offer)
        self._save_offers()

        logger.info(f"Queued research offer for topic: {result.topic}")
        return offer

    def get_pending_offers(self) -> List[ResearchOffer]:
        """Get all pending offers"""
        return [o for o in self._pending_offers if o.status == "pending"]

    def generate_offer_message(self, offer: ResearchOffer) -> str:
        """Generate the offer message for user"""
        return self.OFFER_TEMPLATE.format(topic=offer.topic)

    def on_attention_gained(self) -> Optional[str]:
        """
        Called when user gains attention.

        Returns offer message if there are pending results, None otherwise.
        """
        pending = self.get_pending_offers()
        if not pending:
            return None

        # Get the oldest pending offer
        offer = pending[0]
        return self.generate_offer_message(offer)

    def accept_offer(self, result_id: str) -> Optional[ResearchResult]:
        """
        User accepts offer, return full result.

        Returns the full research result.
        """
        for offer in self._pending_offers:
            if offer.result_id == result_id:
                offer.status = "accepted"
                self._save_offers()

                # Load and return full result
                summarizer = ResearchResultSummarizer(self.senter_root)
                return summarizer.load_result(result_id)

        return None

    def defer_offer(self, result_id: str):
        """User defers offer (will be offered again later)"""
        for offer in self._pending_offers:
            if offer.result_id == result_id:
                offer.status = "deferred"
                # Move to end of queue by updating created_at
                offer.created_at = time.time() + 3600  # Defer for 1 hour
                offer.status = "pending"
                self._save_offers()
                break

    def dismiss_offer(self, result_id: str):
        """User dismisses offer (won't be offered again)"""
        for offer in self._pending_offers:
            if offer.result_id == result_id:
                offer.status = "dismissed"
                self._save_offers()
                break

    def handle_user_response(self, result_id: str, response: str) -> Optional[str]:
        """
        Handle user response to offer.

        Args:
            result_id: The result being responded to
            response: User response (accept, defer, dismiss, or text)

        Returns:
            Response message or None
        """
        response_lower = response.lower().strip()

        if response_lower in ["yes", "sure", "ok", "accept", "show me"]:
            result = self.accept_offer(result_id)
            if result:
                return f"## Research Summary: {result.topic}\n\n{result.summary}\n\n**Key Findings:**\n" + \
                       "\n".join(f"- {f}" for f in result.key_findings[:3])
            return "Sorry, I couldn't find that research result."

        elif response_lower in ["later", "defer", "not now"]:
            self.defer_offer(result_id)
            return "Got it, I'll remind you later."

        elif response_lower in ["no", "dismiss", "skip", "no thanks"]:
            self.dismiss_offer(result_id)
            return "Okay, I've dismissed that research result."

        return None

    def clear_old_offers(self, max_age_hours: int = 72):
        """Clear offers older than max_age_hours"""
        cutoff = time.time() - (max_age_hours * 3600)
        self._pending_offers = [
            o for o in self._pending_offers
            if o.created_at > cutoff or o.status == "pending"
        ]
        self._save_offers()


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    senter_root = Path(__file__).parent.parent

    print("Testing Research Trigger Module")
    print("=" * 50)

    # Test topic extraction
    extractor = ResearchTopicExtractor(senter_root)

    print("\nRecent topics from queries:")
    topics = extractor.get_recent_topics(hours=168)  # Last week
    for t in topics:
        print(f"  - {t['topic']}: {t['query_count']} queries")
        if t['sample_queries']:
            print(f"    Sample: {t['sample_queries'][0][:50]}...")

    print("\nTop patterns:")
    patterns = extractor.get_top_patterns()
    for p in patterns:
        print(f"  - {p['topic']}: {p['count']} occurrences")

    # Test task generation
    print("\nGenerated research tasks:")
    tasks = trigger_background_research(senter_root)
    for task in tasks:
        print(f"  - [{task['id']}] {task['topic']}")
        print(f"    {task['description'][:80]}...")

    print("\nTest complete")
