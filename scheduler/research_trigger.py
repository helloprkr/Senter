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
