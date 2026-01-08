#!/usr/bin/env python3
"""
UI-003: Feedback That Actually Improves Future Research

Feedback collection and analysis for improving research quality.

VALUE: User rates a research 2 stars → Senter learns that topic type
or source type wasn't helpful. Future research improves.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("senter.research.feedback")


@dataclass
class FeedbackStats:
    """Statistics about feedback."""
    total_ratings: int
    average_rating: float
    rating_distribution: Dict[int, int]  # {1: 5, 2: 3, ...}
    top_rated_topics: List[str]
    low_rated_topics: List[str]


@dataclass
class TopicPattern:
    """A learned pattern about topics."""
    pattern: str  # e.g., "kubernetes", "python async"
    avg_rating: float
    count: int
    last_seen: datetime


class FeedbackAnalyzer:
    """
    Analyzes feedback to improve future research.

    Tracks:
    - Which topic types get good/bad ratings
    - Which source domains are trusted
    - Research quality trends over time
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".senter" / "research.db"
        self.db_path = db_path

    def get_stats(self) -> FeedbackStats:
        """Get overall feedback statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total and average
            row = conn.execute("""
                SELECT COUNT(*) as total, AVG(feedback_rating) as avg
                FROM research WHERE feedback_rating IS NOT NULL
            """).fetchone()

            total = row["total"] or 0
            avg = row["avg"] or 0.0

            # Distribution
            distribution = {i: 0 for i in range(1, 6)}
            rows = conn.execute("""
                SELECT feedback_rating, COUNT(*) as count
                FROM research
                WHERE feedback_rating IS NOT NULL
                GROUP BY feedback_rating
            """).fetchall()
            for r in rows:
                distribution[r["feedback_rating"]] = r["count"]

            # Top rated topics
            top_rows = conn.execute("""
                SELECT topic FROM research
                WHERE feedback_rating IS NOT NULL
                ORDER BY feedback_rating DESC
                LIMIT 5
            """).fetchall()
            top_rated = [r["topic"] for r in top_rows]

            # Low rated topics
            low_rows = conn.execute("""
                SELECT topic FROM research
                WHERE feedback_rating IS NOT NULL
                ORDER BY feedback_rating ASC
                LIMIT 5
            """).fetchall()
            low_rated = [r["topic"] for r in low_rows]

            return FeedbackStats(
                total_ratings=total,
                average_rating=round(avg, 2),
                rating_distribution=distribution,
                top_rated_topics=top_rated,
                low_rated_topics=low_rated
            )

    def get_topic_patterns(self, min_count: int = 2) -> List[TopicPattern]:
        """
        Analyze topic patterns to find what works well.

        Returns patterns (keywords) that appear in multiple researches
        with their average ratings.
        """
        patterns = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get all rated research
            rows = conn.execute("""
                SELECT topic, feedback_rating, created_at
                FROM research
                WHERE feedback_rating IS NOT NULL
            """).fetchall()

            if not rows:
                return patterns

            # Extract keywords and aggregate
            keyword_data: Dict[str, Dict] = {}

            for row in rows:
                topic_words = row["topic"].lower().split()
                rating = row["feedback_rating"]
                created = datetime.fromtimestamp(row["created_at"])

                for word in topic_words:
                    if len(word) < 3:  # Skip short words
                        continue
                    if word not in keyword_data:
                        keyword_data[word] = {
                            "ratings": [],
                            "last_seen": created
                        }
                    keyword_data[word]["ratings"].append(rating)
                    if created > keyword_data[word]["last_seen"]:
                        keyword_data[word]["last_seen"] = created

            # Convert to patterns
            for word, data in keyword_data.items():
                if len(data["ratings"]) >= min_count:
                    patterns.append(TopicPattern(
                        pattern=word,
                        avg_rating=sum(data["ratings"]) / len(data["ratings"]),
                        count=len(data["ratings"]),
                        last_seen=data["last_seen"]
                    ))

            # Sort by rating
            patterns.sort(key=lambda p: p.avg_rating, reverse=True)

        return patterns

    def get_source_quality(self) -> Dict[str, float]:
        """
        Analyze which source domains tend to produce good research.

        Returns domain -> average rating mapping.
        """
        domain_ratings: Dict[str, List[int]] = {}

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT sources, feedback_rating
                FROM research
                WHERE feedback_rating IS NOT NULL AND sources IS NOT NULL
            """).fetchall()

            for row in rows:
                sources = json.loads(row[0]) if row[0] else []
                rating = row[1]

                for url in sources:
                    # Extract domain
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        if domain:
                            if domain not in domain_ratings:
                                domain_ratings[domain] = []
                            domain_ratings[domain].append(rating)
                    except:
                        continue

        # Calculate averages
        return {
            domain: round(sum(ratings) / len(ratings), 2)
            for domain, ratings in domain_ratings.items()
            if len(ratings) >= 1
        }

    def should_research_topic(self, topic: str) -> Dict[str, Any]:
        """
        Advise whether a topic is likely to produce good research.

        Based on historical feedback for similar topics.
        """
        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())

        patterns = self.get_topic_patterns(min_count=1)

        # Find matching patterns
        matching = [p for p in patterns if p.pattern in topic_words]

        if not matching:
            return {
                "recommend": True,
                "confidence": "unknown",
                "reason": "No historical data for this topic type"
            }

        avg_rating = sum(p.avg_rating for p in matching) / len(matching)

        if avg_rating >= 4.0:
            return {
                "recommend": True,
                "confidence": "high",
                "reason": f"Similar topics rated well (avg {avg_rating:.1f})"
            }
        elif avg_rating >= 3.0:
            return {
                "recommend": True,
                "confidence": "medium",
                "reason": f"Similar topics rated average (avg {avg_rating:.1f})"
            }
        else:
            return {
                "recommend": False,
                "confidence": "high",
                "reason": f"Similar topics rated poorly (avg {avg_rating:.1f})"
            }

    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for improving research quality."""
        suggestions = []
        stats = self.get_stats()

        if stats.total_ratings < 5:
            suggestions.append(
                "Rate more research to help Senter learn your preferences"
            )
            return suggestions

        if stats.average_rating < 3.0:
            suggestions.append(
                "Research quality is below average - consider adjusting topics"
            )

        # Check source quality
        source_quality = self.get_source_quality()
        poor_sources = [d for d, r in source_quality.items() if r < 3.0]
        if poor_sources:
            suggestions.append(
                f"Consider avoiding sources from: {', '.join(poor_sources[:3])}"
            )

        # Check topic patterns
        patterns = self.get_topic_patterns()
        poor_topics = [p.pattern for p in patterns if p.avg_rating < 3.0]
        if poor_topics:
            suggestions.append(
                f"Topics containing '{poor_topics[0]}' tend to get low ratings"
            )

        if not suggestions:
            suggestions.append(
                f"Research quality is good (avg {stats.average_rating:.1f}/5)"
            )

        return suggestions


# CLI for testing
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Feedback Analysis")
    parser.add_argument("--stats", "-s", action="store_true", help="Show stats")
    parser.add_argument("--patterns", "-p", action="store_true", help="Show patterns")
    parser.add_argument("--sources", action="store_true", help="Show source quality")
    parser.add_argument("--suggest", action="store_true", help="Get suggestions")
    parser.add_argument("--check", type=str, help="Check if topic is recommended")

    args = parser.parse_args()

    analyzer = FeedbackAnalyzer()

    if args.stats:
        stats = analyzer.get_stats()
        print("\n=== Feedback Statistics ===")
        print(f"Total ratings: {stats.total_ratings}")
        print(f"Average rating: {stats.average_rating}/5")
        print(f"\nDistribution:")
        for stars, count in stats.rating_distribution.items():
            bar = "★" * count
            print(f"  {stars} star: {bar} ({count})")

    elif args.patterns:
        patterns = analyzer.get_topic_patterns()
        print("\n=== Topic Patterns ===")
        if patterns:
            for p in patterns[:10]:
                print(f"  '{p.pattern}': {p.avg_rating:.1f}/5 ({p.count} times)")
        else:
            print("  No patterns found yet")

    elif args.sources:
        quality = analyzer.get_source_quality()
        print("\n=== Source Quality ===")
        if quality:
            sorted_q = sorted(quality.items(), key=lambda x: x[1], reverse=True)
            for domain, rating in sorted_q[:10]:
                print(f"  {domain}: {rating}/5")
        else:
            print("  No source data yet")

    elif args.suggest:
        suggestions = analyzer.get_improvement_suggestions()
        print("\n=== Improvement Suggestions ===")
        for s in suggestions:
            print(f"  • {s}")

    elif args.check:
        advice = analyzer.should_research_topic(args.check)
        print(f"\n=== Topic: {args.check} ===")
        print(f"Recommend: {'Yes' if advice['recommend'] else 'No'}")
        print(f"Confidence: {advice['confidence']}")
        print(f"Reason: {advice['reason']}")

    else:
        # Show summary
        stats = analyzer.get_stats()
        print("\n=== Feedback Summary ===")
        print(f"Ratings: {stats.total_ratings}")
        print(f"Average: {stats.average_rating}/5")

        suggestions = analyzer.get_improvement_suggestions()
        print("\nSuggestions:")
        for s in suggestions:
            print(f"  • {s}")
