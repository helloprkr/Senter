#!/usr/bin/env python3
"""
Time-based Pattern Detection (US-009)

Analyzes user_events to detect behavioral patterns:
- Peak usage hours
- Most common topics by time of day
- User activity patterns

Patterns stored in data/learning/patterns.json
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional

logger = logging.getLogger('senter.pattern_detector')


class PatternDetector:
    """
    Analyzes user events to detect time-based patterns.
    """

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.patterns_file = self.senter_root / "data" / "learning" / "patterns.json"
        self.patterns_file.parent.mkdir(parents=True, exist_ok=True)

    def analyze_patterns(self, days: int = 7) -> dict:
        """
        Analyze events from the last N days to detect patterns.

        Returns dict with:
        - peak_hours: List of hours with most activity
        - topics_by_time: Topics most common at each time of day
        - daily_activity: Activity counts by day of week
        - topic_frequency: Overall topic frequency
        """
        from learning.events_db import UserEventsDB

        db = UserEventsDB(senter_root=self.senter_root)
        events = db.get_events_by_time_range(hours=days * 24)

        if not events:
            logger.info("No events to analyze")
            return self._empty_patterns()

        patterns = {
            "analyzed_at": datetime.now().isoformat(),
            "event_count": len(events),
            "days_analyzed": days,
            "peak_hours": self._find_peak_hours(events),
            "topics_by_time_of_day": self._topics_by_time_of_day(events),
            "daily_activity": self._daily_activity(events),
            "topic_frequency": self._topic_frequency(events),
            "hourly_distribution": self._hourly_distribution(events),
            "average_response_time": self._average_response_time(events)
        }

        return patterns

    def _empty_patterns(self) -> dict:
        """Return empty patterns structure"""
        return {
            "analyzed_at": datetime.now().isoformat(),
            "event_count": 0,
            "days_analyzed": 0,
            "peak_hours": [],
            "topics_by_time_of_day": {},
            "daily_activity": {},
            "topic_frequency": {},
            "hourly_distribution": {},
            "average_response_time": 0
        }

    def _find_peak_hours(self, events: list) -> list:
        """Find the hours with most user activity"""
        hour_counts = Counter()

        for event in events:
            if event.event_type == "query":
                hour = datetime.fromtimestamp(event.timestamp).hour
                hour_counts[hour] += 1

        if not hour_counts:
            return []

        # Get top 3 peak hours
        sorted_hours = hour_counts.most_common(3)
        return [{"hour": h, "query_count": c} for h, c in sorted_hours]

    def _topics_by_time_of_day(self, events: list) -> dict:
        """Find most common topics for each time of day"""
        time_topics = defaultdict(Counter)

        for event in events:
            if event.event_type == "query":
                time_of_day = event.metadata.get("time_of_day", "unknown")
                topic = event.metadata.get("topic", "general")
                time_topics[time_of_day][topic] += 1

        result = {}
        for time_of_day, topics in time_topics.items():
            top_topics = topics.most_common(3)
            result[time_of_day] = [{"topic": t, "count": c} for t, c in top_topics]

        return result

    def _daily_activity(self, events: list) -> dict:
        """Count activity by day of week"""
        day_counts = Counter()
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]

        for event in events:
            if event.event_type == "query":
                day = datetime.fromtimestamp(event.timestamp).weekday()
                day_counts[day_names[day]] += 1

        return dict(day_counts.most_common())

    def _topic_frequency(self, events: list) -> dict:
        """Count overall topic frequency"""
        topics = Counter()

        for event in events:
            if event.event_type == "query":
                topic = event.metadata.get("topic", "general")
                topics[topic] += 1

        return dict(topics.most_common(10))

    def _hourly_distribution(self, events: list) -> dict:
        """Get query counts for each hour of the day"""
        hourly = Counter()

        for event in events:
            if event.event_type == "query":
                hour = datetime.fromtimestamp(event.timestamp).hour
                hourly[hour] += 1

        return {str(h): hourly.get(h, 0) for h in range(24)}

    def _average_response_time(self, events: list) -> float:
        """Calculate average response time in ms"""
        latencies = []

        for event in events:
            if event.event_type == "response":
                latency = event.metadata.get("latency_ms", 0)
                if latency > 0:
                    latencies.append(latency)

        if not latencies:
            return 0

        return sum(latencies) / len(latencies)

    def save_patterns(self, patterns: dict):
        """Save patterns to JSON file"""
        self.patterns_file.write_text(json.dumps(patterns, indent=2))
        logger.info(f"Patterns saved to {self.patterns_file}")

    def load_patterns(self) -> Optional[dict]:
        """Load patterns from JSON file"""
        if not self.patterns_file.exists():
            return None

        try:
            return json.loads(self.patterns_file.read_text())
        except Exception as e:
            logger.warning(f"Could not load patterns: {e}")
            return None

    def update_patterns(self, days: int = 7) -> dict:
        """Analyze and save updated patterns"""
        patterns = self.analyze_patterns(days=days)
        self.save_patterns(patterns)
        return patterns

    def get_insights(self) -> dict:
        """Get human-readable insights from patterns"""
        patterns = self.load_patterns()
        if not patterns:
            return {"message": "No patterns detected yet"}

        insights = []

        # Peak hours insight
        peak_hours = patterns.get("peak_hours", [])
        if peak_hours:
            hours = [f"{h['hour']}:00" for h in peak_hours]
            insights.append(f"Your most active hours are: {', '.join(hours)}")

        # Topic insights
        topics = patterns.get("topic_frequency", {})
        if topics:
            top_topic = list(topics.keys())[0] if topics else "general"
            insights.append(f"Your most frequent topic is: {top_topic}")

        # Time of day patterns
        by_time = patterns.get("topics_by_time_of_day", {})
        for time_of_day, topic_list in by_time.items():
            if topic_list:
                top = topic_list[0]["topic"]
                insights.append(f"In the {time_of_day}, you usually ask about: {top}")

        # Response time
        avg_response = patterns.get("average_response_time", 0)
        if avg_response > 0:
            insights.append(f"Average response time: {avg_response:.0f}ms")

        return {
            "patterns": patterns,
            "insights": insights,
            "last_updated": patterns.get("analyzed_at")
        }


def detect_and_save_patterns(senter_root: Path, days: int = 7) -> dict:
    """
    Main function to detect and save patterns.

    Called by scheduler job for daily pattern updates.
    """
    detector = PatternDetector(senter_root)
    return detector.update_patterns(days=days)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)

        # Create test events
        from learning.events_db import UserEventsDB, UserEvent

        db = UserEventsDB(senter_root=senter_root)

        # Log some test events at different times
        now = time.time()
        test_events = [
            # Morning queries about coding
            {"type": "query", "time_offset": -3600*9, "topic": "coding", "tod": "morning"},
            {"type": "query", "time_offset": -3600*9.5, "topic": "coding", "tod": "morning"},
            # Afternoon queries about research
            {"type": "query", "time_offset": -3600*14, "topic": "research", "tod": "afternoon"},
            {"type": "query", "time_offset": -3600*15, "topic": "learning", "tod": "afternoon"},
            # Evening queries
            {"type": "query", "time_offset": -3600*20, "topic": "creative", "tod": "evening"},
            # Responses
            {"type": "response", "time_offset": -3600*9, "latency": 1500},
            {"type": "response", "time_offset": -3600*14, "latency": 1200},
        ]

        for te in test_events:
            if te["type"] == "query":
                event = UserEvent(
                    event_type="query",
                    timestamp=now + te["time_offset"],
                    context={"query": f"Test query about {te['topic']}"},
                    metadata={"topic": te["topic"], "time_of_day": te["tod"]}
                )
            else:
                event = UserEvent(
                    event_type="response",
                    timestamp=now + te["time_offset"],
                    context={"response": "Test response"},
                    metadata={"latency_ms": te["latency"]}
                )
            db.log_event(event)

        # Test pattern detection
        print("\nTesting Pattern Detector...")
        detector = PatternDetector(senter_root)

        patterns = detector.analyze_patterns(days=1)
        print(f"\nPatterns detected:")
        print(f"  Event count: {patterns['event_count']}")
        print(f"  Peak hours: {patterns['peak_hours']}")
        print(f"  Topic frequency: {patterns['topic_frequency']}")
        print(f"  Topics by time: {patterns['topics_by_time_of_day']}")
        print(f"  Avg response time: {patterns['average_response_time']:.0f}ms")

        # Save and reload
        detector.save_patterns(patterns)
        loaded = detector.load_patterns()
        assert loaded is not None, "Should load saved patterns"

        # Get insights
        insights = detector.get_insights()
        print(f"\nInsights:")
        for insight in insights.get("insights", []):
            print(f"  - {insight}")

        print("\nTest complete!")
