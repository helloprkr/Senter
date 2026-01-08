#!/usr/bin/env python3
"""
Time-based Pattern Detection (US-009) and Preference Learning (CG-004)

Analyzes user_events to detect behavioral patterns:
- Peak usage hours
- Most common topics by time of day
- User activity patterns
- Semantic topic clustering via embeddings (CG-004)

Patterns stored in data/learning/patterns.json
Preferences stored in data/learning/preferences.json
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

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


class PreferenceLearner:
    """
    Embedding-based Preference Learning (CG-004)

    Uses semantic similarity to cluster topics and detect user preferences.
    Falls back to TF-IDF clustering if embeddings unavailable.
    """

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.preferences_file = self.senter_root / "data" / "learning" / "preferences.json"
        self.preferences_file.parent.mkdir(parents=True, exist_ok=True)

        # Embedding model path (optional)
        self._embedding_model = None
        self._use_embeddings = self._check_embedding_support()

    def _check_embedding_support(self) -> bool:
        """Check if embedding support is available"""
        try:
            import sys
            sys.path.insert(0, str(self.senter_root / "Functions"))
            from embedding_utils import get_default_embedding_model
            model_path = get_default_embedding_model()
            if model_path:
                self._embedding_model = model_path
                return True
        except Exception as e:
            logger.debug(f"Embedding support not available: {e}")
        return False

    def analyze_preferences(self, min_queries: int = 10) -> Dict[str, Any]:
        """
        Analyze user preferences using semantic similarity.

        Args:
            min_queries: Minimum queries needed for analysis (default 10)

        Returns:
            Preference model with confidence scores
        """
        from learning.events_db import UserEventsDB

        db = UserEventsDB(senter_root=self.senter_root)
        events = db.get_events_by_time_range(hours=24 * 30)  # Last 30 days

        # Extract queries
        queries = []
        for event in events:
            if event.event_type == "query":
                query_text = event.context.get("query", "")
                if query_text:
                    queries.append({
                        "text": query_text,
                        "timestamp": event.timestamp,
                        "topic": event.metadata.get("topic", "general"),
                        "time_of_day": event.metadata.get("time_of_day", "unknown")
                    })

        if len(queries) < min_queries:
            logger.info(f"Not enough queries for analysis ({len(queries)}/{min_queries})")
            return self._empty_preferences(len(queries))

        # Analyze using embeddings or TF-IDF fallback
        if self._use_embeddings:
            topic_clusters = self._cluster_with_embeddings(queries)
        else:
            topic_clusters = self._cluster_with_tfidf(queries)

        # Calculate preference confidence scores
        preferences = self._calculate_preferences(queries, topic_clusters)

        # Add peak hours from timestamps
        preferences["peak_hours"] = self._calculate_peak_hours(queries)

        # Store metadata
        preferences["analyzed_at"] = datetime.now().isoformat()
        preferences["query_count"] = len(queries)
        preferences["method"] = "embeddings" if self._use_embeddings else "tfidf"

        return preferences

    def _empty_preferences(self, query_count: int) -> Dict[str, Any]:
        """Return empty preferences structure"""
        return {
            "analyzed_at": datetime.now().isoformat(),
            "query_count": query_count,
            "method": "none",
            "topic_preferences": [],
            "response_style": {},
            "peak_hours": [],
            "confidence": 0.0
        }

    def _cluster_with_embeddings(self, queries: List[Dict]) -> List[Dict]:
        """Cluster queries using embedding similarity"""
        try:
            import sys
            sys.path.insert(0, str(self.senter_root / "Functions"))
            from embedding_utils import create_embeddings

            # Get query texts
            texts = [q["text"] for q in queries]

            # Create embeddings
            embeddings = create_embeddings(texts, self._embedding_model)

            # Simple clustering: find cluster centroids using k-means-like approach
            n_clusters = min(5, len(texts) // 3 + 1)  # Max 5 clusters
            clusters = self._kmeans_cluster(embeddings, n_clusters)

            # Map clusters to topics
            return self._map_clusters_to_topics(queries, clusters)

        except Exception as e:
            logger.warning(f"Embedding clustering failed: {e}, falling back to TF-IDF")
            return self._cluster_with_tfidf(queries)

    def _kmeans_cluster(self, embeddings: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
        """Simple k-means clustering without sklearn dependency"""
        n_samples = len(embeddings)
        if n_samples <= k:
            return np.arange(n_samples)

        # Initialize centroids randomly
        np.random.seed(42)
        centroid_indices = np.random.choice(n_samples, k, replace=False)
        centroids = embeddings[centroid_indices].copy()

        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max_iter):
            # Assign points to nearest centroid
            new_labels = np.zeros(n_samples, dtype=int)
            for i, emb in enumerate(embeddings):
                distances = [np.linalg.norm(emb - c) for c in centroids]
                new_labels[i] = np.argmin(distances)

            # Check convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Update centroids
            for j in range(k):
                cluster_points = embeddings[labels == j]
                if len(cluster_points) > 0:
                    centroids[j] = cluster_points.mean(axis=0)

        return labels

    def _cluster_with_tfidf(self, queries: List[Dict]) -> List[Dict]:
        """Fallback: cluster queries using TF-IDF similarity"""
        from collections import Counter
        import re

        texts = [q["text"].lower() for q in queries]

        # Simple TF-IDF: word frequency across documents
        word_docs = defaultdict(set)  # word -> set of doc indices
        doc_words = []  # List of word counters per doc

        for i, text in enumerate(texts):
            words = re.findall(r'\b[a-z]+\b', text)
            word_counter = Counter(words)
            doc_words.append(word_counter)
            for word in set(words):
                word_docs[word].add(i)

        # Calculate TF-IDF vectors
        n_docs = len(texts)
        vocab = list(word_docs.keys())
        vocab_idx = {w: i for i, w in enumerate(vocab)}

        if not vocab:
            return [{"cluster": 0, "queries": queries, "representative": queries[0]["text"]}]

        # Build TF-IDF matrix
        tfidf_matrix = np.zeros((n_docs, len(vocab)))
        for i, word_counter in enumerate(doc_words):
            for word, count in word_counter.items():
                if word in vocab_idx:
                    tf = count / (sum(word_counter.values()) + 1)
                    idf = np.log(n_docs / (len(word_docs[word]) + 1))
                    tfidf_matrix[i, vocab_idx[word]] = tf * idf

        # Cluster using simple k-means on TF-IDF
        n_clusters = min(5, len(texts) // 3 + 1)
        labels = self._kmeans_cluster(tfidf_matrix, n_clusters)

        return self._map_clusters_to_topics(queries, labels)

    def _map_clusters_to_topics(self, queries: List[Dict], labels: np.ndarray) -> List[Dict]:
        """Map cluster labels to topic structures"""
        clusters = defaultdict(list)
        for i, q in enumerate(queries):
            clusters[int(labels[i])].append(q)

        result = []
        for cluster_id, cluster_queries in clusters.items():
            # Find representative query (most common topic or first query)
            topics = Counter(q.get("topic", "general") for q in cluster_queries)
            dominant_topic = topics.most_common(1)[0][0] if topics else "general"

            # Get the query closest to cluster center (first one as approximation)
            representative = cluster_queries[0]["text"] if cluster_queries else ""

            result.append({
                "cluster": cluster_id,
                "dominant_topic": dominant_topic,
                "query_count": len(cluster_queries),
                "queries": cluster_queries,
                "representative": representative[:100]
            })

        return sorted(result, key=lambda x: x["query_count"], reverse=True)

    def _calculate_preferences(self, queries: List[Dict], clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate user preferences with confidence scores"""
        total_queries = len(queries)

        # Topic preferences with confidence
        topic_preferences = []
        for cluster in clusters:
            confidence = cluster["query_count"] / total_queries
            topic_preferences.append({
                "topic": cluster["dominant_topic"],
                "confidence": round(confidence, 3),
                "query_count": cluster["query_count"],
                "representative": cluster["representative"]
            })

        # Response style preferences (detected from query patterns)
        response_style = self._detect_response_style(queries)

        # Overall confidence based on data volume
        overall_confidence = min(1.0, total_queries / 50)  # Full confidence at 50+ queries

        return {
            "topic_preferences": topic_preferences,
            "response_style": response_style,
            "confidence": round(overall_confidence, 3)
        }

    def _detect_response_style(self, queries: List[Dict]) -> Dict[str, Any]:
        """Detect preferred response style from query patterns"""
        style = {
            "preferred_length": "medium",
            "formality": "casual",
            "detail_level": "moderate"
        }

        # Analyze query patterns
        query_texts = [q["text"].lower() for q in queries]

        # Check for detail requests
        detail_keywords = ["explain", "detail", "elaborate", "comprehensive", "thorough"]
        brief_keywords = ["quick", "brief", "short", "summary", "tldr"]

        detail_count = sum(1 for q in query_texts if any(k in q for k in detail_keywords))
        brief_count = sum(1 for q in query_texts if any(k in q for k in brief_keywords))

        if detail_count > brief_count * 2:
            style["preferred_length"] = "detailed"
            style["detail_level"] = "high"
        elif brief_count > detail_count * 2:
            style["preferred_length"] = "brief"
            style["detail_level"] = "low"

        # Check for formality (presence of please, thanks, formal language)
        formal_count = sum(1 for q in query_texts if any(k in q for k in ["please", "kindly", "would you"]))
        if formal_count > len(queries) * 0.3:
            style["formality"] = "formal"

        return style

    def _calculate_peak_hours(self, queries: List[Dict]) -> List[Dict]:
        """Calculate peak usage hours from query timestamps"""
        hour_counts = Counter()
        for q in queries:
            hour = datetime.fromtimestamp(q["timestamp"]).hour
            hour_counts[hour] += 1

        if not hour_counts:
            return []

        # Get top 3 peak hours
        total = sum(hour_counts.values())
        peak_hours = []
        for hour, count in hour_counts.most_common(3):
            peak_hours.append({
                "hour": hour,
                "count": count,
                "percentage": round(count / total * 100, 1)
            })

        return peak_hours

    def save_preferences(self, preferences: Dict[str, Any]):
        """Save preferences to JSON file"""
        self.preferences_file.write_text(json.dumps(preferences, indent=2))
        logger.info(f"Preferences saved to {self.preferences_file}")

    def load_preferences(self) -> Optional[Dict[str, Any]]:
        """Load preferences from JSON file"""
        if not self.preferences_file.exists():
            return None
        try:
            return json.loads(self.preferences_file.read_text())
        except Exception as e:
            logger.warning(f"Could not load preferences: {e}")
            return None

    def update_preferences(self, min_queries: int = 10) -> Dict[str, Any]:
        """Analyze and save updated preferences"""
        preferences = self.analyze_preferences(min_queries=min_queries)
        if preferences["query_count"] >= min_queries:
            self.save_preferences(preferences)
        return preferences

    def get_system_prompt_additions(self) -> str:
        """
        Get additions to system prompt based on learned preferences.

        Returns string to append to system prompts for personalization.
        """
        preferences = self.load_preferences()
        if not preferences or preferences.get("confidence", 0) < 0.3:
            return ""

        additions = []

        # Response style
        style = preferences.get("response_style", {})
        if style.get("preferred_length") == "brief":
            additions.append("The user prefers brief, concise responses.")
        elif style.get("preferred_length") == "detailed":
            additions.append("The user appreciates detailed, thorough explanations.")

        if style.get("formality") == "formal":
            additions.append("Maintain a professional, formal tone.")

        # Topic interests
        topics = preferences.get("topic_preferences", [])
        if topics:
            top_topics = [t["topic"] for t in topics[:3] if t["confidence"] > 0.1]
            if top_topics:
                additions.append(f"The user is particularly interested in: {', '.join(top_topics)}.")

        return " ".join(additions)


def learn_preferences(senter_root: Path, min_queries: int = 10) -> Dict[str, Any]:
    """
    Main function to learn and save preferences.

    Called by scheduler job for preference updates.
    """
    learner = PreferenceLearner(senter_root)
    return learner.update_preferences(min_queries=min_queries)


# ========== LS-001: Temporal Pattern Analysis ==========

class TemporalPatternAnalyzer:
    """
    Temporal Pattern Analysis (LS-001)

    Detects time-based patterns in user behavior:
    - Peak usage hours
    - Weekday vs weekend patterns
    - Topic by time of day

    Requires 2+ weeks of data for meaningful analysis.
    """

    MIN_DAYS_FOR_ANALYSIS = 14  # 2 weeks

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.temporal_file = self.senter_root / "data" / "learning" / "temporal_patterns.json"
        self.temporal_file.parent.mkdir(parents=True, exist_ok=True)

    def analyze_temporal_patterns(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze temporal patterns from user behavior.

        Args:
            days: Number of days to analyze (default 30)

        Returns:
            Temporal patterns dict with peak_hours, weekday_patterns, topic_by_time
        """
        from learning.events_db import UserEventsDB

        db = UserEventsDB(senter_root=self.senter_root)
        events = db.get_events_by_time_range(hours=days * 24)

        if not events:
            return self._empty_temporal_patterns()

        # Check if we have enough data
        query_events = [e for e in events if e.event_type == "query"]
        if len(query_events) < 20:  # Need at least 20 queries
            return self._empty_temporal_patterns(insufficient_data=True)

        # Calculate the actual days span
        timestamps = [e.timestamp for e in query_events]
        if timestamps:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            actual_days = (max_ts - min_ts) / (24 * 3600)
        else:
            actual_days = 0

        patterns = {
            "analyzed_at": datetime.now().isoformat(),
            "days_analyzed": round(actual_days, 1),
            "query_count": len(query_events),
            "peak_hours": self._detect_peak_hours(query_events),
            "weekday_vs_weekend": self._weekday_weekend_patterns(query_events),
            "topic_by_time_of_day": self._topic_by_time_of_day(query_events),
            "hourly_heatmap": self._generate_hourly_heatmap(query_events),
            "activity_streaks": self._detect_activity_streaks(query_events),
            "confidence": min(1.0, actual_days / self.MIN_DAYS_FOR_ANALYSIS)
        }

        return patterns

    def _empty_temporal_patterns(self, insufficient_data: bool = False) -> Dict[str, Any]:
        """Return empty temporal patterns structure"""
        return {
            "analyzed_at": datetime.now().isoformat(),
            "days_analyzed": 0,
            "query_count": 0,
            "peak_hours": [],
            "weekday_vs_weekend": {},
            "topic_by_time_of_day": {},
            "hourly_heatmap": {},
            "activity_streaks": [],
            "confidence": 0.0,
            "insufficient_data": insufficient_data
        }

    def _detect_peak_hours(self, events: List) -> List[Dict]:
        """Detect peak activity hours with statistics"""
        hour_counts = Counter()
        hour_days = defaultdict(set)  # Track which days had activity at each hour

        for event in events:
            dt = datetime.fromtimestamp(event.timestamp)
            hour = dt.hour
            day = dt.date()
            hour_counts[hour] += 1
            hour_days[hour].add(day)

        if not hour_counts:
            return []

        total = sum(hour_counts.values())
        peak_hours = []

        for hour, count in hour_counts.most_common(5):
            peak_hours.append({
                "hour": hour,
                "count": count,
                "percentage": round(count / total * 100, 1),
                "days_active": len(hour_days[hour]),
                "time_label": self._hour_to_label(hour)
            })

        return peak_hours

    def _hour_to_label(self, hour: int) -> str:
        """Convert hour to human-readable label"""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _weekday_weekend_patterns(self, events: List) -> Dict[str, Any]:
        """Analyze weekday vs weekend patterns"""
        weekday_events = []
        weekend_events = []
        weekday_hours = Counter()
        weekend_hours = Counter()
        weekday_topics = Counter()
        weekend_topics = Counter()

        for event in events:
            dt = datetime.fromtimestamp(event.timestamp)
            topic = event.metadata.get("topic", "general")
            hour = dt.hour

            if dt.weekday() < 5:  # Monday-Friday
                weekday_events.append(event)
                weekday_hours[hour] += 1
                weekday_topics[topic] += 1
            else:  # Saturday-Sunday
                weekend_events.append(event)
                weekend_hours[hour] += 1
                weekend_topics[topic] += 1

        return {
            "weekday": {
                "count": len(weekday_events),
                "peak_hours": [h for h, _ in weekday_hours.most_common(3)],
                "top_topics": [t for t, _ in weekday_topics.most_common(3)],
                "avg_queries_per_day": round(len(weekday_events) / 5, 1) if weekday_events else 0
            },
            "weekend": {
                "count": len(weekend_events),
                "peak_hours": [h for h, _ in weekend_hours.most_common(3)],
                "top_topics": [t for t, _ in weekend_topics.most_common(3)],
                "avg_queries_per_day": round(len(weekend_events) / 2, 1) if weekend_events else 0
            },
            "weekday_preference": "weekday" if len(weekday_events) > len(weekend_events) * 2.5 else
                                   "weekend" if len(weekend_events) > len(weekday_events) else "balanced"
        }

    def _topic_by_time_of_day(self, events: List) -> Dict[str, List[Dict]]:
        """Analyze which topics are discussed at different times of day"""
        time_topics = {
            "morning": Counter(),      # 5-12
            "afternoon": Counter(),    # 12-17
            "evening": Counter(),      # 17-21
            "night": Counter()         # 21-5
        }

        for event in events:
            dt = datetime.fromtimestamp(event.timestamp)
            hour = dt.hour
            topic = event.metadata.get("topic", "general")

            time_label = self._hour_to_label(hour)
            time_topics[time_label][topic] += 1

        result = {}
        for time_period, topics in time_topics.items():
            if topics:
                total = sum(topics.values())
                result[time_period] = [
                    {"topic": t, "count": c, "percentage": round(c/total*100, 1)}
                    for t, c in topics.most_common(5)
                ]
            else:
                result[time_period] = []

        return result

    def _generate_hourly_heatmap(self, events: List) -> Dict[str, Dict[str, int]]:
        """Generate a heatmap of activity by day of week and hour"""
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap = {day: {str(h): 0 for h in range(24)} for day in day_names}

        for event in events:
            dt = datetime.fromtimestamp(event.timestamp)
            day = day_names[dt.weekday()]
            hour = str(dt.hour)
            heatmap[day][hour] += 1

        return heatmap

    def _detect_activity_streaks(self, events: List) -> List[Dict]:
        """Detect consecutive days of activity"""
        if not events:
            return []

        # Get unique active days
        active_days = set()
        for event in events:
            day = datetime.fromtimestamp(event.timestamp).date()
            active_days.add(day)

        if not active_days:
            return []

        # Sort days and find streaks
        sorted_days = sorted(active_days)
        streaks = []
        current_streak_start = sorted_days[0]
        current_streak_length = 1

        for i in range(1, len(sorted_days)):
            if (sorted_days[i] - sorted_days[i-1]).days == 1:
                current_streak_length += 1
            else:
                if current_streak_length >= 3:  # Only count streaks of 3+ days
                    streaks.append({
                        "start": current_streak_start.isoformat(),
                        "length": current_streak_length
                    })
                current_streak_start = sorted_days[i]
                current_streak_length = 1

        # Don't forget the last streak
        if current_streak_length >= 3:
            streaks.append({
                "start": current_streak_start.isoformat(),
                "length": current_streak_length
            })

        # Sort by length descending
        return sorted(streaks, key=lambda x: x["length"], reverse=True)[:5]

    def save_patterns(self, patterns: Dict[str, Any]):
        """Save temporal patterns to file"""
        self.temporal_file.write_text(json.dumps(patterns, indent=2))
        logger.info(f"Temporal patterns saved to {self.temporal_file}")

    def load_patterns(self) -> Optional[Dict[str, Any]]:
        """Load temporal patterns from file"""
        if not self.temporal_file.exists():
            return None
        try:
            return json.loads(self.temporal_file.read_text())
        except Exception as e:
            logger.warning(f"Could not load temporal patterns: {e}")
            return None

    def update_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze and save updated patterns"""
        patterns = self.analyze_temporal_patterns(days=days)
        if patterns.get("confidence", 0) > 0:
            self.save_patterns(patterns)
        return patterns


# ========== LS-002: Preference Prediction Model ==========

class PreferencePredictionModel:
    """
    Preference Prediction Model (LS-002)

    Predicts user preferences based on historical behavior:
    - Preferred response length
    - Formality level
    - Detail level

    Returns predictions with confidence scores.
    """

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.predictions_file = self.senter_root / "data" / "learning" / "preference_predictions.json"
        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)

    def predict_preferences(self) -> Dict[str, Any]:
        """
        Predict user preferences based on history.

        Returns predictions for response length, formality, detail level.
        """
        from learning.events_db import UserEventsDB

        db = UserEventsDB(senter_root=self.senter_root)
        events = db.get_events_by_time_range(hours=24 * 30)  # 30 days

        # Extract query-response pairs
        queries = []
        responses = []

        for event in events:
            if event.event_type == "query":
                queries.append({
                    "text": event.context.get("query", ""),
                    "timestamp": event.timestamp,
                    "topic": event.metadata.get("topic", "general")
                })
            elif event.event_type == "response":
                responses.append({
                    "text": event.context.get("response", ""),
                    "latency": event.metadata.get("latency_ms", 0)
                })

        if len(queries) < 10:
            return self._empty_predictions()

        # Analyze patterns
        predictions = {
            "predicted_at": datetime.now().isoformat(),
            "sample_size": len(queries),
            "response_length": self._predict_response_length(queries),
            "formality": self._predict_formality(queries),
            "detail_level": self._predict_detail_level(queries),
            "topic_preferences": self._predict_topic_preferences(queries),
            "overall_confidence": 0.0
        }

        # Calculate overall confidence
        confidences = [
            predictions["response_length"]["confidence"],
            predictions["formality"]["confidence"],
            predictions["detail_level"]["confidence"]
        ]
        predictions["overall_confidence"] = round(sum(confidences) / len(confidences), 3)

        return predictions

    def _empty_predictions(self) -> Dict[str, Any]:
        """Return empty predictions structure"""
        return {
            "predicted_at": datetime.now().isoformat(),
            "sample_size": 0,
            "response_length": {"prediction": "medium", "confidence": 0.0},
            "formality": {"prediction": "casual", "confidence": 0.0},
            "detail_level": {"prediction": "moderate", "confidence": 0.0},
            "topic_preferences": [],
            "overall_confidence": 0.0
        }

    def _predict_response_length(self, queries: List[Dict]) -> Dict[str, Any]:
        """Predict preferred response length"""
        texts = [q["text"].lower() for q in queries]

        # Keywords indicating preferences
        brief_keywords = ["quick", "brief", "short", "summary", "tldr", "just tell me", "simple"]
        detailed_keywords = ["explain", "detail", "elaborate", "comprehensive", "thorough", "full", "everything"]

        brief_count = sum(1 for t in texts if any(k in t for k in brief_keywords))
        detailed_count = sum(1 for t in texts if any(k in t for k in detailed_keywords))

        total = len(texts)
        brief_ratio = brief_count / total
        detailed_ratio = detailed_count / total

        if detailed_ratio > 0.2:
            prediction = "detailed"
            confidence = min(0.9, 0.5 + detailed_ratio)
        elif brief_ratio > 0.2:
            prediction = "brief"
            confidence = min(0.9, 0.5 + brief_ratio)
        else:
            prediction = "medium"
            confidence = 0.6

        return {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "brief_signal_count": brief_count,
            "detailed_signal_count": detailed_count
        }

    def _predict_formality(self, queries: List[Dict]) -> Dict[str, Any]:
        """Predict preferred formality level"""
        texts = [q["text"].lower() for q in queries]

        # Formal indicators
        formal_keywords = ["please", "kindly", "would you", "could you", "i would appreciate"]
        casual_keywords = ["hey", "yo", "gimme", "gonna", "wanna", "lol", "btw"]

        formal_count = sum(1 for t in texts if any(k in t for k in formal_keywords))
        casual_count = sum(1 for t in texts if any(k in t for k in casual_keywords))

        total = len(texts)
        formal_ratio = formal_count / total
        casual_ratio = casual_count / total

        if formal_ratio > 0.3:
            prediction = "formal"
            confidence = min(0.9, 0.5 + formal_ratio)
        elif casual_ratio > 0.3:
            prediction = "casual"
            confidence = min(0.9, 0.5 + casual_ratio)
        else:
            prediction = "neutral"
            confidence = 0.6

        return {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "formal_signal_count": formal_count,
            "casual_signal_count": casual_count
        }

    def _predict_detail_level(self, queries: List[Dict]) -> Dict[str, Any]:
        """Predict preferred detail level"""
        texts = [q["text"].lower() for q in queries]

        # Detail level indicators
        high_detail = ["why", "how does", "explain", "what causes", "deep dive", "underlying"]
        low_detail = ["just", "only", "simple", "basic", "quick answer", "yes or no"]

        high_count = sum(1 for t in texts if any(k in t for k in high_detail))
        low_count = sum(1 for t in texts if any(k in t for k in low_detail))

        total = len(texts)
        high_ratio = high_count / total
        low_ratio = low_count / total

        if high_ratio > 0.25:
            prediction = "high"
            confidence = min(0.9, 0.5 + high_ratio)
        elif low_ratio > 0.25:
            prediction = "low"
            confidence = min(0.9, 0.5 + low_ratio)
        else:
            prediction = "moderate"
            confidence = 0.6

        return {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "high_detail_count": high_count,
            "low_detail_count": low_count
        }

    def _predict_topic_preferences(self, queries: List[Dict]) -> List[Dict]:
        """Predict topic preferences with confidence"""
        topic_counts = Counter()
        for q in queries:
            topic = q.get("topic", "general")
            topic_counts[topic] += 1

        total = sum(topic_counts.values())
        predictions = []

        for topic, count in topic_counts.most_common(5):
            predictions.append({
                "topic": topic,
                "count": count,
                "confidence": round(count / total, 3)
            })

        return predictions

    def save_predictions(self, predictions: Dict[str, Any]):
        """Save predictions to file"""
        self.predictions_file.write_text(json.dumps(predictions, indent=2))
        logger.info(f"Predictions saved to {self.predictions_file}")

    def load_predictions(self) -> Optional[Dict[str, Any]]:
        """Load predictions from file"""
        if not self.predictions_file.exists():
            return None
        try:
            return json.loads(self.predictions_file.read_text())
        except Exception as e:
            logger.warning(f"Could not load predictions: {e}")
            return None

    def update_predictions(self) -> Dict[str, Any]:
        """Predict and save updated predictions"""
        predictions = self.predict_preferences()
        if predictions["sample_size"] >= 10:
            self.save_predictions(predictions)
        return predictions


# ========== LS-003: Feedback Collection and Learning ==========

@dataclass
class FeedbackEntry:
    """A single feedback entry"""
    feedback_type: str  # positive, negative, correction
    message_id: str
    timestamp: float
    original_response: str
    user_comment: str = ""
    correction: str = ""

    def to_dict(self) -> Dict:
        return {
            "feedback_type": self.feedback_type,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "original_response": self.original_response[:200],  # Truncate for storage
            "user_comment": self.user_comment,
            "correction": self.correction
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FeedbackEntry":
        return cls(**data)


class FeedbackCollector:
    """
    Feedback Collection and Learning (LS-003)

    Collects and processes user feedback to improve responses:
    - Positive feedback (helpful, great, thanks)
    - Negative feedback (wrong, incorrect, no)
    - Corrections (actually, the correct answer is...)
    """

    # Keywords for detecting feedback
    POSITIVE_KEYWORDS = ["thanks", "helpful", "great", "perfect", "exactly", "good", "awesome", "nice"]
    NEGATIVE_KEYWORDS = ["wrong", "incorrect", "no", "not right", "mistake", "error", "bad"]
    CORRECTION_KEYWORDS = ["actually", "correction", "the correct", "should be", "meant to say"]

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.feedback_file = self.senter_root / "data" / "learning" / "feedback.json"
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        self._feedback_log: List[FeedbackEntry] = []
        self._load_feedback()

    def _load_feedback(self):
        """Load existing feedback from file"""
        if self.feedback_file.exists():
            try:
                data = json.loads(self.feedback_file.read_text())
                self._feedback_log = [FeedbackEntry.from_dict(f) for f in data.get("entries", [])]
            except Exception as e:
                logger.warning(f"Could not load feedback: {e}")
                self._feedback_log = []

    def _save_feedback(self):
        """Save feedback to file"""
        data = {
            "updated_at": datetime.now().isoformat(),
            "count": len(self._feedback_log),
            "entries": [f.to_dict() for f in self._feedback_log[-500:]]  # Keep last 500
        }
        self.feedback_file.write_text(json.dumps(data, indent=2))

    def detect_feedback(self, user_message: str, previous_response: str = "", message_id: str = "") -> Optional[FeedbackEntry]:
        """
        Detect feedback in user message.

        Args:
            user_message: The user's message
            previous_response: The previous AI response (if available)
            message_id: ID of the message being responded to

        Returns:
            FeedbackEntry if feedback detected, None otherwise
        """
        text = user_message.lower()

        # Check for corrections first (most specific)
        if any(k in text for k in self.CORRECTION_KEYWORDS):
            return FeedbackEntry(
                feedback_type="correction",
                message_id=message_id,
                timestamp=time.time(),
                original_response=previous_response,
                user_comment=user_message,
                correction=user_message
            )

        # Check for negative feedback
        if any(k in text for k in self.NEGATIVE_KEYWORDS):
            return FeedbackEntry(
                feedback_type="negative",
                message_id=message_id,
                timestamp=time.time(),
                original_response=previous_response,
                user_comment=user_message
            )

        # Check for positive feedback
        if any(k in text for k in self.POSITIVE_KEYWORDS):
            return FeedbackEntry(
                feedback_type="positive",
                message_id=message_id,
                timestamp=time.time(),
                original_response=previous_response,
                user_comment=user_message
            )

        return None

    def record_feedback(self, feedback: FeedbackEntry):
        """Record a feedback entry"""
        self._feedback_log.append(feedback)
        self._save_feedback()
        logger.info(f"Recorded {feedback.feedback_type} feedback")

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback"""
        if not self._feedback_log:
            return {
                "total_feedback": 0,
                "positive": 0,
                "negative": 0,
                "corrections": 0,
                "sentiment_ratio": 0.0
            }

        positive = sum(1 for f in self._feedback_log if f.feedback_type == "positive")
        negative = sum(1 for f in self._feedback_log if f.feedback_type == "negative")
        corrections = sum(1 for f in self._feedback_log if f.feedback_type == "correction")

        total_sentiment = positive + negative
        sentiment_ratio = positive / total_sentiment if total_sentiment > 0 else 0.5

        return {
            "total_feedback": len(self._feedback_log),
            "positive": positive,
            "negative": negative,
            "corrections": corrections,
            "sentiment_ratio": round(sentiment_ratio, 3)
        }

    def get_recent_feedback(self, count: int = 10) -> List[Dict]:
        """Get recent feedback entries"""
        return [f.to_dict() for f in self._feedback_log[-count:]]

    def should_clarify(self, feedback: FeedbackEntry) -> bool:
        """Determine if clarification should be requested"""
        return feedback.feedback_type == "negative"

    def get_clarification_prompt(self, feedback: FeedbackEntry) -> str:
        """Get a clarification prompt for negative feedback"""
        return "I apologize if my response wasn't helpful. Could you tell me what I got wrong or what you were looking for?"


# ========== LS-004: Topic Expertise Modeling ==========

@dataclass
class ExpertiseLevel:
    """Expertise level for a topic"""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class TopicExpertiseTracker:
    """
    Topic Expertise Modeling (LS-004)

    Tracks user expertise level per topic:
    - Novice: Basic questions, needs explanations
    - Intermediate: Some knowledge, contextual questions
    - Expert: Advanced questions, technical terminology

    Adjusts response complexity based on expertise.
    """

    # Keywords indicating expertise level
    NOVICE_INDICATORS = ["what is", "explain", "how do i", "beginner", "basic", "simple", "new to"]
    INTERMEDIATE_INDICATORS = ["how to", "best practice", "optimize", "improve", "configure"]
    EXPERT_INDICATORS = ["implementation", "architecture", "performance", "edge case", "under the hood", "internals"]

    def __init__(self, senter_root: Path):
        self.senter_root = Path(senter_root)
        self.expertise_file = self.senter_root / "data" / "learning" / "topic_expertise.json"
        self.expertise_file.parent.mkdir(parents=True, exist_ok=True)
        self._expertise: Dict[str, Dict] = {}
        self._load_expertise()

    def _load_expertise(self):
        """Load expertise data from file"""
        if self.expertise_file.exists():
            try:
                self._expertise = json.loads(self.expertise_file.read_text())
            except Exception as e:
                logger.warning(f"Could not load expertise: {e}")
                self._expertise = {}

    def _save_expertise(self):
        """Save expertise data to file"""
        self.expertise_file.write_text(json.dumps(self._expertise, indent=2))

    def analyze_query_expertise(self, query: str, topic: str) -> str:
        """
        Analyze the expertise level indicated by a query.

        Returns: novice, intermediate, or expert
        """
        text = query.lower()

        # Score each level
        scores = {
            "novice": sum(1 for k in self.NOVICE_INDICATORS if k in text),
            "intermediate": sum(1 for k in self.INTERMEDIATE_INDICATORS if k in text),
            "expert": sum(1 for k in self.EXPERT_INDICATORS if k in text)
        }

        # Return highest score, default to intermediate
        max_score = max(scores.values())
        if max_score == 0:
            return "intermediate"

        for level, score in scores.items():
            if score == max_score:
                return level

        return "intermediate"

    def record_query(self, query: str, topic: str):
        """
        Record a query and update expertise for the topic.
        """
        level = self.analyze_query_expertise(query, topic)

        if topic not in self._expertise:
            self._expertise[topic] = {
                "level": level,
                "novice_count": 0,
                "intermediate_count": 0,
                "expert_count": 0,
                "total_queries": 0,
                "last_updated": datetime.now().isoformat()
            }

        # Update counts
        self._expertise[topic][f"{level}_count"] += 1
        self._expertise[topic]["total_queries"] += 1
        self._expertise[topic]["last_updated"] = datetime.now().isoformat()

        # Recalculate level based on weighted history
        self._expertise[topic]["level"] = self._calculate_expertise_level(topic)

        self._save_expertise()

    def _calculate_expertise_level(self, topic: str) -> str:
        """Calculate expertise level based on query history"""
        data = self._expertise.get(topic, {})

        novice = data.get("novice_count", 0)
        intermediate = data.get("intermediate_count", 0)
        expert = data.get("expert_count", 0)

        total = novice + intermediate + expert
        if total == 0:
            return "intermediate"

        # Weight recent queries more heavily (approximated by using raw counts)
        # If expert queries > 40%, user is expert
        # If novice queries > 40%, user is novice
        # Otherwise intermediate

        if expert / total > 0.4:
            return "expert"
        elif novice / total > 0.4:
            return "novice"
        else:
            return "intermediate"

    def get_expertise_level(self, topic: str) -> str:
        """Get the current expertise level for a topic"""
        if topic not in self._expertise:
            return "intermediate"  # Default
        return self._expertise[topic].get("level", "intermediate")

    def get_all_expertise(self) -> Dict[str, Dict]:
        """Get expertise data for all topics"""
        return self._expertise.copy()

    def get_complexity_adjustment(self, topic: str) -> Dict[str, Any]:
        """
        Get response complexity adjustments based on expertise.

        Returns dict with recommendations for response generation.
        """
        level = self.get_expertise_level(topic)

        adjustments = {
            "novice": {
                "use_technical_terms": False,
                "include_examples": True,
                "explain_concepts": True,
                "response_detail": "high",
                "assume_knowledge": False
            },
            "intermediate": {
                "use_technical_terms": True,
                "include_examples": True,
                "explain_concepts": False,
                "response_detail": "moderate",
                "assume_knowledge": True
            },
            "expert": {
                "use_technical_terms": True,
                "include_examples": False,
                "explain_concepts": False,
                "response_detail": "concise",
                "assume_knowledge": True
            }
        }

        return {
            "expertise_level": level,
            "adjustments": adjustments.get(level, adjustments["intermediate"])
        }

    def get_system_prompt_for_topic(self, topic: str) -> str:
        """Get system prompt additions based on topic expertise"""
        level = self.get_expertise_level(topic)

        if level == "novice":
            return f"The user is a beginner with {topic}. Explain concepts clearly, avoid jargon, and provide examples."
        elif level == "expert":
            return f"The user is experienced with {topic}. Be concise, use technical terms, and skip basic explanations."
        else:
            return f"The user has intermediate knowledge of {topic}. Balance detail with conciseness."


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
