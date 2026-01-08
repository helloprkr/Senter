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
