"""
Affective Memory - Emotional context storage.

Tracks the emotional context of interactions - how things felt.
"""

from __future__ import annotations
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.base import ModelInterface


@dataclass
class SentimentAnalysis:
    """Result of LLM-based sentiment analysis."""
    sentiment: float  # -1 to 1 scale
    confidence: float  # 0 to 1
    emotions: List[str]  # Detected emotions
    explanation: str  # Brief explanation


class AffectiveMemory:
    """
    Affective memory for emotional context.

    Tracks:
    - User sentiment over time
    - Frustration events
    - Satisfaction levels
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        config: Dict,
        model: Optional["ModelInterface"] = None,
    ):
        self.conn = conn
        self.config = config
        self.model = model
        self.track_items = config.get("track", ["user_sentiment", "interaction_satisfaction"])

    def store(
        self,
        episode_id: str,
        sentiment: float = 0.5,
        frustration: float = 0.0,
        satisfaction: float = 0.5,
    ) -> str:
        """
        Store affective state for an episode.

        Args:
            episode_id: Associated episode ID
            sentiment: Overall sentiment (-1 to 1, normalized to 0-1)
            frustration: Frustration level (0-1)
            satisfaction: Satisfaction level (0-1)

        Returns:
            Affective record ID
        """
        record_id = str(uuid.uuid4())[:8]

        self.conn.execute(
            """
            INSERT INTO affective (id, episode_id, sentiment, frustration, satisfaction)
            VALUES (?, ?, ?, ?, ?)
        """,
            (record_id, episode_id, sentiment, frustration, satisfaction),
        )
        self.conn.commit()

        return record_id

    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze sentiment of text using LLM.

        Args:
            text: Text to analyze

        Returns:
            SentimentAnalysis with sentiment score and details
        """
        if not self.model:
            return self._analyze_sentiment_heuristic(text)

        prompt = f"""Analyze the sentiment of the following text. Return a JSON object with:
- sentiment: float from -1 (very negative) to 1 (very positive)
- confidence: float from 0 to 1 indicating how confident you are
- emotions: list of detected emotions (e.g., ["happy", "excited"])
- explanation: brief explanation of the sentiment

Text to analyze:
\"\"\"{text}\"\"\"

Return ONLY valid JSON, no other text."""

        try:
            response = await self.model.generate(prompt)
            return self._parse_sentiment_response(response)
        except Exception:
            return self._analyze_sentiment_heuristic(text)

    def _parse_sentiment_response(self, response: str) -> SentimentAnalysis:
        """Parse LLM response into SentimentAnalysis."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._default_sentiment()
            else:
                return self._default_sentiment()

        sentiment = float(data.get("sentiment", 0))
        sentiment = max(-1, min(1, sentiment))  # Clamp to -1 to 1

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0, min(1, confidence))  # Clamp to 0 to 1

        emotions = data.get("emotions", [])
        if not isinstance(emotions, list):
            emotions = []

        explanation = str(data.get("explanation", ""))

        return SentimentAnalysis(
            sentiment=sentiment,
            confidence=confidence,
            emotions=emotions,
            explanation=explanation
        )

    def _analyze_sentiment_heuristic(self, text: str) -> SentimentAnalysis:
        """Fallback heuristic-based sentiment analysis."""
        text_lower = text.lower()

        # Positive indicators
        positive_words = ["thanks", "great", "awesome", "love", "excellent", "perfect",
                         "wonderful", "amazing", "helpful", "good", "happy", "pleased"]
        # Negative indicators
        negative_words = ["frustrated", "annoyed", "angry", "hate", "terrible", "awful",
                         "bad", "wrong", "broken", "useless", "confused", "stuck"]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Calculate sentiment
        if positive_count + negative_count == 0:
            sentiment = 0.0  # Neutral
            emotions = ["neutral"]
        elif positive_count > negative_count:
            sentiment = min(0.5 + (positive_count * 0.1), 1.0)
            emotions = ["positive"]
        else:
            sentiment = max(-0.5 - (negative_count * 0.1), -1.0)
            emotions = ["negative"]

        return SentimentAnalysis(
            sentiment=sentiment,
            confidence=0.4,  # Lower confidence for heuristic
            emotions=emotions,
            explanation="Heuristic-based analysis"
        )

    def _default_sentiment(self) -> SentimentAnalysis:
        """Return default neutral sentiment."""
        return SentimentAnalysis(
            sentiment=0.0,
            confidence=0.5,
            emotions=["neutral"],
            explanation="Default neutral sentiment"
        )

    def get_recent_context(self, days: int = 7) -> Dict[str, float]:
        """
        Get recent emotional context.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with average emotional metrics
        """
        cursor = self.conn.execute(
            """
            SELECT
                AVG(sentiment) as avg_sentiment,
                AVG(frustration) as avg_frustration,
                AVG(satisfaction) as avg_satisfaction,
                COUNT(*) as interaction_count
            FROM affective
            WHERE timestamp > datetime('now', ?)
        """,
            (f"-{days} days",),
        )

        row = cursor.fetchone()
        return {
            "avg_sentiment": row["avg_sentiment"] or 0.5,
            "avg_frustration": row["avg_frustration"] or 0.0,
            "avg_satisfaction": row["avg_satisfaction"] or 0.5,
            "interaction_count": row["interaction_count"],
        }

    def get_frustration_events(self, threshold: float = 0.5, limit: int = 10) -> List[Dict]:
        """
        Get recent high-frustration events.

        Args:
            threshold: Minimum frustration level
            limit: Maximum results

        Returns:
            List of frustration events with episode info
        """
        cursor = self.conn.execute(
            """
            SELECT a.*, e.input, e.response, e.mode
            FROM affective a
            JOIN episodic e ON a.episode_id = e.id
            WHERE a.frustration >= ?
            ORDER BY a.timestamp DESC
            LIMIT ?
        """,
            (threshold, limit),
        )

        events = []
        for row in cursor.fetchall():
            events.append(
                {
                    "timestamp": row["timestamp"],
                    "frustration": row["frustration"],
                    "episode_id": row["episode_id"],
                    "input": row["input"],
                    "response": row["response"],
                    "mode": row["mode"],
                }
            )

        return events

    def get_satisfaction_trend(self, days: int = 30) -> List[Dict]:
        """
        Get satisfaction trend over time.

        Args:
            days: Number of days to analyze

        Returns:
            List of daily satisfaction averages
        """
        cursor = self.conn.execute(
            """
            SELECT
                DATE(timestamp) as date,
                AVG(satisfaction) as avg_satisfaction,
                AVG(frustration) as avg_frustration,
                COUNT(*) as interactions
            FROM affective
            WHERE timestamp > datetime('now', ?)
            GROUP BY DATE(timestamp)
            ORDER BY date
        """,
            (f"-{days} days",),
        )

        return [
            {
                "date": row["date"],
                "satisfaction": row["avg_satisfaction"],
                "frustration": row["avg_frustration"],
                "interactions": row["interactions"],
            }
            for row in cursor.fetchall()
        ]

    def get_episode_affect(self, episode_id: str) -> Dict[str, float]:
        """Get affective data for a specific episode."""
        cursor = self.conn.execute(
            """
            SELECT sentiment, frustration, satisfaction
            FROM affective
            WHERE episode_id = ?
        """,
            (episode_id,),
        )

        row = cursor.fetchone()
        if row:
            return {
                "sentiment": row["sentiment"],
                "frustration": row["frustration"],
                "satisfaction": row["satisfaction"],
            }

        return {"sentiment": 0.5, "frustration": 0.0, "satisfaction": 0.5}

    def compute_overall_health(self) -> Dict[str, Any]:
        """
        Compute overall emotional health metrics.

        Returns:
            Health metrics dictionary
        """
        recent = self.get_recent_context(days=7)
        trend = self.get_satisfaction_trend(days=30)

        # Calculate trend direction
        if len(trend) >= 2:
            first_week = sum(d["satisfaction"] for d in trend[:7]) / min(7, len(trend))
            last_week = sum(d["satisfaction"] for d in trend[-7:]) / min(7, len(trend))
            trend_direction = "improving" if last_week > first_week else "declining"
            if abs(last_week - first_week) < 0.05:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"

        # Determine health status
        health_score = (
            recent["avg_satisfaction"] * 0.5
            + (1 - recent["avg_frustration"]) * 0.3
            + recent["avg_sentiment"] * 0.2
        )

        if health_score > 0.7:
            status = "healthy"
        elif health_score > 0.4:
            status = "moderate"
        else:
            status = "needs_attention"

        return {
            "health_score": health_score,
            "status": status,
            "trend": trend_direction,
            "recent_satisfaction": recent["avg_satisfaction"],
            "recent_frustration": recent["avg_frustration"],
            "interaction_count": recent["interaction_count"],
        }

    def count(self) -> int:
        """Get total affective record count."""
        return self.conn.execute("SELECT COUNT(*) FROM affective").fetchone()[0]
