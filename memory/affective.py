"""
Affective Memory - Emotional context storage.

Tracks the emotional context of interactions - how things felt.
"""

from __future__ import annotations
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List


class AffectiveMemory:
    """
    Affective memory for emotional context.

    Tracks:
    - User sentiment over time
    - Frustration events
    - Satisfaction levels
    """

    def __init__(self, conn: sqlite3.Connection, config: Dict):
        self.conn = conn
        self.config = config
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
