"""
Episodic Memory - Interaction event storage.

Stores what happened - specific interactions and events.
"""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .living_memory import Episode


class EpisodicMemory:
    """
    Episodic memory for interaction events.

    Stores:
    - Complete interaction history
    - Context at time of interaction
    - Fitness scores for learning
    """

    def __init__(self, conn: sqlite3.Connection, config: Dict):
        self.conn = conn
        self.config = config
        self.max_episodes = config.get("max_episodes", 10000)

    def store(self, episode: "Episode") -> None:
        """
        Store an episode.

        Args:
            episode: The episode to store
        """
        self.conn.execute(
            """
            INSERT INTO episodic (id, timestamp, input, response, mode,
                                  cognitive_state, joint_state, fitness, sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                episode.id,
                episode.timestamp.isoformat(),
                episode.input,
                episode.response,
                episode.mode,
                json.dumps(episode.cognitive_state),
                json.dumps(episode.joint_state),
                episode.fitness,
                episode.sentiment,
            ),
        )

        # Cleanup old episodes if over limit
        self._cleanup_if_needed()

    def get_recent(self, limit: int = 100) -> List["Episode"]:
        """
        Get recent episodes.

        Args:
            limit: Maximum episodes to return

        Returns:
            List of recent episodes
        """
        from .living_memory import Episode

        cursor = self.conn.execute(
            """
            SELECT * FROM episodic
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        return [self._row_to_episode(row) for row in cursor.fetchall()]

    def search(self, query: str, limit: int = 5) -> List["Episode"]:
        """
        Search episodes by input text.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching episodes
        """
        from .living_memory import Episode

        cursor = self.conn.execute(
            """
            SELECT * FROM episodic
            WHERE input LIKE ? OR response LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (f"%{query}%", f"%{query}%", limit),
        )

        return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_by_mode(self, mode: str, limit: int = 20) -> List["Episode"]:
        """Get episodes by coupling mode."""
        from .living_memory import Episode

        cursor = self.conn.execute(
            """
            SELECT * FROM episodic
            WHERE mode = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (mode, limit),
        )

        return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_high_fitness(self, threshold: float = 0.7, limit: int = 20) -> List["Episode"]:
        """Get episodes with high fitness scores."""
        from .living_memory import Episode

        cursor = self.conn.execute(
            """
            SELECT * FROM episodic
            WHERE fitness IS NOT NULL AND fitness >= ?
            ORDER BY fitness DESC
            LIMIT ?
        """,
            (threshold, limit),
        )

        return [self._row_to_episode(row) for row in cursor.fetchall()]

    def update_fitness(self, episode_id: str, fitness: float) -> bool:
        """
        Update fitness score for an episode.

        Args:
            episode_id: Episode ID
            fitness: New fitness score

        Returns:
            True if updated
        """
        cursor = self.conn.execute(
            """
            UPDATE episodic
            SET fitness = ?
            WHERE id = ?
        """,
            (fitness, episode_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_episode(self, episode_id: str) -> Optional["Episode"]:
        """Get a specific episode by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM episodic WHERE id = ?", (episode_id,)
        )
        row = cursor.fetchone()
        return self._row_to_episode(row) if row else None

    def _row_to_episode(self, row: sqlite3.Row) -> "Episode":
        """Convert database row to Episode."""
        from .living_memory import Episode

        timestamp = row["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return Episode(
            id=row["id"],
            timestamp=timestamp,
            input=row["input"],
            response=row["response"],
            mode=row["mode"],
            cognitive_state=json.loads(row["cognitive_state"] or "{}"),
            joint_state=json.loads(row["joint_state"] or "{}"),
            fitness=row["fitness"],
            sentiment=row["sentiment"],
        )

    def _cleanup_if_needed(self) -> None:
        """Remove old episodes if over limit."""
        count = self.conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]

        if count > self.max_episodes:
            # Keep most recent episodes
            to_delete = count - self.max_episodes
            self.conn.execute(
                """
                DELETE FROM episodic
                WHERE id IN (
                    SELECT id FROM episodic
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
            """,
                (to_delete,),
            )

    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary statistics for recent period."""
        cursor = self.conn.execute(
            """
            SELECT
                COUNT(*) as total_episodes,
                AVG(fitness) as avg_fitness,
                COUNT(DISTINCT mode) as modes_used
            FROM episodic
            WHERE timestamp > datetime('now', ?)
        """,
            (f"-{days} days",),
        )

        row = cursor.fetchone()
        return {
            "total_episodes": row["total_episodes"],
            "avg_fitness": row["avg_fitness"] or 0,
            "modes_used": row["modes_used"],
        }

    def count(self) -> int:
        """Get total episode count."""
        return self.conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
