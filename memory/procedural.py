"""
Procedural Memory - Pattern and skill storage.

Stores how to help this specific human - what works and what doesn't.
"""

from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List


class ProceduralMemory:
    """
    Procedural memory for patterns and skills.

    Stores:
    - Communication patterns that work
    - Mode-specific success patterns
    - User preference patterns
    """

    def __init__(self, conn: sqlite3.Connection, config: Dict):
        self.conn = conn
        self.config = config

    def store_pattern(
        self,
        pattern_type: str,
        pattern_data: str,
    ) -> str:
        """
        Store a new pattern.

        Args:
            pattern_type: Type of pattern (mode_*, preference_*, etc.)
            pattern_data: JSON-encoded pattern data

        Returns:
            Pattern ID
        """
        pattern_id = str(uuid.uuid4())[:8]

        self.conn.execute(
            """
            INSERT INTO procedural (id, pattern_type, pattern_data, last_used)
            VALUES (?, ?, ?, ?)
        """,
            (pattern_id, pattern_type, pattern_data, datetime.now().isoformat()),
        )
        self.conn.commit()

        return pattern_id

    def update_pattern(
        self,
        pattern_type: str,
        pattern_data: str,
        success: bool = True,
    ) -> None:
        """
        Update or create a pattern based on usage.

        Args:
            pattern_type: Type of pattern
            pattern_data: Pattern data
            success: Whether this usage was successful
        """
        # Check if pattern exists
        cursor = self.conn.execute(
            "SELECT id, success_count, failure_count FROM procedural WHERE pattern_type = ?",
            (pattern_type,),
        )
        row = cursor.fetchone()

        if row:
            # Update existing pattern
            if success:
                self.conn.execute(
                    """
                    UPDATE procedural
                    SET success_count = success_count + 1,
                        last_used = ?,
                        pattern_data = ?
                    WHERE id = ?
                """,
                    (datetime.now().isoformat(), pattern_data, row["id"]),
                )
            else:
                self.conn.execute(
                    """
                    UPDATE procedural
                    SET failure_count = failure_count + 1,
                        last_used = ?
                    WHERE id = ?
                """,
                    (datetime.now().isoformat(), row["id"]),
                )
        else:
            # Create new pattern with initial count
            pattern_id = str(uuid.uuid4())[:8]
            success_count = 1 if success else 0
            failure_count = 0 if success else 1

            self.conn.execute(
                """
                INSERT INTO procedural (id, pattern_type, pattern_data, success_count, failure_count, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (pattern_id, pattern_type, pattern_data, success_count, failure_count, datetime.now().isoformat()),
            )

        self.conn.commit()

    def get_patterns(self, pattern_type: str = None) -> Dict[str, Any]:
        """
        Get stored patterns.

        Args:
            pattern_type: Optional filter by type

        Returns:
            Dictionary of patterns
        """
        if pattern_type:
            cursor = self.conn.execute(
                """
                SELECT pattern_type, pattern_data, success_count, failure_count
                FROM procedural
                WHERE pattern_type = ?
            """,
                (pattern_type,),
            )
        else:
            cursor = self.conn.execute(
                """
                SELECT pattern_type, pattern_data, success_count, failure_count
                FROM procedural
                ORDER BY success_count DESC
            """
            )

        patterns = {}
        for row in cursor.fetchall():
            success_rate = 0.5
            total = row["success_count"] + row["failure_count"]
            if total > 0:
                success_rate = row["success_count"] / total

            patterns[row["pattern_type"]] = {
                "data": json.loads(row["pattern_data"]),
                "success_rate": success_rate,
                "usage_count": total,
            }

        return patterns

    def get_best_pattern(self, pattern_prefix: str) -> Dict[str, Any]:
        """
        Get the best pattern matching a prefix.

        Args:
            pattern_prefix: Pattern type prefix (e.g., "mode_")

        Returns:
            Best pattern data or empty dict
        """
        cursor = self.conn.execute(
            """
            SELECT pattern_type, pattern_data, success_count, failure_count
            FROM procedural
            WHERE pattern_type LIKE ?
            ORDER BY (success_count * 1.0 / NULLIF(success_count + failure_count, 0)) DESC
            LIMIT 1
        """,
            (f"{pattern_prefix}%",),
        )

        row = cursor.fetchone()
        if row:
            total = row["success_count"] + row["failure_count"]
            return {
                "type": row["pattern_type"],
                "data": json.loads(row["pattern_data"]),
                "success_rate": row["success_count"] / total if total > 0 else 0.5,
            }

        return {}

    def get_mode_preferences(self) -> Dict[str, float]:
        """Get success rates for each mode."""
        cursor = self.conn.execute(
            """
            SELECT pattern_type, success_count, failure_count
            FROM procedural
            WHERE pattern_type LIKE 'mode_%'
        """
        )

        preferences = {}
        for row in cursor.fetchall():
            mode = row["pattern_type"].replace("mode_", "")
            total = row["success_count"] + row["failure_count"]
            if total > 0:
                preferences[mode] = row["success_count"] / total

        return preferences

    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern by ID."""
        cursor = self.conn.execute(
            "DELETE FROM procedural WHERE id = ?", (pattern_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def cleanup_low_performers(self, min_success_rate: float = 0.2) -> int:
        """
        Remove patterns with low success rates.

        Args:
            min_success_rate: Minimum success rate to keep

        Returns:
            Number of patterns removed
        """
        cursor = self.conn.execute(
            """
            DELETE FROM procedural
            WHERE (success_count + failure_count) > 10
              AND (success_count * 1.0 / (success_count + failure_count)) < ?
        """,
            (min_success_rate,),
        )
        self.conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        """Get total pattern count."""
        return self.conn.execute("SELECT COUNT(*) FROM procedural").fetchone()[0]
