"""
Procedural Memory - Pattern and skill storage.

Stores how to help this specific human - what works and what doesn't.
"""

from __future__ import annotations
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class UserPreference:
    """A tracked user preference."""
    name: str  # e.g., "length", "formality", "detail_level"
    value: str  # e.g., "concise", "detailed", "formal", "casual"
    confidence: float  # 0-1, increases with consistent signals
    signal_count: int  # Number of signals that contributed to this preference
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "signal_count": self.signal_count,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            confidence=data.get("confidence", 0.5),
            signal_count=data.get("signal_count", 1),
            last_updated=datetime.fromisoformat(data["last_updated"])
            if isinstance(data.get("last_updated"), str)
            else datetime.now(),
        )


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

    # =========================================================================
    # Preference Learning (US-012)
    # =========================================================================

    def track_response_fitness(
        self,
        response_length: int,
        formality: str,
        detail_level: str,
        fitness_score: float,
        min_fitness_threshold: float = 0.6
    ) -> Optional[Dict[str, str]]:
        """
        Track response patterns that get high fitness scores.

        Args:
            response_length: Length of response in characters
            formality: "formal" or "casual"
            detail_level: "concise", "moderate", or "detailed"
            fitness_score: Fitness score for this response (0-1)
            min_fitness_threshold: Minimum score to consider "high fitness"

        Returns:
            Dict of inferred preferences if high fitness, None otherwise
        """
        if fitness_score < min_fitness_threshold:
            return None

        inferred = {}

        # Infer length preference
        if response_length < 200:
            inferred["length"] = "concise"
        elif response_length < 500:
            inferred["length"] = "moderate"
        else:
            inferred["length"] = "detailed"

        inferred["formality"] = formality
        inferred["detail_level"] = detail_level

        # Update preferences based on high-fitness response
        for pref_name, pref_value in inferred.items():
            self.update_preference(pref_name, pref_value, fitness_score)

        return inferred

    def update_preference(
        self,
        preference_name: str,
        preference_value: str,
        signal_strength: float = 1.0
    ) -> UserPreference:
        """
        Update or create a user preference with confidence adjustment.

        Confidence increases with consistent signals, decreases with conflicting ones.

        Args:
            preference_name: Name of preference (e.g., "length", "formality")
            preference_value: Value of preference (e.g., "concise", "formal")
            signal_strength: How strong this signal is (0-1, typically fitness score)

        Returns:
            Updated UserPreference
        """
        pattern_type = f"preference_{preference_name}"

        # Get existing preference
        cursor = self.conn.execute(
            """
            SELECT pattern_data, success_count
            FROM procedural
            WHERE pattern_type = ?
        """,
            (pattern_type,),
        )
        row = cursor.fetchone()

        if row:
            existing_data = json.loads(row["pattern_data"])
            existing_pref = UserPreference.from_dict(existing_data)

            # Check if same value (consistent signal)
            if existing_pref.value == preference_value:
                # Increase confidence (diminishing returns)
                confidence_boost = signal_strength * 0.1 * (1 - existing_pref.confidence)
                new_confidence = min(existing_pref.confidence + confidence_boost, 0.95)
                new_signal_count = existing_pref.signal_count + 1
            else:
                # Conflicting signal - decrease confidence and potentially flip
                confidence_decrease = signal_strength * 0.15
                new_confidence = existing_pref.confidence - confidence_decrease

                if new_confidence < 0.3:
                    # Flip to new value
                    preference_value = preference_value
                    new_confidence = 0.4  # Start with moderate confidence
                    new_signal_count = 1
                else:
                    # Keep old value but with reduced confidence
                    preference_value = existing_pref.value
                    new_signal_count = existing_pref.signal_count

            updated_pref = UserPreference(
                name=preference_name,
                value=preference_value,
                confidence=max(0.1, new_confidence),
                signal_count=new_signal_count,
            )

            self.conn.execute(
                """
                UPDATE procedural
                SET pattern_data = ?, success_count = success_count + 1, last_used = ?
                WHERE pattern_type = ?
            """,
                (json.dumps(updated_pref.to_dict()), datetime.now().isoformat(), pattern_type),
            )
        else:
            # Create new preference
            updated_pref = UserPreference(
                name=preference_name,
                value=preference_value,
                confidence=0.5,  # Start with neutral confidence
                signal_count=1,
            )

            pattern_id = str(uuid.uuid4())[:8]
            self.conn.execute(
                """
                INSERT INTO procedural (id, pattern_type, pattern_data, success_count, failure_count, last_used)
                VALUES (?, ?, ?, 1, 0, ?)
            """,
                (pattern_id, pattern_type, json.dumps(updated_pref.to_dict()), datetime.now().isoformat()),
            )

        self.conn.commit()
        return updated_pref

    def get_preference(self, preference_name: str) -> Optional[UserPreference]:
        """
        Get a stored user preference.

        Args:
            preference_name: Name of preference to retrieve

        Returns:
            UserPreference if found, None otherwise
        """
        pattern_type = f"preference_{preference_name}"

        cursor = self.conn.execute(
            """
            SELECT pattern_data
            FROM procedural
            WHERE pattern_type = ?
        """,
            (pattern_type,),
        )
        row = cursor.fetchone()

        if row:
            data = json.loads(row["pattern_data"])
            return UserPreference.from_dict(data)

        return None

    def get_all_preferences(self) -> Dict[str, UserPreference]:
        """
        Get all stored user preferences.

        Returns:
            Dict of preference_name -> UserPreference
        """
        cursor = self.conn.execute(
            """
            SELECT pattern_type, pattern_data
            FROM procedural
            WHERE pattern_type LIKE 'preference_%'
        """
        )

        preferences = {}
        for row in cursor.fetchall():
            pref_name = row["pattern_type"].replace("preference_", "")
            data = json.loads(row["pattern_data"])
            preferences[pref_name] = UserPreference.from_dict(data)

        return preferences

    def get_preference_instructions(self, min_confidence: float = 0.5) -> List[str]:
        """
        Get preference instructions for LLM prompting.

        Args:
            min_confidence: Minimum confidence to include preference

        Returns:
            List of instruction strings
        """
        instructions = []
        preferences = self.get_all_preferences()

        for name, pref in preferences.items():
            if pref.confidence >= min_confidence:
                if name == "length":
                    if pref.value == "concise":
                        instructions.append("Keep responses concise and to the point.")
                    elif pref.value == "detailed":
                        instructions.append("Provide detailed, comprehensive responses.")
                elif name == "formality":
                    if pref.value == "formal":
                        instructions.append("Use formal, professional language.")
                    elif pref.value == "casual":
                        instructions.append("Use casual, conversational language.")
                elif name == "detail_level":
                    if pref.value == "concise":
                        instructions.append("Focus on key points, avoid excessive detail.")
                    elif pref.value == "detailed":
                        instructions.append("Include thorough explanations and examples.")

        return instructions

    def clear_preferences(self) -> int:
        """
        Clear all stored preferences.

        Returns:
            Number of preferences cleared
        """
        cursor = self.conn.execute(
            """
            DELETE FROM procedural
            WHERE pattern_type LIKE 'preference_%'
        """
        )
        self.conn.commit()
        return cursor.rowcount
