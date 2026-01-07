"""
Trust Tracker - Tracks trust relationship between human and AI.

Trust affects what the AI can do proactively and how confident
it should be in its suggestions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.living_memory import Episode


@dataclass
class TrustEvent:
    """A trust-affecting event."""

    timestamp: datetime
    event_type: str  # successful_task | error | misunderstanding | etc.
    delta: float
    description: str


class TrustTracker:
    """
    Tracks trust level between human and AI.

    Trust affects:
    - Suggestion confidence (higher trust = more confident suggestions)
    - Proactive behavior (only at high trust)
    - Error tolerance (higher trust = more forgiving)
    """

    def __init__(self, config: Dict = None):
        config = config or {}

        self.level = config.get("initial", 0.5)
        self.min_level = config.get("range", [0, 1])[0]
        self.max_level = config.get("range", [0, 1])[1]

        # Parse increase/decrease rules
        self.increase_rules = config.get("increase_on", {})
        self.decrease_rules = config.get("decrease_on", {})

        # Parse effect thresholds
        self.effects = config.get("effects", [])

        # Trust history
        self._history: List[TrustEvent] = []

    def update(self, episode: "Episode") -> float:
        """
        Update trust based on an interaction episode.

        Args:
            episode: The interaction episode

        Returns:
            New trust level
        """
        delta = 0.0
        event_type = "interaction"
        description = "Normal interaction"

        # Check for correction patterns (user correcting the AI)
        if self._detect_correction(episode.input):
            delta -= 0.05
            event_type = "misunderstanding"
            description = "User corrected the AI"

        # Check cognitive state for frustration (indicates potential issue)
        cognitive_state = episode.cognitive_state or {}
        frustration = cognitive_state.get("frustration", 0)

        if frustration > 0.5:
            # High frustration indicates we might have failed
            delta -= 0.03
            event_type = "frustration"
            description = f"User frustrated (level: {frustration:.2f})"
        elif frustration < 0.2 and delta >= 0:
            # Low frustration is good (only if no correction detected)
            delta += 0.01
            event_type = "positive"
            description = "Positive interaction"

        # Check fitness if available
        if episode.fitness is not None:
            if episode.fitness > 0.7:
                delta += self.increase_rules.get("successful_task_completion", 0.02)
                event_type = "success"
                description = f"Successful task (fitness: {episode.fitness:.2f})"
            elif episode.fitness < 0.3:
                delta += self.decrease_rules.get("error", -0.05)
                event_type = "failure"
                description = f"Task failure (fitness: {episode.fitness:.2f})"

        # Apply delta
        old_level = self.level
        self.level = max(self.min_level, min(self.max_level, self.level + delta))

        # Record event
        if delta != 0:
            self._history.append(
                TrustEvent(
                    timestamp=datetime.now(),
                    event_type=event_type,
                    delta=delta,
                    description=description,
                )
            )

        return self.level

    def _detect_correction(self, user_input: str) -> bool:
        """
        Detect if the user is correcting the AI.

        Common patterns that indicate the AI made a mistake.
        """
        input_lower = user_input.lower()

        correction_patterns = [
            "no, my name is",
            "that's not right",
            "that's wrong",
            "you're wrong",
            "i said",
            "i told you",
            "no, i'm",
            "no, it's",
            "actually, ",
            "i already said",
            "i already told",
            "not what i asked",
            "that's not what",
            "you misunderstood",
            "you didn't understand",
            "you forgot",
            "i'm not",
            "my name isn't",
            "that's incorrect",
        ]

        return any(pattern in input_lower for pattern in correction_patterns)

    def record_event(
        self,
        event_type: str,
        delta: Optional[float] = None,
        description: str = "",
    ) -> float:
        """
        Manually record a trust event.

        Args:
            event_type: Type of event (maps to config rules)
            delta: Override delta (uses config if not provided)
            description: Event description

        Returns:
            New trust level
        """
        if delta is None:
            # Look up delta from config
            if event_type in self.increase_rules:
                delta = self.increase_rules[event_type]
            elif event_type in self.decrease_rules:
                delta = self.decrease_rules[event_type]
            else:
                delta = 0.0

        # Apply delta
        self.level = max(self.min_level, min(self.max_level, self.level + delta))

        # Record event
        self._history.append(
            TrustEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                delta=delta,
                description=description or event_type,
            )
        )

        return self.level

    def can_suggest_confidently(self) -> bool:
        """Check if trust is high enough for confident suggestions."""
        threshold = next(
            (e.get("threshold", 0.7) for e in self.effects if e.get("affects") == "suggestion_confidence"),
            0.7,
        )
        return self.level >= threshold

    def can_be_proactive(self) -> bool:
        """Check if trust is high enough for proactive behavior."""
        threshold = next(
            (e.get("threshold", 0.8) for e in self.effects if e.get("affects") == "proactive_behavior"),
            0.8,
        )
        return self.level >= threshold

    def get_trend(self, window: int = 10) -> str:
        """Get trust trend over recent events."""
        if len(self._history) < 2:
            return "stable"

        recent = self._history[-window:]
        total_delta = sum(e.delta for e in recent)

        if total_delta > 0.05:
            return "increasing"
        elif total_delta < -0.05:
            return "decreasing"
        else:
            return "stable"

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trust events."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "type": e.event_type,
                "delta": e.delta,
                "description": e.description,
            }
            for e in self._history[-limit:]
        ]

    def reset(self, level: Optional[float] = None) -> None:
        """Reset trust to initial or specified level."""
        self.level = level if level is not None else 0.5
        self._history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "trend": self.get_trend(),
            "can_suggest_confidently": self.can_suggest_confidently(),
            "can_be_proactive": self.can_be_proactive(),
            "event_count": len(self._history),
        }
