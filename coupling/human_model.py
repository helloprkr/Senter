"""
Human Model - AI's model of the human's cognitive state.

The key insight: The AI must model not just what the human says,
but what cognitive state they're in while saying it.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.living_memory import LivingMemory, Episode


@dataclass
class HumanCognitiveState:
    """Inferred mental state of the human."""

    focus_level: float = 0.7  # 0-1
    energy_level: float = 0.7  # 0-1
    mode: Literal["exploring", "executing", "debugging", "learning", "creating"] = "exploring"
    time_pressure: Literal["none", "low", "moderate", "high", "urgent"] = "moderate"
    frustration: float = 0.0  # 0-1

    # Evidence for this inference
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "focus_level": self.focus_level,
            "energy_level": self.energy_level,
            "mode": self.mode,
            "time_pressure": self.time_pressure,
            "frustration": self.frustration,
        }


@dataclass
class HumanProfile:
    """Persistent profile learned over time."""

    communication_style: str = "neutral"  # concise | detailed | casual | formal
    expertise_areas: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    patterns: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "communication_style": self.communication_style,
            "expertise_areas": self.expertise_areas,
            "preferences": self.preferences,
            "pattern_count": len(self.patterns),
        }


class HumanModel:
    """
    Models the human for better coupling.

    This is bidirectional modeling - the AI maintains a model
    of the human, enabling anticipation and adaptation.
    """

    # Patterns for detecting cognitive states
    FRUSTRATION_PATTERNS = [
        r"frustrat",      # matches frustrated, frustrating, frustration
        r"annoy",         # matches annoying, annoyed
        r"doesn't work",
        r"won't work",
        r"broken",
        r"still not",
        r"again\?",
        r"why won't",
        r"ugh",
        r"argh",
        r"this is stupid",
        r"waste of time",
        r"so tired of",
        r"keeps? (failing|breaking|crashing)",
        r"!{2,}",         # Multiple exclamation marks
    ]

    URGENCY_PATTERNS: Dict[str, List[str]] = {
        "urgent": [r"asap", r"urgent", r"immediately", r"right now", r"emergency"],
        "high": [r"quickly", r"hurry", r"deadline", r"soon", r"today"],
        "moderate": [r"when you can", r"sometime", r"would be nice"],
        "low": [r"no rush", r"whenever", r"eventually"],
    }

    MODE_INDICATORS: Dict[str, List[str]] = {
        "debugging": [r"bug", r"error", r"doesn't work", r"broken", r"fix", r"wrong"],
        "learning": [r"how does", r"explain", r"teach", r"understand", r"what is", r"why"],
        "creating": [r"create", r"make", r"build", r"design", r"write", r"generate"],
        "executing": [r"do", r"run", r"execute", r"implement", r"now"],
        "exploring": [r"what if", r"could you", r"maybe", r"think about", r"ideas"],
    }

    def __init__(self, config: Dict, memory: Optional["LivingMemory"] = None):
        self.config = config
        self.memory = memory
        self.profile = HumanProfile()
        self.cognitive_state = HumanCognitiveState()

        # State history for trend analysis
        self._state_history: List[HumanCognitiveState] = []

    def infer_state(self, input_text: str) -> HumanCognitiveState:
        """
        Infer current cognitive state from input.

        Uses pattern matching and context to determine:
        - What mode the human is in
        - How frustrated they are
        - How much time pressure they're under
        - Their focus and energy levels
        """
        evidence = []
        input_lower = input_text.lower()

        # Detect frustration
        frustration = 0.0
        for pattern in self.FRUSTRATION_PATTERNS:
            if re.search(pattern, input_lower):
                frustration = min(1.0, frustration + 0.2)
                evidence.append(f"Frustration indicator: {pattern}")

        # Detect urgency
        time_pressure: Literal["none", "low", "moderate", "high", "urgent"] = "moderate"
        for level, patterns in self.URGENCY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    time_pressure = level  # type: ignore
                    evidence.append(f"Urgency indicator: {pattern} -> {level}")
                    break

        # Detect mode
        mode: Literal["exploring", "executing", "debugging", "learning", "creating"] = "exploring"
        mode_scores: Dict[str, int] = {m: 0 for m in self.MODE_INDICATORS}
        for mode_name, patterns in self.MODE_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    mode_scores[mode_name] += 1

        if max(mode_scores.values()) > 0:
            mode = max(mode_scores, key=mode_scores.get)  # type: ignore
            evidence.append(f"Mode detected: {mode}")

        # Estimate focus and energy from message characteristics
        focus_level = 0.7  # Default
        energy_level = 0.7  # Default

        # Short, terse messages might indicate low energy or high frustration
        if len(input_text) < 20 and frustration > 0.3:
            energy_level = 0.4
            evidence.append("Short message + frustration -> low energy")

        # Long, detailed messages indicate high focus
        if len(input_text) > 200:
            focus_level = 0.9
            evidence.append("Long detailed message -> high focus")

        # Very short responses might indicate distraction
        if len(input_text) < 10:
            focus_level = 0.5
            evidence.append("Very short message -> possible distraction")

        self.cognitive_state = HumanCognitiveState(
            focus_level=focus_level,
            energy_level=energy_level,
            mode=mode,
            time_pressure=time_pressure,
            frustration=frustration,
            evidence=evidence,
        )

        # Track history
        self._state_history.append(self.cognitive_state)
        if len(self._state_history) > 50:
            self._state_history = self._state_history[-50:]

        return self.cognitive_state

    def update_profile(self, episode: "Episode") -> None:
        """Update persistent profile based on interaction."""
        # Track patterns that work well
        if episode.fitness and episode.fitness > 0.7:
            self.profile.patterns.append(
                {
                    "mode": episode.mode,
                    "success": True,
                    "context": episode.cognitive_state,
                }
            )

        # Limit pattern history
        if len(self.profile.patterns) > 100:
            self.profile.patterns = self.profile.patterns[-100:]

    def update_expertise(self, topic: str, level: float) -> None:
        """Update expertise level for a topic."""
        self.profile.expertise_areas[topic] = level

    def get_frustration_trend(self) -> str:
        """Get trend of frustration over recent interactions."""
        if len(self._state_history) < 3:
            return "insufficient_data"

        recent = self._state_history[-5:]
        first_half = sum(s.frustration for s in recent[: len(recent) // 2])
        second_half = sum(s.frustration for s in recent[len(recent) // 2 :])

        if second_half > first_half + 0.1:
            return "increasing"
        elif second_half < first_half - 0.1:
            return "decreasing"
        else:
            return "stable"

    def get_preferred_communication_style(self) -> str:
        """Infer preferred communication style from patterns."""
        if not self._state_history:
            return "neutral"

        # Check time pressure patterns
        avg_pressure = sum(
            {"none": 0, "low": 1, "moderate": 2, "high": 3, "urgent": 4}.get(
                s.time_pressure, 2
            )
            for s in self._state_history
        ) / len(self._state_history)

        if avg_pressure > 3:
            return "concise"
        elif avg_pressure < 1.5:
            return "detailed"
        else:
            return "balanced"

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "cognitive_state": self.cognitive_state.to_dict(),
            "profile": self.profile.to_dict(),
            "frustration_trend": self.get_frustration_trend(),
            "preferred_style": self.get_preferred_communication_style(),
        }
