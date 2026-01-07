"""
Joint State Surface - The shared cognitive space.

The key insight: Both human and AI need to see the same state
to coordinate effectively. This is the "shared whiteboard."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .human_model import HumanCognitiveState


@dataclass
class Goal:
    """A goal in the shared space."""

    text: str
    proposed_by: str  # "human" or "ai"
    status: str = "active"  # active, completed, abandoned
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class JointState:
    """
    The shared cognitive space between human and AI.

    Both parties can observe this state. This enables:
    - Alignment checking
    - Goal negotiation
    - Uncertainty visibility
    """

    # Current focus
    focus: Optional[str] = None
    focus_since: Optional[datetime] = None

    # Shared goals (negotiated)
    goals: List[Goal] = field(default_factory=list)

    # Alignment score (how well aligned are human and AI)
    alignment: float = 0.8

    # What neither party knows
    uncertainties: List[str] = field(default_factory=list)

    # Context about the current interaction
    context: Dict[str, Any] = field(default_factory=dict)

    def update_from_input(
        self,
        input_text: str,
        cognitive_state: Optional["HumanCognitiveState"] = None,
    ) -> None:
        """Update joint state based on new input."""
        # Update focus
        self.focus = self._extract_focus(input_text)
        self.focus_since = datetime.now()

        # Check for goal mentions
        self._update_goals(input_text)

        # Update alignment based on cognitive state
        if cognitive_state:
            if cognitive_state.frustration > 0.5:
                self.alignment = max(0.3, self.alignment - 0.1)
            else:
                self.alignment = min(1.0, self.alignment + 0.02)

        # Update context
        self.context["last_input_time"] = datetime.now().isoformat()
        self.context["input_length"] = len(input_text)

    def update_from_response(self, response: str) -> None:
        """Update joint state after AI responds."""
        self.context["last_response_time"] = datetime.now().isoformat()
        self.context["response_length"] = len(response)

        # Clear uncertainties that might have been addressed
        self.uncertainties = [u for u in self.uncertainties if len(self.uncertainties) <= 3]

    def _extract_focus(self, input_text: str) -> str:
        """Extract current focus from input."""
        # Simple: use topic indicators
        words = input_text.split()

        # Look for question words to determine focus
        question_words = {"what", "how", "why", "when", "where", "who"}
        for i, word in enumerate(words[:5]):
            if word.lower() in question_words and i + 1 < len(words):
                return " ".join(words[i : i + 4])

        # Default to first few meaningful words
        return " ".join(words[:5]) if words else "general"

    def _update_goals(self, input_text: str) -> None:
        """Update goals based on input."""
        input_lower = input_text.lower()

        # Check for goal completion
        completion_words = ["done", "finished", "completed", "thanks", "perfect", "great"]
        if any(word in input_lower for word in completion_words):
            for goal in self.goals:
                if goal.status == "active":
                    goal.status = "completed"
                    goal.progress = 1.0
                    goal.completed_at = datetime.now()
                    break

        # Check for new goals
        goal_indicators = ["need to", "want to", "help me", "can you", "please"]
        if any(indicator in input_lower for indicator in goal_indicators):
            # Don't add duplicate goals
            goal_text = input_text[:100]
            if not any(g.text == goal_text for g in self.goals):
                self.goals.append(
                    Goal(
                        text=goal_text,
                        proposed_by="human",
                    )
                )

        # Check for goal abandonment
        abandon_words = ["nevermind", "forget it", "cancel", "stop"]
        if any(word in input_lower for word in abandon_words):
            for goal in self.goals:
                if goal.status == "active":
                    goal.status = "abandoned"
                    break

    def add_goal(self, text: str, proposed_by: str = "ai") -> Goal:
        """Add a new goal to the shared space."""
        goal = Goal(text=text, proposed_by=proposed_by)
        self.goals.append(goal)
        return goal

    def add_uncertainty(self, uncertainty: str) -> None:
        """Add an uncertainty to the shared space."""
        if uncertainty not in self.uncertainties:
            self.uncertainties.append(uncertainty)

    def clear_uncertainty(self, uncertainty: str) -> None:
        """Remove an uncertainty from the shared space."""
        if uncertainty in self.uncertainties:
            self.uncertainties.remove(uncertainty)

    def get_active_goals(self) -> List[Goal]:
        """Get list of active goals."""
        return [g for g in self.goals if g.status == "active"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/display."""
        return {
            "focus": self.focus,
            "alignment": self.alignment,
            "goals": [
                {
                    "text": g.text,
                    "status": g.status,
                    "progress": g.progress,
                    "proposed_by": g.proposed_by,
                }
                for g in self.goals
            ],
            "uncertainties": self.uncertainties,
            "context": self.context,
        }

    def to_visible_dict(self) -> Dict[str, Any]:
        """Format for human visibility."""
        active_goals = self.get_active_goals()
        return {
            "Current Focus": self.focus or "None",
            "Alignment": f"{self.alignment:.0%}",
            "Active Goals": len(active_goals),
            "Uncertainties": len(self.uncertainties),
        }

    def get_status_line(self) -> str:
        """Get a single-line status summary."""
        active = len(self.get_active_goals())
        return (
            f"Focus: {self.focus or 'general'} | "
            f"Alignment: {self.alignment:.0%} | "
            f"Goals: {active}"
        )
