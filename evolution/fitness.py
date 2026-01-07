"""
Fitness Tracker - Measures system performance.

Computes fitness scores based on multiple weighted metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.living_memory import Episode
    from coupling.joint_state import JointState
    from coupling.trust import TrustTracker


@dataclass
class FitnessMetric:
    """A single fitness metric."""

    name: str
    weight: float
    value: float
    source: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FitnessRecord:
    """A fitness computation record."""

    episode_id: str
    score: float
    metrics: List[FitnessMetric]
    timestamp: datetime


class FitnessTracker:
    """
    Tracks and computes fitness scores.

    Fitness measures how well the system is serving the user,
    considering multiple weighted factors.
    """

    def __init__(self, config: Dict = None):
        config = config or {}

        # Parse metric configurations
        self.metric_configs = config.get("metrics", [])
        if not self.metric_configs:
            # Default metrics
            self.metric_configs = [
                {"name": "goal_achievement", "weight": 0.3, "source": "goal_tracker"},
                {"name": "coupling_depth", "weight": 0.3, "source": "joint_state.alignment"},
                {"name": "trust_stability", "weight": 0.2, "source": "trust_tracker"},
                {"name": "user_satisfaction", "weight": 0.2, "source": "affective_memory"},
            ]

        # Fitness history
        self._history: List[FitnessRecord] = []

    def compute(
        self,
        episode: "Episode",
        joint_state: Optional["JointState"] = None,
        trust: Optional["TrustTracker"] = None,
    ) -> float:
        """
        Compute fitness score for an episode.

        Args:
            episode: The interaction episode
            joint_state: Current joint state
            trust: Trust tracker

        Returns:
            Fitness score (0-1)
        """
        metrics = []

        for config in self.metric_configs:
            name = config["name"]
            weight = config["weight"]
            source = config["source"]

            value = self._get_metric_value(name, source, episode, joint_state, trust)
            metrics.append(
                FitnessMetric(
                    name=name,
                    weight=weight,
                    value=value,
                    source=source,
                )
            )

        # Compute weighted sum
        total_weight = sum(m.weight for m in metrics)
        if total_weight > 0:
            score = sum(m.weight * m.value for m in metrics) / total_weight
        else:
            score = 0.5  # Default neutral score

        # Record
        record = FitnessRecord(
            episode_id=episode.id,
            score=score,
            metrics=metrics,
            timestamp=datetime.now(),
        )
        self._history.append(record)

        # Limit history size
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return score

    def _get_metric_value(
        self,
        name: str,
        source: str,
        episode: "Episode",
        joint_state: Optional["JointState"],
        trust: Optional["TrustTracker"],
    ) -> float:
        """Get value for a specific metric."""
        if name == "goal_achievement" or source == "goal_tracker":
            # Check if goals were completed
            if joint_state:
                active = joint_state.get_active_goals()
                completed = [g for g in joint_state.goals if g.status == "completed"]
                total = len(active) + len(completed)
                if total > 0:
                    return len(completed) / total
            return 0.5

        elif name == "coupling_depth" or source == "joint_state.alignment":
            # Use alignment from joint state
            if joint_state:
                return joint_state.alignment
            return 0.5

        elif name == "trust_stability" or source == "trust_tracker":
            # Use trust level
            if trust:
                return trust.level
            return 0.5

        elif name == "user_satisfaction" or source == "affective_memory":
            # Infer from cognitive state
            cognitive_state = episode.cognitive_state or {}
            frustration = cognitive_state.get("frustration", 0.5)
            return 1.0 - frustration

        else:
            return 0.5  # Default neutral value

    def get_average(self, window: int = 50) -> float:
        """Get average fitness over recent episodes."""
        if not self._history:
            return 0.5

        recent = self._history[-window:]
        return sum(r.score for r in recent) / len(recent)

    def get_trend(self, window: int = 20) -> str:
        """Get fitness trend."""
        if len(self._history) < window:
            return "insufficient_data"

        first_half = self._history[-window : -window // 2]
        second_half = self._history[-window // 2 :]

        avg_first = sum(r.score for r in first_half) / len(first_half)
        avg_second = sum(r.score for r in second_half) / len(second_half)

        if avg_second > avg_first + 0.05:
            return "improving"
        elif avg_second < avg_first - 0.05:
            return "declining"
        else:
            return "stable"

    def get_metric_breakdown(self, window: int = 50) -> Dict[str, float]:
        """Get average values for each metric."""
        if not self._history:
            return {}

        recent = self._history[-window:]
        breakdown = {}

        for config in self.metric_configs:
            name = config["name"]
            values = []
            for record in recent:
                for metric in record.metrics:
                    if metric.name == name:
                        values.append(metric.value)
            if values:
                breakdown[name] = sum(values) / len(values)

        return breakdown

    def get_low_performing_metrics(self, threshold: float = 0.4) -> List[str]:
        """Get metrics that are underperforming."""
        breakdown = self.get_metric_breakdown()
        return [name for name, value in breakdown.items() if value < threshold]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "average": self.get_average(),
            "trend": self.get_trend(),
            "breakdown": self.get_metric_breakdown(),
            "history_size": len(self._history),
        }

    @classmethod
    def trend(cls) -> str:
        """Class method to get trend (for CLI compatibility)."""
        # This would typically load from persisted state
        return "stable"
