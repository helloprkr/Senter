"""
Selection Pressure - Determines which configurations survive.

Applies selection pressure based on fitness to keep improvements
and discard regressions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mutations import Mutation, MutationResult
    from coupling.trust import TrustTracker


@dataclass
class Experiment:
    """An ongoing configuration experiment."""

    id: str
    mutation: "Mutation"
    start_fitness: float
    interactions_count: int = 0
    fitness_samples: List[float] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    kept: bool = False


class SelectionPressure:
    """
    Applies selection pressure to mutations.

    Tracks experiments and decides which mutations to keep
    based on observed fitness changes.
    """

    def __init__(self, config: Dict = None):
        config = config or {}

        self.pressure_metric = config.get("pressure", "user_satisfaction")
        self.rollback_on_trust_drop = config.get("rollback_on_trust_drop", True)
        self.experiment_duration = config.get("experiment_duration", 10)

        # Active experiments
        self._experiments: Dict[str, Experiment] = {}

        # History of selections
        self._selection_history: List[Dict[str, Any]] = []

    def start_experiment(self, mutation: "Mutation", current_fitness: float) -> str:
        """
        Start an experiment with a mutation.

        Args:
            mutation: The mutation being tested
            current_fitness: Fitness before mutation

        Returns:
            Experiment ID
        """
        experiment = Experiment(
            id=mutation.id,
            mutation=mutation,
            start_fitness=current_fitness,
        )
        self._experiments[mutation.id] = experiment
        return mutation.id

    def record_interaction(
        self,
        experiment_id: str,
        fitness: float,
    ) -> Optional[bool]:
        """
        Record an interaction during an experiment.

        Args:
            experiment_id: The experiment ID
            fitness: Fitness of this interaction

        Returns:
            None if experiment ongoing, True/False if decision made
        """
        if experiment_id not in self._experiments:
            return None

        experiment = self._experiments[experiment_id]
        experiment.interactions_count += 1
        experiment.fitness_samples.append(fitness)

        # Check if experiment duration reached
        if experiment.interactions_count >= self.experiment_duration:
            return self._evaluate_experiment(experiment)

        return None

    def _evaluate_experiment(self, experiment: Experiment) -> bool:
        """Evaluate experiment and decide if mutation is kept."""
        experiment.completed = True

        if not experiment.fitness_samples:
            # No data - keep mutation with low confidence
            experiment.kept = True
            return True

        avg_fitness = sum(experiment.fitness_samples) / len(experiment.fitness_samples)
        improvement = avg_fitness - experiment.start_fitness

        # Decision criteria
        if improvement > 0.03:
            # Clear improvement - keep
            experiment.kept = True
            decision = True
        elif improvement < -0.03:
            # Clear regression - rollback
            experiment.kept = False
            decision = False
        else:
            # Neutral - keep with some probability
            experiment.kept = improvement >= 0
            decision = experiment.kept

        # Record decision
        self._selection_history.append({
            "experiment_id": experiment.id,
            "mutation_type": experiment.mutation.mutation_type,
            "start_fitness": experiment.start_fitness,
            "end_fitness": avg_fitness,
            "improvement": improvement,
            "kept": decision,
            "interactions": experiment.interactions_count,
            "timestamp": datetime.now().isoformat(),
        })

        # Clean up
        del self._experiments[experiment.id]

        return decision

    def record_experiment_result(self, result: "MutationResult") -> None:
        """
        Record a mutation experiment result from MutationEngine.

        Args:
            result: The mutation result containing success/failure info
        """
        # Record in selection history
        self._selection_history.append({
            "experiment_id": result.mutation.id,
            "mutation_type": result.mutation.mutation_type,
            "target": result.mutation.target,
            "start_fitness": result.mutation.fitness_at_proposal,
            "end_fitness": result.fitness_after,
            "improvement": result.fitness_after - result.mutation.fitness_at_proposal,
            "kept": result.success,
            "rolled_back": result.rolled_back,
            "interactions": result.interactions_tested,
            "timestamp": datetime.now().isoformat(),
        })

        # Clean up experiment if it exists
        if result.mutation.id in self._experiments:
            del self._experiments[result.mutation.id]

    def check_trust_threshold(
        self,
        trust: "TrustTracker",
        threshold: float = 0.3,
    ) -> bool:
        """
        Check if trust has dropped below threshold.

        If so, all active experiments should be rolled back.
        """
        if not self.rollback_on_trust_drop:
            return False

        if trust.level < threshold:
            # Rollback all experiments
            for exp_id in list(self._experiments.keys()):
                experiment = self._experiments[exp_id]
                experiment.completed = True
                experiment.kept = False

                self._selection_history.append({
                    "experiment_id": exp_id,
                    "mutation_type": experiment.mutation.mutation_type,
                    "reason": "trust_drop_rollback",
                    "trust_level": trust.level,
                    "kept": False,
                    "timestamp": datetime.now().isoformat(),
                })

                del self._experiments[exp_id]

            return True

        return False

    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get list of active experiments."""
        return [
            {
                "id": exp.id,
                "mutation_type": exp.mutation.mutation_type,
                "interactions": exp.interactions_count,
                "current_avg": (
                    sum(exp.fitness_samples) / len(exp.fitness_samples)
                    if exp.fitness_samples
                    else exp.start_fitness
                ),
                "started": exp.started_at.isoformat(),
            }
            for exp in self._experiments.values()
        ]

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get selection statistics."""
        if not self._selection_history:
            return {"total": 0, "kept": 0, "rejected": 0, "keep_rate": 0}

        kept = sum(1 for h in self._selection_history if h.get("kept", False))
        rejected = len(self._selection_history) - kept

        return {
            "total": len(self._selection_history),
            "kept": kept,
            "rejected": rejected,
            "keep_rate": kept / len(self._selection_history),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "active_experiments": len(self._experiments),
            "stats": self.get_selection_stats(),
            "experiments": self.get_active_experiments(),
        }
