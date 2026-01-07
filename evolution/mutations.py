"""
Mutation Engine - Actually applies configuration changes.

Generates mutations to improve system behavior based on fitness feedback,
applies them to genome.yaml, and tracks results to keep what works.
"""

from __future__ import annotations
import copy
import json
import random
import shutil
import uuid
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.living_memory import Episode


@dataclass
class Mutation:
    """A proposed mutation to the system."""

    id: str
    mutation_type: str  # prompt_refinement | capability_adjustment | etc.
    target: str  # What to mutate (path in genome, e.g., "coupling.trust.initial")
    old_value: Any
    new_value: Any
    reason: str
    fitness_at_proposal: float
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False
    fitness_after: Optional[float] = None


@dataclass
class MutationResult:
    """Result of applying a mutation."""

    mutation: Mutation
    fitness_after: float
    interactions_tested: int
    success: bool
    rolled_back: bool


class MutationEngine:
    """
    Actually applies mutations to genome and tracks results.

    The key insight: Evolution only works if mutations are APPLIED,
    TESTED, and SELECTED based on outcomes.
    """

    MUTATION_TYPES = [
        "prompt_refinement",
        "capability_adjustment",
        "protocol_tuning",
        "threshold_modification",
    ]

    def __init__(
        self,
        config: Dict = None,
        genome: Dict = None,
        genome_path: Optional[Path] = None,
    ):
        config = config or {}

        self.rate = config.get("rate", 0.05)
        self.enabled_types = config.get("types", self.MUTATION_TYPES)
        self.genome = genome or {}
        self.genome_path = Path(genome_path) if genome_path else None
        self.experiment_duration = config.get("experiment_duration", 10)

        # Active experiment tracking
        self.active_mutation: Optional[Mutation] = None
        self.experiment_start_fitness: float = 0.0
        self.experiment_interactions: int = 0

        # Mutation history
        self._proposed: List[Mutation] = []
        self._applied: List[Mutation] = []
        self._rolled_back: List[Mutation] = []
        self.history: List[MutationResult] = []

        # Persistence path
        self._persistence_path: Optional[Path] = None

    def set_persistence_path(self, path: Path) -> None:
        """Set path for persisting mutation history."""
        self._persistence_path = path
        path.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def set_genome_path(self, path: Path) -> None:
        """Set the genome.yaml path for persistence."""
        self.genome_path = path

    def _load_history(self) -> None:
        """Load mutation history from disk."""
        if not self._persistence_path:
            return

        history_file = self._persistence_path / "evolution_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    self.history = [self._dict_to_result(r) for r in data.get("results", [])]
                    self._applied = [self._dict_to_mutation(m) for m in data.get("applied", [])]
                    self._rolled_back = [self._dict_to_mutation(m) for m in data.get("rolled_back", [])]
            except Exception as e:
                print(f"Warning: Could not load evolution history: {e}")

    def should_mutate(self, fitness: float) -> bool:
        """
        Determine if a mutation should be proposed.

        Don't mutate if experiment in progress.
        Mutations are more likely when fitness is low.
        """
        # Don't mutate if experiment in progress
        if self.active_mutation:
            return False

        if fitness > 0.8:
            # High fitness - rare mutations
            return random.random() < self.rate * 0.2
        elif fitness > 0.5:
            # Medium fitness - normal rate
            return random.random() < self.rate
        else:
            # Low fitness - increased rate
            adjusted_rate = self.rate * (2.0 - fitness)
            return random.random() < adjusted_rate

    def propose(
        self,
        fitness: float,
        episode: Optional["Episode"] = None,
    ) -> Optional[Mutation]:
        """
        Propose a mutation based on current fitness and episode context.

        Args:
            fitness: Current fitness score
            episode: Recent episode for context

        Returns:
            Proposed mutation or None
        """
        if not self.enabled_types:
            return None

        # Analyze episode for targeted mutations
        mutation_data = self._analyze_for_mutation(fitness, episode)

        if mutation_data[0] is None:
            # Random mutation if no targeted one found
            mutation_type = random.choice(self.enabled_types)
            if mutation_type == "threshold_modification":
                return self._propose_threshold_mutation(fitness)
            elif mutation_type == "prompt_refinement":
                return self._propose_prompt_mutation(fitness, episode)
            elif mutation_type == "capability_adjustment":
                return self._propose_capability_mutation(fitness)
            elif mutation_type == "protocol_tuning":
                return self._propose_protocol_mutation(fitness)
            return None

        mutation_type, target, old_val, new_val, reason = mutation_data

        return Mutation(
            id=str(uuid.uuid4())[:8],
            mutation_type=mutation_type,
            target=target,
            old_value=old_val,
            new_value=new_val,
            reason=reason,
            fitness_at_proposal=fitness,
        )

    def _analyze_for_mutation(
        self,
        fitness: float,
        episode: Optional["Episode"],
    ) -> Tuple[Optional[str], Optional[str], Any, Any, Optional[str]]:
        """Analyze episode to determine what mutation would help."""
        if episode is None:
            return (None, None, None, None, None)

        cognitive_state = getattr(episode, "cognitive_state", {})
        if isinstance(cognitive_state, dict):
            pass
        elif hasattr(cognitive_state, "to_dict"):
            cognitive_state = cognitive_state.to_dict()
        else:
            cognitive_state = {}

        mode = getattr(episode, "mode", "DIALOGUE")
        input_text = getattr(episode, "input", "").lower()

        # If user was frustrated, adjust frustration detection threshold
        frustration = cognitive_state.get("frustration", 0)
        if frustration > 0.3:
            current_threshold = self._get_genome_value(
                "coupling.human_model.frustration_threshold", 0.3
            )
            return (
                "threshold_modification",
                "coupling.human_model.frustration_threshold",
                current_threshold,
                max(0.1, current_threshold - 0.05),
                f"User showed frustration ({frustration:.2f}) - lowering detection threshold",
            )

        # If mode seems mismatched
        if "teach" in input_text and mode != "TEACHING":
            current_triggers = self._get_genome_value(
                "coupling.protocols.0.triggers", []
            )
            if isinstance(current_triggers, list) and "teach" not in current_triggers:
                return (
                    "trigger_addition",
                    "coupling.protocols.0.triggers",
                    current_triggers,
                    current_triggers + ["teach"],
                    f"User said 'teach' but mode was {mode} - adding trigger",
                )

        # If trust is stagnant, adjust increment
        joint_state = getattr(episode, "joint_state", {})
        if isinstance(joint_state, dict):
            trust_level = joint_state.get("trust_level", 0.5)
        elif hasattr(joint_state, "to_dict"):
            trust_level = joint_state.to_dict().get("trust_level", 0.5)
        else:
            trust_level = 0.5

        if 0.45 < trust_level < 0.55:  # Stuck around default
            current_increment = self._get_genome_value(
                "coupling.trust.increase_on.successful_task_completion", 0.02
            )
            return (
                "threshold_modification",
                "coupling.trust.increase_on.successful_task_completion",
                current_increment,
                current_increment + 0.01,
                "Trust seems stagnant - increasing success reward",
            )

        return (None, None, None, None, None)

    def _get_genome_value(self, path: str, default: Any) -> Any:
        """Get a value from genome by dot-notation path."""
        parts = path.split(".")
        current = self.genome
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError):
                    return default
            else:
                return default
        return current

    def _set_genome_value(self, path: str, value: Any) -> None:
        """Set a value in genome by dot-notation path."""
        parts = path.split(".")
        current = self.genome
        for part in parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError):
                    return

        final_key = parts[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list):
            try:
                idx = int(final_key)
                current[idx] = value
            except (ValueError, IndexError):
                pass

    def _propose_threshold_mutation(self, fitness: float) -> Mutation:
        """Propose a threshold adjustment."""
        targets = [
            ("coupling.trust.effects.0.threshold", 0.7, 0.05),
            ("coupling.trust.effects.1.threshold", 0.8, 0.05),
            ("evolution.mutations.rate", self.rate, 0.01),
            ("coupling.trust.initial", 0.5, 0.05),
        ]

        target, default, delta = random.choice(targets)
        current = self._get_genome_value(target, default)

        direction = random.choice([-1, 1])
        new_value = current + (direction * delta)
        new_value = max(0.1, min(0.95, new_value))

        return Mutation(
            id=str(uuid.uuid4())[:8],
            mutation_type="threshold_modification",
            target=target,
            old_value=current,
            new_value=round(new_value, 3),
            reason=f"Fitness is {fitness:.2f}, adjusting {target}",
            fitness_at_proposal=fitness,
        )

    def _propose_prompt_mutation(
        self,
        fitness: float,
        episode: Optional["Episode"],
    ) -> Mutation:
        """Propose a prompt refinement."""
        refinements = [
            "Add more empathy markers",
            "Be more concise",
            "Add clarifying questions",
            "Include more examples",
        ]

        refinement = random.choice(refinements)

        return Mutation(
            id=str(uuid.uuid4())[:8],
            mutation_type="prompt_refinement",
            target="system_prompt",
            old_value="current_prompt",
            new_value=refinement,
            reason=f"Fitness is {fitness:.2f}, trying: {refinement}",
            fitness_at_proposal=fitness,
        )

    def _propose_capability_mutation(self, fitness: float) -> Mutation:
        """Propose a capability adjustment."""
        adjustments = [
            ("capabilities.builtin.1.priority", 0.5, 0.1),  # web_search
            ("memory.semantic.decay_rate", 0.001, 0.0002),
        ]

        target, default, delta = random.choice(adjustments)
        current = self._get_genome_value(target, default)
        direction = random.choice([-1, 1])
        new_value = max(0.0001, min(0.9, current + (direction * delta)))

        return Mutation(
            id=str(uuid.uuid4())[:8],
            mutation_type="capability_adjustment",
            target=target,
            old_value=current,
            new_value=round(new_value, 4),
            reason=f"Fitness is {fitness:.2f}, adjusting capability",
            fitness_at_proposal=fitness,
        )

    def _propose_protocol_mutation(self, fitness: float) -> Mutation:
        """Propose a protocol tuning."""
        tunings = [
            ("interface.cli.show_thinking", False, [True, False]),
            ("interface.cli.show_ai_state", True, [True, False]),
        ]

        target, default, options = random.choice(tunings)
        current = self._get_genome_value(target, default)
        new_value = random.choice([o for o in options if o != current])

        return Mutation(
            id=str(uuid.uuid4())[:8],
            mutation_type="protocol_tuning",
            target=target,
            old_value=current,
            new_value=new_value,
            reason=f"Fitness is {fitness:.2f}, trying protocol change",
            fitness_at_proposal=fitness,
        )

    def apply(self, mutation: Mutation) -> None:
        """
        Apply a mutation to the genome and start experiment.

        THIS IS THE KEY: We actually modify the configuration.
        """
        # Store original for rollback
        self.active_mutation = mutation
        self.experiment_start_fitness = mutation.fitness_at_proposal
        self.experiment_interactions = 0

        # Apply the mutation to in-memory genome
        self._set_genome_value(mutation.target, mutation.new_value)

        mutation.applied = True
        self._applied.append(mutation)
        self._proposed.append(mutation)

        print(f"[EVOLUTION] Applied mutation {mutation.id}: {mutation.reason}")
        print(f"[EVOLUTION] Changed {mutation.target}: {mutation.old_value} -> {mutation.new_value}")

    def record_interaction(self, fitness: float) -> Optional[MutationResult]:
        """
        Record an interaction during an experiment.

        Returns MutationResult if experiment completed, None otherwise.
        """
        if not self.active_mutation:
            return None

        self.experiment_interactions += 1

        # Check if experiment is complete
        if self.experiment_interactions >= self.experiment_duration:
            return self._evaluate_experiment(fitness)

        return None

    def _evaluate_experiment(self, final_fitness: float) -> MutationResult:
        """Evaluate if the mutation improved things."""
        mutation = self.active_mutation
        fitness_change = final_fitness - mutation.fitness_at_proposal

        success = fitness_change > 0
        rolled_back = False

        if not success:
            # Rollback
            self._set_genome_value(mutation.target, mutation.old_value)
            rolled_back = True
            self._applied.remove(mutation)
            self._rolled_back.append(mutation)
            print(f"[EVOLUTION] Rolled back mutation {mutation.id}: fitness dropped by {-fitness_change:.3f}")
        else:
            # Persist successful mutation to genome file
            self._persist_genome()
            print(f"[EVOLUTION] Kept mutation {mutation.id}: fitness improved by {fitness_change:.3f}")

        # Record result
        result = MutationResult(
            mutation=mutation,
            fitness_after=final_fitness,
            interactions_tested=self.experiment_interactions,
            success=success,
            rolled_back=rolled_back,
        )
        self.history.append(result)

        # Update mutation
        mutation.fitness_after = final_fitness

        # Clear active experiment
        self.active_mutation = None
        self.experiment_interactions = 0

        return result

    def _persist_genome(self) -> None:
        """Write modified genome back to file."""
        if not self.genome_path:
            print("[EVOLUTION] No genome path set, skipping persistence")
            return

        try:
            # Create backup first
            backup_dir = self.genome_path.parent / "data" / "genome_backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir / f"genome_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"

            shutil.copy(self.genome_path, backup_file)

            # Write updated genome
            with open(self.genome_path, "w") as f:
                yaml.dump(self.genome, f, default_flow_style=False, sort_keys=False)

            print(f"[EVOLUTION] Persisted genome changes (backup: {backup_file.name})")
        except Exception as e:
            print(f"[EVOLUTION] Error persisting genome: {e}")

    def rollback(self, mutation: Mutation) -> None:
        """Rollback a mutation."""
        if mutation in self._applied:
            self._applied.remove(mutation)

        # Restore old value
        self._set_genome_value(mutation.target, mutation.old_value)

        mutation.applied = False
        self._rolled_back.append(mutation)

        print(f"[EVOLUTION] Rolled back mutation {mutation.id}")

    def evaluate(self, mutation: Mutation, new_fitness: float) -> bool:
        """
        Evaluate if a mutation was successful.

        Args:
            mutation: The mutation to evaluate
            new_fitness: Fitness after mutation

        Returns:
            True if mutation should be kept
        """
        mutation.fitness_after = new_fitness

        # Keep if fitness improved
        if new_fitness > mutation.fitness_at_proposal + 0.02:
            return True

        # Rollback if fitness dropped significantly
        if new_fitness < mutation.fitness_at_proposal - 0.05:
            self.rollback(mutation)
            return False

        # Neutral - keep with some probability
        return random.random() < 0.5

    def persist(self) -> None:
        """Persist mutation history to disk."""
        if not self._persistence_path:
            return

        history_file = self._persistence_path / "evolution_history.json"

        data = {
            "results": [self._result_to_dict(r) for r in self.history[-100:]],
            "applied": [self._mutation_to_dict(m) for m in self._applied[-100:]],
            "rolled_back": [self._mutation_to_dict(m) for m in self._rolled_back[-50:]],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not persist evolution history: {e}")

    def _mutation_to_dict(self, mutation: Mutation) -> Dict[str, Any]:
        """Convert mutation to dictionary."""
        return {
            "id": mutation.id,
            "type": mutation.mutation_type,
            "target": mutation.target,
            "old_value": mutation.old_value,
            "new_value": mutation.new_value,
            "reason": mutation.reason,
            "fitness_at_proposal": mutation.fitness_at_proposal,
            "fitness_after": mutation.fitness_after,
            "applied": mutation.applied,
            "timestamp": mutation.timestamp.isoformat(),
        }

    def _dict_to_mutation(self, data: Dict) -> Mutation:
        """Convert dictionary to mutation."""
        return Mutation(
            id=data["id"],
            mutation_type=data.get("type", "unknown"),
            target=data["target"],
            old_value=data["old_value"],
            new_value=data["new_value"],
            reason=data.get("reason", ""),
            fitness_at_proposal=data.get("fitness_at_proposal", 0.5),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            applied=data.get("applied", False),
            fitness_after=data.get("fitness_after"),
        )

    def _result_to_dict(self, result: MutationResult) -> Dict[str, Any]:
        """Convert mutation result to dictionary."""
        return {
            "mutation": self._mutation_to_dict(result.mutation),
            "fitness_after": result.fitness_after,
            "interactions_tested": result.interactions_tested,
            "success": result.success,
            "rolled_back": result.rolled_back,
        }

    def _dict_to_result(self, data: Dict) -> MutationResult:
        """Convert dictionary to mutation result."""
        return MutationResult(
            mutation=self._dict_to_mutation(data["mutation"]),
            fitness_after=data["fitness_after"],
            interactions_tested=data["interactions_tested"],
            success=data["success"],
            rolled_back=data["rolled_back"],
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get mutation statistics."""
        return {
            "total_proposed": len(self._proposed),
            "total_applied": len(self._applied),
            "total_rolled_back": len(self._rolled_back),
            "success_rate": (
                len([r for r in self.history if r.success]) / len(self.history)
                if self.history
                else 0
            ),
            "active_experiment": self.active_mutation.id if self.active_mutation else None,
            "experiment_progress": f"{self.experiment_interactions}/{self.experiment_duration}"
            if self.active_mutation
            else None,
        }

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution progress."""
        if not self.history:
            return {"total": 0, "successful": 0, "rolled_back": 0}

        successful = [r for r in self.history if r.success]

        return {
            "total": len(self.history),
            "successful": len(successful),
            "rolled_back": sum(1 for r in self.history if r.rolled_back),
            "avg_fitness_improvement": (
                sum(r.fitness_after - r.mutation.fitness_at_proposal for r in successful)
                / len(successful)
                if successful
                else 0
            ),
            "recent_mutations": [
                {
                    "type": r.mutation.mutation_type,
                    "target": r.mutation.target,
                    "success": r.success,
                    "reason": r.mutation.reason,
                }
                for r in self.history[-5:]
            ],
        }
