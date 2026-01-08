"""
Mutation Engine - Actually applies configuration changes.

Generates mutations to improve system behavior based on fitness feedback,
applies them to genome.yaml, and tracks results to keep what works.
"""

from __future__ import annotations
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
class FailureAnalysis:
    """Analysis of failure patterns in low-fitness episodes."""

    total_episodes: int
    patterns: Dict[str, int]  # failure_type -> count
    suggested_fixes: List[Dict[str, Any]]
    avg_fitness: float
    worst_episode_id: Optional[str] = None
    analysis_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_episodes": self.total_episodes,
            "patterns": self.patterns,
            "suggested_fixes": self.suggested_fixes,
            "avg_fitness": self.avg_fitness,
            "worst_episode_id": self.worst_episode_id,
            "analysis_summary": self.analysis_summary,
        }


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

    def analyze_low_fitness_episodes(
        self,
        episodes: List["Episode"],
        fitness_threshold: float = 0.5
    ) -> FailureAnalysis:
        """
        Analyze low-fitness episodes to identify failure patterns.

        Categorizes failures into types:
        - too_long: Response was too verbose
        - too_short: Response was too terse
        - wrong_mode: Coupling mode didn't match user intent
        - missed_frustration: Failed to detect user frustration
        - low_engagement: User seemed disengaged
        - off_topic: Response didn't address user's question

        Args:
            episodes: List of Episode objects to analyze
            fitness_threshold: Episodes below this fitness are considered failures

        Returns:
            FailureAnalysis with patterns, counts, and suggested fixes
        """
        if not episodes:
            return FailureAnalysis(
                total_episodes=0,
                patterns={},
                suggested_fixes=[],
                avg_fitness=0.0,
                analysis_summary="No episodes to analyze"
            )

        patterns: Dict[str, int] = {
            "too_long": 0,
            "too_short": 0,
            "wrong_mode": 0,
            "missed_frustration": 0,
            "low_engagement": 0,
            "off_topic": 0,
        }

        fitnesses = []
        worst_fitness = 1.0
        worst_episode_id = None

        for ep in episodes:
            # Get episode fitness from cognitive state or joint state
            cognitive_state = getattr(ep, "cognitive_state", {})
            if hasattr(cognitive_state, "to_dict"):
                cognitive_state = cognitive_state.to_dict()
            elif not isinstance(cognitive_state, dict):
                cognitive_state = {}

            joint_state = getattr(ep, "joint_state", {})
            if hasattr(joint_state, "to_dict"):
                joint_state = joint_state.to_dict()
            elif not isinstance(joint_state, dict):
                joint_state = {}

            fitness = joint_state.get("fitness", cognitive_state.get("fitness", 0.5))
            fitnesses.append(fitness)

            # Track worst episode
            if fitness < worst_fitness:
                worst_fitness = fitness
                worst_episode_id = getattr(ep, "id", None)

            # Skip if above threshold
            if fitness >= fitness_threshold:
                continue

            # Analyze failure patterns
            response = getattr(ep, "response", "")
            input_text = getattr(ep, "input", "").lower()
            mode = getattr(ep, "mode", "DIALOGUE")

            # Too long check
            if len(response) > 2000:
                patterns["too_long"] += 1

            # Too short check
            if len(response) < 50 and len(input_text) > 50:
                patterns["too_short"] += 1

            # Wrong mode check
            mode_signals = {
                "teach": "TEACHING",
                "explain": "DIALOGUE",
                "help me": "COLLABORATIVE",
                "let's work on": "COLLABORATIVE",
                "can you": "DIALOGUE",
                "show me": "TEACHING",
            }
            for signal, expected_mode in mode_signals.items():
                if signal in input_text and mode != expected_mode:
                    patterns["wrong_mode"] += 1
                    break

            # Missed frustration check
            frustration_signals = ["frustrated", "annoyed", "stuck", "not working", "doesn't work", "broken"]
            frustration = cognitive_state.get("frustration", 0)
            if any(sig in input_text for sig in frustration_signals) and frustration < 0.3:
                patterns["missed_frustration"] += 1

            # Low engagement check
            engagement = cognitive_state.get("engagement", joint_state.get("engagement", 0.5))
            if engagement < 0.3:
                patterns["low_engagement"] += 1

            # Off topic check (simple heuristic)
            input_words = set(input_text.split())
            response_lower = response.lower()
            response_words = set(response_lower.split())
            overlap = len(input_words & response_words)
            if len(input_words) > 5 and overlap < 2:
                patterns["off_topic"] += 1

        # Generate suggested fixes based on patterns
        suggested_fixes = self._generate_fixes_from_patterns(patterns)

        # Calculate average fitness
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

        # Generate summary
        total_failures = sum(patterns.values())
        dominant_pattern = max(patterns, key=patterns.get) if any(patterns.values()) else None
        summary = f"Analyzed {len(episodes)} episodes. Found {total_failures} failure patterns."
        if dominant_pattern and patterns[dominant_pattern] > 0:
            summary += f" Most common issue: {dominant_pattern} ({patterns[dominant_pattern]} occurrences)."

        return FailureAnalysis(
            total_episodes=len(episodes),
            patterns=patterns,
            suggested_fixes=suggested_fixes,
            avg_fitness=avg_fitness,
            worst_episode_id=worst_episode_id,
            analysis_summary=summary,
        )

    def _generate_fixes_from_patterns(self, patterns: Dict[str, int]) -> List[Dict[str, Any]]:
        """Generate suggested fixes based on failure patterns."""
        fixes = []

        if patterns.get("too_long", 0) >= 2:
            fixes.append({
                "pattern": "too_long",
                "fix_type": "prompt_refinement",
                "target": "response.max_length",
                "suggestion": "Add conciseness instruction to system prompt",
                "priority": patterns["too_long"],
            })

        if patterns.get("too_short", 0) >= 2:
            fixes.append({
                "pattern": "too_short",
                "fix_type": "prompt_refinement",
                "target": "response.min_length",
                "suggestion": "Add detail instruction to system prompt",
                "priority": patterns["too_short"],
            })

        if patterns.get("wrong_mode", 0) >= 2:
            fixes.append({
                "pattern": "wrong_mode",
                "fix_type": "protocol_tuning",
                "target": "coupling.protocols",
                "suggestion": "Adjust mode detection triggers",
                "priority": patterns["wrong_mode"],
            })

        if patterns.get("missed_frustration", 0) >= 2:
            fixes.append({
                "pattern": "missed_frustration",
                "fix_type": "threshold_modification",
                "target": "coupling.human_model.frustration_threshold",
                "suggestion": "Lower frustration detection threshold",
                "priority": patterns["missed_frustration"],
            })

        if patterns.get("low_engagement", 0) >= 2:
            fixes.append({
                "pattern": "low_engagement",
                "fix_type": "prompt_refinement",
                "target": "response.engagement",
                "suggestion": "Add engagement hooks to responses",
                "priority": patterns["low_engagement"],
            })

        if patterns.get("off_topic", 0) >= 2:
            fixes.append({
                "pattern": "off_topic",
                "fix_type": "prompt_refinement",
                "target": "response.relevance",
                "suggestion": "Improve context understanding in prompt",
                "priority": patterns["off_topic"],
            })

        # Sort by priority
        fixes.sort(key=lambda x: x["priority"], reverse=True)

        return fixes

    async def propose_intelligent_mutation(
        self,
        episodes: List["Episode"],
        model: Optional[Any] = None
    ) -> Optional[Mutation]:
        """
        Use LLM to propose a targeted mutation based on failure analysis.

        Instead of random mutations, this analyzes recent low-fitness episodes
        and uses LLM to determine the best configuration change.

        Args:
            episodes: Recent episodes to analyze
            model: LLM model with async generate() method

        Returns:
            Targeted Mutation or None if no model/analysis available
        """
        if not episodes:
            return None

        # First, get failure analysis
        analysis = self.analyze_low_fitness_episodes(episodes)

        if analysis.total_episodes == 0 or not any(analysis.patterns.values()):
            # No failures to analyze
            return None

        # If no LLM available, use heuristic based on analysis
        if model is None:
            return self._propose_mutation_from_analysis(analysis)

        # Build context for LLM
        patterns_summary = ", ".join([
            f"{k}: {v}" for k, v in analysis.patterns.items() if v > 0
        ])

        # Sample some low-fitness episodes for context
        sample_episodes = []
        for ep in episodes[:5]:
            input_text = getattr(ep, "input", "")[:100]
            response = getattr(ep, "response", "")[:100]
            mode = getattr(ep, "mode", "DIALOGUE")
            sample_episodes.append(f"- Input: '{input_text}...' Mode: {mode} Response: '{response}...'")

        episodes_context = "\n".join(sample_episodes) if sample_episodes else "No samples"

        prompt = f"""Analyze these failure patterns in an AI assistant system and propose a specific configuration change.

Failure Analysis:
- Total episodes analyzed: {analysis.total_episodes}
- Average fitness: {analysis.avg_fitness:.2f}
- Failure patterns: {patterns_summary}

Sample low-fitness interactions:
{episodes_context}

Current suggested fixes from pattern analysis:
{json.dumps(analysis.suggested_fixes[:3], indent=2) if analysis.suggested_fixes else "None"}

Based on this analysis, propose ONE specific configuration change. Return JSON with:
- mutation_type: One of (threshold_modification, prompt_refinement, protocol_tuning, capability_adjustment)
- target: Config path to modify (e.g., "coupling.trust.initial", "coupling.human_model.frustration_threshold")
- direction: "increase" or "decrease" for numeric values, or specific value for others
- magnitude: How much to change (small, medium, large)
- reason: Why this change would help

Example output:
{{"mutation_type": "threshold_modification", "target": "coupling.human_model.frustration_threshold", "direction": "decrease", "magnitude": "medium", "reason": "Missed frustration signals suggest threshold is too high"}}

JSON:"""

        try:
            response = await model.generate(prompt)

            # Parse LLM response
            mutation_data = self._parse_llm_mutation_response(response)

            if mutation_data is None:
                return self._propose_mutation_from_analysis(analysis)

            # Create mutation from LLM suggestion
            mutation = self._create_mutation_from_llm(mutation_data, analysis.avg_fitness)

            if mutation:
                return mutation

        except Exception as e:
            print(f"Warning: LLM mutation proposal failed: {e}")

        # Fall back to heuristic
        return self._propose_mutation_from_analysis(analysis)

    def _parse_llm_mutation_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response to extract mutation proposal."""
        if not response:
            return None

        import re

        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response
        json_match = re.search(r'\{[\s\S]*?\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _create_mutation_from_llm(
        self,
        data: Dict[str, Any],
        current_fitness: float
    ) -> Optional[Mutation]:
        """Create a Mutation object from LLM suggestion data."""
        mutation_type = data.get("mutation_type", "threshold_modification")
        target = data.get("target", "")
        direction = data.get("direction", "")
        magnitude = data.get("magnitude", "small")
        reason = data.get("reason", "LLM-suggested improvement")

        if not target:
            return None

        # Calculate value change based on direction and magnitude
        current_value = self._get_genome_value(target, 0.5)

        # Check bool first since bool is subclass of int in Python
        if isinstance(current_value, bool):
            # Boolean toggle
            new_value = not current_value
        elif isinstance(current_value, (int, float)):
            # Numeric value
            deltas = {"small": 0.05, "medium": 0.1, "large": 0.2}
            delta = deltas.get(magnitude, 0.05)

            if direction == "decrease":
                new_value = max(0.01, current_value - delta)
            else:  # increase or default
                new_value = min(0.99, current_value + delta)

            new_value = round(new_value, 3)
        elif isinstance(current_value, list):
            # List - not modifying via LLM for safety
            return None
        else:
            # String or other - not modifying
            return None

        return Mutation(
            id=str(uuid.uuid4())[:8],
            mutation_type=mutation_type,
            target=target,
            old_value=current_value,
            new_value=new_value,
            reason=f"LLM Analysis: {reason}",
            fitness_at_proposal=current_fitness,
        )

    def _propose_mutation_from_analysis(
        self,
        analysis: FailureAnalysis
    ) -> Optional[Mutation]:
        """
        Propose a mutation based on failure analysis without LLM.

        Uses pattern-based heuristics to suggest targeted changes.
        """
        if not analysis.suggested_fixes:
            return None

        # Take the highest priority fix
        fix = analysis.suggested_fixes[0]
        pattern = fix["pattern"]
        fix_type = fix["fix_type"]

        # Map patterns to specific mutations
        mutations = {
            "too_long": (
                "threshold_modification",
                "response.conciseness_weight",
                0.5,
                0.7,
                "Response verbosity too high - increasing conciseness weight"
            ),
            "too_short": (
                "threshold_modification",
                "response.detail_weight",
                0.5,
                0.7,
                "Responses too terse - increasing detail weight"
            ),
            "wrong_mode": (
                "protocol_tuning",
                "coupling.mode_detection.sensitivity",
                0.5,
                0.7,
                "Mode detection missing signals - increasing sensitivity"
            ),
            "missed_frustration": (
                "threshold_modification",
                "coupling.human_model.frustration_threshold",
                0.3,
                0.2,
                "Missing frustration signals - lowering detection threshold"
            ),
            "low_engagement": (
                "prompt_refinement",
                "response.engagement_hooks",
                0.3,
                0.5,
                "Low engagement detected - adding engagement hooks"
            ),
            "off_topic": (
                "prompt_refinement",
                "response.relevance_weight",
                0.5,
                0.7,
                "Responses off-topic - increasing relevance weight"
            ),
        }

        if pattern not in mutations:
            return None

        mut_type, target, default, new_val, reason = mutations[pattern]
        current = self._get_genome_value(target, default)

        return Mutation(
            id=str(uuid.uuid4())[:8],
            mutation_type=mut_type,
            target=target,
            old_value=current,
            new_value=new_val,
            reason=reason,
            fitness_at_proposal=analysis.avg_fitness,
        )
