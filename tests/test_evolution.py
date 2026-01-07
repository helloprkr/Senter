"""
Tests for evolution module.
"""

import pytest
from datetime import datetime

from evolution.fitness import FitnessTracker, FitnessMetric
from evolution.mutations import MutationEngine, Mutation
from evolution.selection import SelectionPressure


class TestFitnessTracker:
    """Tests for FitnessTracker."""

    def test_compute_fitness(self):
        """Test fitness computation."""
        from memory.living_memory import Episode
        from coupling.joint_state import JointState
        from coupling.trust import TrustTracker

        tracker = FitnessTracker()

        episode = Episode(
            id="test1",
            timestamp=datetime.now(),
            input="Test input",
            response="Test response",
            mode="dialogue",
            cognitive_state={"frustration": 0.1},
            joint_state={},
        )

        joint_state = JointState()
        joint_state.alignment = 0.8

        trust = TrustTracker({"initial": 0.7})

        fitness = tracker.compute(episode, joint_state, trust)

        assert 0 <= fitness <= 1

    def test_fitness_average(self):
        """Test fitness average calculation."""
        from memory.living_memory import Episode

        tracker = FitnessTracker()

        # Add some fitness scores
        for i in range(10):
            episode = Episode(
                id=f"ep{i}",
                timestamp=datetime.now(),
                input=f"Input {i}",
                response=f"Response {i}",
                mode="dialogue",
                cognitive_state={"frustration": 0.0},
                joint_state={},
            )
            tracker.compute(episode, None, None)

        avg = tracker.get_average()
        assert 0 <= avg <= 1

    def test_fitness_trend(self):
        """Test fitness trend detection."""
        tracker = FitnessTracker()

        # Not enough data
        trend = tracker.get_trend()
        assert trend == "insufficient_data"


class TestMutationEngine:
    """Tests for MutationEngine."""

    def test_should_mutate(self):
        """Test mutation probability."""
        engine = MutationEngine({"rate": 1.0})  # 100% rate for testing

        # Low fitness should increase mutation chance
        assert engine.should_mutate(0.2) or engine.should_mutate(0.2)

    def test_propose_mutation(self):
        """Test mutation proposal."""
        engine = MutationEngine({
            "rate": 0.05,
            "types": ["threshold_modification"],
        })

        mutation = engine.propose(0.3, None)

        assert mutation is not None
        assert mutation.mutation_type == "threshold_modification"
        assert mutation.fitness_at_proposal == 0.3

    def test_mutation_apply(self):
        """Test applying mutation."""
        engine = MutationEngine({})

        mutation = Mutation(
            id="mut1",
            mutation_type="threshold_modification",
            target="test.threshold",
            old_value=0.5,
            new_value=0.6,
            reason="Test mutation",
            fitness_at_proposal=0.4,
        )

        engine.apply(mutation)

        assert mutation.applied
        assert mutation in engine._applied

    def test_mutation_rollback(self):
        """Test rolling back mutation."""
        engine = MutationEngine({})

        mutation = Mutation(
            id="mut1",
            mutation_type="test",
            target="test",
            old_value=0.5,
            new_value=0.6,
            reason="Test",
            fitness_at_proposal=0.4,
        )

        engine.apply(mutation)
        engine.rollback(mutation)

        assert not mutation.applied
        assert mutation in engine._rolled_back

    def test_mutation_evaluation(self):
        """Test mutation evaluation."""
        engine = MutationEngine({})

        mutation = Mutation(
            id="mut1",
            mutation_type="test",
            target="test",
            old_value=0.5,
            new_value=0.6,
            reason="Test",
            fitness_at_proposal=0.4,
        )

        engine.apply(mutation)

        # Improved fitness - should keep
        result = engine.evaluate(mutation, 0.6)
        assert result is True

        # Create new mutation for regression test
        mutation2 = Mutation(
            id="mut2",
            mutation_type="test",
            target="test",
            old_value=0.5,
            new_value=0.6,
            reason="Test",
            fitness_at_proposal=0.5,
        )
        engine.apply(mutation2)

        # Regression - should rollback
        result = engine.evaluate(mutation2, 0.3)
        assert result is False


class TestSelectionPressure:
    """Tests for SelectionPressure."""

    def test_start_experiment(self):
        """Test starting an experiment."""
        selection = SelectionPressure({"experiment_duration": 5})

        mutation = Mutation(
            id="mut1",
            mutation_type="test",
            target="test",
            old_value=0.5,
            new_value=0.6,
            reason="Test",
            fitness_at_proposal=0.4,
        )

        exp_id = selection.start_experiment(mutation, 0.4)

        assert exp_id == "mut1"
        assert len(selection._experiments) == 1

    def test_record_interaction(self):
        """Test recording interactions during experiment."""
        selection = SelectionPressure({"experiment_duration": 3})

        mutation = Mutation(
            id="mut1",
            mutation_type="test",
            target="test",
            old_value=0.5,
            new_value=0.6,
            reason="Test",
            fitness_at_proposal=0.4,
        )

        selection.start_experiment(mutation, 0.4)

        # Record interactions
        result = selection.record_interaction("mut1", 0.5)
        assert result is None  # Not done yet

        result = selection.record_interaction("mut1", 0.6)
        assert result is None  # Still not done

        result = selection.record_interaction("mut1", 0.7)
        # Now should be evaluated (3 interactions = experiment_duration)
        assert result is not None

    def test_experiment_evaluation(self):
        """Test experiment evaluation."""
        selection = SelectionPressure({"experiment_duration": 2})

        mutation = Mutation(
            id="mut1",
            mutation_type="test",
            target="test",
            old_value=0.5,
            new_value=0.6,
            reason="Test",
            fitness_at_proposal=0.4,
        )

        selection.start_experiment(mutation, 0.4)

        # Record good fitness
        selection.record_interaction("mut1", 0.6)
        result = selection.record_interaction("mut1", 0.7)

        # Should keep mutation (improvement)
        assert result is True

    def test_trust_threshold_rollback(self):
        """Test rollback on trust drop."""
        from coupling.trust import TrustTracker

        selection = SelectionPressure({
            "experiment_duration": 10,
            "rollback_on_trust_drop": True,
        })

        mutation = Mutation(
            id="mut1",
            mutation_type="test",
            target="test",
            old_value=0.5,
            new_value=0.6,
            reason="Test",
            fitness_at_proposal=0.4,
        )

        selection.start_experiment(mutation, 0.4)

        trust = TrustTracker({"initial": 0.2})  # Below default threshold

        rolled_back = selection.check_trust_threshold(trust, threshold=0.3)

        assert rolled_back
        assert len(selection._experiments) == 0
