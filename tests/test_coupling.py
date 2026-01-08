"""
Tests for coupling module.
"""


from coupling.joint_state import JointState, Goal
from coupling.human_model import HumanModel, HumanCognitiveState
from coupling.protocols import CouplingFacilitator, CouplingMode
from coupling.trust import TrustTracker


class TestJointState:
    """Tests for JointState."""

    def test_update_from_input(self):
        """Test updating joint state from input."""
        state = JointState()

        state.update_from_input("Help me debug this error")

        assert state.focus is not None
        assert state.focus_since is not None

    def test_goal_detection(self):
        """Test automatic goal detection."""
        state = JointState()

        state.update_from_input("Can you help me write a function?")

        assert len(state.goals) == 1
        assert state.goals[0].proposed_by == "human"
        assert state.goals[0].status == "active"

    def test_goal_completion(self):
        """Test goal completion detection."""
        state = JointState()
        state.goals.append(Goal(text="Test goal", proposed_by="human"))

        state.update_from_input("Thanks, that's perfect!")

        assert state.goals[0].status == "completed"

    def test_alignment_update(self):
        """Test alignment score updates."""
        state = JointState()
        initial_alignment = state.alignment

        # Simulate frustration
        cognitive_state = HumanCognitiveState(frustration=0.7)
        state.update_from_input("This is frustrating!", cognitive_state)

        assert state.alignment < initial_alignment

    def test_to_visible_dict(self):
        """Test visible state formatting."""
        state = JointState()
        state.focus = "testing"
        state.alignment = 0.85

        visible = state.to_visible_dict()

        assert visible["Current Focus"] == "testing"
        assert "85%" in visible["Alignment"]


class TestHumanModel:
    """Tests for HumanModel."""

    def test_frustration_detection(self):
        """Test detecting frustration."""
        model = HumanModel({})

        state = model.infer_state("This is so frustrating! Why won't it work?!")

        assert state.frustration > 0.3
        assert "Frustration indicator" in " ".join(state.evidence)

    def test_mode_detection(self):
        """Test detecting cognitive mode."""
        model = HumanModel({})

        # Debugging mode
        state = model.infer_state("There's a bug in this code, it's broken")
        assert state.mode == "debugging"

        # Learning mode
        state = model.infer_state("Can you explain how this works?")
        assert state.mode == "learning"

        # Creating mode
        state = model.infer_state("I want to create a new feature")
        assert state.mode == "creating"

    def test_urgency_detection(self):
        """Test detecting time pressure."""
        model = HumanModel({})

        # High urgency
        state = model.infer_state("I need this ASAP!")
        assert state.time_pressure in ("high", "urgent")

        # Low urgency
        state = model.infer_state("No rush, whenever you can")
        assert state.time_pressure == "low"

    def test_focus_estimation(self):
        """Test focus level estimation."""
        model = HumanModel({})

        # Long detailed message = high focus
        long_message = "I need help with this complex problem. " * 20
        state = model.infer_state(long_message)
        assert state.focus_level >= 0.8

        # Very short message = lower focus
        state = model.infer_state("ok")
        assert state.focus_level <= 0.6


class TestCouplingProtocols:
    """Tests for CouplingFacilitator."""

    def test_mode_selection(self):
        """Test automatic mode selection."""
        facilitator = CouplingFacilitator()

        # Teaching mode
        mode = facilitator.select_mode("Can you explain how this works?", None)
        assert mode == CouplingMode.TEACHING

        # Directing mode
        mode = facilitator.select_mode("Please run this command", None)
        assert mode == CouplingMode.DIRECTING

        # Parallel mode
        mode = facilitator.select_mode("Do some research while I work on this", None)
        assert mode == CouplingMode.PARALLEL

    def test_protocol_application(self):
        """Test protocol-specific response modifications."""
        facilitator = CouplingFacilitator()

        # Teaching mode should offer to explain reasoning
        response = facilitator._apply_teaching_protocol("Here's the answer.")
        assert "reasoning" in response.lower()

        # Parallel mode should add sync point
        response = facilitator._apply_parallel_protocol("I'll work on that.")
        assert "sync" in response.lower()

        # Directing mode should add confirmation
        response = facilitator._apply_directing_protocol("Doing that now.")
        assert response.startswith("Understood.")


class TestTrustTracker:
    """Tests for TrustTracker."""

    def test_initial_trust(self):
        """Test initial trust level."""
        tracker = TrustTracker({"initial": 0.5})
        assert tracker.level == 0.5

    def test_trust_increase(self):
        """Test trust increase on success."""
        tracker = TrustTracker({"initial": 0.5})

        tracker.record_event("successful_task_completion", delta=0.05)

        assert tracker.level == 0.55

    def test_trust_decrease(self):
        """Test trust decrease on error."""
        tracker = TrustTracker({"initial": 0.5})

        tracker.record_event("error", delta=-0.1)

        assert tracker.level == 0.4

    def test_trust_bounds(self):
        """Test trust stays within bounds."""
        tracker = TrustTracker({"initial": 0.5, "range": [0, 1]})

        # Try to exceed max
        for _ in range(20):
            tracker.record_event("success", delta=0.1)
        assert tracker.level <= 1.0

        # Try to go below min
        for _ in range(30):
            tracker.record_event("error", delta=-0.1)
        assert tracker.level >= 0.0

    def test_capability_thresholds(self):
        """Test capability thresholds."""
        tracker = TrustTracker({
            "initial": 0.5,
            "effects": [
                {"affects": "suggestion_confidence", "threshold": 0.7},
                {"affects": "proactive_behavior", "threshold": 0.8},
            ],
        })

        assert not tracker.can_suggest_confidently()
        assert not tracker.can_be_proactive()

        tracker.level = 0.75
        assert tracker.can_suggest_confidently()
        assert not tracker.can_be_proactive()

        tracker.level = 0.85
        assert tracker.can_suggest_confidently()
        assert tracker.can_be_proactive()
