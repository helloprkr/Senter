#!/usr/bin/env python3
"""
Tests for State Manager (DI-002)
Tests crash recovery with state serialization and restoration.
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_state_manager_init():
    """Test StateManager initialization"""
    from daemon.state_manager import StateManager

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))

        assert manager.state_dir.exists()
        assert manager.checkpoint_interval == 30

        return True


def test_state_serialization():
    """Test saving and loading component state"""
    from daemon.state_manager import StateManager

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))

        # Save state
        test_state = {
            "items": [1, 2, 3],
            "config": {"enabled": True},
            "count": 42
        }
        result = manager.save_component_state("test_component", test_state)
        assert result is True

        # Verify file exists
        state_file = state_dir / "test_component.json"
        assert state_file.exists()

        # Load state
        loaded = manager.load_component_state("test_component")
        assert loaded is not None
        assert loaded["items"] == [1, 2, 3]
        assert loaded["config"]["enabled"] is True
        assert loaded["count"] == 42

        return True


def test_state_restoration():
    """Test state restoration for components"""
    from daemon.state_manager import StateManager, StatefulComponent

    class TestComponent(StatefulComponent):
        def __init__(self):
            self.items = []
            self.value = 0

        def get_state(self):
            return {"items": self.items, "value": self.value}

        def restore_state(self, state):
            self.items = state.get("items", [])
            self.value = state.get("value", 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))

        # Create and save component state
        comp = TestComponent()
        comp.items = ["a", "b", "c"]
        comp.value = 100

        manager.save_component_state("test", comp.get_state())

        # Create new component and restore
        comp2 = TestComponent()
        state = manager.load_component_state("test")
        comp2.restore_state(state)

        assert comp2.items == ["a", "b", "c"]
        assert comp2.value == 100

        return True


def test_daemon_state_capture():
    """Test DaemonState capture"""
    from daemon.state_manager import StateManager, DaemonState

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))
        daemon_state = DaemonState(manager)

        # Capture state
        state = daemon_state.capture(
            session_id="test-session-123",
            active_goals=["goal_1", "goal_2"],
            pending_tasks=[{"id": "task_1"}, {"id": "task_2"}],
            scheduler_jobs=["job_1"],
            component_status={"audio": "healthy"}
        )

        assert state["session_id"] == "test-session-123"
        assert len(state["active_goals"]) == 2
        assert len(state["pending_tasks"]) == 2
        assert state["recovery_info"]["tasks_in_flight"] == 2

        return True


def test_daemon_state_save_load():
    """Test DaemonState save and load"""
    from daemon.state_manager import StateManager, DaemonState

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))
        daemon_state = DaemonState(manager)

        # Capture and save
        daemon_state.capture(
            session_id="save-load-test",
            active_goals=["goal_x"],
            pending_tasks=[{"id": "task_x"}]
        )
        result = daemon_state.save()
        assert result is True

        # Load with new instance
        daemon_state2 = DaemonState(manager)
        loaded = daemon_state2.load()

        assert loaded is not None
        assert loaded["session_id"] == "save-load-test"
        assert loaded["active_goals"] == ["goal_x"]

        return True


def test_daemon_state_recovery():
    """Test DaemonState recovery for re-queuing tasks"""
    from daemon.state_manager import StateManager, DaemonState

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))
        daemon_state = DaemonState(manager)

        # Simulate crash: save state with pending tasks
        daemon_state.capture(
            session_id="crashed-session",
            active_goals=["goal_recovery"],
            pending_tasks=[
                {"id": "task_1", "description": "Task 1"},
                {"id": "task_2", "description": "Task 2"}
            ],
            scheduler_jobs=["daily_report"]
        )
        daemon_state.save()

        # New session: recover
        daemon_state2 = DaemonState(manager)
        recovery = daemon_state2.recover()

        assert recovery["previous_session"] == "crashed-session"
        assert len(recovery["pending_tasks"]) == 2
        assert len(recovery["active_goals"]) == 1
        assert len(recovery["scheduler_jobs"]) == 1
        assert "recovered_at" in recovery

        return True


def test_daemon_state_clear():
    """Test DaemonState clear on clean shutdown"""
    from daemon.state_manager import StateManager, DaemonState

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))
        daemon_state = DaemonState(manager)

        # Save state
        daemon_state.capture(session_id="to-clear")
        daemon_state.save()

        assert daemon_state.state_file.exists()

        # Clear (clean shutdown)
        daemon_state.clear()

        assert not daemon_state.state_file.exists()

        return True


def test_state_age_warning():
    """Test that old state triggers warning"""
    from daemon.state_manager import StateManager
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))

        # Create artificially old state
        state_file = state_dir / "old_component.json"
        old_state = {
            "component": "old_component",
            "state": {"data": "old"},
            "saved_at": time.time() - 7200,  # 2 hours ago
            "version": 1
        }
        state_file.write_text(json.dumps(old_state))

        # Load should work but may log warning
        loaded = manager.load_component_state("old_component")
        assert loaded is not None
        assert loaded["data"] == "old"

        return True


def test_corrupted_state_handling():
    """Test handling of corrupted state files"""
    from daemon.state_manager import StateManager

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))

        # Create corrupted state file
        state_file = state_dir / "corrupted.json"
        state_file.write_text("{ invalid json }")

        # Load should return None and backup file
        loaded = manager.load_component_state("corrupted")
        assert loaded is None

        # Original should be renamed to .corrupted
        assert state_file.with_suffix('.corrupted').exists()

        return True


def test_has_saved_state():
    """Test has_saved_state check"""
    from daemon.state_manager import StateManager

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(str(state_dir))

        # Initially no state
        assert manager.has_saved_state() is False

        # Save some state
        manager.save_component_state("test", {"data": 1})

        # Now should have state
        assert manager.has_saved_state() is True

        return True


if __name__ == "__main__":
    tests = [
        test_state_manager_init,
        test_state_serialization,
        test_state_restoration,
        test_daemon_state_capture,
        test_daemon_state_save_load,
        test_daemon_state_recovery,
        test_daemon_state_clear,
        test_state_age_warning,
        test_corrupted_state_handling,
        test_has_saved_state,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
