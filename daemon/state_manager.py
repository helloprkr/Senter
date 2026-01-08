#!/usr/bin/env python3
"""
State Persistence for Crash Recovery

Saves component state periodically and restores on restart.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger('senter.state')


class StateManager:
    """
    Manages state persistence for daemon components.

    Features:
    - Periodic checkpointing
    - Atomic file writes
    - State age warnings
    - Clean shutdown handling
    """

    def __init__(self, state_dir: str = None):
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            self.state_dir = Path(__file__).parent.parent / "data" / "state"

        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = 30  # seconds
        self.last_checkpoint = 0
        self.max_state_age = 3600  # 1 hour

    def save_component_state(self, component: str, state: dict) -> bool:
        """
        Save state for a specific component.

        Uses atomic write (write to temp, then rename) to prevent corruption.
        """
        component_file = self.state_dir / f"{component}.json"

        data = {
            "component": component,
            "state": state,
            "saved_at": time.time(),
            "version": 1
        }

        try:
            # Write atomically
            temp_file = component_file.with_suffix('.tmp')
            temp_file.write_text(json.dumps(data, indent=2, default=str))
            temp_file.rename(component_file)

            logger.debug(f"Saved state for {component}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state for {component}: {e}")
            return False

    def load_component_state(self, component: str) -> Optional[dict]:
        """
        Load state for a specific component.

        Returns None if no state exists or state is corrupted.
        """
        component_file = self.state_dir / f"{component}.json"

        if not component_file.exists():
            return None

        try:
            data = json.loads(component_file.read_text())
            age = time.time() - data.get("saved_at", 0)

            if age > self.max_state_age:
                logger.warning(
                    f"State for {component} is {age/3600:.1f} hours old - may be stale"
                )

            logger.info(f"Loaded state for {component} (age: {age:.0f}s)")
            return data.get("state", {})

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted state file for {component}: {e}")
            # Backup corrupted file
            backup = component_file.with_suffix('.corrupted')
            component_file.rename(backup)
            return None

        except Exception as e:
            logger.error(f"Failed to load state for {component}: {e}")
            return None

    def checkpoint(self, components: dict[str, Any]) -> int:
        """
        Checkpoint all component states.

        Returns number of components checkpointed.
        """
        now = time.time()

        if now - self.last_checkpoint < self.checkpoint_interval:
            return 0

        saved = 0
        for name, component in components.items():
            if hasattr(component, 'get_state'):
                try:
                    state = component.get_state()
                    if self.save_component_state(name, state):
                        saved += 1
                except Exception as e:
                    logger.error(f"Error getting state from {name}: {e}")

        self.last_checkpoint = now
        if saved > 0:
            logger.info(f"Checkpoint completed: {saved} components")

        return saved

    def restore_all(self, components: dict[str, Any]) -> int:
        """
        Restore state for all components.

        Returns number of components restored.
        """
        restored = 0

        for name, component in components.items():
            if hasattr(component, 'restore_state'):
                state = self.load_component_state(name)
                if state:
                    try:
                        component.restore_state(state)
                        restored += 1
                        logger.info(f"Restored state for {name}")
                    except Exception as e:
                        logger.error(f"Error restoring state for {name}: {e}")

        return restored

    def clear_state(self, component: str = None):
        """
        Clear state files.

        Args:
            component: Specific component to clear, or None for all
        """
        if component:
            state_file = self.state_dir / f"{component}.json"
            if state_file.exists():
                state_file.unlink()
                logger.info(f"Cleared state for {component}")
        else:
            for f in self.state_dir.glob("*.json"):
                f.unlink()
            logger.info("Cleared all state files")

    def get_state_info(self) -> dict:
        """Get information about stored state"""
        info = {"components": {}}

        for f in self.state_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                age = time.time() - data.get("saved_at", 0)
                info["components"][f.stem] = {
                    "age_seconds": age,
                    "saved_at": data.get("saved_at"),
                    "size_bytes": f.stat().st_size
                }
            except:
                info["components"][f.stem] = {"error": "corrupted"}

        return info

    def has_saved_state(self) -> bool:
        """Check if there is any saved state"""
        return any(self.state_dir.glob("*.json"))


# DI-002: Full daemon state class
class DaemonState:
    """
    Captures complete daemon state for crash recovery (DI-002).

    Includes:
    - Session ID for tracking
    - Active goals
    - Pending tasks
    - Scheduler jobs
    - Component health status
    """

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.state_file = state_manager.state_dir / "daemon_state.json"
        self.session_id: str = ""
        self._state: dict = {}

    def capture(
        self,
        session_id: str,
        active_goals: list = None,
        pending_tasks: list = None,
        scheduler_jobs: list = None,
        component_status: dict = None
    ) -> dict:
        """Capture current daemon state."""
        import uuid

        self._state = {
            "session_id": session_id,
            "captured_at": time.time(),
            "active_goals": active_goals or [],
            "pending_tasks": pending_tasks or [],
            "scheduler_jobs": scheduler_jobs or [],
            "component_status": component_status or {},
            "recovery_info": {
                "tasks_in_flight": len(pending_tasks or []),
                "goals_active": len(active_goals or [])
            }
        }
        return self._state

    def save(self) -> bool:
        """Save daemon state to file."""
        if not self._state:
            return False

        try:
            temp_file = self.state_file.with_suffix('.tmp')
            temp_file.write_text(json.dumps(self._state, indent=2, default=str))
            temp_file.rename(self.state_file)
            logger.debug(f"Daemon state saved (session: {self._state.get('session_id', 'unknown')[:8]})")
            return True
        except Exception as e:
            logger.error(f"Failed to save daemon state: {e}")
            return False

    def load(self) -> Optional[dict]:
        """Load daemon state from file."""
        if not self.state_file.exists():
            return None

        try:
            state = json.loads(self.state_file.read_text())
            age = time.time() - state.get("captured_at", 0)
            logger.info(f"Loaded daemon state (age: {age:.0f}s, session: {state.get('session_id', 'unknown')[:8]})")
            return state
        except Exception as e:
            logger.error(f"Failed to load daemon state: {e}")
            return None

    def recover(self) -> dict:
        """
        Recover daemon state and return tasks to re-queue.

        Returns dict with:
        - pending_tasks: Tasks to re-execute
        - active_goals: Goals to continue
        - scheduler_jobs: Jobs to reschedule
        """
        state = self.load()
        if not state:
            return {"pending_tasks": [], "active_goals": [], "scheduler_jobs": []}

        recovery = {
            "pending_tasks": state.get("pending_tasks", []),
            "active_goals": state.get("active_goals", []),
            "scheduler_jobs": state.get("scheduler_jobs", []),
            "previous_session": state.get("session_id"),
            "recovered_at": time.time()
        }

        logger.info(
            f"Recovery: {len(recovery['pending_tasks'])} tasks, "
            f"{len(recovery['active_goals'])} goals to restore"
        )

        return recovery

    def clear(self):
        """Clear saved daemon state (normal shutdown)."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("Daemon state cleared (clean shutdown)")


# Mixin class for components to add state support
class StatefulComponent:
    """
    Mixin that adds state persistence to a component.

    Usage:
        class MyComponent(StatefulComponent):
            def get_state(self):
                return {"items": self.items}

            def restore_state(self, state):
                self.items = state.get("items", [])
    """

    def get_state(self) -> dict:
        """Override to return component state"""
        return {}

    def restore_state(self, state: dict):
        """Override to restore component state"""
        pass
