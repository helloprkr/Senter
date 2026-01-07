#!/usr/bin/env python3
"""
Health Monitor

Watches all daemon components and handles failures.
Provides health status and restart capabilities.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional
from multiprocessing import Process
from datetime import datetime

logger = logging.getLogger('senter.health')


@dataclass
class ComponentHealth:
    """Health status for a single component"""
    name: str
    is_alive: bool = False
    last_check: float = 0
    last_response: float = 0
    restart_count: int = 0
    last_restart: Optional[float] = None
    errors: list[str] = field(default_factory=list)
    pid: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_alive": self.is_alive,
            "last_check": datetime.fromtimestamp(self.last_check).isoformat() if self.last_check else None,
            "last_response": datetime.fromtimestamp(self.last_response).isoformat() if self.last_response else None,
            "restart_count": self.restart_count,
            "pid": self.pid,
            "recent_errors": self.errors[-3:] if self.errors else []
        }


class HealthMonitor:
    """
    Monitors health of all daemon components.

    Features:
    - Process liveness checks
    - Health endpoint responses
    - Automatic restart on failure
    - Restart rate limiting
    """

    def __init__(
        self,
        check_interval: int = 30,
        max_restarts: int = 5,
        restart_window: int = 300
    ):
        self.check_interval = check_interval
        self.max_restarts = max_restarts
        self.restart_window = restart_window  # seconds

        self.components: dict[str, ComponentHealth] = {}
        self.processes: dict[str, Process] = {}
        self.restart_callbacks: dict[str, Callable] = {}
        self.restart_times: dict[str, list[float]] = {}

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def register_component(
        self,
        name: str,
        process: Optional[Process] = None,
        restart_callback: Optional[Callable] = None
    ):
        """Register a component for monitoring"""
        with self._lock:
            self.components[name] = ComponentHealth(name=name)
            if process:
                self.processes[name] = process
                self.components[name].pid = process.pid
            if restart_callback:
                self.restart_callbacks[name] = restart_callback
            self.restart_times[name] = []
            logger.info(f"Registered component for monitoring: {name}")

    def unregister_component(self, name: str):
        """Unregister a component"""
        with self._lock:
            self.components.pop(name, None)
            self.processes.pop(name, None)
            self.restart_callbacks.pop(name, None)
            self.restart_times.pop(name, None)

    def start(self):
        """Start the health monitor"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitor started")

    def stop(self):
        """Stop the health monitor"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Health monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            self._check_all_components()
            time.sleep(self.check_interval)

    def _check_all_components(self):
        """Check health of all registered components"""
        now = time.time()

        with self._lock:
            for name, health in list(self.components.items()):
                health.last_check = now

                # Check process if registered
                if name in self.processes:
                    process = self.processes[name]
                    health.is_alive = process.is_alive()
                    health.pid = process.pid if process.is_alive() else None

                    if not health.is_alive:
                        logger.warning(f"Component {name} is not alive")
                        self._handle_dead_component(name)

    def _handle_dead_component(self, name: str):
        """Handle a dead component"""
        health = self.components.get(name)
        if not health:
            return

        now = time.time()

        # Check restart limits
        if name not in self.restart_times:
            self.restart_times[name] = []

        # Remove old restart times
        self.restart_times[name] = [
            t for t in self.restart_times[name]
            if now - t < self.restart_window
        ]

        if len(self.restart_times[name]) >= self.max_restarts:
            error_msg = f"Component {name} exceeded max restarts ({self.max_restarts} in {self.restart_window}s)"
            logger.error(error_msg)
            health.errors.append(error_msg)
            return

        # Attempt restart
        if name in self.restart_callbacks:
            logger.info(f"Attempting to restart {name}")
            self.restart_times[name].append(now)
            health.restart_count += 1
            health.last_restart = now

            try:
                # Call restart callback
                new_process = self.restart_callbacks[name]()
                if new_process and isinstance(new_process, Process):
                    self.processes[name] = new_process
                    health.pid = new_process.pid
                    health.is_alive = True
                    logger.info(f"Successfully restarted {name}")
            except Exception as e:
                error_msg = f"Failed to restart {name}: {e}"
                logger.error(error_msg)
                health.errors.append(error_msg)

    def report_alive(self, name: str):
        """Report that a component is alive (called by component)"""
        with self._lock:
            if name in self.components:
                self.components[name].last_response = time.time()
                self.components[name].is_alive = True

    def report_error(self, name: str, error: str):
        """Report an error from a component"""
        with self._lock:
            if name in self.components:
                self.components[name].errors.append(f"{datetime.now().isoformat()}: {error}")
                # Keep only last 10 errors
                self.components[name].errors = self.components[name].errors[-10:]

    def get_status(self) -> dict:
        """Get health status of all components"""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "overall": all(h.is_alive for h in self.components.values()),
                "components": {
                    name: health.to_dict()
                    for name, health in self.components.items()
                }
            }

    def is_healthy(self, name: str = None) -> bool:
        """Check if a component (or all components) are healthy"""
        with self._lock:
            if name:
                return name in self.components and self.components[name].is_alive
            return all(h.is_alive for h in self.components.values())


# Test
if __name__ == "__main__":
    import multiprocessing

    logging.basicConfig(level=logging.DEBUG)

    def dummy_worker():
        """Dummy worker for testing"""
        time.sleep(10)

    monitor = HealthMonitor(check_interval=2)

    # Create a test process
    p = multiprocessing.Process(target=dummy_worker)
    p.start()

    monitor.register_component("test_worker", process=p)
    monitor.start()

    # Check status
    time.sleep(1)
    print(f"Status: {monitor.get_status()}")

    # Stop test
    p.terminate()
    time.sleep(3)
    print(f"After terminate: {monitor.get_status()}")

    monitor.stop()
    print("Test complete")
