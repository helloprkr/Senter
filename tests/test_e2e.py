#!/usr/bin/env python3
"""
End-to-End Integration Tests

Tests complete flows that exercise multiple components together.
Run with: python3 -m pytest tests/test_e2e.py -v --timeout=120
"""

import os
import sys
import time
import json
import signal
import subprocess
from pathlib import Path
from multiprocessing import Process, Queue

import pytest

# Setup path
SENTER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SENTER_ROOT))


class DaemonManager:
    """Helper to manage daemon for tests"""

    def __init__(self):
        self.pid = None
        self.pid_file = SENTER_ROOT / "data" / "senter.pid"

    def start(self, timeout: float = 10.0) -> bool:
        """Start daemon and wait for it to be ready"""
        if self.is_running():
            return True

        # Start daemon
        cmd = [sys.executable, "scripts/senter_ctl.py", "start"]
        subprocess.run(cmd, cwd=str(SENTER_ROOT), capture_output=True)

        # Wait for startup
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                # Wait a bit more for components
                time.sleep(2)
                return True
            time.sleep(0.5)

        return False

    def stop(self, timeout: float = 10.0) -> bool:
        """Stop daemon"""
        if not self.is_running():
            return True

        pid = self._get_pid()
        if pid:
            os.kill(pid, signal.SIGTERM)

        # Wait for stop
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.is_running():
                return True
            time.sleep(0.5)

        # Force kill
        if self.is_running():
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)

        return not self.is_running()

    def is_running(self) -> bool:
        """Check if daemon is running"""
        pid = self._get_pid()
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, OSError):
            return False

    def _get_pid(self) -> int:
        """Get daemon PID"""
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except:
            return None


@pytest.fixture(scope="module")
def daemon():
    """Fixture to ensure daemon is running for tests"""
    manager = DaemonManager()

    # Start if needed
    if not manager.is_running():
        assert manager.start(timeout=15), "Failed to start daemon"
        # Wait for health check
        time.sleep(5)

    yield manager

    # Don't stop after tests - leave running for other tests


@pytest.fixture
def ipc_client():
    """Fixture for IPC client"""
    from daemon.ipc_client import IPCClient
    return IPCClient()


class TestQueryFlow:
    """Tests for query → model → response flow"""

    def test_basic_query(self, daemon, ipc_client):
        """Test a simple query gets a response"""
        result = ipc_client.query("What is 1+1?")

        assert "error" not in result, f"Query failed: {result}"
        assert "response" in result
        assert len(result["response"]) > 0
        assert result.get("latency", 0) > 0

    def test_query_with_context(self, daemon, ipc_client):
        """Test query with more complex context"""
        result = ipc_client.query(
            "Explain in one sentence: Why is the sky blue?"
        )

        assert "error" not in result
        assert "response" in result
        assert len(result["response"]) > 10  # Should have meaningful response

    def test_multiple_queries(self, daemon, ipc_client):
        """Test multiple sequential queries"""
        queries = [
            "What is 2+2?",
            "What is 3+3?",
            "What is 4+4?"
        ]

        for query in queries:
            result = ipc_client.query(query)
            assert "error" not in result, f"Query '{query}' failed: {result}"
            assert "response" in result


class TestStatusFlow:
    """Tests for status and health checks"""

    def test_status_returns_data(self, daemon, ipc_client):
        """Test status endpoint returns expected data"""
        result = ipc_client.status()

        assert "error" not in result or result.get("error") == "", \
            f"Status failed: {result}"
        assert result.get("running") == True
        assert "components" in result
        assert "pid" in result

    def test_health_check(self, daemon, ipc_client):
        """Test health check endpoint"""
        result = ipc_client.health()

        assert result.get("healthy") == True
        assert "timestamp" in result

    def test_all_components_healthy(self, daemon, ipc_client):
        """Test all components are running"""
        result = ipc_client.status()

        components = result.get("components", {})
        assert len(components) >= 6, "Not all components registered"

        # Allow some time for health check to run
        time.sleep(1)

        # Re-check
        result = ipc_client.status()
        components = result.get("components", {})

        for name, info in components.items():
            assert info.get("is_alive") or info.get("pid"), \
                f"Component {name} not healthy"


class TestGoalFlow:
    """Tests for goal creation and task engine"""

    def test_goals_endpoint(self, daemon, ipc_client):
        """Test goals endpoint responds"""
        result = ipc_client.goals()

        # Should return empty list or list of goals
        assert "error" not in result or "goals" in result


class TestProgressFlow:
    """Tests for progress reporting"""

    def test_progress_endpoint(self, daemon, ipc_client):
        """Test progress endpoint responds"""
        result = ipc_client.progress(hours=24)

        # Should return some data
        assert isinstance(result, dict)


class TestFailureRecovery:
    """Tests for failure handling and recovery"""

    def test_daemon_survives_bad_query(self, daemon, ipc_client):
        """Test daemon survives malformed input"""
        # Send very long query
        long_text = "test " * 1000
        result = ipc_client.query(long_text)

        # Should either process or return error, not crash
        assert isinstance(result, dict)

        # Daemon should still be running
        assert daemon.is_running()

        # Should still respond to normal queries
        result = ipc_client.health()
        assert result.get("healthy") == True


class TestGracefulShutdown:
    """Tests for graceful shutdown"""

    @pytest.mark.skip(reason="Would stop daemon needed by other tests")
    def test_graceful_shutdown(self, daemon, ipc_client):
        """Test daemon stops gracefully"""
        # Verify running
        assert daemon.is_running()

        # Stop
        assert daemon.stop(timeout=15)

        # Verify stopped
        assert not daemon.is_running()

        # Restart for other tests
        assert daemon.start(timeout=15)


class TestConcurrentQueries:
    """Tests for concurrent query handling"""

    def test_two_concurrent_queries(self, daemon, ipc_client):
        """Test two queries can run concurrently"""
        import concurrent.futures

        queries = [
            "What is the capital of France?",
            "What is the capital of Germany?"
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(ipc_client.query, q) for q in queries]
            results = [f.result(timeout=120) for f in futures]

        for result in results:
            assert "error" not in result, f"Query failed: {result}"
            assert "response" in result


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=120"])
