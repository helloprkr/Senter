#!/usr/bin/env python3
"""
IPC Client for CLI Tools

Provides a simple interface to communicate with the running Senter daemon.
"""

import json
import socket
import time
from pathlib import Path
from typing import Optional

# Default socket path
DEFAULT_SOCKET_PATH = "/tmp/senter.sock"


class IPCClient:
    """
    Client for communicating with Senter daemon via Unix socket.

    Usage:
        client = IPCClient()
        result = client.query("What is Python?")
        print(result)
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = Path(socket_path)
        self.timeout = 60.0  # seconds

    def is_daemon_running(self) -> bool:
        """Check if daemon is running"""
        return self.socket_path.exists()

    def send(self, command: str, **kwargs) -> dict:
        """
        Send command to daemon and get response.

        Args:
            command: Command name (query, status, goals, progress, etc.)
            **kwargs: Additional parameters for the command

        Returns:
            Response dictionary from daemon
        """
        if not self.socket_path.exists():
            return {"error": "Daemon not running", "code": "NOT_RUNNING"}

        request = {"command": command, **kwargs}

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(self.socket_path))
            sock.settimeout(self.timeout)

            # Send request
            sock.sendall(json.dumps(request).encode())

            # Receive response
            response_data = self._recv_all(sock)
            return json.loads(response_data.decode())

        except socket.timeout:
            return {"error": "Request timed out", "code": "TIMEOUT"}
        except ConnectionRefusedError:
            return {"error": "Connection refused - daemon may be starting", "code": "REFUSED"}
        except Exception as e:
            return {"error": str(e), "code": "CLIENT_ERROR"}
        finally:
            sock.close()

    def _recv_all(self, sock: socket.socket, buffer_size: int = 65536) -> bytes:
        """Receive all data from socket"""
        data = b""
        while True:
            chunk = sock.recv(buffer_size)
            if not chunk:
                break
            data += chunk
            if len(chunk) < buffer_size:
                break
        return data

    # Convenience methods

    def query(self, text: str) -> dict:
        """Send a query to the model"""
        return self.send("query", text=text)

    def status(self) -> dict:
        """Get daemon status"""
        return self.send("status")

    def health(self) -> dict:
        """Quick health check"""
        return self.send("health")

    def goals(self) -> dict:
        """Get active goals"""
        return self.send("goals")

    def progress(self, hours: int = 24) -> dict:
        """Get progress report"""
        return self.send("progress", hours=hours)

    def schedule_list(self) -> dict:
        """List scheduled jobs"""
        return self.send("schedule", action="list")

    def schedule_add(self, job: dict) -> dict:
        """Add a scheduled job"""
        return self.send("schedule", action="add", payload=job)

    def config(self) -> dict:
        """Get daemon configuration"""
        return self.send("config")

    def add_research_task(self, description: str, priority: int = 5) -> dict:
        """Add a research task to the background queue (US-002)"""
        return self.send("add_research_task", description=description, priority=priority)

    def research_queue_status(self) -> dict:
        """Get research queue status (US-002)"""
        return self.send("research_queue_status")

    def get_results(self, task_id: str = None, goal_id: str = None,
                    limit: int = 20, hours: int = None) -> dict:
        """Get task results (US-003)"""
        return self.send("get_results", task_id=task_id, goal_id=goal_id,
                        limit=limit, hours=hours)


# Convenience function for scripts
def get_client(socket_path: str = DEFAULT_SOCKET_PATH) -> IPCClient:
    """Get an IPC client instance"""
    return IPCClient(socket_path)


# CLI test
if __name__ == "__main__":
    import sys

    client = IPCClient()

    if not client.is_daemon_running():
        print("Daemon is not running")
        sys.exit(1)

    # Test commands
    print("Testing status...")
    result = client.status()
    print(json.dumps(result, indent=2))

    print("\nTesting health...")
    result = client.health()
    print(json.dumps(result, indent=2))

    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        print(f"\nTesting query: {query_text}")
        result = client.query(query_text)
        print(json.dumps(result, indent=2))
