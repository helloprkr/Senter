#!/usr/bin/env python3
"""
IPC Server for Daemon Communication

Provides Unix socket server for CLI tools to communicate with the running daemon.
"""

import json
import socket
import threading
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, Callable
from multiprocessing import Queue, Event
from queue import Empty

logger = logging.getLogger('senter.ipc')

# Default socket path
DEFAULT_SOCKET_PATH = "/tmp/senter.sock"


class IPCServer:
    """
    Unix socket server for daemon IPC.

    Handles:
    - Query requests → Model workers
    - Status requests → Health monitor
    - Goal requests → Task engine
    - Progress requests → Reporter
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        shutdown_event: Event = None,
        daemon_ref = None
    ):
        self.socket_path = Path(socket_path)
        self.shutdown_event = shutdown_event or Event()
        self.daemon = daemon_ref

        self.server_socket = None
        self.pending_requests: dict[str, dict] = {}
        self.request_timeout = 60.0  # seconds

        # Handler registry
        self.handlers: dict[str, Callable] = {
            "query": self._handle_query,
            "status": self._handle_status,
            "goals": self._handle_goals,
            "progress": self._handle_progress,
            "schedule": self._handle_schedule,
            "health": self._handle_health,
            "config": self._handle_config,
            "add_research_task": self._handle_add_research_task,  # US-002
            "research_queue_status": self._handle_research_queue_status,  # US-002
            "get_results": self._handle_get_results,  # US-003
        }

    def run(self):
        """Main server loop"""
        self._cleanup_socket()
        self._create_socket()

        logger.info(f"IPC server listening on {self.socket_path}")

        while not self.shutdown_event.is_set():
            try:
                self.server_socket.settimeout(1.0)
                try:
                    conn, addr = self.server_socket.accept()
                    # Handle client in thread
                    threading.Thread(
                        target=self._handle_client,
                        args=(conn,),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
            except Exception as e:
                if not self.shutdown_event.is_set():
                    logger.error(f"IPC server error: {e}")

        self._cleanup_socket()
        logger.info("IPC server stopped")

    def _create_socket(self):
        """Create Unix socket"""
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(str(self.socket_path))
        self.server_socket.listen(5)

    def _cleanup_socket(self):
        """Remove socket file"""
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except:
                pass

    def _handle_client(self, conn: socket.socket):
        """Handle a client connection"""
        try:
            conn.settimeout(self.request_timeout)

            # Read request
            data = self._recv_all(conn)
            if not data:
                return

            request = json.loads(data.decode())
            request_id = request.get("id", str(uuid.uuid4())[:8])

            logger.debug(f"IPC request [{request_id}]: {request.get('command')}")

            # Process request
            response = self._process_request(request)
            response["id"] = request_id

            # Send response
            conn.sendall(json.dumps(response).encode())

        except socket.timeout:
            error_response = {"error": "Request timed out", "code": "TIMEOUT"}
            conn.sendall(json.dumps(error_response).encode())
        except json.JSONDecodeError as e:
            error_response = {"error": f"Invalid JSON: {e}", "code": "PARSE_ERROR"}
            conn.sendall(json.dumps(error_response).encode())
        except Exception as e:
            logger.error(f"Client error: {e}")
            error_response = {"error": str(e), "code": "SERVER_ERROR"}
            try:
                conn.sendall(json.dumps(error_response).encode())
            except:
                pass
        finally:
            conn.close()

    def _recv_all(self, conn: socket.socket, buffer_size: int = 65536) -> bytes:
        """Receive all data from socket"""
        data = b""
        while True:
            chunk = conn.recv(buffer_size)
            if not chunk:
                break
            data += chunk
            if len(chunk) < buffer_size:
                break
        return data

    def _process_request(self, request: dict) -> dict:
        """Process an IPC request"""
        cmd = request.get("command")

        if not cmd:
            return {"error": "Missing 'command' field", "code": "INVALID_REQUEST"}

        handler = self.handlers.get(cmd)
        if not handler:
            return {"error": f"Unknown command: {cmd}", "code": "UNKNOWN_COMMAND"}

        try:
            return handler(request)
        except Exception as e:
            logger.error(f"Handler error for {cmd}: {e}")
            return {"error": str(e), "code": "HANDLER_ERROR"}

    def _handle_query(self, request: dict) -> dict:
        """Handle query request"""
        text = request.get("text")
        if not text:
            return {"error": "Missing 'text' field"}

        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            # Send to model worker
            request_id = str(uuid.uuid4())[:8]
            start_time = time.time()

            # Put query in model queue
            if "model_primary" in self.daemon.queues:
                self.daemon.queues["model_primary"].put({
                    "type": "user_query",
                    "id": request_id,
                    "payload": {"text": text}
                }, timeout=5.0)
            else:
                return {"error": "Model worker not available"}

            # Wait for response from output queue
            output_queue = self.daemon.output_queues.get("model_primary")
            if output_queue:
                try:
                    response_msg = output_queue.get(timeout=self.request_timeout)
                    latency = time.time() - start_time

                    return {
                        "response": response_msg.get("payload", {}).get("response", ""),
                        "latency": latency,
                        "worker": response_msg.get("source", "unknown")
                    }
                except Empty:
                    return {"error": "Request timed out waiting for model"}
            else:
                return {"error": "Output queue not available"}

        except Exception as e:
            return {"error": str(e)}

    def _handle_status(self, request: dict = None) -> dict:
        """Handle status request"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            status = {
                "running": True,
                "pid": self.daemon.pid if hasattr(self.daemon, 'pid') else None,
                "uptime": time.time() - self.daemon.start_time if hasattr(self.daemon, 'start_time') and self.daemon.start_time else 0,
                "components": {}
            }

            # Get component status from health monitor
            if hasattr(self.daemon, 'health_monitor') and self.daemon.health_monitor:
                try:
                    health = self.daemon.health_monitor.get_status()
                    status["components"] = health.get("components", {})
                    status["overall_health"] = health.get("overall", False)
                except Exception as he:
                    logger.error(f"Health monitor error: {he}")
                    status["health_error"] = str(he) or repr(he)

            # Add queue sizes
            if hasattr(self.daemon, 'queues') and self.daemon.queues:
                try:
                    status["queue_sizes"] = {}
                    for name, q in self.daemon.queues.items():
                        try:
                            status["queue_sizes"][name] = q.qsize()
                        except:
                            status["queue_sizes"][name] = -1
                except Exception as qe:
                    logger.error(f"Queue size error: {qe}")

            return status

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Status handler error: {e}\n{tb}")
            return {"error": str(e) or repr(e) or "Unknown error"}

    def _handle_goals(self, request: dict = None) -> dict:
        """Handle goals request"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            # Send request to task engine
            if "task_engine" in self.daemon.queues:
                request_id = str(uuid.uuid4())[:8]
                self.daemon.queues["task_engine"].put({
                    "type": "get_goals",
                    "id": request_id
                }, timeout=5.0)

                # Wait for response
                output_queue = self.daemon.output_queues.get("task_engine")
                if output_queue:
                    try:
                        response = output_queue.get(timeout=10.0)
                        return {"goals": response.get("payload", {}).get("goals", [])}
                    except Empty:
                        return {"goals": [], "note": "Task engine did not respond"}

            return {"goals": []}

        except Exception as e:
            return {"error": str(e)}

    def _handle_progress(self, request: dict = None) -> dict:
        """Handle progress request"""
        hours = request.get("hours", 24) if request else 24

        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            # Send request to reporter
            if "reporter" in self.daemon.queues:
                request_id = str(uuid.uuid4())[:8]
                self.daemon.queues["reporter"].put({
                    "type": "get_progress",
                    "id": request_id,
                    "payload": {"hours": hours}
                }, timeout=5.0)

                # Wait for response
                output_queue = self.daemon.output_queues.get("reporter")
                if output_queue:
                    try:
                        response = output_queue.get(timeout=10.0)
                        return response.get("payload", {})
                    except Empty:
                        return {"summary": "No progress data available"}

            return {"summary": "Reporter not available"}

        except Exception as e:
            return {"error": str(e)}

    def _handle_schedule(self, request: dict) -> dict:
        """Handle schedule request"""
        action = request.get("action", "list")

        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            if "scheduler" in self.daemon.queues:
                self.daemon.queues["scheduler"].put({
                    "type": "schedule_" + action,
                    "payload": request.get("payload", {})
                }, timeout=5.0)

                if action == "list":
                    output_queue = self.daemon.output_queues.get("scheduler")
                    if output_queue:
                        try:
                            response = output_queue.get(timeout=10.0)
                            return {"jobs": response.get("payload", {}).get("jobs", [])}
                        except Empty:
                            return {"jobs": []}

                return {"status": "ok"}

            return {"error": "Scheduler not available"}

        except Exception as e:
            return {"error": str(e)}

    def _handle_health(self, request: dict = None) -> dict:
        """Handle health check request"""
        return {
            "healthy": True,
            "timestamp": time.time()
        }

    def _handle_config(self, request: dict = None) -> dict:
        """Handle config request"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            if hasattr(self.daemon, 'config'):
                return {"config": self.daemon.config}
            return {"config": {}}
        except Exception as e:
            return {"error": str(e)}

    def _handle_add_research_task(self, request: dict) -> dict:
        """Handle add_research_task request (US-002)"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        description = request.get("description")
        if not description:
            return {"error": "Missing 'description' field"}

        try:
            task = {
                "id": request.get("id", str(uuid.uuid4())[:8]),
                "description": description,
                "priority": request.get("priority", 5),
                "source": request.get("source", "ipc"),
            }

            success = self.daemon.add_research_task(task)
            if success:
                return {"status": "ok", "task_id": task["id"]}
            else:
                return {"error": "Failed to add task to queue"}

        except Exception as e:
            return {"error": str(e)}

    def _handle_research_queue_status(self, request: dict = None) -> dict:
        """Handle research_queue_status request (US-002)"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            return self.daemon.get_research_queue_status()
        except Exception as e:
            return {"error": str(e)}

    def _handle_get_results(self, request: dict) -> dict:
        """Handle get_results request (US-003)"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from engine.task_results import TaskResultStorage
            from pathlib import Path

            # Get result storage path from daemon
            results_dir = Path(self.daemon.senter_root) / "data" / "tasks" / "results"
            storage = TaskResultStorage(results_dir)

            # Query parameters
            task_id = request.get("task_id")
            goal_id = request.get("goal_id")
            limit = request.get("limit", 20)
            hours = request.get("hours")

            # Query by task_id
            if task_id:
                result = storage.get_by_task_id(task_id)
                if result:
                    return {"result": result.to_dict()}
                else:
                    return {"error": f"No result found for task {task_id}"}

            # Query by goal_id
            elif goal_id:
                results = storage.get_by_goal_id(goal_id)
                return {
                    "results": [r.to_dict() for r in results],
                    "count": len(results)
                }

            # Recent results
            else:
                import time
                since = time.time() - (hours * 3600) if hours else None
                results = storage.get_recent(limit=limit, since=since)
                summary = storage.get_summary()
                return {
                    "results": [r.to_dict() for r in results],
                    "count": len(results),
                    "summary": summary
                }

        except Exception as e:
            return {"error": str(e)}
