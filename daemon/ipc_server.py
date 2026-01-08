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
            "trigger_research": self._handle_trigger_research,  # US-005
            "activity_report": self._handle_activity_report,  # US-006
            "get_events": self._handle_get_events,  # US-008
            "get_conversations": self._handle_get_conversations,  # P1-001
            "save_conversation": self._handle_save_conversation,  # P1-001
            "get_tasks": self._handle_get_tasks,  # P2-001
            "get_journal": self._handle_get_journal,  # P2-004
            "generate_journal": self._handle_generate_journal,  # P2-004
            "get_context_sources": self._handle_get_context_sources,  # P3-001
            "add_context_source": self._handle_add_context_source,  # P3-001
            "remove_context_source": self._handle_remove_context_source,  # P3-001
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

            # Detect topic using NLP-like extraction (US-008)
            topic = self._detect_topic(text)

            # Log query to events database (US-008)
            self._log_user_query(text, topic)

            # P2-002: Check for research intent and create task instead
            research_intent = self._detect_research_intent(text)
            if research_intent:
                # Create research task instead of responding immediately
                task_result = self._handle_add_research_task({
                    "description": research_intent["description"],
                    "goal_id": "chat_research"
                })

                if "error" not in task_result:
                    task_id = task_result.get("task_id", "unknown")
                    return {
                        "response": f"I'll research that for you. I've created a task: '{research_intent['description']}'. Check the Tasks tab to see the progress and results.",
                        "latency": time.time() - start_time,
                        "worker": "task_creator",
                        "topic": "research",
                        "task_created": True,
                        "task_id": task_id
                    }

            # P1-003: Build context from conversation history
            conversation_id = request.get("conversation_id")
            context_prompt = self._build_conversation_context(conversation_id, text)

            # P3-002: Build context from pinned sources
            sources_context = self._build_context_sources_prompt()

            # Build system prompt with context sources
            system_prompt = "You are Senter, a helpful AI assistant. You remember the context of our conversation."
            if sources_context:
                system_prompt += f"\n\n{sources_context}"

            # Put query in model queue
            if "model_primary" in self.daemon.queues:
                self.daemon.queues["model_primary"].put({
                    "type": "user_query",
                    "id": request_id,
                    "payload": {
                        "text": context_prompt if context_prompt else text,
                        "system_prompt": system_prompt
                    },
                    "correlation_id": request_id
                }, timeout=5.0)
            else:
                return {"error": "Model worker not available"}

            # Wait for response from output queue
            output_queue = self.daemon.output_queues.get("model_primary")
            if output_queue:
                try:
                    response_msg = output_queue.get(timeout=self.request_timeout)
                    latency = time.time() - start_time
                    latency_ms = int(latency * 1000)

                    response_text = response_msg.get("payload", {}).get("response", "")
                    worker = response_msg.get("source", "unknown")

                    # Log response to events database (US-008)
                    self._log_user_response(text, response_text, latency_ms, worker, topic)

                    return {
                        "response": response_text,
                        "latency": latency,
                        "worker": worker,
                        "topic": topic
                    }
                except Empty:
                    return {"error": "Request timed out waiting for model"}
            else:
                return {"error": "Output queue not available"}

        except Exception as e:
            return {"error": str(e)}

    def _detect_topic(self, text: str) -> str:
        """Detect topic from text using NLP-like extraction (US-008)"""
        text_lower = text.lower()

        # Topic detection with patterns (more sophisticated than simple keywords)
        topic_patterns = {
            "coding": [
                r"\b(code|coding|program|debug|function|class|variable)\b",
                r"\b(python|javascript|java|c\+\+|rust|go|typescript)\b",
                r"\b(error|exception|bug|fix|compile|runtime)\b",
                r"\b(api|library|framework|module|package)\b"
            ],
            "research": [
                r"\b(research|study|analyze|investigate|explore)\b",
                r"\b(data|statistics|findings|results|report)\b",
                r"\b(compare|contrast|evaluate|assess)\b"
            ],
            "writing": [
                r"\b(write|draft|compose|edit|proofread)\b",
                r"\b(essay|article|blog|document|email)\b",
                r"\b(paragraph|sentence|grammar|style)\b"
            ],
            "creative": [
                r"\b(creative|story|poem|design|art|music)\b",
                r"\b(imagine|brainstorm|idea|concept)\b"
            ],
            "productivity": [
                r"\b(task|todo|schedule|plan|organize)\b",
                r"\b(goal|deadline|priority|workflow)\b",
                r"\b(time|manage|efficient|productive)\b"
            ],
            "learning": [
                r"\b(learn|understand|explain|teach|tutorial)\b",
                r"\b(how\s+(do|does|to|can)|what\s+is|why\s+is)\b",
                r"\b(concept|theory|practice|example)\b"
            ],
            "general": []  # Fallback
        }

        import re
        matched_topics = []
        for topic, patterns in topic_patterns.items():
            if patterns:
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        matched_topics.append(topic)
                        break

        # Return most specific topic or general
        if matched_topics:
            # Priority order: coding > research > learning > writing > creative > productivity
            priority = ["coding", "research", "learning", "writing", "creative", "productivity"]
            for p in priority:
                if p in matched_topics:
                    return p
            return matched_topics[0]

        return "general"

    def _detect_research_intent(self, text: str) -> dict | None:
        """Detect if user wants to create a research task (P2-002)"""
        import re
        text_lower = text.lower().strip()

        # Patterns that indicate research intent
        research_patterns = [
            # "research X", "research about X"
            (r"^research\s+(?:about\s+)?(.+)$", lambda m: m.group(1)),
            # "look up X", "look into X"
            (r"^look\s+(?:up|into)\s+(.+)$", lambda m: m.group(1)),
            # "find out about X", "find information on X"
            (r"^find\s+(?:out\s+(?:about|more\s+about)?|information\s+(?:on|about))\s+(.+)$", lambda m: m.group(1)),
            # "search for X"
            (r"^search\s+for\s+(.+)$", lambda m: m.group(1)),
            # "what are the best practices for X"
            (r"^what\s+are\s+(?:the\s+)?best\s+practices\s+for\s+(.+)$", lambda m: m.group(1)),
            # "can you research X"
            (r"^can\s+you\s+research\s+(.+)$", lambda m: m.group(1)),
            # "I need research on X"
            (r"^i\s+need\s+(?:some\s+)?research\s+(?:on|about)\s+(.+)$", lambda m: m.group(1)),
        ]

        for pattern, extractor in research_patterns:
            match = re.match(pattern, text_lower, re.IGNORECASE)
            if match:
                topic = extractor(match).strip()
                # Clean up the topic
                topic = re.sub(r'[?.!]+$', '', topic).strip()
                if topic:
                    return {
                        "description": f"Research: {topic}",
                        "original_query": text,
                        "topic": topic
                    }

        return None

    def _log_user_query(self, query: str, topic: str):
        """Log user query to events database (US-008)"""
        try:
            from learning.events_db import UserEventsDB
            from pathlib import Path

            db = UserEventsDB(senter_root=Path(self.daemon.senter_root))
            db.log_query(query, topic=topic)
        except Exception as e:
            logger.warning(f"Failed to log query event: {e}")

    def _log_user_response(self, query: str, response: str, latency_ms: int,
                           worker: str, topic: str):
        """Log response to events database (US-008)"""
        try:
            from learning.events_db import UserEventsDB
            from pathlib import Path

            db = UserEventsDB(senter_root=Path(self.daemon.senter_root))
            db.log_response(query, response, latency_ms=latency_ms,
                           worker=worker, topic=topic)
        except Exception as e:
            logger.warning(f"Failed to log response event: {e}")

    def _build_conversation_context(self, conversation_id: str, current_query: str) -> str:
        """Build context-aware prompt from conversation history (P1-003)"""
        if not conversation_id:
            return None

        try:
            senter_root = Path(self.daemon.senter_root)
            conv_file = senter_root / "data" / "conversations" / f"{conversation_id}.json"

            if not conv_file.exists():
                logger.debug(f"No conversation file found: {conversation_id}")
                return None

            conv_data = json.loads(conv_file.read_text())
            messages = conv_data.get("messages", [])

            if not messages:
                return None

            # Limit to last 20 messages to fit context window
            messages = messages[-20:]

            # Build context string
            context_parts = ["Previous conversation:"]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"User: {content}")
                else:
                    context_parts.append(f"Assistant: {content}")

            context_parts.append("")
            context_parts.append(f"Current question: {current_query}")

            return "\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Failed to build conversation context: {e}")
            return None

    def _build_context_sources_prompt(self) -> str:
        """Build context from pinned sources (P3-002)"""
        if not self.daemon:
            return ""

        try:
            from learning.context_sources_db import ContextSourcesDB

            db = ContextSourcesDB(Path(self.daemon.senter_root))
            sources = db.get_all_sources(active_only=True)

            if not sources:
                return ""

            context_parts = ["The user has provided the following context sources for reference:"]

            for source in sources:
                content = self._get_source_content(source)
                if content:
                    context_parts.append(f"\n--- {source.title} ({source.type}: {source.path}) ---")
                    # Truncate to avoid context overflow
                    max_content_len = 2000
                    if len(content) > max_content_len:
                        content = content[:max_content_len] + "... [truncated]"
                    context_parts.append(content)

            if len(context_parts) > 1:
                context_parts.append("\n--- End of context sources ---")
                context_parts.append("Use these sources to inform your responses when relevant.")
                return "\n".join(context_parts)

            return ""

        except Exception as e:
            logger.warning(f"Failed to build context sources: {e}")
            return ""

    def _get_source_content(self, source) -> str:
        """Get content from a context source (P3-002)"""
        try:
            if source.type == "file":
                # Read file content
                file_path = Path(source.path).expanduser()
                if file_path.exists() and file_path.is_file():
                    return file_path.read_text()
            elif source.type == "web":
                # For URLs, use stored preview or fetch (simple approach)
                if source.content_preview:
                    return source.content_preview
                # Could add httpx fetch here for live content
            else:
                # Return stored preview for other types
                return source.content_preview or ""
        except Exception as e:
            logger.warning(f"Failed to get source content ({source.path}): {e}")

        return source.content_preview or ""

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

    def _handle_trigger_research(self, request: dict = None) -> dict:
        """Handle trigger_research request (US-005) - generate and queue research tasks"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from scheduler.research_trigger import trigger_background_research
            from pathlib import Path

            # Generate research tasks from recent user queries
            tasks = trigger_background_research(Path(self.daemon.senter_root))

            if not tasks:
                return {
                    "status": "ok",
                    "message": "No research topics found from recent queries",
                    "tasks_added": 0
                }

            # Add tasks to research queue
            added = 0
            for task in tasks:
                if self.daemon.add_research_task(task):
                    added += 1

            return {
                "status": "ok",
                "tasks_added": added,
                "topics": [t.get("topic", "unknown") for t in tasks]
            }

        except Exception as e:
            return {"error": str(e)}

    def _handle_activity_report(self, request: dict = None) -> dict:
        """Handle activity_report request (US-006) - 'what did Senter do' report"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from pathlib import Path
            from datetime import datetime
            import json

            hours = request.get("hours", 24) if request else 24
            since = time.time() - (hours * 3600)

            report = {
                "period_hours": hours,
                "since": datetime.fromtimestamp(since).isoformat(),
                "until": datetime.now().isoformat(),
                "tasks_completed": [],
                "research_done": [],
                "activity_summary": {},
                "time_stats": {}
            }

            senter_root = Path(self.daemon.senter_root)

            # 1. Get completed tasks from task result storage
            try:
                from engine.task_results import TaskResultStorage
                results_dir = senter_root / "data" / "tasks" / "results"
                if results_dir.exists():
                    storage = TaskResultStorage(results_dir)
                    task_results = storage.get_recent(limit=100, since=since)
                    report["tasks_completed"] = [
                        {
                            "task_id": r.task_id,
                            "description": r.description,
                            "worker": r.worker,
                            "latency_ms": r.latency_ms,
                            "timestamp": r.timestamp
                        }
                        for r in task_results
                    ]
            except Exception as e:
                report["tasks_error"] = str(e)

            # 2. Get research results from research output folder
            try:
                research_dir = senter_root / "data" / "research" / "results"
                if research_dir.exists():
                    research_results = []
                    for date_dir in sorted(research_dir.iterdir(), reverse=True):
                        if not date_dir.is_dir():
                            continue
                        for result_file in date_dir.glob("*.json"):
                            try:
                                data = json.loads(result_file.read_text())
                                if data.get("timestamp", 0) >= since:
                                    research_results.append({
                                        "task_id": data.get("task_id"),
                                        "description": data.get("description", "")[:100],
                                        "topic": data.get("topic", "unknown"),
                                        "timestamp": data.get("timestamp"),
                                        "result_preview": data.get("result", "")[:200] + "..."
                                    })
                            except:
                                continue
                        if len(research_results) >= 20:  # Limit
                            break
                    report["research_done"] = research_results
            except Exception as e:
                report["research_error"] = str(e)

            # 3. Get activity summary from activity log
            try:
                from reporter.progress_reporter import ActivityLog
                log_dir = senter_root / "data" / "progress" / "activity"
                if log_dir.exists():
                    activity_log = ActivityLog(log_dir)
                    summary = activity_log.get_summary(since=since)
                    report["activity_summary"] = summary
            except Exception as e:
                report["activity_error"] = str(e)

            # 4. Calculate time stats
            try:
                total_task_time = sum(t.get("latency_ms", 0) for t in report["tasks_completed"])
                report["time_stats"] = {
                    "total_task_time_ms": total_task_time,
                    "total_task_time_seconds": total_task_time / 1000,
                    "tasks_completed_count": len(report["tasks_completed"]),
                    "research_completed_count": len(report["research_done"]),
                    "total_activities": report.get("activity_summary", {}).get("total_activities", 0)
                }
            except Exception as e:
                report["time_stats_error"] = str(e)

            return report

        except Exception as e:
            return {"error": str(e)}

    def _handle_get_events(self, request: dict = None) -> dict:
        """Handle get_events request (US-008) - retrieve user interaction events"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from learning.events_db import UserEventsDB
            from pathlib import Path

            hours = request.get("hours", 24) if request else 24
            event_type = request.get("event_type") if request else None
            limit = request.get("limit", 50) if request else 50

            db = UserEventsDB(senter_root=Path(self.daemon.senter_root))

            # Get events
            events = db.get_events_by_time_range(hours=hours)
            if event_type:
                events = [e for e in events if e.event_type == event_type]

            events = events[:limit]

            # Get stats
            stats = db.get_stats()
            counts = db.get_event_counts(hours=hours)

            return {
                "events": [e.to_dict() for e in events],
                "count": len(events),
                "period_hours": hours,
                "event_counts": counts,
                "total_events": stats.get("total_events", 0),
                "database_path": stats.get("database_path")
            }

        except Exception as e:
            return {"error": str(e)}

    def _handle_get_conversations(self, request: dict = None) -> dict:
        """Handle get_conversations request (P1-001) - retrieve conversation history"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            senter_root = Path(self.daemon.senter_root)
            conv_dir = senter_root / "data" / "conversations"
            index_file = conv_dir / "index.json"

            if not index_file.exists():
                return {"conversations": [], "count": 0}

            # Read index
            index_data = json.loads(index_file.read_text())
            conversations = index_data.get("conversations", [])

            # If specific conversation requested, return full messages
            conv_id = request.get("conversation_id") if request else None
            if conv_id:
                conv_file = conv_dir / f"{conv_id}.json"
                if conv_file.exists():
                    conv_data = json.loads(conv_file.read_text())
                    return {"conversation": conv_data}
                else:
                    return {"error": f"Conversation {conv_id} not found"}

            # If include_messages flag, load all messages
            include_messages = request.get("include_messages", False) if request else False
            if include_messages:
                full_conversations = []
                for conv in conversations:
                    conv_file = conv_dir / f"{conv['id']}.json"
                    if conv_file.exists():
                        full_conversations.append(json.loads(conv_file.read_text()))
                    else:
                        full_conversations.append(conv)
                return {"conversations": full_conversations, "count": len(full_conversations)}

            # Return just the index (metadata only)
            return {"conversations": conversations, "count": len(conversations)}

        except Exception as e:
            logger.error(f"get_conversations error: {e}")
            return {"error": str(e)}

    def _handle_save_conversation(self, request: dict) -> dict:
        """Handle save_conversation request (P1-001) - persist conversation to storage"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        conversation = request.get("conversation")
        if not conversation:
            return {"error": "Missing 'conversation' field"}

        try:
            from datetime import datetime

            senter_root = Path(self.daemon.senter_root)
            conv_dir = senter_root / "data" / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)

            index_file = conv_dir / "index.json"

            # Load or create index
            if index_file.exists():
                index_data = json.loads(index_file.read_text())
            else:
                index_data = {"conversations": [], "last_id": 0}

            conv_id = conversation.get("id")
            messages = conversation.get("messages", [])
            focus = conversation.get("focus", "general")

            # Generate new ID if not provided
            if not conv_id:
                today = datetime.now().strftime("%Y-%m-%d")
                index_data["last_id"] = index_data.get("last_id", 0) + 1
                conv_id = f"{today}_{index_data['last_id']:04d}"
                conversation["id"] = conv_id

            # Generate summary from first user message
            summary = ""
            for msg in messages:
                if msg.get("role") == "user":
                    summary = msg.get("content", "")[:100]
                    break

            # Prepare metadata
            conv_meta = {
                "id": conv_id,
                "focus": focus,
                "created": conversation.get("created", datetime.now().isoformat()),
                "message_count": len(messages),
                "summary": summary
            }

            # Update or add to index
            existing_idx = next(
                (i for i, c in enumerate(index_data["conversations"]) if c["id"] == conv_id),
                None
            )
            if existing_idx is not None:
                index_data["conversations"][existing_idx] = conv_meta
            else:
                index_data["conversations"].append(conv_meta)

            # Save index
            index_file.write_text(json.dumps(index_data, indent=2))

            # Save full conversation
            conv_file = conv_dir / f"{conv_id}.json"
            full_conv = {
                "id": conv_id,
                "messages": messages,
                "focus": focus,
                "created": conv_meta["created"],
                "summary": summary
            }
            conv_file.write_text(json.dumps(full_conv, indent=2))

            logger.info(f"Saved conversation {conv_id} with {len(messages)} messages")

            return {
                "status": "ok",
                "conversation_id": conv_id,
                "message_count": len(messages)
            }

        except Exception as e:
            logger.error(f"save_conversation error: {e}")
            return {"error": str(e)}

    def _handle_get_tasks(self, request: dict = None) -> dict:
        """Handle get_tasks request (P2-001) - get all tasks with status"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from engine.task_results import TaskResultStorage
            from datetime import datetime

            tasks = []

            # 1. Get pending tasks from research queue
            try:
                queue_status = self.daemon.get_research_queue_status()
                pending_tasks = queue_status.get("tasks", [])
                for task in pending_tasks:
                    tasks.append({
                        "id": task.get("id", ""),
                        "title": task.get("description", "")[:50],
                        "description": task.get("description", ""),
                        "status": "pending",
                        "timestamp": task.get("added_at", time.time()),
                        "datetime": datetime.fromtimestamp(task.get("added_at", time.time())).isoformat(),
                        "result": None
                    })
            except Exception as e:
                logger.warning(f"Failed to get pending tasks: {e}")

            # 2. Get completed tasks from result storage
            try:
                results_dir = Path(self.daemon.senter_root) / "data" / "tasks" / "results"
                if results_dir.exists():
                    storage = TaskResultStorage(results_dir)
                    limit = request.get("limit", 20) if request else 20
                    completed = storage.get_recent(limit=limit)

                    for result in completed:
                        # Check if this task is not already in pending
                        existing_ids = {t["id"] for t in tasks}
                        if result.task_id not in existing_ids:
                            # Truncate result for preview
                            result_preview = str(result.result)
                            if len(result_preview) > 200:
                                result_preview = result_preview[:200] + "..."

                            tasks.append({
                                "id": result.task_id,
                                "title": result.description[:50] if result.description else result.task_id,
                                "description": result.description,
                                "status": result.status,
                                "timestamp": result.timestamp,
                                "datetime": datetime.fromtimestamp(result.timestamp).isoformat(),
                                "result": result_preview,
                                "worker": result.worker,
                                "latency_ms": result.latency_ms
                            })
            except Exception as e:
                logger.warning(f"Failed to get completed tasks: {e}")

            # Sort by timestamp (newest first)
            tasks.sort(key=lambda x: x["timestamp"], reverse=True)

            return {
                "tasks": tasks,
                "count": len(tasks),
                "pending_count": sum(1 for t in tasks if t["status"] == "pending"),
                "completed_count": sum(1 for t in tasks if t["status"] == "completed")
            }

        except Exception as e:
            logger.error(f"get_tasks error: {e}")
            return {"error": str(e)}

    def _handle_get_journal(self, request: dict = None) -> dict:
        """Handle get_journal request (P2-004) - retrieve journal entries"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from learning.journal_db import JournalDB

            journal = JournalDB(Path(self.daemon.senter_root))

            # Get specific date or recent entries
            entry_date = request.get("date") if request else None
            limit = request.get("limit", 7) if request else 7

            if entry_date:
                entry = journal.get_entry(entry_date)
                if entry:
                    return {"entry": entry.to_dict()}
                else:
                    return {"entry": None, "message": f"No entry for {entry_date}"}
            else:
                entries = journal.get_recent_entries(limit)
                return {
                    "entries": [e.to_dict() for e in entries],
                    "count": len(entries)
                }

        except Exception as e:
            logger.error(f"get_journal error: {e}")
            return {"error": str(e)}

    def _handle_generate_journal(self, request: dict = None) -> dict:
        """Handle generate_journal request (P2-004) - generate journal entry for a date"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from learning.journal_db import JournalDB
            from datetime import datetime

            journal = JournalDB(Path(self.daemon.senter_root))

            # Get target date (default: today)
            target_date = request.get("date") if request else None
            if not target_date:
                target_date = datetime.now().strftime("%Y-%m-%d")

            # Generate entry from activity data
            entry = journal.generate_entry_for_date(target_date)

            # Save to database
            journal.save_entry(entry)

            return {
                "entry": entry.to_dict(),
                "generated": True
            }

        except Exception as e:
            logger.error(f"generate_journal error: {e}")
            return {"error": str(e)}

    # ========== P3-001: Context Sources ==========

    def _handle_get_context_sources(self, request: dict = None) -> dict:
        """Handle get_context_sources request (P3-001) - get all pinned context sources"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from learning.context_sources_db import ContextSourcesDB

            db = ContextSourcesDB(Path(self.daemon.senter_root))

            # Get active sources
            sources = db.get_all_sources(active_only=True)

            return {
                "sources": [s.to_dict() for s in sources],
                "count": len(sources)
            }

        except Exception as e:
            logger.error(f"get_context_sources error: {e}")
            return {"error": str(e)}

    def _handle_add_context_source(self, request: dict = None) -> dict:
        """Handle add_context_source request (P3-001) - add a new context source"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from learning.context_sources_db import ContextSourcesDB, ContextSource
            import uuid

            db = ContextSourcesDB(Path(self.daemon.senter_root))

            # Parse request
            source_type = request.get("type", "file")
            title = request.get("title", "Untitled")
            path = request.get("path", "")
            description = request.get("description", "")

            if not path:
                return {"error": "Path is required"}

            # Generate preview for files
            content_preview = ""
            if source_type == "file":
                try:
                    file_path = Path(path).expanduser()
                    if file_path.exists() and file_path.is_file():
                        content = file_path.read_text()
                        content_preview = content[:500] + "..." if len(content) > 500 else content
                except Exception as e:
                    logger.warning(f"Could not read file for preview: {e}")

            # Create and save source
            source = ContextSource(
                id=str(uuid.uuid4()),
                type=source_type,
                title=title,
                path=path,
                description=description,
                content_preview=content_preview
            )

            success = db.add_source(source)

            if success:
                return {
                    "source": source.to_dict(),
                    "added": True
                }
            else:
                return {"error": "Failed to add source"}

        except Exception as e:
            logger.error(f"add_context_source error: {e}")
            return {"error": str(e)}

    def _handle_remove_context_source(self, request: dict = None) -> dict:
        """Handle remove_context_source request (P3-001) - remove a context source"""
        if not self.daemon:
            return {"error": "Daemon not available"}

        try:
            from learning.context_sources_db import ContextSourcesDB

            db = ContextSourcesDB(Path(self.daemon.senter_root))

            source_id = request.get("id") if request else None
            if not source_id:
                return {"error": "Source ID is required"}

            success = db.remove_source(source_id)

            return {
                "removed": success,
                "id": source_id
            }

        except Exception as e:
            logger.error(f"remove_context_source error: {e}")
            return {"error": str(e)}
