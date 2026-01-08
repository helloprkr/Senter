#!/usr/bin/env python3
"""
Message Bus for Inter-Process Communication

All components communicate through this central message bus.
Messages are typed and routed to appropriate handlers.

Features (MB-001, MB-002, MB-003):
- Dead letter queue for failed messages
- Correlation ID tracking for request/response matching
- Message persistence for crash recovery
"""

import json
import time
import logging
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Optional, Callable
from multiprocessing import Queue
from queue import Empty, Full
from pathlib import Path
import threading

logger = logging.getLogger('senter.message_bus')


class MessageType(Enum):
    """All message types in the system"""
    # User interaction
    USER_QUERY = "user_query"
    USER_VOICE = "user_voice"

    # Attention
    ATTENTION_GAINED = "attention_gained"
    ATTENTION_LOST = "attention_lost"

    # Model requests/responses
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"

    # Task management
    TASK_CREATE = "task_create"
    TASK_UPDATE = "task_update"
    TASK_COMPLETE = "task_complete"
    TASK_PROGRESS = "task_progress"

    # Goal management (CG-008)
    GOAL_DETECTED = "goal_detected"
    GOAL_COMPLETE = "goal_complete"

    # Scheduler
    SCHEDULE_JOB = "schedule_job"
    CANCEL_JOB = "cancel_job"
    JOB_TRIGGERED = "job_triggered"

    # Learning
    LEARN_EVENT = "learn_event"
    PROFILE_UPDATE = "profile_update"

    # Progress/Activity
    ACTIVITY_LOG = "activity_log"
    DIGEST_REQUEST = "digest_request"
    DIGEST_READY = "digest_ready"

    # Audio
    SPEAK = "speak"
    AUDIO_LEVEL = "audio_level"

    # System
    HEALTH_CHECK = "health_check"
    HEALTH_RESPONSE = "health_response"
    COMPONENT_READY = "component_ready"
    SHUTDOWN = "shutdown"
    ERROR = "error"


@dataclass
class Message:
    """Base message structure for all IPC"""
    type: MessageType
    source: str  # Component that sent the message
    payload: dict = field(default_factory=dict)
    target: Optional[str] = None  # Target component (None = broadcast)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None  # For request/response matching
    priority: int = 5  # 1-10, higher = more urgent

    def to_dict(self) -> dict:
        """Convert to dict for queue transport"""
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        """Create from dict"""
        d = d.copy()
        d["type"] = MessageType(d["type"])
        return cls(**d)

    def __repr__(self):
        return f"Message({self.type.value}, {self.source}→{self.target or '*'})"


# MB-001: Dead Letter Queue Entry
@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue."""
    message: dict
    error: str
    failure_count: int
    first_failure: float
    last_failure: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DeadLetterEntry":
        return cls(**d)


# MB-002: Correlation Registry Entry
@dataclass
class PendingRequest:
    """Tracks a pending request awaiting response."""
    correlation_id: str
    source: str
    msg_type: str
    timestamp: float
    callback: Optional[Callable] = None


class MessageBus:
    """
    Central message bus for inter-component communication.

    Features:
    - Pub/sub model with typed messages
    - Priority queue support
    - Routing table for message dispatch
    - Thread-safe operations
    - Dead letter queue for failed messages (MB-001)
    - Correlation ID tracking (MB-002)
    - Message persistence for crash recovery (MB-003)
    """

    # MB-001: DLQ configuration
    DLQ_MAX_SIZE = 1000
    MAX_RETRIES = 3

    # MB-002: Correlation timeout
    CORRELATION_TIMEOUT = 120  # seconds

    def __init__(self, max_size: int = 10000, data_dir: Path = None):
        self.queue = Queue(maxsize=max_size)
        self.subscribers: dict[str, Queue] = {}  # component_name -> queue
        self.handlers: dict[MessageType, list[Callable]] = {}
        self._running = False
        self._router_thread: Optional[threading.Thread] = None
        self._correlation_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # MB-001: Dead letter queue
        self._dead_letter_queue: list[DeadLetterEntry] = []
        self._failure_counts: dict[str, int] = {}  # message_id -> count

        # MB-002: Correlation registry
        self._correlation_registry: dict[str, PendingRequest] = {}

        # MB-003: Persistence
        self._data_dir = data_dir or Path("data/message_bus")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._dlq_file = self._data_dir / "dlq.json"
        self._pending_file = self._data_dir / "pending.json"

        # Load persisted state
        self._load_dlq()
        self._load_pending_messages()

        # Default routing table
        self.routing_table = {
            MessageType.USER_QUERY: ["model_primary", "task_engine", "learning"],
            MessageType.USER_VOICE: ["model_primary", "audio"],
            MessageType.ATTENTION_GAINED: ["audio", "model_primary"],
            MessageType.ATTENTION_LOST: ["audio"],
            MessageType.MODEL_REQUEST: ["model_primary", "model_research"],
            MessageType.MODEL_RESPONSE: ["task_engine", "audio", "cli"],
            MessageType.TASK_CREATE: ["task_engine"],
            MessageType.TASK_UPDATE: ["reporter", "cli"],
            MessageType.TASK_COMPLETE: ["reporter", "learning", "cli"],
            MessageType.GOAL_DETECTED: ["task_engine"],  # CG-008: Goal triggers plan creation
            MessageType.GOAL_COMPLETE: ["reporter", "learning", "cli"],  # CG-008: Goal completion notification
            MessageType.SCHEDULE_JOB: ["scheduler"],
            MessageType.JOB_TRIGGERED: ["task_engine"],
            MessageType.LEARN_EVENT: ["learning"],
            MessageType.ACTIVITY_LOG: ["reporter"],
            MessageType.DIGEST_REQUEST: ["reporter"],
            MessageType.SPEAK: ["audio"],
            MessageType.HEALTH_CHECK: ["*"],  # Broadcast
            MessageType.SHUTDOWN: ["*"],
        }

    def start(self):
        """Start the message router"""
        if self._running:
            return

        self._running = True
        self._router_thread = threading.Thread(target=self._router_loop, daemon=True)
        self._router_thread.start()

        # MB-002: Start correlation timeout checker
        self._correlation_thread = threading.Thread(target=self._correlation_timeout_loop, daemon=True)
        self._correlation_thread.start()

        logger.info("Message bus started")

    def stop(self):
        """Stop the message router"""
        self._running = False

        # MB-003: Persist messages before stopping
        self._save_pending_messages()
        self._save_dlq()

        if self._router_thread:
            self._router_thread.join(timeout=2.0)
        if self._correlation_thread:
            self._correlation_thread.join(timeout=2.0)
        logger.info("Message bus stopped")

    def register(self, component_name: str) -> Queue:
        """Register a component and return its message queue"""
        with self._lock:
            if component_name not in self.subscribers:
                self.subscribers[component_name] = Queue(maxsize=1000)
                logger.debug(f"Registered component: {component_name}")
            return self.subscribers[component_name]

    def unregister(self, component_name: str):
        """Unregister a component"""
        with self._lock:
            if component_name in self.subscribers:
                del self.subscribers[component_name]
                logger.debug(f"Unregistered component: {component_name}")

    def publish(self, message: Message):
        """Publish a message to the bus"""
        try:
            self.queue.put_nowait(message.to_dict())
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")

    def send(self, msg_type: MessageType, source: str, payload: dict = None,
             target: str = None, correlation_id: str = None):
        """Convenience method to create and publish a message"""
        msg = Message(
            type=msg_type,
            source=source,
            payload=payload or {},
            target=target,
            correlation_id=correlation_id
        )
        self.publish(msg)

    def subscribe(self, msg_type: MessageType, handler: Callable[[Message], None]):
        """Subscribe to a message type with a handler function"""
        with self._lock:
            if msg_type not in self.handlers:
                self.handlers[msg_type] = []
            self.handlers[msg_type].append(handler)

    def _router_loop(self):
        """Main routing loop"""
        while self._running:
            try:
                msg_dict = self.queue.get(timeout=0.1)
                message = Message.from_dict(msg_dict)
                self._route_message(message)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Router error: {e}")

    def _route_message(self, message: Message):
        """Route a message to appropriate subscribers"""
        # If specific target, send only there
        if message.target:
            self._deliver_to(message.target, message)
            return

        # Use routing table
        targets = self.routing_table.get(message.type, [])

        if "*" in targets:
            # Broadcast to all
            targets = list(self.subscribers.keys())

        for target in targets:
            self._deliver_to(target, message)

        # Call registered handlers
        handlers = self.handlers.get(message.type, [])
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Handler error for {message.type}: {e}")

    def _deliver_to(self, target: str, message: Message):
        """Deliver message to a specific subscriber"""
        with self._lock:
            if target in self.subscribers:
                try:
                    self.subscribers[target].put_nowait(message.to_dict())
                    logger.debug(f"Delivered {message.type.value} to {target}")
                except Exception as e:
                    logger.warning(f"Failed to deliver to {target}: {e}")
                    # MB-001: Track delivery failures
                    self._record_failure(message.to_dict(), str(e))

    # ========== MB-001: Dead Letter Queue ==========

    def _record_failure(self, msg_dict: dict, error: str):
        """Record a message delivery failure. After MAX_RETRIES, move to DLQ."""
        # Create unique ID for tracking
        msg_id = msg_dict.get("correlation_id") or f"{msg_dict.get('type')}_{msg_dict.get('timestamp')}"

        if msg_id not in self._failure_counts:
            self._failure_counts[msg_id] = 0
        self._failure_counts[msg_id] += 1

        if self._failure_counts[msg_id] >= self.MAX_RETRIES:
            self._move_to_dlq(msg_dict, error, self._failure_counts[msg_id])
            del self._failure_counts[msg_id]
            logger.warning(f"Message moved to DLQ after {self.MAX_RETRIES} failures: {msg_id}")

    def _move_to_dlq(self, msg_dict: dict, error: str, failure_count: int):
        """Move a failed message to the dead letter queue."""
        now = time.time()
        entry = DeadLetterEntry(
            message=msg_dict,
            error=error,
            failure_count=failure_count,
            first_failure=now,
            last_failure=now
        )

        with self._lock:
            self._dead_letter_queue.append(entry)
            # Enforce size limit
            if len(self._dead_letter_queue) > self.DLQ_MAX_SIZE:
                self._dead_letter_queue = self._dead_letter_queue[-self.DLQ_MAX_SIZE:]

        # Persist DLQ to disk
        self._save_dlq()

    def get_dlq_entries(self) -> list[DeadLetterEntry]:
        """Get all entries in the dead letter queue."""
        with self._lock:
            return list(self._dead_letter_queue)

    def clear_dlq(self):
        """Clear the dead letter queue."""
        with self._lock:
            self._dead_letter_queue.clear()
        self._save_dlq()
        logger.info("Dead letter queue cleared")

    def retry_dlq_message(self, index: int) -> bool:
        """Retry a message from the DLQ."""
        entry = None
        with self._lock:
            if 0 <= index < len(self._dead_letter_queue):
                entry = self._dead_letter_queue.pop(index)

        if entry is None:
            return False

        try:
            self.queue.put_nowait(entry.message)
            logger.info(f"Retrying DLQ message: {entry.message.get('type')}")
            self._save_dlq()  # Now called outside the lock
            return True
        except Full:
            # Put it back
            with self._lock:
                self._dead_letter_queue.insert(index, entry)
            return False

    def _save_dlq(self):
        """Persist DLQ to disk."""
        try:
            with self._lock:
                data = [e.to_dict() for e in self._dead_letter_queue]
            self._dlq_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save DLQ: {e}")

    def _load_dlq(self):
        """Load DLQ from disk."""
        try:
            if self._dlq_file.exists():
                data = json.loads(self._dlq_file.read_text())
                self._dead_letter_queue = [DeadLetterEntry.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self._dead_letter_queue)} DLQ entries")
        except Exception as e:
            logger.warning(f"Failed to load DLQ: {e}")
            self._dead_letter_queue = []

    # ========== MB-002: Correlation ID Tracking ==========

    def send_with_correlation(
        self,
        msg_type: MessageType,
        source: str,
        payload: dict = None,
        target: str = None,
        callback: Callable[[Message], None] = None,
        timeout: float = None
    ) -> str:
        """
        Send a message with correlation ID tracking.

        Returns the correlation_id for later matching.
        If callback provided, it will be called when response arrives.
        """
        correlation_id = str(uuid.uuid4())
        timeout = timeout or self.CORRELATION_TIMEOUT

        # Register in correlation registry
        pending = PendingRequest(
            correlation_id=correlation_id,
            source=source,
            msg_type=msg_type.value,
            timestamp=time.time(),
            callback=callback
        )

        with self._lock:
            self._correlation_registry[correlation_id] = pending

        # Send the message
        self.send(msg_type, source, payload, target, correlation_id)

        return correlation_id

    def respond(self, original_msg: Message, response_payload: dict, source: str):
        """Send a response message preserving the correlation ID."""
        if not original_msg.correlation_id:
            logger.warning("Cannot respond to message without correlation_id")
            return

        # Determine response type
        response_type = self._get_response_type(original_msg.type)

        self.send(
            msg_type=response_type,
            source=source,
            payload=response_payload,
            target=original_msg.source,
            correlation_id=original_msg.correlation_id
        )

    def _get_response_type(self, request_type: MessageType) -> MessageType:
        """Map request types to response types."""
        response_map = {
            MessageType.MODEL_REQUEST: MessageType.MODEL_RESPONSE,
            MessageType.HEALTH_CHECK: MessageType.HEALTH_RESPONSE,
            MessageType.DIGEST_REQUEST: MessageType.DIGEST_READY,
        }
        return response_map.get(request_type, MessageType.MODEL_RESPONSE)

    def check_correlation(self, correlation_id: str) -> Optional[PendingRequest]:
        """Check if a correlation ID is pending."""
        with self._lock:
            return self._correlation_registry.get(correlation_id)

    def complete_correlation(self, correlation_id: str, response: Message):
        """Mark a correlation as complete and call callback if registered."""
        with self._lock:
            pending = self._correlation_registry.pop(correlation_id, None)

        if pending and pending.callback:
            try:
                pending.callback(response)
            except Exception as e:
                logger.error(f"Correlation callback error: {e}")

    def _correlation_timeout_loop(self):
        """Background thread to check for timed-out correlations."""
        while self._running:
            now = time.time()
            expired = []

            with self._lock:
                for corr_id, pending in list(self._correlation_registry.items()):
                    if now - pending.timestamp > self.CORRELATION_TIMEOUT:
                        expired.append(corr_id)

            for corr_id in expired:
                with self._lock:
                    pending = self._correlation_registry.pop(corr_id, None)
                if pending:
                    logger.error(
                        f"Correlation timeout: {corr_id} "
                        f"(type={pending.msg_type}, source={pending.source})"
                    )

            time.sleep(5)  # Check every 5 seconds

    def get_pending_correlations(self) -> dict[str, PendingRequest]:
        """Get all pending correlations."""
        with self._lock:
            return dict(self._correlation_registry)

    # ========== MB-003: Message Persistence ==========

    def _save_pending_messages(self):
        """Save unprocessed messages to disk for crash recovery."""
        messages = []

        # Drain the main queue
        while True:
            try:
                msg = self.queue.get_nowait()
                messages.append(msg)
            except Empty:
                break

        # Drain subscriber queues
        with self._lock:
            for name, q in self.subscribers.items():
                while True:
                    try:
                        msg = q.get_nowait()
                        messages.append(msg)
                    except Empty:
                        break

        if messages:
            try:
                self._pending_file.write_text(json.dumps(messages, indent=2))
                logger.info(f"Saved {len(messages)} pending messages")
            except Exception as e:
                logger.error(f"Failed to save pending messages: {e}")
        else:
            # Clear the file if no messages
            if self._pending_file.exists():
                self._pending_file.unlink()

    def _load_pending_messages(self):
        """Load pending messages from disk on startup."""
        try:
            if self._pending_file.exists():
                data = json.loads(self._pending_file.read_text())
                for msg_dict in data:
                    try:
                        self.queue.put_nowait(msg_dict)
                    except Full:
                        logger.warning("Queue full, some messages not restored")
                        break
                logger.info(f"Restored {len(data)} pending messages")
                # Clear the file after loading
                self._pending_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to load pending messages: {e}")

    def get_pending_count(self) -> int:
        """Get count of messages in main queue."""
        return self.queue.qsize()

    def persist_now(self):
        """Force immediate persistence of all messages."""
        self._save_pending_messages()
        self._save_dlq()
        logger.info("Message bus state persisted")


# Singleton instance for easy access
_bus_instance: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get or create the global message bus instance"""
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = MessageBus()
    return _bus_instance


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    bus = get_message_bus()
    bus.start()

    # Register test components
    q1 = bus.register("test_component")
    q2 = bus.register("model_primary")

    # Test publish
    bus.send(
        MessageType.USER_QUERY,
        source="cli",
        payload={"text": "Hello, Senter!"}
    )

    time.sleep(0.2)

    # Check delivery
    try:
        msg = q2.get_nowait()
        print(f"Received: {msg}")
    except Empty:
        print("No message received")

    bus.stop()
    print("Test complete")
