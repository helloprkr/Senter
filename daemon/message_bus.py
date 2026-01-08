#!/usr/bin/env python3
"""
Message Bus for Inter-Process Communication

All components communicate through this central message bus.
Messages are typed and routed to appropriate handlers.
"""

import json
import time
import logging
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Optional, Callable
from multiprocessing import Queue
from queue import Empty
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


class MessageBus:
    """
    Central message bus for inter-component communication.

    Features:
    - Pub/sub model with typed messages
    - Priority queue support
    - Routing table for message dispatch
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 10000):
        self.queue = Queue(maxsize=max_size)
        self.subscribers: dict[str, Queue] = {}  # component_name -> queue
        self.handlers: dict[MessageType, list[Callable]] = {}
        self._running = False
        self._router_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

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
        logger.info("Message bus started")

    def stop(self):
        """Stop the message router"""
        self._running = False
        if self._router_thread:
            self._router_thread.join(timeout=2.0)
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
