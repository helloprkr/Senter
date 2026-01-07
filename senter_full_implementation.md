# SENTER: Full Autonomous Implementation Blueprint

**Version**: 4.0 - "The Real Vision"  
**Scope**: Complete implementation of all autonomous features  
**Estimated Effort**: 120-160 hours of focused development  
**Goal**: A truly autonomous AI assistant that works 24/7

---

## TABLE OF CONTENTS

1. [Architecture Overview](#1-architecture-overview)
2. [Component 1: Persistent Background Daemon](#2-persistent-background-daemon)
3. [Component 2: Task Execution Engine](#3-task-execution-engine)
4. [Component 3: Audio Pipeline](#4-audio-pipeline)
5. [Component 4: Gaze Detection](#5-gaze-detection)
6. [Component 5: Dual Model Workers](#6-dual-model-workers)
7. [Component 6: Learning Database](#7-learning-database)
8. [Component 7: Action Scheduler](#8-action-scheduler)
9. [Component 8: Progress Reporter](#9-progress-reporter)
10. [Integration Map](#10-integration-map)
11. [Implementation Order](#11-implementation-order)

---

## 1. ARCHITECTURE OVERVIEW

### The Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SENTER DAEMON (24/7)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   AUDIO     │    │    GAZE     │    │   ACTION    │    │  PROGRESS   │  │
│  │  PIPELINE   │───▶│  DETECTOR   │───▶│  SCHEDULER  │───▶│  REPORTER   │  │
│  │ (STT/TTS)   │    │  (Camera)   │    │  (Cron)     │    │  (Notify)   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        MESSAGE BUS (Event Queue)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   MODEL     │    │   MODEL     │    │    TASK     │    │  LEARNING   │  │
│  │  WORKER 1   │    │  WORKER 2   │    │   ENGINE    │    │  DATABASE   │  │
│  │ (Response)  │    │ (Research)  │    │  (Execute)  │    │  (Profile)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure After Implementation

```
Senter/
├── daemon/
│   ├── __init__.py
│   ├── senter_daemon.py      # Main daemon process
│   ├── process_manager.py    # Manages all child processes
│   ├── message_bus.py        # Inter-process communication
│   └── health_monitor.py     # Watchdog for all components
│
├── workers/
│   ├── __init__.py
│   ├── model_worker.py       # LLM inference worker
│   ├── research_worker.py    # Background research worker
│   └── worker_pool.py        # Worker pool management
│
├── audio/
│   ├── __init__.py
│   ├── audio_pipeline.py     # Main audio controller
│   ├── stt_engine.py         # Speech-to-text (Whisper)
│   ├── tts_engine.py         # Text-to-speech (Piper)
│   ├── vad.py                # Voice activity detection
│   └── audio_buffer.py       # Audio stream buffer
│
├── vision/
│   ├── __init__.py
│   ├── gaze_detector.py      # Gaze/attention detection
│   ├── face_tracker.py       # Face detection and tracking
│   └── camera_manager.py     # Camera capture management
│
├── engine/
│   ├── __init__.py
│   ├── task_engine.py        # Goal → Plan → Execute → Report
│   ├── planner.py            # Breaks goals into tasks
│   ├── executor.py           # Executes individual tasks
│   └── tool_registry.py      # Available tools/actions
│
├── scheduler/
│   ├── __init__.py
│   ├── action_scheduler.py   # Cron-like scheduler
│   ├── job_store.py          # Persistent job storage
│   └── triggers.py           # Time/event triggers
│
├── learning/
│   ├── __init__.py
│   ├── learning_db.py        # Time-series database
│   ├── behavior_analyzer.py  # Pattern detection
│   ├── preference_model.py   # User preference ML model
│   └── profile_builder.py    # Builds user profile over time
│
├── reporter/
│   ├── __init__.py
│   ├── progress_reporter.py  # Main reporter
│   ├── notification.py       # Desktop/mobile notifications
│   ├── digest_generator.py   # Daily/session digests
│   └── activity_log.py       # Detailed activity logging
│
├── Functions/                 # Existing functions (keep)
├── Focuses/                   # Existing focuses (keep)
├── scripts/
│   ├── senter.py             # CLI (updated to connect to daemon)
│   ├── senter_app.py         # TUI (updated)
│   └── senter_ctl.py         # NEW: Daemon control script
│
├── config/
│   ├── user_profile.json     # User config
│   ├── daemon_config.json    # NEW: Daemon configuration
│   └── audio_config.json     # NEW: Audio settings
│
└── data/
    ├── conversations/         # Existing
    ├── goals.json            # Existing
    ├── learning/             # NEW: Learning database
    │   ├── behavior.db       # SQLite time-series
    │   ├── embeddings/       # User behavior embeddings
    │   └── models/           # Trained preference models
    ├── scheduler/            # NEW: Scheduled jobs
    │   └── jobs.json
    └── progress/             # NEW: Progress reports
        ├── daily/
        └── session/
```

---

## 2. PERSISTENT BACKGROUND DAEMON

### Purpose
A daemon process that runs 24/7, managing all Senter subsystems, surviving terminal closure, and providing always-on AI assistance.

### Implementation

#### File: `daemon/senter_daemon.py`

```python
#!/usr/bin/env python3
"""
Senter Background Daemon

A persistent process that runs 24/7, managing:
- Model workers (inference)
- Audio pipeline (STT/TTS)
- Gaze detection (attention)
- Task execution engine
- Action scheduler
- Progress reporting

Usage:
    python -m daemon.senter_daemon start
    python -m daemon.senter_daemon stop
    python -m daemon.senter_daemon status
"""

import os
import sys
import signal
import time
import json
import atexit
import logging
from pathlib import Path
from multiprocessing import Process, Queue, Event
from typing import Optional
import daemon  # python-daemon package
import lockfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('data/daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('senter.daemon')


class SenterDaemon:
    """
    Main Senter daemon process.
    
    Manages lifecycle of all subsystems:
    - Starts/stops child processes
    - Monitors health
    - Handles graceful shutdown
    - Provides IPC via message bus
    """
    
    def __init__(self, config_path: str = "config/daemon_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Process management
        self.processes: dict[str, Process] = {}
        self.shutdown_event = Event()
        
        # Message queues for IPC
        self.message_bus = Queue(maxsize=1000)
        self.response_queues: dict[str, Queue] = {}
        
        # PID file for daemon management
        self.pid_file = Path("data/senter.pid")
        
        # Component references
        self.model_workers = []
        self.audio_pipeline = None
        self.gaze_detector = None
        self.task_engine = None
        self.scheduler = None
        self.reporter = None
    
    def _load_config(self) -> dict:
        """Load daemon configuration"""
        default_config = {
            "components": {
                "model_workers": {
                    "enabled": True,
                    "count": 2,  # Dual workers
                    "models": {
                        "primary": "llama3.2",
                        "research": "llama3.2"
                    }
                },
                "audio_pipeline": {
                    "enabled": True,
                    "stt_model": "whisper-small",
                    "tts_model": "piper",
                    "vad_threshold": 0.5
                },
                "gaze_detection": {
                    "enabled": True,
                    "camera_id": 0,
                    "attention_threshold": 0.7
                },
                "task_engine": {
                    "enabled": True,
                    "max_concurrent_tasks": 3
                },
                "scheduler": {
                    "enabled": True,
                    "check_interval": 60  # seconds
                },
                "reporter": {
                    "enabled": True,
                    "digest_hour": 9,  # 9 AM daily digest
                    "notifications": True
                }
            },
            "ipc": {
                "socket_path": "/tmp/senter.sock",
                "http_port": 8765
            },
            "health_check_interval": 30  # seconds
        }
        
        if self.config_path.exists():
            with open(self.config_path) as f:
                user_config = json.load(f)
                # Deep merge
                self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base: dict, overlay: dict):
        """Deep merge overlay into base"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def start(self):
        """Start the daemon and all components"""
        logger.info("Starting Senter daemon...")
        
        # Check if already running
        if self._is_running():
            logger.error("Senter daemon is already running")
            return False
        
        # Write PID file
        self._write_pid()
        
        # Register cleanup
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        try:
            # Start components in order
            self._start_message_bus()
            self._start_model_workers()
            self._start_audio_pipeline()
            self._start_gaze_detection()
            self._start_task_engine()
            self._start_scheduler()
            self._start_reporter()
            self._start_health_monitor()
            self._start_ipc_server()
            
            logger.info("Senter daemon started successfully")
            
            # Main loop
            self._main_loop()
            
        except Exception as e:
            logger.error(f"Daemon startup failed: {e}")
            self._cleanup()
            raise
    
    def _start_message_bus(self):
        """Initialize the message bus for IPC"""
        logger.info("Starting message bus...")
        # Message bus is a multiprocessing Queue, already initialized
        # Start the message router process
        from daemon.message_bus import MessageRouter
        router = MessageRouter(self.message_bus, self.response_queues)
        p = Process(target=router.run, name="message_router")
        p.start()
        self.processes["message_router"] = p
    
    def _start_model_workers(self):
        """Start dual model workers"""
        if not self.config["components"]["model_workers"]["enabled"]:
            logger.info("Model workers disabled in config")
            return
        
        logger.info("Starting model workers...")
        from workers.model_worker import ModelWorker
        
        worker_config = self.config["components"]["model_workers"]
        
        # Worker 1: Primary (user responses)
        response_queue = Queue()
        self.response_queues["model_primary"] = response_queue
        worker1 = ModelWorker(
            name="primary",
            model=worker_config["models"]["primary"],
            input_queue=self.message_bus,
            output_queue=response_queue,
            shutdown_event=self.shutdown_event
        )
        p1 = Process(target=worker1.run, name="model_worker_primary")
        p1.start()
        self.processes["model_worker_primary"] = p1
        
        # Worker 2: Research (background tasks)
        research_queue = Queue()
        self.response_queues["model_research"] = research_queue
        worker2 = ModelWorker(
            name="research",
            model=worker_config["models"]["research"],
            input_queue=self.message_bus,
            output_queue=research_queue,
            shutdown_event=self.shutdown_event
        )
        p2 = Process(target=worker2.run, name="model_worker_research")
        p2.start()
        self.processes["model_worker_research"] = p2
        
        logger.info("Model workers started")
    
    def _start_audio_pipeline(self):
        """Start audio pipeline (STT/TTS/VAD)"""
        if not self.config["components"]["audio_pipeline"]["enabled"]:
            logger.info("Audio pipeline disabled in config")
            return
        
        logger.info("Starting audio pipeline...")
        from audio.audio_pipeline import AudioPipeline
        
        audio_config = self.config["components"]["audio_pipeline"]
        audio_queue = Queue()
        self.response_queues["audio"] = audio_queue
        
        pipeline = AudioPipeline(
            stt_model=audio_config["stt_model"],
            tts_model=audio_config["tts_model"],
            vad_threshold=audio_config["vad_threshold"],
            output_queue=self.message_bus,
            input_queue=audio_queue,
            shutdown_event=self.shutdown_event
        )
        p = Process(target=pipeline.run, name="audio_pipeline")
        p.start()
        self.processes["audio_pipeline"] = p
        logger.info("Audio pipeline started")
    
    def _start_gaze_detection(self):
        """Start gaze/attention detection"""
        if not self.config["components"]["gaze_detection"]["enabled"]:
            logger.info("Gaze detection disabled in config")
            return
        
        logger.info("Starting gaze detection...")
        from vision.gaze_detector import GazeDetector
        
        gaze_config = self.config["components"]["gaze_detection"]
        gaze_queue = Queue()
        self.response_queues["gaze"] = gaze_queue
        
        detector = GazeDetector(
            camera_id=gaze_config["camera_id"],
            attention_threshold=gaze_config["attention_threshold"],
            output_queue=self.message_bus,
            shutdown_event=self.shutdown_event
        )
        p = Process(target=detector.run, name="gaze_detector")
        p.start()
        self.processes["gaze_detector"] = p
        logger.info("Gaze detection started")
    
    def _start_task_engine(self):
        """Start task execution engine"""
        if not self.config["components"]["task_engine"]["enabled"]:
            logger.info("Task engine disabled in config")
            return
        
        logger.info("Starting task engine...")
        from engine.task_engine import TaskEngine
        
        engine_config = self.config["components"]["task_engine"]
        engine_queue = Queue()
        self.response_queues["task_engine"] = engine_queue
        
        engine = TaskEngine(
            max_concurrent=engine_config["max_concurrent_tasks"],
            message_bus=self.message_bus,
            input_queue=engine_queue,
            shutdown_event=self.shutdown_event
        )
        p = Process(target=engine.run, name="task_engine")
        p.start()
        self.processes["task_engine"] = p
        logger.info("Task engine started")
    
    def _start_scheduler(self):
        """Start action scheduler"""
        if not self.config["components"]["scheduler"]["enabled"]:
            logger.info("Scheduler disabled in config")
            return
        
        logger.info("Starting scheduler...")
        from scheduler.action_scheduler import ActionScheduler
        
        sched_config = self.config["components"]["scheduler"]
        sched_queue = Queue()
        self.response_queues["scheduler"] = sched_queue
        
        scheduler = ActionScheduler(
            check_interval=sched_config["check_interval"],
            message_bus=self.message_bus,
            input_queue=sched_queue,
            shutdown_event=self.shutdown_event
        )
        p = Process(target=scheduler.run, name="scheduler")
        p.start()
        self.processes["scheduler"] = p
        logger.info("Scheduler started")
    
    def _start_reporter(self):
        """Start progress reporter"""
        if not self.config["components"]["reporter"]["enabled"]:
            logger.info("Reporter disabled in config")
            return
        
        logger.info("Starting progress reporter...")
        from reporter.progress_reporter import ProgressReporter
        
        reporter_config = self.config["components"]["reporter"]
        reporter_queue = Queue()
        self.response_queues["reporter"] = reporter_queue
        
        reporter = ProgressReporter(
            digest_hour=reporter_config["digest_hour"],
            notifications=reporter_config["notifications"],
            message_bus=self.message_bus,
            input_queue=reporter_queue,
            shutdown_event=self.shutdown_event
        )
        p = Process(target=reporter.run, name="reporter")
        p.start()
        self.processes["reporter"] = p
        logger.info("Progress reporter started")
    
    def _start_health_monitor(self):
        """Start health monitoring"""
        logger.info("Starting health monitor...")
        from daemon.health_monitor import HealthMonitor
        
        monitor = HealthMonitor(
            processes=self.processes,
            check_interval=self.config["health_check_interval"],
            restart_callback=self._restart_component,
            shutdown_event=self.shutdown_event
        )
        p = Process(target=monitor.run, name="health_monitor")
        p.start()
        self.processes["health_monitor"] = p
        logger.info("Health monitor started")
    
    def _start_ipc_server(self):
        """Start IPC server for CLI/TUI communication"""
        logger.info("Starting IPC server...")
        from daemon.ipc_server import IPCServer
        
        server = IPCServer(
            socket_path=self.config["ipc"]["socket_path"],
            http_port=self.config["ipc"]["http_port"],
            message_bus=self.message_bus,
            response_queues=self.response_queues,
            shutdown_event=self.shutdown_event
        )
        p = Process(target=server.run, name="ipc_server")
        p.start()
        self.processes["ipc_server"] = p
        logger.info("IPC server started")
    
    def _main_loop(self):
        """Main daemon loop"""
        logger.info("Entering main loop...")
        while not self.shutdown_event.is_set():
            time.sleep(1)
            # Main loop just waits; actual work happens in child processes
    
    def _restart_component(self, name: str):
        """Restart a failed component"""
        logger.warning(f"Restarting component: {name}")
        if name in self.processes:
            old_process = self.processes[name]
            if old_process.is_alive():
                old_process.terminate()
                old_process.join(timeout=5)
        
        # Map name to start function
        start_functions = {
            "model_worker_primary": self._start_model_workers,
            "model_worker_research": self._start_model_workers,
            "audio_pipeline": self._start_audio_pipeline,
            "gaze_detector": self._start_gaze_detection,
            "task_engine": self._start_task_engine,
            "scheduler": self._start_scheduler,
            "reporter": self._start_reporter,
        }
        
        if name in start_functions:
            start_functions[name]()
    
    def stop(self):
        """Stop the daemon gracefully"""
        logger.info("Stopping Senter daemon...")
        self.shutdown_event.set()
        
        # Give processes time to clean up
        time.sleep(2)
        
        # Terminate any remaining processes
        for name, process in self.processes.items():
            if process.is_alive():
                logger.info(f"Terminating {name}...")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    logger.warning(f"Force killing {name}")
                    process.kill()
        
        self._cleanup()
        logger.info("Senter daemon stopped")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.pid_file.exists():
            self.pid_file.unlink()
    
    def _write_pid(self):
        """Write PID to file"""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))
    
    def _is_running(self) -> bool:
        """Check if daemon is already running"""
        if not self.pid_file.exists():
            return False
        
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            return True
        except (ProcessLookupError, ValueError):
            # Process doesn't exist or invalid PID
            self.pid_file.unlink()
            return False
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
    
    def status(self) -> dict:
        """Get daemon status"""
        if not self._is_running():
            return {"running": False}
        
        pid = int(self.pid_file.read_text().strip())
        return {
            "running": True,
            "pid": pid,
            "components": {
                name: {"alive": p.is_alive(), "pid": p.pid}
                for name, p in self.processes.items()
            }
        }


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Senter Daemon Control")
    parser.add_argument("command", choices=["start", "stop", "status", "restart"])
    parser.add_argument("--foreground", "-f", action="store_true", 
                       help="Run in foreground (don't daemonize)")
    args = parser.parse_args()
    
    daemon_instance = SenterDaemon()
    
    if args.command == "start":
        if args.foreground:
            daemon_instance.start()
        else:
            # Daemonize
            with daemon.DaemonContext(
                working_directory=os.getcwd(),
                pidfile=lockfile.FileLock(daemon_instance.pid_file),
                stdout=open('data/daemon_stdout.log', 'w+'),
                stderr=open('data/daemon_stderr.log', 'w+'),
            ):
                daemon_instance.start()
    
    elif args.command == "stop":
        if daemon_instance._is_running():
            pid = int(daemon_instance.pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print("Senter daemon stopped")
        else:
            print("Senter daemon is not running")
    
    elif args.command == "status":
        status = daemon_instance.status()
        if status["running"]:
            print(f"Senter daemon is running (PID: {status['pid']})")
            for name, info in status.get("components", {}).items():
                status_str = "✓" if info["alive"] else "✗"
                print(f"  {status_str} {name} (PID: {info['pid']})")
        else:
            print("Senter daemon is not running")
    
    elif args.command == "restart":
        if daemon_instance._is_running():
            pid = int(daemon_instance.pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        daemon_instance.start()


if __name__ == "__main__":
    main()
```

#### File: `daemon/message_bus.py`

```python
"""
Message Bus for Inter-Process Communication

All components communicate through this central message bus.
Messages are typed and routed to appropriate handlers.
"""

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Optional
from multiprocessing import Queue
import logging

logger = logging.getLogger('senter.message_bus')


class MessageType(Enum):
    # User interaction
    USER_QUERY = "user_query"
    USER_VOICE = "user_voice"
    
    # Attention
    ATTENTION_GAINED = "attention_gained"
    ATTENTION_LOST = "attention_lost"
    
    # Model requests
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    
    # Task management
    TASK_CREATE = "task_create"
    TASK_UPDATE = "task_update"
    TASK_COMPLETE = "task_complete"
    
    # Scheduler
    SCHEDULE_JOB = "schedule_job"
    JOB_TRIGGERED = "job_triggered"
    
    # Learning
    LEARN_EVENT = "learn_event"
    PROFILE_UPDATE = "profile_update"
    
    # Progress
    ACTIVITY_LOG = "activity_log"
    DIGEST_REQUEST = "digest_request"
    
    # System
    HEALTH_CHECK = "health_check"
    SHUTDOWN = "shutdown"


@dataclass
class Message:
    """Base message structure"""
    type: MessageType
    source: str  # Component that sent the message
    target: Optional[str]  # Target component (None = broadcast)
    payload: dict
    timestamp: float = None
    correlation_id: Optional[str] = None  # For request/response matching
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        d["type"] = MessageType(d["type"])
        return cls(**d)


class MessageRouter:
    """
    Routes messages between components.
    
    Runs as a separate process, reading from main queue
    and dispatching to component queues.
    """
    
    def __init__(self, main_queue: Queue, component_queues: dict[str, Queue]):
        self.main_queue = main_queue
        self.component_queues = component_queues
        self.running = True
        
        # Message type to component mapping
        self.routing_table = {
            MessageType.USER_QUERY: ["model_primary", "task_engine", "learning"],
            MessageType.USER_VOICE: ["model_primary", "audio"],
            MessageType.ATTENTION_GAINED: ["audio", "model_primary"],
            MessageType.ATTENTION_LOST: ["audio"],
            MessageType.MODEL_REQUEST: ["model_primary", "model_research"],
            MessageType.TASK_CREATE: ["task_engine"],
            MessageType.SCHEDULE_JOB: ["scheduler"],
            MessageType.JOB_TRIGGERED: ["task_engine"],
            MessageType.LEARN_EVENT: ["learning"],
            MessageType.ACTIVITY_LOG: ["reporter"],
            MessageType.DIGEST_REQUEST: ["reporter"],
            MessageType.HEALTH_CHECK: ["*"],  # Broadcast
            MessageType.SHUTDOWN: ["*"],
        }
    
    def run(self):
        """Main routing loop"""
        logger.info("Message router started")
        
        while self.running:
            try:
                # Get message with timeout
                message_dict = self.main_queue.get(timeout=1.0)
                message = Message.from_dict(message_dict)
                
                # Route message
                self._route_message(message)
                
            except Exception as e:
                if "Empty" not in str(type(e)):
                    logger.error(f"Router error: {e}")
    
    def _route_message(self, message: Message):
        """Route message to appropriate queues"""
        # If specific target, send only there
        if message.target:
            if message.target in self.component_queues:
                self.component_queues[message.target].put(message.to_dict())
            return
        
        # Otherwise, use routing table
        targets = self.routing_table.get(message.type, [])
        
        if "*" in targets:
            # Broadcast to all
            targets = list(self.component_queues.keys())
        
        for target in targets:
            if target in self.component_queues:
                self.component_queues[target].put(message.to_dict())
                logger.debug(f"Routed {message.type.value} to {target}")
```

#### File: `daemon/health_monitor.py`

```python
"""
Health Monitor

Watches all daemon components and restarts failed ones.
"""

import time
import logging
from multiprocessing import Process, Event
from typing import Callable

logger = logging.getLogger('senter.health')


class HealthMonitor:
    """
    Monitors health of all daemon components.
    
    - Checks if processes are alive
    - Monitors resource usage
    - Triggers restarts on failure
    - Logs health metrics
    """
    
    def __init__(
        self,
        processes: dict[str, Process],
        check_interval: int,
        restart_callback: Callable[[str], None],
        shutdown_event: Event
    ):
        self.processes = processes
        self.check_interval = check_interval
        self.restart_callback = restart_callback
        self.shutdown_event = shutdown_event
        
        # Track restart counts to prevent restart loops
        self.restart_counts: dict[str, int] = {}
        self.max_restarts = 5
        self.restart_window = 300  # 5 minutes
        self.restart_times: dict[str, list[float]] = {}
    
    def run(self):
        """Main monitoring loop"""
        logger.info("Health monitor started")
        
        while not self.shutdown_event.is_set():
            self._check_all_processes()
            time.sleep(self.check_interval)
    
    def _check_all_processes(self):
        """Check health of all processes"""
        for name, process in list(self.processes.items()):
            if name == "health_monitor":
                continue  # Don't monitor self
            
            if not process.is_alive():
                logger.warning(f"Process {name} is not alive!")
                self._handle_dead_process(name)
    
    def _handle_dead_process(self, name: str):
        """Handle a dead process"""
        # Check restart limits
        now = time.time()
        
        if name not in self.restart_times:
            self.restart_times[name] = []
        
        # Remove old restart times
        self.restart_times[name] = [
            t for t in self.restart_times[name]
            if now - t < self.restart_window
        ]
        
        if len(self.restart_times[name]) >= self.max_restarts:
            logger.error(
                f"Process {name} has restarted {self.max_restarts} times "
                f"in {self.restart_window}s - not restarting"
            )
            return
        
        # Attempt restart
        logger.info(f"Attempting to restart {name}")
        self.restart_times[name].append(now)
        
        try:
            self.restart_callback(name)
            logger.info(f"Successfully restarted {name}")
        except Exception as e:
            logger.error(f"Failed to restart {name}: {e}")
```

### Integration Points

The daemon integrates with:
- **CLI/TUI**: Via IPC server (Unix socket or HTTP)
- **All Components**: Via message bus
- **System**: Via systemd/launchd for auto-start

### Configuration File: `config/daemon_config.json`

```json
{
  "components": {
    "model_workers": {
      "enabled": true,
      "count": 2,
      "models": {
        "primary": "llama3.2",
        "research": "llama3.2"
      }
    },
    "audio_pipeline": {
      "enabled": true,
      "stt_model": "whisper-small",
      "tts_model": "piper",
      "vad_threshold": 0.5
    },
    "gaze_detection": {
      "enabled": true,
      "camera_id": 0,
      "attention_threshold": 0.7
    },
    "task_engine": {
      "enabled": true,
      "max_concurrent_tasks": 3
    },
    "scheduler": {
      "enabled": true,
      "check_interval": 60
    },
    "reporter": {
      "enabled": true,
      "digest_hour": 9,
      "notifications": true
    }
  },
  "ipc": {
    "socket_path": "/tmp/senter.sock",
    "http_port": 8765
  },
  "health_check_interval": 30
}
```

---

## 3. TASK EXECUTION ENGINE

### Purpose
A complete Goal → Plan → Execute → Report pipeline that takes user goals and autonomously works on them.

### Implementation

#### File: `engine/task_engine.py`

```python
"""
Task Execution Engine

The brain of Senter's autonomous operation.
Takes goals and executes them without user intervention.

Pipeline:
1. Goal received (from user or scheduler)
2. Planner breaks goal into tasks
3. Executor runs each task
4. Progress reported back
"""

import json
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from multiprocessing import Queue, Event
from pathlib import Path
import uuid

logger = logging.getLogger('senter.task_engine')


class TaskStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"  # Waiting for external input
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    RESEARCH = "research"       # Web search, information gathering
    GENERATE = "generate"       # Create content (text, code, etc.)
    ANALYZE = "analyze"         # Analyze data or content
    COMMUNICATE = "communicate" # Draft emails, messages
    ORGANIZE = "organize"       # File management, scheduling
    CUSTOM = "custom"           # Custom tool execution


@dataclass
class Task:
    """Individual executable task"""
    id: str
    goal_id: str
    description: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10, higher = more urgent
    
    # Execution details
    tool: Optional[str] = None
    tool_params: dict = field(default_factory=dict)
    
    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "description": self.description,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "priority": self.priority,
            "tool": self.tool,
            "tool_params": self.tool_params,
            "depends_on": self.depends_on,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class ExecutionPlan:
    """Plan for executing a goal"""
    goal_id: str
    goal_description: str
    tasks: list[Task]
    created_at: float = field(default_factory=time.time)
    
    def get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to execute (dependencies met)"""
        completed_ids = {
            t.id for t in self.tasks 
            if t.status == TaskStatus.COMPLETED
        }
        
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.depends_on)
        ]


class TaskEngine:
    """
    Main task execution engine.
    
    Responsibilities:
    - Receive goals from message bus
    - Create execution plans
    - Execute tasks
    - Report progress
    - Handle failures and retries
    """
    
    def __init__(
        self,
        max_concurrent: int,
        message_bus: Queue,
        input_queue: Queue,
        shutdown_event: Event
    ):
        self.max_concurrent = max_concurrent
        self.message_bus = message_bus
        self.input_queue = input_queue
        self.shutdown_event = shutdown_event
        
        # Active plans and tasks
        self.plans: dict[str, ExecutionPlan] = {}
        self.running_tasks: dict[str, Task] = {}
        
        # Components
        self.planner = TaskPlanner(message_bus)
        self.executor = TaskExecutor(message_bus)
        
        # Persistence
        self.state_file = Path("data/task_engine_state.json")
        self._load_state()
    
    def run(self):
        """Main engine loop"""
        logger.info("Task engine started")
        
        while not self.shutdown_event.is_set():
            try:
                # Check for new messages
                self._process_messages()
                
                # Execute ready tasks
                self._execute_ready_tasks()
                
                # Small sleep to prevent busy loop
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Task engine error: {e}")
        
        self._save_state()
        logger.info("Task engine stopped")
    
    def _process_messages(self):
        """Process incoming messages"""
        try:
            while True:
                msg = self.input_queue.get_nowait()
                self._handle_message(msg)
        except:
            pass  # Queue empty
    
    def _handle_message(self, msg: dict):
        """Handle a single message"""
        msg_type = msg.get("type")
        payload = msg.get("payload", {})
        
        if msg_type == "task_create":
            # New goal to execute
            goal_id = payload.get("goal_id")
            goal_desc = payload.get("description")
            self._create_plan(goal_id, goal_desc)
        
        elif msg_type == "job_triggered":
            # Scheduled job triggered
            job_id = payload.get("job_id")
            self._handle_scheduled_job(job_id, payload)
        
        elif msg_type == "task_update":
            # External update to a task
            task_id = payload.get("task_id")
            status = payload.get("status")
            self._update_task_status(task_id, status, payload)
    
    def _create_plan(self, goal_id: str, goal_description: str):
        """Create an execution plan for a goal"""
        logger.info(f"Creating plan for goal: {goal_description}")
        
        # Use planner to break down goal
        plan = self.planner.create_plan(goal_id, goal_description)
        self.plans[goal_id] = plan
        
        # Log activity
        self._log_activity("plan_created", {
            "goal_id": goal_id,
            "description": goal_description,
            "task_count": len(plan.tasks)
        })
        
        logger.info(f"Plan created with {len(plan.tasks)} tasks")
    
    def _execute_ready_tasks(self):
        """Execute tasks that are ready"""
        # Check capacity
        available_slots = self.max_concurrent - len(self.running_tasks)
        if available_slots <= 0:
            return
        
        # Get ready tasks from all plans
        ready_tasks = []
        for plan in self.plans.values():
            ready_tasks.extend(plan.get_ready_tasks())
        
        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Execute up to available slots
        for task in ready_tasks[:available_slots]:
            self._execute_task(task)
    
    def _execute_task(self, task: Task):
        """Execute a single task"""
        logger.info(f"Executing task: {task.description}")
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        self.running_tasks[task.id] = task
        
        try:
            # Execute via executor
            result = self.executor.execute(task)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            logger.info(f"Task completed: {task.description}")
            
            # Log activity
            self._log_activity("task_completed", {
                "task_id": task.id,
                "description": task.description,
                "duration": task.completed_at - task.started_at
            })
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            logger.error(f"Task failed: {task.description} - {e}")
            
            self._log_activity("task_failed", {
                "task_id": task.id,
                "description": task.description,
                "error": str(e)
            })
        
        finally:
            del self.running_tasks[task.id]
            self._check_plan_completion(task.goal_id)
    
    def _check_plan_completion(self, goal_id: str):
        """Check if a plan is complete"""
        plan = self.plans.get(goal_id)
        if not plan:
            return
        
        all_done = all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            for t in plan.tasks
        )
        
        if all_done:
            # Generate completion report
            completed = sum(1 for t in plan.tasks if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
            
            logger.info(
                f"Plan complete for goal {goal_id}: "
                f"{completed} succeeded, {failed} failed"
            )
            
            self._log_activity("goal_completed", {
                "goal_id": goal_id,
                "description": plan.goal_description,
                "tasks_completed": completed,
                "tasks_failed": failed
            })
            
            # Clean up
            del self.plans[goal_id]
    
    def _handle_scheduled_job(self, job_id: str, payload: dict):
        """Handle a triggered scheduled job"""
        job_type = payload.get("job_type")
        
        if job_type == "research":
            # Background research task
            topic = payload.get("topic")
            self._create_plan(
                goal_id=f"scheduled_{job_id}_{int(time.time())}",
                goal_description=f"Research: {topic}"
            )
        
        elif job_type == "digest":
            # Generate daily digest
            self._generate_digest()
    
    def _generate_digest(self):
        """Generate activity digest"""
        self.message_bus.put({
            "type": "digest_request",
            "source": "task_engine",
            "target": "reporter",
            "payload": {},
            "timestamp": time.time()
        })
    
    def _log_activity(self, activity_type: str, details: dict):
        """Log activity to reporter"""
        self.message_bus.put({
            "type": "activity_log",
            "source": "task_engine",
            "target": "reporter",
            "payload": {
                "activity_type": activity_type,
                "details": details,
                "timestamp": time.time()
            },
            "timestamp": time.time()
        })
    
    def _update_task_status(self, task_id: str, status: str, payload: dict):
        """Update task status from external source"""
        for plan in self.plans.values():
            for task in plan.tasks:
                if task.id == task_id:
                    task.status = TaskStatus(status)
                    if "result" in payload:
                        task.result = payload["result"]
                    return
    
    def _load_state(self):
        """Load persisted state"""
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text())
                # Restore plans (simplified - would need full deserialization)
                logger.info(f"Loaded {len(state.get('plans', []))} plans from state")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def _save_state(self):
        """Save state for persistence"""
        state = {
            "plans": [
                {
                    "goal_id": p.goal_id,
                    "goal_description": p.goal_description,
                    "tasks": [t.to_dict() for t in p.tasks]
                }
                for p in self.plans.values()
            ]
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2))


class TaskPlanner:
    """
    Breaks goals into executable tasks.
    
    Uses LLM to understand goal and create task sequence.
    """
    
    def __init__(self, message_bus: Queue):
        self.message_bus = message_bus
    
    def create_plan(self, goal_id: str, goal_description: str) -> ExecutionPlan:
        """Create an execution plan for a goal"""
        # Ask LLM to break down the goal
        planning_prompt = f"""
        Break down this goal into specific, executable tasks:
        
        Goal: {goal_description}
        
        For each task, specify:
        1. Description (what to do)
        2. Type: research, generate, analyze, communicate, organize
        3. Dependencies (which tasks must complete first)
        4. Tool to use (if applicable): web_search, file_write, email_draft, etc.
        
        Return as JSON:
        {{
            "tasks": [
                {{
                    "description": "...",
                    "type": "research|generate|analyze|communicate|organize",
                    "depends_on": [],
                    "tool": "tool_name or null",
                    "tool_params": {{}}
                }}
            ]
        }}
        """
        
        # For now, create a simple plan
        # In production, this would call the LLM via message bus
        tasks = self._generate_tasks(goal_id, goal_description)
        
        return ExecutionPlan(
            goal_id=goal_id,
            goal_description=goal_description,
            tasks=tasks
        )
    
    def _generate_tasks(self, goal_id: str, goal_description: str) -> list[Task]:
        """Generate tasks for a goal"""
        # Simple heuristic-based planning
        # In production, use LLM for intelligent planning
        
        tasks = []
        desc_lower = goal_description.lower()
        
        # Research-type goals
        if any(kw in desc_lower for kw in ["research", "find", "search", "learn about"]):
            tasks.append(Task(
                id=f"{goal_id}_research",
                goal_id=goal_id,
                description=f"Search for information about: {goal_description}",
                task_type=TaskType.RESEARCH,
                tool="web_search",
                tool_params={"query": goal_description}
            ))
            tasks.append(Task(
                id=f"{goal_id}_summarize",
                goal_id=goal_id,
                description="Summarize research findings",
                task_type=TaskType.GENERATE,
                depends_on=[f"{goal_id}_research"]
            ))
        
        # Writing-type goals
        elif any(kw in desc_lower for kw in ["write", "draft", "create", "compose"]):
            tasks.append(Task(
                id=f"{goal_id}_outline",
                goal_id=goal_id,
                description=f"Create outline for: {goal_description}",
                task_type=TaskType.GENERATE
            ))
            tasks.append(Task(
                id=f"{goal_id}_draft",
                goal_id=goal_id,
                description="Write first draft",
                task_type=TaskType.GENERATE,
                depends_on=[f"{goal_id}_outline"]
            ))
            tasks.append(Task(
                id=f"{goal_id}_review",
                goal_id=goal_id,
                description="Review and refine draft",
                task_type=TaskType.ANALYZE,
                depends_on=[f"{goal_id}_draft"]
            ))
        
        # Default: single task
        else:
            tasks.append(Task(
                id=f"{goal_id}_main",
                goal_id=goal_id,
                description=goal_description,
                task_type=TaskType.CUSTOM
            ))
        
        return tasks


class TaskExecutor:
    """
    Executes individual tasks.
    
    Maps task types to execution strategies.
    """
    
    def __init__(self, message_bus: Queue):
        self.message_bus = message_bus
        
        # Tool registry
        self.tools = {
            "web_search": self._execute_web_search,
            "file_write": self._execute_file_write,
            "file_read": self._execute_file_read,
            "email_draft": self._execute_email_draft,
            "calendar_check": self._execute_calendar_check,
        }
    
    def execute(self, task: Task) -> Any:
        """Execute a task and return result"""
        if task.tool and task.tool in self.tools:
            return self.tools[task.tool](task)
        
        # Default: use LLM to execute
        return self._execute_with_llm(task)
    
    def _execute_web_search(self, task: Task) -> dict:
        """Execute web search task"""
        from Functions.web_search import search_web
        
        query = task.tool_params.get("query", task.description)
        results = search_web(query, max_results=5)
        
        return {
            "query": query,
            "results": results,
            "result_count": len(results)
        }
    
    def _execute_file_write(self, task: Task) -> dict:
        """Execute file write task"""
        path = task.tool_params.get("path")
        content = task.tool_params.get("content")
        
        if not path or not content:
            raise ValueError("File write requires 'path' and 'content' params")
        
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        
        return {"path": str(path), "bytes_written": len(content)}
    
    def _execute_file_read(self, task: Task) -> dict:
        """Execute file read task"""
        path = task.tool_params.get("path")
        
        if not path:
            raise ValueError("File read requires 'path' param")
        
        content = Path(path).read_text()
        return {"path": str(path), "content": content}
    
    def _execute_email_draft(self, task: Task) -> dict:
        """Execute email draft task"""
        # Would integrate with email service
        # For now, generate draft content
        return self._execute_with_llm(task)
    
    def _execute_calendar_check(self, task: Task) -> dict:
        """Execute calendar check task"""
        # Would integrate with calendar service
        return {"message": "Calendar integration not configured"}
    
    def _execute_with_llm(self, task: Task) -> dict:
        """Execute task using LLM"""
        # Send to model worker via message bus
        correlation_id = str(uuid.uuid4())
        
        self.message_bus.put({
            "type": "model_request",
            "source": "task_executor",
            "target": "model_research",
            "payload": {
                "prompt": f"Execute this task: {task.description}",
                "task_context": task.to_dict()
            },
            "correlation_id": correlation_id,
            "timestamp": time.time()
        })
        
        # In production, would wait for response
        # For now, return placeholder
        return {"status": "submitted", "correlation_id": correlation_id}
```

---

## 4. AUDIO PIPELINE

### Purpose
Always-on speech recognition and synthesis for voice interaction without wake words.

### Implementation

#### File: `audio/audio_pipeline.py`

```python
"""
Audio Pipeline

Handles all audio I/O for Senter:
- Voice Activity Detection (VAD)
- Speech-to-Text (STT) via Whisper
- Text-to-Speech (TTS) via Piper
- Audio stream management

Works with gaze detection: only transcribes when user is paying attention.
"""

import time
import logging
import numpy as np
from typing import Optional
from multiprocessing import Queue, Event
from pathlib import Path
import threading

logger = logging.getLogger('senter.audio')


class AudioPipeline:
    """
    Main audio pipeline controller.
    
    Manages:
    - Continuous audio capture
    - Voice activity detection
    - Speech transcription (when attention detected)
    - Speech synthesis for responses
    """
    
    def __init__(
        self,
        stt_model: str,
        tts_model: str,
        vad_threshold: float,
        output_queue: Queue,  # Message bus
        input_queue: Queue,   # Commands/text to speak
        shutdown_event: Event
    ):
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.vad_threshold = vad_threshold
        self.output_queue = output_queue
        self.input_queue = input_queue
        self.shutdown_event = shutdown_event
        
        # State
        self.is_listening = False
        self.has_attention = False
        self.audio_buffer = AudioBuffer(sample_rate=16000, buffer_seconds=30)
        
        # Components (lazy loaded)
        self._stt_engine = None
        self._tts_engine = None
        self._vad = None
    
    @property
    def stt_engine(self):
        if self._stt_engine is None:
            self._stt_engine = STTEngine(self.stt_model)
        return self._stt_engine
    
    @property
    def tts_engine(self):
        if self._tts_engine is None:
            self._tts_engine = TTSEngine(self.tts_model)
        return self._tts_engine
    
    @property
    def vad(self):
        if self._vad is None:
            self._vad = VoiceActivityDetector(self.vad_threshold)
        return self._vad
    
    def run(self):
        """Main pipeline loop"""
        logger.info("Audio pipeline started")
        
        # Start audio capture thread
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()
        
        # Start TTS thread
        tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
        tts_thread.start()
        
        # Main processing loop
        while not self.shutdown_event.is_set():
            try:
                self._process_messages()
                self._process_audio()
                time.sleep(0.05)  # 50ms loop
            except Exception as e:
                logger.error(f"Audio pipeline error: {e}")
        
        logger.info("Audio pipeline stopped")
    
    def _capture_loop(self):
        """Continuous audio capture"""
        try:
            import sounddevice as sd
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                self.audio_buffer.write(indata.copy())
            
            with sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype=np.float32,
                callback=audio_callback,
                blocksize=1600  # 100ms chunks
            ):
                while not self.shutdown_event.is_set():
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
    
    def _process_messages(self):
        """Process incoming messages"""
        try:
            while True:
                msg = self.input_queue.get_nowait()
                self._handle_message(msg)
        except:
            pass
    
    def _handle_message(self, msg: dict):
        """Handle a message"""
        msg_type = msg.get("type")
        payload = msg.get("payload", {})
        
        if msg_type == "attention_gained":
            self.has_attention = True
            logger.info("Attention gained - activating voice")
        
        elif msg_type == "attention_lost":
            self.has_attention = False
            logger.info("Attention lost - deactivating voice")
        
        elif msg_type == "speak":
            # Queue text for TTS
            text = payload.get("text")
            if text:
                self._speak(text)
        
        elif msg_type == "model_response":
            # Speak model response
            text = payload.get("response")
            if text and self.has_attention:
                self._speak(text)
    
    def _process_audio(self):
        """Process captured audio"""
        if not self.has_attention:
            return
        
        # Get recent audio
        audio_data = self.audio_buffer.get_recent(seconds=2)
        if audio_data is None or len(audio_data) == 0:
            return
        
        # Check for voice activity
        if not self.vad.is_speech(audio_data):
            if self.is_listening:
                # Speech ended - transcribe
                self._transcribe_and_send()
            return
        
        # Voice detected
        if not self.is_listening:
            logger.debug("Speech detected - starting transcription buffer")
            self.is_listening = True
            self.audio_buffer.mark_speech_start()
    
    def _transcribe_and_send(self):
        """Transcribe buffered speech and send to message bus"""
        self.is_listening = False
        
        # Get speech segment
        speech_audio = self.audio_buffer.get_speech_segment()
        if speech_audio is None or len(speech_audio) < 1600:  # Min 100ms
            return
        
        # Transcribe
        try:
            text = self.stt_engine.transcribe(speech_audio)
            if text and len(text.strip()) > 0:
                logger.info(f"Transcribed: {text}")
                
                # Send to message bus
                self.output_queue.put({
                    "type": "user_voice",
                    "source": "audio_pipeline",
                    "target": None,
                    "payload": {
                        "text": text,
                        "audio_duration": len(speech_audio) / 16000
                    },
                    "timestamp": time.time()
                })
        except Exception as e:
            logger.error(f"Transcription error: {e}")
    
    def _speak(self, text: str):
        """Synthesize and play speech"""
        try:
            audio = self.tts_engine.synthesize(text)
            self._play_audio(audio)
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    def _play_audio(self, audio: np.ndarray):
        """Play audio through speakers"""
        try:
            import sounddevice as sd
            sd.play(audio, samplerate=22050)
            sd.wait()
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def _tts_loop(self):
        """TTS processing loop"""
        # Handle queued TTS requests
        pass


class AudioBuffer:
    """
    Ring buffer for audio data.
    
    Maintains recent audio history for:
    - Voice activity detection
    - Speech transcription
    """
    
    def __init__(self, sample_rate: int, buffer_seconds: int):
        self.sample_rate = sample_rate
        self.buffer_size = sample_rate * buffer_seconds
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.speech_start_pos = None
        self._lock = threading.Lock()
    
    def write(self, data: np.ndarray):
        """Write audio data to buffer"""
        with self._lock:
            data = data.flatten()
            data_len = len(data)
            
            if data_len >= self.buffer_size:
                self.buffer[:] = data[-self.buffer_size:]
                self.write_pos = 0
            else:
                end_pos = self.write_pos + data_len
                if end_pos <= self.buffer_size:
                    self.buffer[self.write_pos:end_pos] = data
                else:
                    first_part = self.buffer_size - self.write_pos
                    self.buffer[self.write_pos:] = data[:first_part]
                    self.buffer[:data_len - first_part] = data[first_part:]
                self.write_pos = end_pos % self.buffer_size
    
    def get_recent(self, seconds: float) -> np.ndarray:
        """Get recent audio data"""
        with self._lock:
            samples = int(seconds * self.sample_rate)
            if samples > self.buffer_size:
                samples = self.buffer_size
            
            if self.write_pos >= samples:
                return self.buffer[self.write_pos - samples:self.write_pos].copy()
            else:
                return np.concatenate([
                    self.buffer[-(samples - self.write_pos):],
                    self.buffer[:self.write_pos]
                ])
    
    def mark_speech_start(self):
        """Mark current position as speech start"""
        with self._lock:
            self.speech_start_pos = self.write_pos
    
    def get_speech_segment(self) -> Optional[np.ndarray]:
        """Get audio from speech start to now"""
        with self._lock:
            if self.speech_start_pos is None:
                return None
            
            if self.write_pos > self.speech_start_pos:
                segment = self.buffer[self.speech_start_pos:self.write_pos].copy()
            else:
                segment = np.concatenate([
                    self.buffer[self.speech_start_pos:],
                    self.buffer[:self.write_pos]
                ])
            
            self.speech_start_pos = None
            return segment


class VoiceActivityDetector:
    """
    Detects voice activity in audio.
    
    Uses Silero VAD for accurate detection.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            try:
                import torch
                self._model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
                self._get_speech_timestamps = utils[0]
            except Exception as e:
                logger.warning(f"Could not load Silero VAD: {e}")
                self._model = "fallback"
        return self._model
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech"""
        if self.model == "fallback":
            # Fallback: simple energy-based detection
            energy = np.sqrt(np.mean(audio ** 2))
            return energy > 0.01
        
        try:
            import torch
            audio_tensor = torch.from_numpy(audio)
            speech_prob = self.model(audio_tensor, 16000).item()
            return speech_prob > self.threshold
        except:
            return False


class STTEngine:
    """
    Speech-to-Text engine using Whisper.
    """
    
    def __init__(self, model_name: str = "whisper-small"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            try:
                import whisper
                # Map friendly names to whisper model names
                model_map = {
                    "whisper-tiny": "tiny",
                    "whisper-base": "base",
                    "whisper-small": "small",
                    "whisper-medium": "medium",
                    "whisper-large": "large",
                }
                whisper_model = model_map.get(self.model_name, "small")
                self._model = whisper.load_model(whisper_model)
                logger.info(f"Loaded Whisper model: {whisper_model}")
            except Exception as e:
                logger.error(f"Could not load Whisper: {e}")
                raise
        return self._model
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text"""
        result = self.model.transcribe(
            audio,
            language="en",
            fp16=False
        )
        return result["text"].strip()


class TTSEngine:
    """
    Text-to-Speech engine using Piper.
    """
    
    def __init__(self, model_name: str = "piper"):
        self.model_name = model_name
        self._model = None
    
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio"""
        try:
            # Try piper-tts
            from piper import PiperVoice
            
            if self._model is None:
                # Load default voice
                voice_path = Path("models/piper/en_US-lessac-medium.onnx")
                if voice_path.exists():
                    self._model = PiperVoice.load(str(voice_path))
                else:
                    raise FileNotFoundError("Piper voice not found")
            
            # Synthesize
            audio_data = []
            for audio_bytes in self._model.synthesize_stream_raw(text):
                audio_data.append(np.frombuffer(audio_bytes, dtype=np.int16))
            
            audio = np.concatenate(audio_data).astype(np.float32) / 32768.0
            return audio
            
        except ImportError:
            logger.warning("Piper not available, using fallback")
            return self._fallback_synthesize(text)
    
    def _fallback_synthesize(self, text: str) -> np.ndarray:
        """Fallback TTS using system speech"""
        try:
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
                temp_path = f.name
            
            # macOS say command
            subprocess.run(
                ["say", "-o", temp_path, text],
                capture_output=True,
                timeout=30
            )
            
            # Load audio file
            import soundfile as sf
            audio, sr = sf.read(temp_path)
            Path(temp_path).unlink()
            
            return audio
            
        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")
            return np.zeros(1000, dtype=np.float32)
```

---

## 5. GAZE DETECTION

### Purpose
Detect when user is looking at the camera to enable wake-word-free interaction.

### Implementation

#### File: `vision/gaze_detector.py`

```python
"""
Gaze Detection

Determines if user is paying attention to Senter by:
- Detecting face presence
- Tracking eye gaze direction
- Estimating attention level

When attention detected, enables voice interaction.
"""

import time
import logging
import numpy as np
from typing import Optional, Tuple
from multiprocessing import Queue, Event
import threading

logger = logging.getLogger('senter.gaze')


class GazeDetector:
    """
    Main gaze detection controller.
    
    Uses MediaPipe Face Mesh for:
    - Face detection
    - Eye landmark tracking
    - Gaze direction estimation
    """
    
    def __init__(
        self,
        camera_id: int,
        attention_threshold: float,
        output_queue: Queue,
        shutdown_event: Event
    ):
        self.camera_id = camera_id
        self.attention_threshold = attention_threshold
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event
        
        # State
        self.has_attention = False
        self.attention_start_time = None
        self.last_attention_time = None
        
        # Smoothing
        self.attention_history = []
        self.history_size = 10  # Frames to average
        
        # Attention timeout (seconds without attention before declaring lost)
        self.attention_timeout = 2.0
        
        # Components
        self._face_mesh = None
        self._camera = None
    
    @property
    def face_mesh(self):
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except ImportError:
                logger.error("MediaPipe not installed")
                raise
        return self._face_mesh
    
    @property
    def camera(self):
        if self._camera is None:
            import cv2
            self._camera = cv2.VideoCapture(self.camera_id)
            if not self._camera.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_id}")
            # Set resolution
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._camera.set(cv2.CAP_PROP_FPS, 15)
        return self._camera
    
    def run(self):
        """Main detection loop"""
        logger.info("Gaze detector started")
        
        try:
            while not self.shutdown_event.is_set():
                self._process_frame()
                time.sleep(0.066)  # ~15 FPS
        except Exception as e:
            logger.error(f"Gaze detector error: {e}")
        finally:
            self._cleanup()
        
        logger.info("Gaze detector stopped")
    
    def _process_frame(self):
        """Process a single camera frame"""
        import cv2
        
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh
        results = self.face_mesh.process(rgb_frame)
        
        # Calculate attention score
        attention_score = self._calculate_attention(results, frame.shape)
        
        # Update history
        self.attention_history.append(attention_score)
        if len(self.attention_history) > self.history_size:
            self.attention_history.pop(0)
        
        # Smooth score
        smoothed_score = np.mean(self.attention_history)
        
        # Update attention state
        self._update_attention_state(smoothed_score)
    
    def _calculate_attention(self, results, frame_shape) -> float:
        """
        Calculate attention score from face mesh results.
        
        Factors:
        - Face presence (required)
        - Face orientation (looking at camera)
        - Eye openness
        - Gaze direction
        """
        if not results.multi_face_landmarks:
            return 0.0
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_shape[:2]
        
        # Get key landmarks
        nose_tip = landmarks[1]
        left_eye_center = self._get_eye_center(landmarks, "left")
        right_eye_center = self._get_eye_center(landmarks, "right")
        
        # Calculate face orientation
        # If nose is roughly centered, face is pointing at camera
        nose_x = nose_tip.x
        face_center_score = 1.0 - abs(nose_x - 0.5) * 2  # 1.0 at center, 0.0 at edges
        face_center_score = max(0, face_center_score)
        
        # Calculate eye openness (using eye aspect ratio)
        left_ear = self._eye_aspect_ratio(landmarks, "left")
        right_ear = self._eye_aspect_ratio(landmarks, "right")
        avg_ear = (left_ear + right_ear) / 2
        
        # Eyes open: EAR > 0.2, closed: EAR < 0.1
        eye_open_score = min(1.0, avg_ear / 0.25)
        
        # Calculate gaze direction
        gaze_score = self._estimate_gaze_score(landmarks, frame_shape)
        
        # Combine scores
        # Face must be present and oriented toward camera
        # Eyes should be open
        # Gaze should be toward camera
        attention_score = (
            face_center_score * 0.3 +
            eye_open_score * 0.3 +
            gaze_score * 0.4
        )
        
        return attention_score
    
    def _get_eye_center(self, landmarks, eye: str) -> Tuple[float, float]:
        """Get eye center coordinates"""
        if eye == "left":
            indices = [33, 133, 160, 159, 158, 144, 145, 153]
        else:
            indices = [362, 263, 387, 386, 385, 373, 374, 380]
        
        x = np.mean([landmarks[i].x for i in indices])
        y = np.mean([landmarks[i].y for i in indices])
        return (x, y)
    
    def _eye_aspect_ratio(self, landmarks, eye: str) -> float:
        """Calculate eye aspect ratio (EAR)"""
        if eye == "left":
            # Vertical landmarks
            p1 = landmarks[159]  # Top
            p2 = landmarks[145]  # Bottom
            # Horizontal landmarks
            p3 = landmarks[33]   # Left corner
            p4 = landmarks[133]  # Right corner
        else:
            p1 = landmarks[386]
            p2 = landmarks[374]
            p3 = landmarks[362]
            p4 = landmarks[263]
        
        # Calculate distances
        vertical = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        horizontal = np.sqrt((p3.x - p4.x)**2 + (p3.y - p4.y)**2)
        
        if horizontal == 0:
            return 0
        
        return vertical / horizontal
    
    def _estimate_gaze_score(self, landmarks, frame_shape) -> float:
        """
        Estimate if user is looking at camera.
        
        Uses iris position relative to eye corners.
        """
        try:
            # Get iris landmarks (indices 468-477 for refined landmarks)
            left_iris = landmarks[468]
            right_iris = landmarks[473]
            
            # Get eye corners
            left_inner = landmarks[133]
            left_outer = landmarks[33]
            right_inner = landmarks[362]
            right_outer = landmarks[263]
            
            # Calculate iris position relative to eye width
            # 0.5 = center = looking straight
            left_eye_width = abs(left_outer.x - left_inner.x)
            right_eye_width = abs(right_outer.x - right_inner.x)
            
            if left_eye_width == 0 or right_eye_width == 0:
                return 0.5
            
            left_iris_pos = (left_iris.x - left_outer.x) / left_eye_width
            right_iris_pos = (right_iris.x - right_outer.x) / right_eye_width
            
            # Average position
            avg_iris_pos = (left_iris_pos + right_iris_pos) / 2
            
            # Score: 1.0 at center (0.5), lower toward edges
            gaze_score = 1.0 - abs(avg_iris_pos - 0.5) * 2
            gaze_score = max(0, min(1, gaze_score))
            
            return gaze_score
            
        except (IndexError, AttributeError):
            # Refined landmarks not available
            return 0.5
    
    def _update_attention_state(self, score: float):
        """Update attention state based on score"""
        now = time.time()
        
        if score >= self.attention_threshold:
            self.last_attention_time = now
            
            if not self.has_attention:
                # Attention gained
                self.has_attention = True
                self.attention_start_time = now
                
                logger.info(f"Attention gained (score: {score:.2f})")
                
                self.output_queue.put({
                    "type": "attention_gained",
                    "source": "gaze_detector",
                    "target": None,
                    "payload": {
                        "score": score,
                        "timestamp": now
                    },
                    "timestamp": now
                })
        
        else:
            # Check timeout
            if self.has_attention and self.last_attention_time:
                time_since_attention = now - self.last_attention_time
                
                if time_since_attention > self.attention_timeout:
                    # Attention lost
                    self.has_attention = False
                    duration = now - self.attention_start_time if self.attention_start_time else 0
                    
                    logger.info(f"Attention lost after {duration:.1f}s")
                    
                    self.output_queue.put({
                        "type": "attention_lost",
                        "source": "gaze_detector",
                        "target": None,
                        "payload": {
                            "duration": duration,
                            "timestamp": now
                        },
                        "timestamp": now
                    })
                    
                    self.attention_start_time = None
    
    def _cleanup(self):
        """Clean up resources"""
        if self._camera is not None:
            self._camera.release()
        if self._face_mesh is not None:
            self._face_mesh.close()
```

---

## 6. DUAL MODEL WORKERS

### Purpose
Run two separate LLM inference processes simultaneously - one for user responses, one for background research.

### Implementation

#### File: `workers/model_worker.py`

```python
"""
Model Worker

Dedicated process for LLM inference.

Multiple workers can run simultaneously:
- Primary worker: User-facing responses (low latency priority)
- Research worker: Background tasks (throughput priority)
"""

import time
import logging
from typing import Optional
from multiprocessing import Queue, Event
from pathlib import Path

logger = logging.getLogger('senter.worker')


class ModelWorker:
    """
    LLM inference worker process.
    
    Each worker:
    - Has its own model instance
    - Processes requests from queue
    - Returns responses to output queue
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        input_queue: Queue,
        output_queue: Queue,
        shutdown_event: Event,
        priority: str = "balanced"  # "latency", "throughput", "balanced"
    ):
        self.name = name
        self.model_name = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event
        self.priority = priority
        
        # Model instance
        self._client = None
        
        # Statistics
        self.requests_processed = 0
        self.total_tokens = 0
        self.total_latency = 0
    
    @property
    def client(self):
        """Lazy load model client"""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self):
        """Create model client based on configuration"""
        # Try Ollama first
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            # Test connection
            client.models.list()
            logger.info(f"Worker {self.name}: Connected to Ollama")
            return client
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        
        # Fallback to other providers
        raise RuntimeError("No model provider available")
    
    def run(self):
        """Main worker loop"""
        logger.info(f"Model worker '{self.name}' started with model '{self.model_name}'")
        
        while not self.shutdown_event.is_set():
            try:
                # Get request with timeout
                try:
                    msg = self.input_queue.get(timeout=1.0)
                except:
                    continue
                
                # Check if this message is for us
                if not self._should_process(msg):
                    continue
                
                # Process request
                self._process_request(msg)
                
            except Exception as e:
                logger.error(f"Worker {self.name} error: {e}")
        
        logger.info(f"Model worker '{self.name}' stopped")
    
    def _should_process(self, msg: dict) -> bool:
        """Check if we should process this message"""
        msg_type = msg.get("type")
        target = msg.get("target")
        
        # Only process model requests
        if msg_type not in ("model_request", "user_query", "user_voice"):
            return False
        
        # Check target
        if target:
            return target == f"model_{self.name}"
        
        # Route based on worker role
        if msg_type in ("user_query", "user_voice"):
            return self.name == "primary"
        
        return True
    
    def _process_request(self, msg: dict):
        """Process a model request"""
        start_time = time.time()
        
        payload = msg.get("payload", {})
        correlation_id = msg.get("correlation_id")
        
        # Build prompt
        if msg.get("type") in ("user_query", "user_voice"):
            prompt = payload.get("text", payload.get("query", ""))
            messages = [{"role": "user", "content": prompt}]
        else:
            prompt = payload.get("prompt", "")
            messages = payload.get("messages", [{"role": "user", "content": prompt}])
        
        # Add system prompt
        system_prompt = self._get_system_prompt(msg)
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        try:
            # Call model
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=self._get_max_tokens()
            )
            
            response_text = response.choices[0].message.content
            
            # Calculate stats
            latency = time.time() - start_time
            tokens = response.usage.total_tokens if response.usage else 0
            
            self.requests_processed += 1
            self.total_tokens += tokens
            self.total_latency += latency
            
            logger.debug(
                f"Worker {self.name}: Generated {tokens} tokens in {latency:.2f}s"
            )
            
            # Send response
            self.output_queue.put({
                "type": "model_response",
                "source": f"model_{self.name}",
                "target": msg.get("source"),
                "payload": {
                    "response": response_text,
                    "tokens": tokens,
                    "latency": latency,
                    "model": self.model_name
                },
                "correlation_id": correlation_id,
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            
            self.output_queue.put({
                "type": "model_response",
                "source": f"model_{self.name}",
                "target": msg.get("source"),
                "payload": {
                    "error": str(e),
                    "model": self.model_name
                },
                "correlation_id": correlation_id,
                "timestamp": time.time()
            })
    
    def _get_system_prompt(self, msg: dict) -> Optional[str]:
        """Get appropriate system prompt"""
        if self.name == "primary":
            return """You are Senter, a helpful AI assistant. 
            Be concise, friendly, and helpful.
            If you don't know something, say so."""
        
        elif self.name == "research":
            return """You are a research assistant for Senter.
            Your job is to gather information, analyze data, and prepare summaries.
            Be thorough but concise. Focus on facts and key insights."""
        
        return None
    
    def _get_max_tokens(self) -> int:
        """Get max tokens based on priority"""
        if self.priority == "latency":
            return 500  # Shorter for faster response
        elif self.priority == "throughput":
            return 2000  # Longer for thorough research
        return 1000  # Balanced


class WorkerPool:
    """
    Manages multiple model workers.
    
    Provides:
    - Worker lifecycle management
    - Load balancing
    - Health monitoring
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.workers: dict[str, ModelWorker] = {}
        self.processes: dict[str, 'Process'] = {}
    
    def start_workers(
        self,
        message_bus: Queue,
        response_queues: dict[str, Queue],
        shutdown_event: Event
    ):
        """Start all configured workers"""
        from multiprocessing import Process
        
        worker_config = self.config.get("model_workers", {})
        
        if not worker_config.get("enabled", True):
            logger.info("Model workers disabled")
            return
        
        models = worker_config.get("models", {})
        
        # Primary worker
        if "primary" in models:
            response_queue = Queue()
            response_queues["model_primary"] = response_queue
            
            worker = ModelWorker(
                name="primary",
                model=models["primary"],
                input_queue=message_bus,
                output_queue=response_queue,
                shutdown_event=shutdown_event,
                priority="latency"
            )
            
            p = Process(target=worker.run, name="model_worker_primary")
            p.start()
            
            self.workers["primary"] = worker
            self.processes["primary"] = p
        
        # Research worker
        if "research" in models:
            response_queue = Queue()
            response_queues["model_research"] = response_queue
            
            worker = ModelWorker(
                name="research",
                model=models["research"],
                input_queue=message_bus,
                output_queue=response_queue,
                shutdown_event=shutdown_event,
                priority="throughput"
            )
            
            p = Process(target=worker.run, name="model_worker_research")
            p.start()
            
            self.workers["research"] = worker
            self.processes["research"] = p
        
        logger.info(f"Started {len(self.workers)} model workers")
    
    def stop_workers(self):
        """Stop all workers"""
        for name, process in self.processes.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        
        self.workers.clear()
        self.processes.clear()
```

---

## 7. LEARNING DATABASE

### Purpose
Store and analyze user behavior over time to build a rich understanding of preferences, patterns, and habits.

### Implementation

#### File: `learning/learning_db.py`

```python
"""
Learning Database

Time-series storage for user behavior data.

Stores:
- Interaction events (queries, responses, feedback)
- Timing patterns (when user is active)
- Topic preferences (what user asks about)
- Style preferences (how user likes responses)
- Goal patterns (types of goals, completion rates)
"""

import json
import time
import sqlite3
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
import numpy as np

logger = logging.getLogger('senter.learning')


@dataclass
class InteractionEvent:
    """Single interaction event"""
    timestamp: float
    event_type: str  # query, response, feedback, goal_created, etc.
    content: str
    metadata: dict
    
    # Derived features
    hour_of_day: int = None
    day_of_week: int = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        dt = datetime.fromtimestamp(self.timestamp)
        self.hour_of_day = dt.hour
        self.day_of_week = dt.weekday()


class LearningDatabase:
    """
    SQLite-based learning database.
    
    Stores interaction history and derived insights.
    """
    
    def __init__(self, db_path: str = "data/learning/behavior.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = None
        self._init_db()
    
    @property
    def conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _init_db(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                content TEXT,
                metadata TEXT,
                hour_of_day INTEGER,
                day_of_week INTEGER,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Topics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT UNIQUE NOT NULL,
                count INTEGER DEFAULT 1,
                last_seen REAL,
                embedding BLOB
            )
        """)
        
        # Preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_type TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 1,
                last_updated REAL,
                UNIQUE(preference_type, preference_key)
            )
        """)
        
        # Activity patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                last_updated REAL,
                UNIQUE(pattern_type, pattern_key)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_hour ON events(hour_of_day)")
        
        self.conn.commit()
    
    def log_event(self, event: InteractionEvent):
        """Log an interaction event"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO events (timestamp, event_type, content, metadata, 
                               hour_of_day, day_of_week, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            event.event_type,
            event.content,
            json.dumps(event.metadata),
            event.hour_of_day,
            event.day_of_week,
            event.session_id
        ))
        
        self.conn.commit()
        
        # Update derived data
        self._update_topics(event)
        self._update_activity_patterns(event)
    
    def _update_topics(self, event: InteractionEvent):
        """Extract and update topics from event"""
        if event.event_type != "query":
            return
        
        # Simple topic extraction (would use NLP in production)
        topics = self._extract_topics(event.content)
        
        cursor = self.conn.cursor()
        for topic in topics:
            cursor.execute("""
                INSERT INTO topics (topic, count, last_seen)
                VALUES (?, 1, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    count = count + 1,
                    last_seen = ?
            """, (topic, event.timestamp, event.timestamp))
        
        self.conn.commit()
    
    def _extract_topics(self, text: str) -> list[str]:
        """Extract topics from text"""
        # Simple keyword extraction
        # In production, use spaCy or similar
        import re
        
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'it', 'its', 'they', 'them', 'their', 'about', 'help',
            'please', 'thanks', 'thank', 'hello', 'hi', 'hey'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter
        topics = [w for w in words if w not in stop_words]
        
        return topics[:5]  # Top 5 topics
    
    def _update_activity_patterns(self, event: InteractionEvent):
        """Update activity pattern statistics"""
        cursor = self.conn.cursor()
        
        # Hour of day pattern
        cursor.execute("""
            INSERT INTO activity_patterns (pattern_type, pattern_key, count, last_updated)
            VALUES ('hour', ?, 1, ?)
            ON CONFLICT(pattern_type, pattern_key) DO UPDATE SET
                count = count + 1,
                last_updated = ?
        """, (str(event.hour_of_day), event.timestamp, event.timestamp))
        
        # Day of week pattern
        cursor.execute("""
            INSERT INTO activity_patterns (pattern_type, pattern_key, count, last_updated)
            VALUES ('day', ?, 1, ?)
            ON CONFLICT(pattern_type, pattern_key) DO UPDATE SET
                count = count + 1,
                last_updated = ?
        """, (str(event.day_of_week), event.timestamp, event.timestamp))
        
        self.conn.commit()
    
    def update_preference(
        self, 
        pref_type: str, 
        pref_key: str, 
        pref_value: str,
        confidence_delta: float = 0.1
    ):
        """Update a learned preference"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO preferences (preference_type, preference_key, preference_value,
                                    confidence, evidence_count, last_updated)
            VALUES (?, ?, ?, 0.5, 1, ?)
            ON CONFLICT(preference_type, preference_key) DO UPDATE SET
                preference_value = ?,
                confidence = MIN(1.0, confidence + ?),
                evidence_count = evidence_count + 1,
                last_updated = ?
        """, (
            pref_type, pref_key, pref_value, time.time(),
            pref_value, confidence_delta, time.time()
        ))
        
        self.conn.commit()
    
    def get_preferences(self, pref_type: Optional[str] = None) -> list[dict]:
        """Get learned preferences"""
        cursor = self.conn.cursor()
        
        if pref_type:
            cursor.execute("""
                SELECT * FROM preferences WHERE preference_type = ?
                ORDER BY confidence DESC
            """, (pref_type,))
        else:
            cursor.execute("""
                SELECT * FROM preferences ORDER BY confidence DESC
            """)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_top_topics(self, limit: int = 10) -> list[dict]:
        """Get most frequent topics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT topic, count, last_seen
            FROM topics
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_activity_pattern(self, pattern_type: str) -> dict:
        """Get activity pattern (e.g., by hour or day)"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT pattern_key, count
            FROM activity_patterns
            WHERE pattern_type = ?
        """, (pattern_type,))
        
        return {row['pattern_key']: row['count'] for row in cursor.fetchall()}
    
    def get_peak_hours(self) -> list[int]:
        """Get hours when user is most active"""
        pattern = self.get_activity_pattern('hour')
        if not pattern:
            return [9, 10, 14, 15]  # Default
        
        # Sort by count and get top 4
        sorted_hours = sorted(pattern.items(), key=lambda x: x[1], reverse=True)
        return [int(h) for h, _ in sorted_hours[:4]]
    
    def get_recent_events(
        self, 
        event_type: Optional[str] = None,
        hours: int = 24
    ) -> list[dict]:
        """Get recent events"""
        cursor = self.conn.cursor()
        
        cutoff = time.time() - (hours * 3600)
        
        if event_type:
            cursor.execute("""
                SELECT * FROM events
                WHERE timestamp > ? AND event_type = ?
                ORDER BY timestamp DESC
            """, (cutoff, event_type))
        else:
            cursor.execute("""
                SELECT * FROM events
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> dict:
        """Get overall statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total events
        cursor.execute("SELECT COUNT(*) as count FROM events")
        stats['total_events'] = cursor.fetchone()['count']
        
        # Events by type
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM events
            GROUP BY event_type
        """)
        stats['events_by_type'] = {row['event_type']: row['count'] for row in cursor.fetchall()}
        
        # Unique topics
        cursor.execute("SELECT COUNT(*) as count FROM topics")
        stats['unique_topics'] = cursor.fetchone()['count']
        
        # Preferences count
        cursor.execute("SELECT COUNT(*) as count FROM preferences")
        stats['preferences_count'] = cursor.fetchone()['count']
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
```

#### File: `learning/behavior_analyzer.py`

```python
"""
Behavior Analyzer

Analyzes user behavior data to extract insights and predictions.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
import numpy as np

from .learning_db import LearningDatabase, InteractionEvent

logger = logging.getLogger('senter.analyzer')


class BehaviorAnalyzer:
    """
    Analyzes user behavior patterns.
    
    Provides:
    - Activity predictions
    - Topic trend analysis
    - Style preference detection
    - Engagement patterns
    """
    
    def __init__(self, db: LearningDatabase):
        self.db = db
    
    def analyze_session(self, session_events: list[InteractionEvent]) -> dict:
        """Analyze a single session"""
        if not session_events:
            return {}
        
        analysis = {
            "event_count": len(session_events),
            "duration_seconds": 0,
            "topics": [],
            "style_signals": {},
            "engagement_score": 0.5
        }
        
        # Duration
        if len(session_events) > 1:
            analysis["duration_seconds"] = (
                session_events[-1].timestamp - session_events[0].timestamp
            )
        
        # Topics
        all_topics = []
        for event in session_events:
            if event.event_type == "query":
                all_topics.extend(self.db._extract_topics(event.content))
        
        # Count topics
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        analysis["topics"] = sorted(
            topic_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Style signals
        analysis["style_signals"] = self._detect_style_signals(session_events)
        
        # Engagement
        analysis["engagement_score"] = self._calculate_engagement(session_events)
        
        return analysis
    
    def _detect_style_signals(self, events: list[InteractionEvent]) -> dict:
        """Detect style preferences from events"""
        signals = {}
        
        user_text = " ".join(
            e.content.lower() 
            for e in events 
            if e.event_type == "query"
        )
        
        # Response length preference
        if any(kw in user_text for kw in ["brief", "short", "tldr", "summarize"]):
            signals["response_length"] = "brief"
        elif any(kw in user_text for kw in ["detail", "explain", "elaborate", "expand"]):
            signals["response_length"] = "detailed"
        
        # Formality
        if any(kw in user_text for kw in ["please", "could you", "would you"]):
            signals["formality"] = "formal"
        elif any(kw in user_text for kw in ["hey", "yo", "gimme", "wanna"]):
            signals["formality"] = "casual"
        
        # Technical level
        if any(kw in user_text for kw in ["code", "function", "api", "debug", "implement"]):
            signals["technical_level"] = "high"
        elif any(kw in user_text for kw in ["explain like", "simple", "basic", "beginner"]):
            signals["technical_level"] = "low"
        
        return signals
    
    def _calculate_engagement(self, events: list[InteractionEvent]) -> float:
        """Calculate engagement score for session"""
        if not events:
            return 0.0
        
        # Factors:
        # - Query count
        # - Session duration
        # - Follow-up questions
        # - Positive feedback
        
        query_count = sum(1 for e in events if e.event_type == "query")
        
        # More queries = more engaged (up to a point)
        query_score = min(1.0, query_count / 10)
        
        # Longer sessions = more engaged
        if len(events) > 1:
            duration = events[-1].timestamp - events[0].timestamp
            duration_score = min(1.0, duration / 1800)  # 30 min max
        else:
            duration_score = 0.1
        
        # Check for feedback events
        feedback_events = [e for e in events if e.event_type == "feedback"]
        feedback_score = 0.5
        for fe in feedback_events:
            if fe.metadata.get("positive"):
                feedback_score = min(1.0, feedback_score + 0.2)
            else:
                feedback_score = max(0.0, feedback_score - 0.1)
        
        # Combine scores
        engagement = (query_score * 0.4 + duration_score * 0.3 + feedback_score * 0.3)
        
        return engagement
    
    def predict_next_active_time(self) -> Optional[datetime]:
        """Predict when user will next be active"""
        pattern = self.db.get_activity_pattern('hour')
        
        if not pattern:
            return None
        
        now = datetime.now()
        current_hour = now.hour
        
        # Find next peak hour
        peak_hours = self.db.get_peak_hours()
        
        for hour in peak_hours:
            if hour > current_hour:
                return now.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # Next peak is tomorrow
        next_peak = peak_hours[0] if peak_hours else 9
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=next_peak, minute=0, second=0, microsecond=0)
    
    def get_user_summary(self) -> dict:
        """Get comprehensive user behavior summary"""
        stats = self.db.get_statistics()
        preferences = self.db.get_preferences()
        topics = self.db.get_top_topics(10)
        peak_hours = self.db.get_peak_hours()
        
        return {
            "total_interactions": stats.get('total_events', 0),
            "unique_topics": stats.get('unique_topics', 0),
            "top_topics": [t['topic'] for t in topics[:5]],
            "peak_hours": peak_hours,
            "preferences": {
                p['preference_key']: p['preference_value']
                for p in preferences
                if p['confidence'] > 0.6
            },
            "next_active": self.predict_next_active_time()
        }
```

---

## 8. ACTION SCHEDULER

### Purpose
Cron-like system for scheduling background tasks, research, and maintenance.

### Implementation

#### File: `scheduler/action_scheduler.py`

```python
"""
Action Scheduler

Cron-like scheduling for Senter background tasks.

Supports:
- Time-based triggers (cron expressions)
- Event-based triggers
- One-time and recurring jobs
- Job persistence across restarts
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable
from pathlib import Path
from multiprocessing import Queue, Event
import threading

logger = logging.getLogger('senter.scheduler')


class TriggerType(Enum):
    CRON = "cron"       # Cron expression
    INTERVAL = "interval"  # Every N seconds
    ONCE = "once"       # One-time at specific datetime
    EVENT = "event"     # On specific event


@dataclass
class ScheduledJob:
    """A scheduled job"""
    id: str
    name: str
    job_type: str  # research, digest, maintenance, custom
    trigger_type: TriggerType
    trigger_config: dict
    payload: dict = field(default_factory=dict)
    
    # State
    enabled: bool = True
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    run_count: int = 0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['trigger_type'] = self.trigger_type.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledJob":
        d['trigger_type'] = TriggerType(d['trigger_type'])
        return cls(**d)


class ActionScheduler:
    """
    Main scheduler process.
    
    Manages scheduled jobs and triggers execution.
    """
    
    def __init__(
        self,
        check_interval: int,
        message_bus: Queue,
        input_queue: Queue,
        shutdown_event: Event
    ):
        self.check_interval = check_interval
        self.message_bus = message_bus
        self.input_queue = input_queue
        self.shutdown_event = shutdown_event
        
        # Jobs
        self.jobs: dict[str, ScheduledJob] = {}
        self.job_store_path = Path("data/scheduler/jobs.json")
        
        # Load existing jobs
        self._load_jobs()
        
        # Add default jobs
        self._add_default_jobs()
    
    def _load_jobs(self):
        """Load jobs from persistent storage"""
        if self.job_store_path.exists():
            try:
                data = json.loads(self.job_store_path.read_text())
                for job_data in data.get('jobs', []):
                    job = ScheduledJob.from_dict(job_data)
                    self.jobs[job.id] = job
                logger.info(f"Loaded {len(self.jobs)} scheduled jobs")
            except Exception as e:
                logger.error(f"Failed to load jobs: {e}")
    
    def _save_jobs(self):
        """Save jobs to persistent storage"""
        self.job_store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'jobs': [job.to_dict() for job in self.jobs.values()],
            'saved_at': time.time()
        }
        self.job_store_path.write_text(json.dumps(data, indent=2))
    
    def _add_default_jobs(self):
        """Add default system jobs"""
        # Daily digest
        if 'daily_digest' not in self.jobs:
            self.add_job(ScheduledJob(
                id='daily_digest',
                name='Daily Activity Digest',
                job_type='digest',
                trigger_type=TriggerType.CRON,
                trigger_config={'hour': 9, 'minute': 0},  # 9 AM
                payload={'type': 'daily'}
            ))
        
        # Hourly background research
        if 'background_research' not in self.jobs:
            self.add_job(ScheduledJob(
                id='background_research',
                name='Background Research',
                job_type='research',
                trigger_type=TriggerType.INTERVAL,
                trigger_config={'seconds': 3600},  # Every hour
                payload={'type': 'interests'}
            ))
        
        # Profile update
        if 'profile_update' not in self.jobs:
            self.add_job(ScheduledJob(
                id='profile_update',
                name='Profile Learning Update',
                job_type='learning',
                trigger_type=TriggerType.INTERVAL,
                trigger_config={'seconds': 7200},  # Every 2 hours
                payload={}
            ))
    
    def add_job(self, job: ScheduledJob):
        """Add a new job"""
        self.jobs[job.id] = job
        self._calculate_next_run(job)
        self._save_jobs()
        logger.info(f"Added job: {job.name}")
    
    def remove_job(self, job_id: str):
        """Remove a job"""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._save_jobs()
            logger.info(f"Removed job: {job_id}")
    
    def run(self):
        """Main scheduler loop"""
        logger.info("Action scheduler started")
        
        while not self.shutdown_event.is_set():
            try:
                # Process incoming messages
                self._process_messages()
                
                # Check for due jobs
                self._check_jobs()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
        
        self._save_jobs()
        logger.info("Action scheduler stopped")
    
    def _process_messages(self):
        """Process incoming messages"""
        try:
            while True:
                msg = self.input_queue.get_nowait()
                self._handle_message(msg)
        except:
            pass
    
    def _handle_message(self, msg: dict):
        """Handle a message"""
        msg_type = msg.get("type")
        payload = msg.get("payload", {})
        
        if msg_type == "schedule_job":
            # Create new scheduled job
            job = ScheduledJob(
                id=payload.get("id", f"custom_{int(time.time())}"),
                name=payload.get("name", "Custom Job"),
                job_type=payload.get("job_type", "custom"),
                trigger_type=TriggerType(payload.get("trigger_type", "once")),
                trigger_config=payload.get("trigger_config", {}),
                payload=payload.get("job_payload", {})
            )
            self.add_job(job)
    
    def _check_jobs(self):
        """Check and execute due jobs"""
        now = time.time()
        
        for job in list(self.jobs.values()):
            if not job.enabled:
                continue
            
            if job.next_run is None:
                self._calculate_next_run(job)
                continue
            
            if now >= job.next_run:
                self._execute_job(job)
    
    def _calculate_next_run(self, job: ScheduledJob):
        """Calculate next run time for a job"""
        now = datetime.now()
        
        if job.trigger_type == TriggerType.INTERVAL:
            seconds = job.trigger_config.get('seconds', 3600)
            if job.last_run:
                job.next_run = job.last_run + seconds
            else:
                job.next_run = time.time() + seconds
        
        elif job.trigger_type == TriggerType.CRON:
            # Simple cron (hour/minute only)
            hour = job.trigger_config.get('hour', 0)
            minute = job.trigger_config.get('minute', 0)
            
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if next_time <= now:
                next_time += timedelta(days=1)
            
            job.next_run = next_time.timestamp()
        
        elif job.trigger_type == TriggerType.ONCE:
            timestamp = job.trigger_config.get('timestamp')
            if timestamp and timestamp > time.time():
                job.next_run = timestamp
            else:
                job.enabled = False  # Past or invalid
        
        else:
            # Event-based - no scheduled time
            job.next_run = None
    
    def _execute_job(self, job: ScheduledJob):
        """Execute a scheduled job"""
        logger.info(f"Executing job: {job.name}")
        
        job.last_run = time.time()
        job.run_count += 1
        
        # Send to task engine
        self.message_bus.put({
            "type": "job_triggered",
            "source": "scheduler",
            "target": "task_engine",
            "payload": {
                "job_id": job.id,
                "job_type": job.job_type,
                "job_name": job.name,
                **job.payload
            },
            "timestamp": time.time()
        })
        
        # Calculate next run
        self._calculate_next_run(job)
        
        # Disable one-time jobs
        if job.trigger_type == TriggerType.ONCE:
            job.enabled = False
        
        self._save_jobs()
```

---

## 9. PROGRESS REPORTER

### Purpose
Surface what Senter accomplished while user was away, including daily digests and activity logs.

### Implementation

#### File: `reporter/progress_reporter.py`

```python
"""
Progress Reporter

Reports on Senter's autonomous activity.

Provides:
- Real-time activity notifications
- Session summaries
- Daily digests
- Goal progress reports
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from multiprocessing import Queue, Event
from dataclasses import dataclass, field, asdict

logger = logging.getLogger('senter.reporter')


@dataclass
class ActivityEntry:
    """Single activity log entry"""
    timestamp: float
    activity_type: str
    summary: str
    details: dict = field(default_factory=dict)
    importance: str = "normal"  # low, normal, high, critical
    
    def to_dict(self) -> dict:
        return asdict(self)


class ProgressReporter:
    """
    Main progress reporter.
    
    Collects activity from all components and generates reports.
    """
    
    def __init__(
        self,
        digest_hour: int,
        notifications: bool,
        message_bus: Queue,
        input_queue: Queue,
        shutdown_event: Event
    ):
        self.digest_hour = digest_hour
        self.notifications_enabled = notifications
        self.message_bus = message_bus
        self.input_queue = input_queue
        self.shutdown_event = shutdown_event
        
        # Activity log
        self.activities: list[ActivityEntry] = []
        self.activity_log_path = Path("data/progress/activity.json")
        
        # Digest state
        self.last_digest_date: Optional[str] = None
        
        self._load_activities()
    
    def _load_activities(self):
        """Load today's activities from storage"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_file = Path(f"data/progress/daily/{today}.json")
        
        if today_file.exists():
            try:
                data = json.loads(today_file.read_text())
                self.activities = [
                    ActivityEntry(**a) for a in data.get('activities', [])
                ]
            except Exception as e:
                logger.warning(f"Could not load activities: {e}")
    
    def _save_activities(self):
        """Save activities to storage"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_file = Path(f"data/progress/daily/{today}.json")
        today_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'date': today,
            'activities': [a.to_dict() for a in self.activities],
            'saved_at': time.time()
        }
        today_file.write_text(json.dumps(data, indent=2))
    
    def run(self):
        """Main reporter loop"""
        logger.info("Progress reporter started")
        
        while not self.shutdown_event.is_set():
            try:
                # Process incoming messages
                self._process_messages()
                
                # Check for digest time
                self._check_digest_time()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Reporter error: {e}")
        
        self._save_activities()
        logger.info("Progress reporter stopped")
    
    def _process_messages(self):
        """Process incoming messages"""
        try:
            while True:
                msg = self.input_queue.get_nowait()
                self._handle_message(msg)
        except:
            pass
    
    def _handle_message(self, msg: dict):
        """Handle a message"""
        msg_type = msg.get("type")
        payload = msg.get("payload", {})
        
        if msg_type == "activity_log":
            # Log activity
            activity = ActivityEntry(
                timestamp=payload.get("timestamp", time.time()),
                activity_type=payload.get("activity_type", "unknown"),
                summary=self._generate_summary(payload),
                details=payload.get("details", {}),
                importance=payload.get("importance", "normal")
            )
            self._log_activity(activity)
        
        elif msg_type == "digest_request":
            # Generate and send digest
            digest = self.generate_digest()
            self._send_notification(digest)
    
    def _log_activity(self, activity: ActivityEntry):
        """Log an activity"""
        self.activities.append(activity)
        self._save_activities()
        
        # Send notification for high importance
        if activity.importance in ("high", "critical"):
            self._send_notification({
                "type": "activity",
                "summary": activity.summary,
                "importance": activity.importance
            })
    
    def _generate_summary(self, payload: dict) -> str:
        """Generate human-readable summary from activity"""
        activity_type = payload.get("activity_type", "")
        details = payload.get("details", {})
        
        summaries = {
            "plan_created": f"Created plan for: {details.get('description', 'unknown goal')}",
            "task_completed": f"Completed: {details.get('description', 'task')}",
            "task_failed": f"Failed: {details.get('description', 'task')} - {details.get('error', '')}",
            "goal_completed": f"Goal achieved: {details.get('description', 'unknown goal')}",
            "research_complete": f"Researched: {details.get('topic', 'unknown topic')}",
            "learning_update": f"Learned: {details.get('insight', 'new preference')}",
        }
        
        return summaries.get(activity_type, f"Activity: {activity_type}")
    
    def _check_digest_time(self):
        """Check if it's time for daily digest"""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        
        if now.hour == self.digest_hour and self.last_digest_date != today:
            self.last_digest_date = today
            digest = self.generate_digest()
            self._send_notification(digest)
    
    def generate_digest(self, hours: int = 24) -> dict:
        """Generate activity digest"""
        cutoff = time.time() - (hours * 3600)
        
        # Filter recent activities
        recent = [a for a in self.activities if a.timestamp > cutoff]
        
        # Group by type
        by_type: dict[str, list[ActivityEntry]] = {}
        for activity in recent:
            if activity.activity_type not in by_type:
                by_type[activity.activity_type] = []
            by_type[activity.activity_type].append(activity)
        
        # Generate summary
        summary_lines = []
        
        if not recent:
            summary_lines.append("No activity in the past 24 hours.")
        else:
            summary_lines.append(f"📊 Activity Summary ({len(recent)} events)")
            summary_lines.append("")
            
            # Tasks
            completed = by_type.get("task_completed", [])
            failed = by_type.get("task_failed", [])
            if completed or failed:
                summary_lines.append(f"✅ Tasks: {len(completed)} completed, {len(failed)} failed")
            
            # Goals
            goals_completed = by_type.get("goal_completed", [])
            if goals_completed:
                summary_lines.append(f"🎯 Goals achieved: {len(goals_completed)}")
                for g in goals_completed[:3]:
                    summary_lines.append(f"   • {g.summary}")
            
            # Research
            research = by_type.get("research_complete", [])
            if research:
                summary_lines.append(f"🔍 Research completed: {len(research)} topics")
            
            # Learning
            learning = by_type.get("learning_update", [])
            if learning:
                summary_lines.append(f"📚 New learnings: {len(learning)}")
        
        return {
            "type": "digest",
            "period_hours": hours,
            "activity_count": len(recent),
            "summary": "\n".join(summary_lines),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "generated_at": time.time()
        }
    
    def _send_notification(self, data: dict):
        """Send notification to user"""
        if not self.notifications_enabled:
            return
        
        # Log to console
        logger.info(f"Notification: {data.get('summary', data.get('type', 'unknown'))}")
        
        # Try desktop notification
        try:
            self._desktop_notification(data)
        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")
    
    def _desktop_notification(self, data: dict):
        """Send desktop notification"""
        import subprocess
        import platform
        
        title = "Senter"
        message = data.get('summary', str(data))[:200]
        
        system = platform.system()
        
        if system == "Darwin":  # macOS
            script = f'display notification "{message}" with title "{title}"'
            subprocess.run(["osascript", "-e", script], capture_output=True)
        
        elif system == "Linux":
            subprocess.run(["notify-send", title, message], capture_output=True)
        
        elif system == "Windows":
            # Windows toast notification (requires win10toast)
            try:
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast(title, message, duration=5)
            except ImportError:
                pass
    
    def get_session_summary(self) -> str:
        """Generate summary for current session"""
        # Get activities since last user interaction
        # This would be called when user returns
        
        if not self.activities:
            return "No autonomous activity since last session."
        
        # Find activities since they might have been away
        # For simplicity, get last hour
        hour_ago = time.time() - 3600
        recent = [a for a in self.activities if a.timestamp > hour_ago]
        
        if not recent:
            return "No recent autonomous activity."
        
        lines = [f"While you were away ({len(recent)} activities):"]
        
        for activity in recent[-5:]:  # Last 5
            lines.append(f"  • {activity.summary}")
        
        if len(recent) > 5:
            lines.append(f"  ... and {len(recent) - 5} more")
        
        return "\n".join(lines)
```

---

## 10. INTEGRATION MAP

### How Components Connect

```
┌─────────────────────────────────────────────────────────────────┐
│                        MESSAGE BUS                               │
│  (All components communicate through typed messages)             │
└─────────────────────────────────────────────────────────────────┘
        ↑↓              ↑↓              ↑↓              ↑↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    AUDIO     │ │    GAZE      │ │   WORKERS    │ │   TASK       │
│   PIPELINE   │ │  DETECTOR    │ │  (Dual LLM)  │ │  ENGINE      │
│              │ │              │ │              │ │              │
│ • STT/TTS    │ │ • Face track │ │ • Primary    │ │ • Planner    │
│ • VAD        │ │ • Eye track  │ │ • Research   │ │ • Executor   │
│ • Buffer     │ │ • Attention  │ │ • Inference  │ │ • Progress   │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        ↓               ↓               ↑↓              ↑
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────────┐
│                        LEARNING DATABASE                          │
│  • Events storage    • Behavior analysis    • Pattern detection   │
└──────────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  SCHEDULER   │ │   REPORTER   │ │   CLI/TUI    │
│              │ │              │ │              │
│ • Cron jobs  │ │ • Digest     │ │ • User I/O   │
│ • Triggers   │ │ • Notify     │ │ • Commands   │
│ • Recurring  │ │ • Log        │ │ • Display    │
└──────────────┘ └──────────────┘ └──────────────┘
```

### Message Flow Examples

#### Voice Query Flow
```
1. Gaze Detector → attention_gained → Audio Pipeline
2. Audio Pipeline activates listening
3. User speaks
4. Audio Pipeline → user_voice → Message Bus
5. Message Bus → Model Worker (Primary)
6. Model Worker → model_response → Message Bus
7. Message Bus → Audio Pipeline (TTS)
8. Audio Pipeline speaks response
9. Learning DB logs interaction
```

#### Scheduled Research Flow
```
1. Scheduler → job_triggered → Task Engine
2. Task Engine → creates plan
3. Task Engine → tasks to execute
4. Task Engine → model_request → Model Worker (Research)
5. Model Worker → research results
6. Task Engine → stores results
7. Reporter logs activity
8. Reporter → notification (if important)
```

---

## 11. IMPLEMENTATION ORDER

### Phase 1: Foundation (Week 1)
1. **Daemon Framework**
   - `daemon/senter_daemon.py`
   - `daemon/message_bus.py`
   - `daemon/health_monitor.py`
   - Basic process management

2. **Model Workers**
   - `workers/model_worker.py`
   - Dual worker support
   - Message routing

### Phase 2: Task System (Week 2)
3. **Task Engine**
   - `engine/task_engine.py`
   - `engine/planner.py`
   - `engine/executor.py`
   - Goal → Plan → Execute pipeline

4. **Scheduler**
   - `scheduler/action_scheduler.py`
   - Cron expressions
   - Job persistence

### Phase 3: Learning (Week 3)
5. **Learning Database**
   - `learning/learning_db.py`
   - `learning/behavior_analyzer.py`
   - Event storage
   - Pattern detection

6. **Progress Reporter**
   - `reporter/progress_reporter.py`
   - Activity logging
   - Digest generation
   - Notifications

### Phase 4: Multimodal (Week 4)
7. **Audio Pipeline**
   - `audio/audio_pipeline.py`
   - `audio/stt_engine.py`
   - `audio/tts_engine.py`
   - `audio/vad.py`

8. **Gaze Detection**
   - `vision/gaze_detector.py`
   - Camera capture
   - Attention detection

### Phase 5: Integration (Week 5)
9. **Update CLI/TUI**
   - Connect to daemon via IPC
   - New commands
   - Status display

10. **Testing & Polish**
    - Integration tests
    - Error handling
    - Documentation

---

## DEPENDENCIES TO INSTALL

```bash
# Core
pip install python-daemon lockfile

# Audio
pip install sounddevice numpy openai-whisper piper-tts silero-vad torch

# Vision
pip install opencv-python mediapipe

# Notifications (optional)
pip install win10toast  # Windows only

# Database
# SQLite is built-in to Python
```

---

## SUCCESS CRITERIA

When complete, you should be able to:

1. **Start daemon**: `python -m daemon.senter_daemon start`
2. **See status**: `python -m daemon.senter_daemon status`
3. **Voice interact**: Look at camera, speak, get response
4. **Create goal**: "I need to research AI trends this week"
5. **See progress**: `/progress` command shows what Senter did
6. **Get digest**: Automatic 9 AM summary of overnight activity
7. **Background work**: Close laptop, Senter continues researching

---

**This is the complete blueprint. Execute in order, test each phase, and you'll have a truly autonomous Senter.**

### Context background from Chat

"<This is the big one. Let me create the comprehensive implementation blueprint for the full autonomous Senter vision.Here's **senter_full_implementation.md** — the complete autonomous implementation blueprint.

## What You're Getting

This is ~3,000 lines of detailed implementation specifications covering all 8 missing components. Each section includes:

- **Full Python code** (not pseudocode — actual implementation)
- **Integration points** with other components
- **Configuration files**
- **Dependencies**

### The 8 Components

| Component | What It Does | Key Files |
|-----------|--------------|-----------|
| **1. Persistent Daemon** | 24/7 process manager, survives terminal close | `daemon/senter_daemon.py`, `message_bus.py`, `health_monitor.py` |
| **2. Task Engine** | Goal → Plan → Execute → Report pipeline | `engine/task_engine.py`, `planner.py`, `executor.py` |
| **3. Audio Pipeline** | Always-on STT/TTS with VAD | `audio/audio_pipeline.py`, `stt_engine.py` (Whisper), `tts_engine.py` (Piper) |
| **4. Gaze Detection** | Camera-based attention tracking | `vision/gaze_detector.py` using MediaPipe Face Mesh |
| **5. Dual Workers** | Two LLM processes (response + research) | `workers/model_worker.py`, `worker_pool.py` |
| **6. Learning Database** | Time-series behavior storage | `learning/learning_db.py` (SQLite), `behavior_analyzer.py` |
| **7. Action Scheduler** | Cron-like background tasks | `scheduler/action_scheduler.py` with cron/interval/event triggers |
| **8. Progress Reporter** | Surface what Senter did while away | `reporter/progress_reporter.py`, desktop notifications, daily digests |

### The Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SENTER DAEMON (24/7)                        │
├─────────────────────────────────────────────────────────────────┤
│  AUDIO ──→ GAZE ──→ SCHEDULER ──→ REPORTER                     │
│    ↓         ↓          ↓            ↓                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MESSAGE BUS (Event Queue)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│    ↓         ↓          ↓            ↓                          │
│  WORKER 1  WORKER 2  TASK ENGINE  LEARNING DB                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Order (5 Weeks)

| Week | Components | Result |
|------|------------|--------|
| 1 | Daemon + Message Bus + Workers | Background process runs, dual inference works |
| 2 | Task Engine + Scheduler | Goals execute autonomously, jobs run on schedule |
| 3 | Learning DB + Behavior Analyzer | Patterns tracked, preferences learned |
| 4 | Audio Pipeline + Gaze Detection | Voice interaction, attention-based wake |
| 5 | Integration + Polish | Everything connected, tested, documented |

### Total Effort Estimate

**120-160 hours** of focused development to implement all 8 components.

### What Success Looks Like

After implementation:

```bash
# Start the daemon
python -m daemon.senter_daemon start

# Check status
python -m daemon.senter_daemon status
# → Shows all 8 components running

# Look at camera, speak
"Research AI trends for my investor deck"

# Close laptop, go to sleep

# Next morning at 9 AM
# → Desktop notification: "Daily Digest: 3 research tasks completed..."

# Open Senter
/progress
# → "While you were away: Researched 3 topics, saved to Focuses/research/"
```

That's the vision made real.>"

#### Chat background Context:
"<Now you need the **integration and hardening prompts** — taking it from "components exist" to "bulletproof system."Here's **senter_integration_hardening.md** — the prompts to take it from "components exist" to "bulletproof production system."

## Where You Are Now

| ✅ Done | ❌ Still Needed |
|---------|-----------------|
| 8 components implemented | Components tested together |
| Individual tests pass | End-to-end flows verified |
| Daemon starts | Daemon recovers from failures |
| Basic CLI works | Rich CLI with feedback |
| Happy path works | Edge cases handled |

## The 5 Prompt Sets

### Set 1: End-to-End Integration
- **1A**: Full flow integration tests (query→response→learning)
- **1B**: IPC communication layer (CLI↔daemon robust connection)

### Set 2: Error Handling & Recovery
- **2A**: Comprehensive error handling (retries, circuit breakers)
- **2B**: State persistence & crash recovery

### Set 3: Real-World Testing
- **3A**: Manual test playbook (6 real scenarios)
- **3B**: Stress testing (concurrent queries, memory, load)

### Set 4: User Experience
- **4A**: Rich CLI output (colors, tables, spinners)
- **4B**: Interactive shell mode (persistent session)

### Set 5: Documentation
- **5A**: DAEMON.md, ARCHITECTURE.md, DEVELOPMENT.md

## Priority Order

**Do Immediately:**
1. **IPC Communication (1B)** — CLI literally can't talk to daemon without this
2. **Error Handling (2A)** — Prevents production crashes

**This Week:**
3. E2E Integration Tests (1A)
4. State Persistence (2B)
5. Rich CLI (4A)

**Next Week:**
6. Manual Test Playbook (3A)
7. Interactive Shell (4B)
8. Stress Testing (3B)
9. Documentation (5A)

## The End Result

After completing all prompts:

```bash
$ python3 scripts/senter_ctl.py shell

┌─────────────────────────────────────┐
│     Senter Interactive Shell        │
└─────────────────────────────────────┘

[general] > /status
┌─────────────┬───────────┬───────┐
│ Component   │ Status    │ PID   │
├─────────────┼───────────┼───────┤
│ model_primary   │ ✓ Running │ 12346 │
│ task_engine     │ ✓ Running │ 12348 │
│ scheduler       │ ✓ Running │ 12349 │
│ ...             │ ...       │ ...   │
└─────────────┴───────────┴───────┘

[general] > Research AI trends for my deck

⠋ Processing...

╭─────────────────────────────────────────╮
│ Goal created: "Research AI trends"     │
│ Background research starting now.      │
│ Check /progress for updates.           │
╰─────────────────────────────────────────╯
```

That's production-ready.>""