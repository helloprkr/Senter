#!/usr/bin/env python3
"""
Senter Background Daemon

A persistent process that runs 24/7, managing:
- Model workers (inference)
- Task execution engine
- Action scheduler
- Progress reporting
- Learning system
- Audio pipeline (optional)
- Gaze detection (optional)

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
from queue import Empty
import threading

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.health_monitor import HealthMonitor
from daemon.ipc_server import IPCServer
from daemon.state_manager import StateManager
from daemon.circuit_breaker import CircuitBreaker, with_retry

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('senter.daemon')


# ============================================================
# WORKER FUNCTIONS (run in separate processes)
# ============================================================

def model_worker_process(name: str, model: str, input_queue: Queue,
                         output_queue: Queue, shutdown_event: Event):
    """Model worker process function"""
    import requests

    logger.info(f"Model worker '{name}' starting...")

    OLLAMA_URL = "http://localhost:11434"

    # Test connection
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            logger.info(f"Worker {name}: Connected to Ollama")
        else:
            logger.error(f"Worker {name}: Ollama not responding")
            return
    except Exception as e:
        logger.error(f"Worker {name}: Could not connect to Ollama: {e}")
        return

    while not shutdown_event.is_set():
        try:
            # Get request
            try:
                msg = input_queue.get(timeout=1.0)
            except Empty:
                continue

            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            if msg_type in ("model_request", "user_query", "user_voice"):
                prompt = payload.get("prompt") or payload.get("text", "")
                system_prompt = payload.get("system_prompt", "You are Senter, a helpful AI assistant.")

                logger.info(f"Worker {name}: Processing - {prompt[:50]}...")

                try:
                    # Use Ollama chat API directly
                    resp = requests.post(
                        f"{OLLAMA_URL}/api/chat",
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            "stream": False,
                            "options": {
                                "temperature": payload.get("temperature", 0.7),
                                "num_predict": payload.get("max_tokens", 1024)
                            }
                        },
                        timeout=120
                    )

                    if resp.status_code == 200:
                        result = resp.json().get("message", {}).get("content", "")

                        output_queue.put({
                            "type": "model_response",
                            "source": f"model_{name}",
                            "payload": {"response": result, "worker": name}
                        })

                        logger.info(f"Worker {name}: Response sent")
                    else:
                        logger.error(f"Worker {name}: Ollama error {resp.status_code}")

                except Exception as e:
                    logger.error(f"Worker {name}: Generation failed - {e}")

        except Exception as e:
            logger.error(f"Worker {name} error: {e}")

    logger.info(f"Model worker '{name}' stopped")


def task_engine_process(input_queue: Queue, output_queue: Queue,
                        shutdown_event: Event, senter_root: str):
    """Task engine process function"""
    logger.info("Task engine starting...")

    senter_root = Path(senter_root)
    sys.path.insert(0, str(senter_root))

    from engine.task_engine import TaskEngine, TaskPlanner, TaskExecutor
    from daemon.message_bus import MessageBus, MessageType

    # Create a local message bus for this process
    bus = MessageBus()
    bus.start()

    planner = TaskPlanner(bus, senter_root)
    executor = TaskExecutor(bus, senter_root)

    plans = {}

    while not shutdown_event.is_set():
        try:
            try:
                msg = input_queue.get(timeout=1.0)
            except Empty:
                continue

            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            if msg_type == "task_create":
                goal_id = payload.get("goal_id", f"goal_{int(time.time())}")
                description = payload.get("description", "")

                logger.info(f"Task engine: Creating plan for '{description}'")

                plan = planner.create_plan(goal_id, description)
                plans[goal_id] = plan

                output_queue.put({
                    "type": "activity_log",
                    "source": "task_engine",
                    "payload": {
                        "activity_type": "plan_created",
                        "details": {"goal_id": goal_id, "tasks": len(plan.tasks)}
                    }
                })

                logger.info(f"Task engine: Plan created with {len(plan.tasks)} tasks")

        except Exception as e:
            logger.error(f"Task engine error: {e}")

    bus.stop()
    logger.info("Task engine stopped")


def scheduler_process(input_queue: Queue, output_queue: Queue,
                      shutdown_event: Event, senter_root: str):
    """Scheduler process function"""
    logger.info("Scheduler starting...")

    senter_root = Path(senter_root)
    sys.path.insert(0, str(senter_root))

    from scheduler.action_scheduler import ActionScheduler, ScheduledJob, TriggerType
    from daemon.message_bus import MessageBus

    # Create local message bus
    bus = MessageBus()
    bus.start()

    jobs_file = senter_root / "data" / "scheduler" / "jobs.json"
    jobs = {}

    # Load or create default jobs
    if jobs_file.exists():
        try:
            data = json.loads(jobs_file.read_text())
            for job_data in data.get("jobs", []):
                job = ScheduledJob.from_dict(job_data)
                jobs[job.id] = job
        except:
            pass

    if not jobs:
        from scheduler.action_scheduler import JobStatus
        jobs["daily_digest"] = ScheduledJob(
            id="daily_digest",
            name="Daily Digest",
            description="Generate daily activity summary",
            trigger_type=TriggerType.CRON,
            trigger_config={"hour": 9, "minute": 0},
            job_type="digest"
        )
        jobs["goal_check"] = ScheduledJob(
            id="goal_check",
            name="Goal Check",
            description="Review active goals",
            trigger_type=TriggerType.INTERVAL,
            trigger_config={"seconds": 3600},
            job_type="check"
        )

        # Save
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        jobs_file.write_text(json.dumps({
            "jobs": [j.to_dict() for j in jobs.values()]
        }, indent=2))

    logger.info(f"Scheduler started with {len(jobs)} jobs")

    last_check = 0
    check_interval = 60

    while not shutdown_event.is_set():
        try:
            # Process messages
            try:
                msg = input_queue.get(timeout=1.0)
                # Handle schedule_job, cancel_job messages
            except Empty:
                pass

            # Check triggers
            now = time.time()
            if now - last_check >= check_interval:
                last_check = now
                # Would check job triggers here

        except Exception as e:
            logger.error(f"Scheduler error: {e}")

    bus.stop()
    logger.info("Scheduler stopped")


def reporter_process(input_queue: Queue, output_queue: Queue,
                     shutdown_event: Event, senter_root: str):
    """Progress reporter process function"""
    logger.info("Reporter starting...")

    senter_root = Path(senter_root)
    sys.path.insert(0, str(senter_root))

    from reporter.progress_reporter import ActivityLog, ActivityEntry

    log_dir = senter_root / "data" / "progress" / "activity"
    activity_log = ActivityLog(log_dir)

    while not shutdown_event.is_set():
        try:
            try:
                msg = input_queue.get(timeout=1.0)
            except Empty:
                continue

            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            if msg_type == "activity_log":
                entry = ActivityEntry(
                    activity_type=payload.get("activity_type", "unknown"),
                    timestamp=payload.get("timestamp", time.time()),
                    details=payload.get("details", {}),
                    source=msg.get("source", "unknown")
                )
                activity_log.log(entry)
                logger.debug(f"Reporter: Logged {entry.activity_type}")

        except Exception as e:
            logger.error(f"Reporter error: {e}")

    logger.info("Reporter stopped")


def learning_process(input_queue: Queue, output_queue: Queue,
                     shutdown_event: Event, senter_root: str):
    """Learning service process function"""
    logger.info("Learning service starting...")

    senter_root = Path(senter_root)
    sys.path.insert(0, str(senter_root))

    from learning.learning_db import LearningDatabase, BehaviorAnalyzer, LearningEvent

    db_path = senter_root / "data" / "learning" / "behavior.db"
    db = LearningDatabase(db_path)
    analyzer = BehaviorAnalyzer(db)

    last_analysis = 0
    analysis_interval = 300

    while not shutdown_event.is_set():
        try:
            try:
                msg = input_queue.get(timeout=1.0)
            except Empty:
                # Periodic analysis
                now = time.time()
                if now - last_analysis >= analysis_interval:
                    analyzer.update_preferences()
                    last_analysis = now
                continue

            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            if msg_type == "user_query":
                query = payload.get("text", "")
                analyzer.analyze_query(query)

            elif msg_type == "learn_event":
                event = LearningEvent(
                    event_type=payload.get("event_type", "custom"),
                    timestamp=payload.get("timestamp", time.time()),
                    data=payload.get("data", {})
                )
                db.store_event(event)

        except Exception as e:
            logger.error(f"Learning error: {e}")

    logger.info("Learning service stopped")


# ============================================================
# MAIN DAEMON CLASS
# ============================================================

class SenterDaemon:
    """Main Senter daemon process."""

    def __init__(self, config_path: str = None):
        self.senter_root = Path(__file__).parent.parent
        self.config_path = Path(config_path) if config_path else self.senter_root / "config" / "daemon_config.json"
        self.config = self._load_config()

        # Process management
        self.processes: dict[str, Process] = {}
        self.shutdown_event = Event()

        # Queues for IPC - input and output per component
        self.queues: dict[str, Queue] = {}  # Input queues
        self.output_queues: dict[str, Queue] = {}  # Output queues for responses
        self.main_output_queue = Queue(maxsize=10000)  # Shared output for routing

        # Health monitor
        self.health_monitor = HealthMonitor(
            check_interval=self.config.get("health_check_interval", 30)
        )

        # State manager for crash recovery
        self.state_manager = StateManager(
            state_dir=str(self.senter_root / "data" / "state")
        )

        # IPC server for CLI communication
        self.ipc_server = None
        self.ipc_thread = None

        # Timing
        self.start_time = None
        self.pid = os.getpid()

        # PID file
        self.pid_file = self.senter_root / "data" / "senter.pid"
        self.is_running = False

    def _load_config(self) -> dict:
        """Load daemon configuration"""
        default_config = {
            "components": {
                "model_workers": {
                    "enabled": True,
                    "count": 2,
                    "models": {"primary": "llama3.2", "research": "llama3.2"}
                },
                "audio_pipeline": {"enabled": False},
                "gaze_detection": {"enabled": False},
                "task_engine": {"enabled": True, "max_concurrent_tasks": 3},
                "scheduler": {"enabled": True, "check_interval": 60},
                "reporter": {"enabled": True},
                "learning": {"enabled": True}
            },
            "health_check_interval": 30
        }

        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    user_config = json.load(f)
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        return default_config

    def start(self, foreground: bool = True):
        """Start the daemon"""
        logger.info("=" * 50)
        logger.info("Starting Senter Daemon...")
        logger.info("=" * 50)

        if self._is_running():
            logger.error("Senter daemon is already running")
            return False

        self._write_pid()
        self.start_time = time.time()
        self.pid = os.getpid()
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        try:
            self.is_running = True
            self.health_monitor.start()

            # Start IPC server for CLI communication
            self._start_ipc_server()

            # Check for saved state
            if self.state_manager.has_saved_state():
                logger.info("Found saved state from previous run")

            # Start components
            self._start_model_workers()
            self._start_task_engine()
            self._start_scheduler()
            self._start_reporter()
            self._start_learning()

            # Optional components
            if self.config["components"]["audio_pipeline"]["enabled"]:
                self._start_audio_pipeline()
            if self.config["components"]["gaze_detection"]["enabled"]:
                self._start_gaze_detection()

            logger.info("=" * 50)
            logger.info(f"Senter daemon started - {len(self.processes)} components")
            logger.info(f"Components: {', '.join(self.processes.keys())}")
            logger.info(f"IPC socket: /tmp/senter.sock")
            logger.info("=" * 50)

            if foreground:
                self._main_loop()

            return True

        except Exception as e:
            logger.error(f"Daemon startup failed: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup()
            raise

    def _start_ipc_server(self):
        """Start IPC server in background thread"""
        logger.info("Starting IPC server...")
        self.ipc_server = IPCServer(
            socket_path="/tmp/senter.sock",
            shutdown_event=self.shutdown_event,
            daemon_ref=self
        )
        self.ipc_thread = threading.Thread(
            target=self.ipc_server.run,
            name="ipc_server",
            daemon=True
        )
        self.ipc_thread.start()
        logger.info("IPC server started")

    def _start_model_workers(self):
        """Start model workers"""
        if not self.config["components"]["model_workers"]["enabled"]:
            return

        logger.info("Starting model workers...")
        worker_config = self.config["components"]["model_workers"]

        # Primary worker
        q1 = Queue(maxsize=1000)
        out1 = Queue(maxsize=1000)
        self.queues["model_primary"] = q1
        self.output_queues["model_primary"] = out1
        p1 = Process(
            target=model_worker_process,
            args=("primary", worker_config["models"]["primary"],
                  q1, out1, self.shutdown_event),
            name="model_worker_primary"
        )
        p1.start()
        self.processes["model_primary"] = p1
        self.health_monitor.register_component("model_primary", p1)

        # Research worker
        q2 = Queue(maxsize=1000)
        out2 = Queue(maxsize=1000)
        self.queues["model_research"] = q2
        self.output_queues["model_research"] = out2
        p2 = Process(
            target=model_worker_process,
            args=("research", worker_config["models"]["research"],
                  q2, out2, self.shutdown_event),
            name="model_worker_research"
        )
        p2.start()
        self.processes["model_research"] = p2
        self.health_monitor.register_component("model_research", p2)

        logger.info("Model workers started")

    def _start_task_engine(self):
        """Start task engine"""
        if not self.config["components"]["task_engine"]["enabled"]:
            return

        logger.info("Starting task engine...")
        q = Queue(maxsize=1000)
        out = Queue(maxsize=1000)
        self.queues["task_engine"] = q
        self.output_queues["task_engine"] = out
        p = Process(
            target=task_engine_process,
            args=(q, out, self.shutdown_event, str(self.senter_root)),
            name="task_engine"
        )
        p.start()
        self.processes["task_engine"] = p
        self.health_monitor.register_component("task_engine", p)
        logger.info("Task engine started")

    def _start_scheduler(self):
        """Start scheduler"""
        if not self.config["components"]["scheduler"]["enabled"]:
            return

        logger.info("Starting scheduler...")
        q = Queue(maxsize=1000)
        out = Queue(maxsize=1000)
        self.queues["scheduler"] = q
        self.output_queues["scheduler"] = out
        p = Process(
            target=scheduler_process,
            args=(q, out, self.shutdown_event, str(self.senter_root)),
            name="scheduler"
        )
        p.start()
        self.processes["scheduler"] = p
        self.health_monitor.register_component("scheduler", p)
        logger.info("Scheduler started")

    def _start_reporter(self):
        """Start reporter"""
        if not self.config["components"]["reporter"]["enabled"]:
            return

        logger.info("Starting reporter...")
        q = Queue(maxsize=1000)
        out = Queue(maxsize=1000)
        self.queues["reporter"] = q
        self.output_queues["reporter"] = out
        p = Process(
            target=reporter_process,
            args=(q, out, self.shutdown_event, str(self.senter_root)),
            name="reporter"
        )
        p.start()
        self.processes["reporter"] = p
        self.health_monitor.register_component("reporter", p)
        logger.info("Reporter started")

    def _start_learning(self):
        """Start learning service"""
        if not self.config["components"]["learning"]["enabled"]:
            return

        logger.info("Starting learning service...")
        q = Queue(maxsize=1000)
        out = Queue(maxsize=1000)
        self.queues["learning"] = q
        self.output_queues["learning"] = out
        p = Process(
            target=learning_process,
            args=(q, out, self.shutdown_event, str(self.senter_root)),
            name="learning"
        )
        p.start()
        self.processes["learning"] = p
        self.health_monitor.register_component("learning", p)
        logger.info("Learning service started")

    def _start_audio_pipeline(self):
        """Start audio pipeline"""
        logger.info("Starting audio pipeline...")
        # Audio pipeline would be started here
        logger.info("Audio pipeline started (stub)")

    def _start_gaze_detection(self):
        """Start gaze detection"""
        logger.info("Starting gaze detection...")
        # Gaze detection would be started here
        logger.info("Gaze detection started (stub)")

    def _main_loop(self):
        """Main daemon loop - routes messages between components"""
        logger.info("Entering main loop (Ctrl+C to stop)...")

        last_checkpoint = time.time()
        checkpoint_interval = 30  # seconds

        while not self.shutdown_event.is_set():
            try:
                # Check all output queues for messages to route
                for name, queue in self.output_queues.items():
                    try:
                        msg = queue.get_nowait()
                        self._route_message(msg)
                    except Empty:
                        pass

                # Periodic state checkpoint
                now = time.time()
                if now - last_checkpoint >= checkpoint_interval:
                    # Would checkpoint component states here
                    last_checkpoint = now

                # Small sleep to prevent busy loop
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Main loop error: {e}")

        logger.info("Main loop exited")

    def _route_message(self, msg: dict):
        """Route a message to appropriate components"""
        msg_type = msg.get("type")
        target = msg.get("target")

        # Route to specific target
        if target and target in self.queues:
            self.queues[target].put(msg)
            return

        # Route by type
        routing = {
            "model_response": ["task_engine", "reporter"],
            "activity_log": ["reporter"],
            "task_create": ["task_engine"],
            "user_query": ["model_primary", "learning"],
        }

        targets = routing.get(msg_type, [])
        for t in targets:
            if t in self.queues:
                self.queues[t].put(msg)

    def stop(self):
        """Stop the daemon"""
        logger.info("Stopping Senter daemon...")
        self.shutdown_event.set()
        self.is_running = False

        # Give components time to save state
        time.sleep(2)

        # Clear state on clean shutdown
        self.state_manager.clear_state()

        for name, process in list(self.processes.items()):
            if process.is_alive():
                logger.info(f"Terminating {name}...")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

        self.health_monitor.stop()
        self._cleanup()
        logger.info("Senter daemon stopped")

    def _cleanup(self):
        # Clean up PID file
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except:
                pass

        # Clean up IPC socket
        ipc_socket = Path("/tmp/senter.sock")
        if ipc_socket.exists():
            try:
                ipc_socket.unlink()
            except:
                pass

    def _write_pid(self):
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

    def _is_running(self) -> bool:
        if not self.pid_file.exists():
            return False
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except:
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self.stop()

    def status(self) -> dict:
        if not self._is_running():
            return {"running": False}
        try:
            pid = int(self.pid_file.read_text().strip())
            return {
                "running": True,
                "pid": pid,
                "health": self.health_monitor.get_status() if self.is_running else {}
            }
        except:
            return {"running": False}


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Senter Daemon Control")
    parser.add_argument("command", choices=["start", "stop", "status", "restart"])
    parser.add_argument("--foreground", "-f", action="store_true")
    parser.add_argument("--config", "-c", type=str)
    args = parser.parse_args()

    daemon = SenterDaemon(config_path=args.config)

    if args.command == "start":
        daemon.start(foreground=True)
    elif args.command == "stop":
        if daemon._is_running():
            pid = int(daemon.pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print("Senter daemon stopped")
        else:
            print("Senter daemon is not running")
    elif args.command == "status":
        status = daemon.status()
        if status.get("running"):
            print(f"Senter daemon is running (PID: {status['pid']})")
        else:
            print("Senter daemon is not running")
    elif args.command == "restart":
        if daemon._is_running():
            pid = int(daemon.pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        daemon.start(foreground=True)


if __name__ == "__main__":
    main()
