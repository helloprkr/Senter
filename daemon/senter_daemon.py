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
import uuid
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
    """Primary model worker - handles user queries only (US-004)"""
    import requests

    logger.info(f"Model worker '{name}' starting...")

    OLLAMA_URL = "http://localhost:11434"
    PRIMARY_SYSTEM_PROMPT = """You are Senter, a helpful AI assistant. You provide clear,
concise, and accurate responses to user queries. Focus on being helpful and direct."""

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

            # Primary worker ONLY handles user queries and voice (US-004)
            if msg_type in ("user_query", "user_voice"):
                prompt = payload.get("prompt") or payload.get("text", "")
                system_prompt = payload.get("system_prompt", PRIMARY_SYSTEM_PROMPT)

                logger.info(f"Worker {name}: Processing user query - {prompt[:50]}...")

                try:
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
                            "payload": {"response": result, "worker": name},
                            "correlation_id": msg.get("correlation_id")
                        })

                        logger.info(f"Worker {name}: Response sent")
                    else:
                        logger.error(f"Worker {name}: Ollama error {resp.status_code}")

                except Exception as e:
                    logger.error(f"Worker {name}: Generation failed - {e}")

        except Exception as e:
            logger.error(f"Worker {name} error: {e}")

    logger.info(f"Model worker '{name}' stopped")


def research_worker_process(name: str, model: str, research_queue: Queue,
                            output_queue: Queue, shutdown_event: Event,
                            senter_root: str):
    """Research worker - pulls from research_tasks queue, stores to research folder (US-004)"""
    import requests

    logger.info(f"Research worker '{name}' starting...")

    OLLAMA_URL = "http://localhost:11434"
    RESEARCH_SYSTEM_PROMPT = """You are a research assistant for Senter. Your role is to:
1. Conduct thorough background research on topics
2. Gather comprehensive information and insights
3. Synthesize findings into clear, well-organized reports
4. Cite sources and provide context where relevant
5. Focus on depth and accuracy over speed

When researching, explore multiple angles and provide nuanced analysis."""

    senter_root = Path(senter_root)
    research_output_dir = senter_root / "data" / "research" / "results"
    research_output_dir.mkdir(parents=True, exist_ok=True)

    # Test connection
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            logger.info(f"Research worker {name}: Connected to Ollama")
        else:
            logger.error(f"Research worker {name}: Ollama not responding")
            return
    except Exception as e:
        logger.error(f"Research worker {name}: Could not connect to Ollama: {e}")
        return

    while not shutdown_event.is_set():
        try:
            # Pull from research_tasks queue (US-004)
            try:
                task = research_queue.get(timeout=1.0)
            except Empty:
                continue

            # Process research task
            task_id = task.get("id", str(uuid.uuid4())[:8])
            description = task.get("description", "")

            logger.info(f"Research worker: Processing task {task_id} - {description[:50]}...")

            try:
                resp = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Research the following topic thoroughly:\n\n{description}"}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.5,  # Lower temp for research accuracy
                            "num_predict": 2048  # Longer output for research
                        }
                    },
                    timeout=180  # Longer timeout for research
                )

                if resp.status_code == 200:
                    result = resp.json().get("message", {}).get("content", "")

                    # Store result to research output folder (US-004)
                    result_data = {
                        "task_id": task_id,
                        "description": description,
                        "result": result,
                        "timestamp": time.time(),
                        "worker": name,
                        "source": task.get("source", "unknown")
                    }

                    # Save to research results folder
                    from datetime import datetime
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    date_dir = research_output_dir / date_str
                    date_dir.mkdir(parents=True, exist_ok=True)
                    result_file = date_dir / f"{task_id}.json"
                    result_file.write_text(json.dumps(result_data, indent=2))

                    # Also send response to output queue
                    output_queue.put({
                        "type": "research_complete",
                        "source": f"research_{name}",
                        "payload": {
                            "task_id": task_id,
                            "response": result[:500],  # Summary
                            "result_file": str(result_file),
                            "worker": name
                        }
                    })

                    logger.info(f"Research worker: Task {task_id} complete, saved to {result_file}")
                else:
                    logger.error(f"Research worker: Ollama error {resp.status_code}")

            except Exception as e:
                logger.error(f"Research worker: Task {task_id} failed - {e}")

        except Exception as e:
            logger.error(f"Research worker error: {e}")

    logger.info(f"Research worker '{name}' stopped")


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


def audio_pipeline_process(input_queue: Queue, output_queue: Queue,
                           shutdown_event: Event, config: dict):
    """Audio pipeline process function (US-010)"""
    logger.info("Audio pipeline process starting...")

    try:
        from audio.audio_pipeline import (
            AudioPipeline, AudioBuffer, VoiceActivityDetector,
            STTEngine, TTSEngine, NUMPY_AVAILABLE, SOUNDDEVICE_AVAILABLE
        )
        from daemon.message_bus import MessageBus, MessageType, Message
    except ImportError as e:
        logger.error(f"Audio pipeline import error: {e}")
        return

    if not NUMPY_AVAILABLE:
        logger.error("Audio pipeline requires numpy")
        return

    # Create local message bus for this process
    bus = MessageBus()
    bus.start()

    # Create pipeline
    pipeline_config = config.get("audio_pipeline", {})
    stt_model = pipeline_config.get("stt_model", "whisper-small")
    tts_model = pipeline_config.get("tts_model", "system")
    vad_threshold = pipeline_config.get("vad_threshold", 0.5)

    # Initialize components
    audio_buffer = AudioBuffer() if NUMPY_AVAILABLE else None
    vad = VoiceActivityDetector(vad_threshold)
    stt = STTEngine(stt_model)
    tts = TTSEngine(tts_model)

    # State
    has_attention = False
    is_listening = False

    # Start audio capture thread
    capture_thread = None
    if SOUNDDEVICE_AVAILABLE and audio_buffer:
        import sounddevice as sd
        import numpy as np
        import threading

        def capture_loop():
            try:
                def audio_callback(indata, frames, time_info, status):
                    if status:
                        logger.warning(f"Audio status: {status}")
                    audio_buffer.write(indata.copy())

                with sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    dtype=np.float32,
                    callback=audio_callback,
                    blocksize=1600
                ):
                    while not shutdown_event.is_set():
                        time.sleep(0.1)
            except Exception as e:
                logger.error(f"Audio capture error: {e}")

        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()
        logger.info("Audio capture started")
    else:
        logger.warning("Audio capture disabled (sounddevice not available)")

    logger.info("Audio pipeline started")

    while not shutdown_event.is_set():
        try:
            # Process incoming messages from daemon
            try:
                msg = input_queue.get(timeout=0.05)
                msg_type = msg.get("type")
                payload = msg.get("payload", {})

                if msg_type == "attention_gained":
                    has_attention = True
                    logger.info("Audio: Attention gained - voice active")

                elif msg_type == "attention_lost":
                    has_attention = False
                    is_listening = False
                    logger.info("Audio: Attention lost - voice inactive")

                elif msg_type == "speak":
                    text = payload.get("text")
                    if text:
                        tts.speak(text)

                elif msg_type == "model_response":
                    # Speak response if we have attention
                    if has_attention:
                        text = payload.get("response")
                        if text and len(text) < 500:  # Don't speak long responses
                            tts.speak(text)

            except Empty:
                pass

            # Process audio if we have attention and audio buffer
            if has_attention and audio_buffer and SOUNDDEVICE_AVAILABLE:
                import numpy as np
                audio_data = audio_buffer.get_recent(seconds=2)

                if audio_data is not None and len(audio_data) > 0:
                    # Check for voice activity
                    if vad.is_speech(audio_data):
                        if not is_listening:
                            logger.debug("Speech detected")
                            is_listening = True
                            audio_buffer.mark_speech_start()
                    else:
                        if is_listening:
                            # Speech ended - transcribe
                            is_listening = False
                            speech_audio = audio_buffer.get_speech_segment()

                            if speech_audio is not None and len(speech_audio) >= 1600:
                                text = stt.transcribe(speech_audio)
                                if text and len(text.strip()) > 0:
                                    logger.info(f"Transcribed: {text}")

                                    # Send to daemon for routing to model
                                    output_queue.put({
                                        "type": "user_voice",
                                        "source": "audio_pipeline",
                                        "payload": {
                                            "text": text,
                                            "audio_duration": len(speech_audio) / 16000
                                        }
                                    })

        except Exception as e:
            logger.error(f"Audio pipeline error: {e}")

    bus.stop()
    logger.info("Audio pipeline stopped")


def gaze_detector_process(input_queue: Queue, output_queue: Queue,
                          shutdown_event: Event, config: dict):
    """Gaze detection process function (US-011)"""
    logger.info("Gaze detector process starting...")

    try:
        from daemon.message_bus import MessageBus, MessageType
        import numpy as np
    except ImportError as e:
        logger.error(f"Gaze detector import error: {e}")
        return

    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        logger.error("Gaze detector requires opencv-python")
        return

    # Check for mediapipe with legacy solutions API
    MEDIAPIPE_AVAILABLE = False
    mp = None
    try:
        import mediapipe as mp_module
        # Check for legacy solutions API (pre-0.10.0)
        if hasattr(mp_module, "solutions") and hasattr(mp_module.solutions, "face_mesh"):
            mp = mp_module
            MEDIAPIPE_AVAILABLE = True
            logger.info("MediaPipe legacy solutions API available")
        else:
            logger.warning("mediapipe installed but no legacy solutions API - using basic face detection")
    except ImportError:
        logger.warning("mediapipe not available - using basic face detection")

    # Get config
    gaze_config = config.get("gaze_detection", {})
    camera_id = gaze_config.get("camera_id", 0)
    attention_threshold = gaze_config.get("attention_threshold", 0.7)

    # State
    has_attention = False
    attention_start_time = None
    last_attention_time = None
    attention_history = []
    history_size = 10
    attention_timeout = 2.0

    # Initialize camera
    camera = None
    face_mesh = None

    try:
        camera = cv2.VideoCapture(camera_id)
        if not camera.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        logger.info(f"Camera {camera_id} initialized")

        # Initialize face mesh if available
        if MEDIAPIPE_AVAILABLE:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Face Mesh initialized")

    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        return

    def calculate_attention_mediapipe(frame, face_mesh_instance) -> float:
        """Calculate attention using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_instance.process(rgb_frame)

        if not results.multi_face_landmarks:
            return 0.0

        landmarks = results.multi_face_landmarks[0].landmark

        # Face center score
        nose_tip = landmarks[1]
        face_center_score = 1.0 - abs(nose_tip.x - 0.5) * 2
        face_center_score = max(0, face_center_score)

        # Eye openness score
        def eye_aspect_ratio(eye_side):
            if eye_side == "left":
                p1, p2 = landmarks[159], landmarks[145]
                p3, p4 = landmarks[33], landmarks[133]
            else:
                p1, p2 = landmarks[386], landmarks[374]
                p3, p4 = landmarks[362], landmarks[263]
            vertical = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            horizontal = np.sqrt((p3.x - p4.x)**2 + (p3.y - p4.y)**2)
            return vertical / horizontal if horizontal > 0 else 0

        avg_ear = (eye_aspect_ratio("left") + eye_aspect_ratio("right")) / 2
        eye_open_score = min(1.0, avg_ear / 0.25)

        # Gaze score
        try:
            left_iris = landmarks[468]
            right_iris = landmarks[473]
            left_inner, left_outer = landmarks[133], landmarks[33]
            right_inner, right_outer = landmarks[362], landmarks[263]

            left_eye_width = abs(left_outer.x - left_inner.x)
            right_eye_width = abs(right_outer.x - right_inner.x)

            if left_eye_width > 0 and right_eye_width > 0:
                left_iris_pos = (left_iris.x - left_outer.x) / left_eye_width
                right_iris_pos = (right_iris.x - right_outer.x) / right_eye_width
                avg_iris_pos = (left_iris_pos + right_iris_pos) / 2
                gaze_score = 1.0 - abs(avg_iris_pos - 0.5) * 2
                gaze_score = max(0, min(1, gaze_score))
            else:
                gaze_score = 0.5
        except (IndexError, AttributeError):
            gaze_score = 0.5

        return face_center_score * 0.3 + eye_open_score * 0.3 + gaze_score * 0.4

    def calculate_attention_basic(frame) -> float:
        """Calculate attention using Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return 0.0

        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        frame_h, frame_w = frame.shape[:2]

        face_center_x = (x + w/2) / frame_w
        center_score = 1.0 - abs(face_center_x - 0.5) * 2
        center_score = max(0, center_score)

        size_score = min(1.0, (w * h) / (frame_w * frame_h) * 10)

        return center_score * 0.6 + size_score * 0.4

    logger.info("Gaze detector started")

    try:
        while not shutdown_event.is_set():
            try:
                # Process frame
                ret, frame = camera.read()
                if not ret:
                    time.sleep(0.066)
                    continue

                # Calculate attention score
                if MEDIAPIPE_AVAILABLE and face_mesh:
                    attention_score = calculate_attention_mediapipe(frame, face_mesh)
                else:
                    attention_score = calculate_attention_basic(frame)

                # Update history
                attention_history.append(attention_score)
                if len(attention_history) > history_size:
                    attention_history.pop(0)

                # Smooth score
                smoothed_score = np.mean(attention_history)

                # Update attention state
                now = time.time()

                if smoothed_score >= attention_threshold:
                    last_attention_time = now

                    if not has_attention:
                        # Attention gained!
                        has_attention = True
                        attention_start_time = now
                        logger.info(f"Attention gained (score: {smoothed_score:.2f})")

                        output_queue.put({
                            "type": "attention_gained",
                            "source": "gaze_detector",
                            "payload": {
                                "score": smoothed_score,
                                "timestamp": now
                            }
                        })

                else:
                    # Check timeout
                    if has_attention and last_attention_time:
                        time_since = now - last_attention_time

                        if time_since > attention_timeout:
                            # Attention lost
                            has_attention = False
                            duration = now - attention_start_time if attention_start_time else 0
                            logger.info(f"Attention lost after {duration:.1f}s")

                            output_queue.put({
                                "type": "attention_lost",
                                "source": "gaze_detector",
                                "payload": {
                                    "duration": duration,
                                    "timestamp": now
                                }
                            })

                            attention_start_time = None

                time.sleep(0.066)  # ~15 FPS

            except Exception as e:
                logger.error(f"Gaze frame error: {e}")
                time.sleep(0.1)

    finally:
        if camera is not None:
            camera.release()
        if face_mesh is not None:
            face_mesh.close()

    logger.info("Gaze detector stopped")


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

        # Research task queue for background work (US-002)
        self.research_tasks_queue = Queue(maxsize=1000)
        self.research_tasks_file = self.senter_root / "data" / "research_tasks.json"

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

    def _load_research_tasks(self):
        """Load pending research tasks from disk (US-002)"""
        if self.research_tasks_file.exists():
            try:
                data = json.loads(self.research_tasks_file.read_text())
                tasks = data.get("pending_tasks", [])
                for task in tasks:
                    self.research_tasks_queue.put(task)
                logger.info(f"Loaded {len(tasks)} pending research tasks")
            except Exception as e:
                logger.warning(f"Could not load research tasks: {e}")

    def _save_research_tasks(self):
        """Save pending research tasks to disk (US-002)"""
        tasks = []
        while True:
            try:
                task = self.research_tasks_queue.get_nowait()
                tasks.append(task)
            except Empty:
                break

        # Put tasks back in queue
        for task in tasks:
            self.research_tasks_queue.put(task)

        # Save to disk
        self.research_tasks_file.parent.mkdir(parents=True, exist_ok=True)
        self.research_tasks_file.write_text(json.dumps({
            "pending_tasks": tasks,
            "saved_at": time.time()
        }, indent=2))
        logger.info(f"Saved {len(tasks)} research tasks")

    def add_research_task(self, task: dict) -> bool:
        """Add a task to the research queue (US-002)"""
        try:
            # Ensure task has required fields
            if "id" not in task:
                task["id"] = str(uuid.uuid4())[:8]
            if "created_at" not in task:
                task["created_at"] = time.time()
            if "status" not in task:
                task["status"] = "pending"

            self.research_tasks_queue.put_nowait(task)
            logger.info(f"Added research task: {task.get('description', task['id'])[:50]}")
            return True
        except Exception as e:
            logger.error(f"Failed to add research task: {e}")
            return False

    def get_research_queue_status(self) -> dict:
        """Get status of research task queue (US-002)"""
        try:
            # qsize() is unreliable on macOS, so we count by draining and refilling
            tasks = []
            while True:
                try:
                    task = self.research_tasks_queue.get_nowait()
                    tasks.append(task)
                except Empty:
                    break

            # Put tasks back
            for task in tasks:
                self.research_tasks_queue.put(task)

            return {
                "queue_size": len(tasks),
                "file_exists": self.research_tasks_file.exists()
            }
        except Exception as e:
            return {"queue_size": -1, "error": str(e)}

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

            # Load pending research tasks (US-002)
            self._load_research_tasks()

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

        # Research worker (US-004) - uses separate function, pulls from research_tasks_queue
        out2 = Queue(maxsize=1000)
        self.output_queues["model_research"] = out2
        p2 = Process(
            target=research_worker_process,
            args=("research", worker_config["models"]["research"],
                  self.research_tasks_queue, out2, self.shutdown_event,
                  str(self.senter_root)),
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
        """Start audio pipeline (US-010)"""
        logger.info("Starting audio pipeline...")

        # Create queues for audio pipeline
        q = Queue(maxsize=1000)
        out = Queue(maxsize=1000)
        self.queues["audio"] = q
        self.output_queues["audio"] = out

        # Start process with config
        p = Process(
            target=audio_pipeline_process,
            args=(q, out, self.shutdown_event, self.config.get("components", {})),
            name="audio_pipeline"
        )
        p.start()
        self.processes["audio"] = p
        self.health_monitor.register_component("audio", p)
        logger.info("Audio pipeline started")

    def _start_gaze_detection(self):
        """Start gaze detection (US-011)"""
        logger.info("Starting gaze detection...")

        # Create queues for gaze detector
        q = Queue(maxsize=1000)
        out = Queue(maxsize=1000)
        self.queues["gaze"] = q
        self.output_queues["gaze"] = out

        # Start process with config
        p = Process(
            target=gaze_detector_process,
            args=(q, out, self.shutdown_event, self.config.get("components", {})),
            name="gaze_detector"
        )
        p.start()
        self.processes["gaze"] = p
        self.health_monitor.register_component("gaze", p)
        logger.info("Gaze detection started")

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
        """Route a message to appropriate components (US-012 adds attention/voice routing)"""
        msg_type = msg.get("type")
        target = msg.get("target")

        # Route to specific target
        if target and target in self.queues:
            self.queues[target].put(msg)
            return

        # Route by type - includes US-012 attention/voice wiring
        routing = {
            "model_response": ["task_engine", "reporter", "audio"],  # Audio for TTS
            "activity_log": ["reporter"],
            "task_create": ["task_engine"],
            "user_query": ["model_primary", "learning"],
            # US-012: Attention events from gaze -> audio pipeline
            "attention_gained": ["audio", "model_primary"],
            "attention_lost": ["audio"],
            # US-012: Voice input from audio -> model
            "user_voice": ["model_primary", "learning"],
            # TTS request
            "speak": ["audio"],
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

        # Save research tasks (US-002)
        self._save_research_tasks()

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
