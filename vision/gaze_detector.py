#!/usr/bin/env python3
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
import sys
from typing import Optional, Tuple
from pathlib import Path
from multiprocessing import Event
from queue import Empty

# Check dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.message_bus import MessageBus, MessageType, Message

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
        message_bus: MessageBus,
        shutdown_event: Event
    ):
        self.camera_id = camera_id
        self.attention_threshold = attention_threshold
        self.message_bus = message_bus
        self.shutdown_event = shutdown_event

        # State
        self.has_attention = False
        self.attention_start_time = None
        self.last_attention_time = None

        # Smoothing
        self.attention_history = []
        self.history_size = 10

        # Attention timeout
        self.attention_timeout = 2.0

        # Components
        self._face_mesh = None
        self._camera = None

        # Message queue
        self._queue = None

    def run(self):
        """Main detection loop"""
        logger.info("Gaze detector starting...")

        # Check dependencies
        if not CV2_AVAILABLE:
            logger.error("opencv-python required for gaze detection")
            return

        if not MEDIAPIPE_AVAILABLE:
            logger.warning("mediapipe not available - using basic face detection")

        if not NUMPY_AVAILABLE:
            logger.error("numpy required for gaze detection")
            return

        # Initialize camera
        try:
            self._init_camera()
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return

        # Initialize face mesh
        if MEDIAPIPE_AVAILABLE:
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                logger.warning(f"Face mesh init failed: {e}")

        # Register with message bus
        self._queue = self.message_bus.register("gaze")

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

    def _init_camera(self):
        """Initialize camera"""
        self._camera = cv2.VideoCapture(self.camera_id)
        if not self._camera.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        # Set resolution
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._camera.set(cv2.CAP_PROP_FPS, 15)

        logger.info(f"Camera {self.camera_id} initialized")

    def _process_frame(self):
        """Process a single camera frame"""
        if self._camera is None:
            return

        ret, frame = self._camera.read()
        if not ret:
            return

        # Calculate attention score
        if MEDIAPIPE_AVAILABLE and self._face_mesh:
            attention_score = self._calculate_attention_mediapipe(frame)
        else:
            attention_score = self._calculate_attention_basic(frame)

        # Update history
        self.attention_history.append(attention_score)
        if len(self.attention_history) > self.history_size:
            self.attention_history.pop(0)

        # Smooth score
        smoothed_score = np.mean(self.attention_history)

        # Update attention state
        self._update_attention_state(smoothed_score)

    def _calculate_attention_mediapipe(self, frame) -> float:
        """Calculate attention using MediaPipe Face Mesh"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return 0.0

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Calculate face orientation
        nose_tip = landmarks[1]
        nose_x = nose_tip.x
        face_center_score = 1.0 - abs(nose_x - 0.5) * 2
        face_center_score = max(0, face_center_score)

        # Calculate eye openness
        left_ear = self._eye_aspect_ratio(landmarks, "left")
        right_ear = self._eye_aspect_ratio(landmarks, "right")
        avg_ear = (left_ear + right_ear) / 2
        eye_open_score = min(1.0, avg_ear / 0.25)

        # Calculate gaze score
        gaze_score = self._estimate_gaze_score(landmarks)

        # Combine scores
        attention_score = (
            face_center_score * 0.3 +
            eye_open_score * 0.3 +
            gaze_score * 0.4
        )

        return attention_score

    def _calculate_attention_basic(self, frame) -> float:
        """Calculate attention using basic face detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return 0.0

        # Get largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Calculate face center position
        frame_h, frame_w = frame.shape[:2]
        face_center_x = (x + w/2) / frame_w

        # Score based on how centered the face is
        center_score = 1.0 - abs(face_center_x - 0.5) * 2
        center_score = max(0, center_score)

        # Score based on face size (larger = closer = more attention)
        size_score = min(1.0, (w * h) / (frame_w * frame_h) * 10)

        return (center_score * 0.6 + size_score * 0.4)

    def _eye_aspect_ratio(self, landmarks, eye: str) -> float:
        """Calculate eye aspect ratio (EAR)"""
        if eye == "left":
            p1 = landmarks[159]  # Top
            p2 = landmarks[145]  # Bottom
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

    def _estimate_gaze_score(self, landmarks) -> float:
        """Estimate if user is looking at camera"""
        try:
            # Get iris landmarks (468-477 for refined landmarks)
            left_iris = landmarks[468]
            right_iris = landmarks[473]

            # Get eye corners
            left_inner = landmarks[133]
            left_outer = landmarks[33]
            right_inner = landmarks[362]
            right_outer = landmarks[263]

            # Calculate iris position
            left_eye_width = abs(left_outer.x - left_inner.x)
            right_eye_width = abs(right_outer.x - right_inner.x)

            if left_eye_width == 0 or right_eye_width == 0:
                return 0.5

            left_iris_pos = (left_iris.x - left_outer.x) / left_eye_width
            right_iris_pos = (right_iris.x - right_outer.x) / right_eye_width

            avg_iris_pos = (left_iris_pos + right_iris_pos) / 2

            # Score: 1.0 at center (0.5), lower toward edges
            gaze_score = 1.0 - abs(avg_iris_pos - 0.5) * 2
            gaze_score = max(0, min(1, gaze_score))

            return gaze_score

        except (IndexError, AttributeError):
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

                self.message_bus.send(
                    MessageType.ATTENTION_GAINED,
                    source="gaze_detector",
                    payload={
                        "score": score,
                        "timestamp": now
                    }
                )

        else:
            # Check timeout
            if self.has_attention and self.last_attention_time:
                time_since = now - self.last_attention_time

                if time_since > self.attention_timeout:
                    # Attention lost
                    self.has_attention = False
                    duration = now - self.attention_start_time if self.attention_start_time else 0

                    logger.info(f"Attention lost after {duration:.1f}s")

                    self.message_bus.send(
                        MessageType.ATTENTION_LOST,
                        source="gaze_detector",
                        payload={
                            "duration": duration,
                            "timestamp": now
                        }
                    )

                    self.attention_start_time = None

    def _cleanup(self):
        """Clean up resources"""
        if self._camera is not None:
            self._camera.release()
        if self._face_mesh is not None:
            self._face_mesh.close()


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Gaze Detection Test")
    print("=" * 40)
    print(f"numpy available: {NUMPY_AVAILABLE}")
    print(f"opencv available: {CV2_AVAILABLE}")
    print(f"mediapipe available: {MEDIAPIPE_AVAILABLE}")

    if CV2_AVAILABLE and NUMPY_AVAILABLE:
        # Test camera
        print("\nTesting camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera works! Frame shape: {frame.shape}")

                # Test face detection
                if MEDIAPIPE_AVAILABLE:
                    face_mesh = mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=1,
                        min_detection_confidence=0.5
                    )
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        print("Face detected with MediaPipe!")
                    else:
                        print("No face detected")
                    face_mesh.close()
                else:
                    # Basic detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    print(f"Detected {len(faces)} face(s) with Haar cascade")

            cap.release()
        else:
            print("Could not open camera")

    print("\nTest complete")
