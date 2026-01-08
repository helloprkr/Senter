#!/usr/bin/env python3
"""
Gaze Detection

Determines if user is paying attention to Senter by:
- Detecting face presence
- Tracking eye gaze direction
- Estimating attention level

When attention detected, enables voice interaction.
"""

import os
import time
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
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


def is_gaze_disabled() -> bool:
    """Check if gaze detection is disabled via environment variable (GD-001)"""
    return os.environ.get("SENTER_DISABLE_GAZE", "").lower() in ("1", "true", "yes")


def check_camera_available(camera_id: int = 0) -> bool:
    """Check if camera is available (GD-001)"""
    if not CV2_AVAILABLE:
        return False

    try:
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
    except Exception as e:
        logger.warning(f"Camera check failed: {e}")

    return False


@dataclass
class FaceInfo:
    """Tracked face information (GD-003)"""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: int
    center: Tuple[float, float]
    landmarks: Optional[Any] = None
    is_primary: bool = False
    occlusion_score: float = 1.0  # 1.0 = no occlusion, 0.0 = fully occluded (GD-002)
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": self.bbox,
            "area": self.area,
            "center": self.center,
            "is_primary": self.is_primary,
            "occlusion_score": self.occlusion_score,
            "last_seen": self.last_seen
        }


class GazeDetector:
    """
    Main gaze detection controller (GD-001, GD-002, GD-003).

    Uses MediaPipe Face Mesh for:
    - Face detection
    - Eye landmark tracking
    - Gaze direction estimation

    Features:
    - GD-001: Enable by default with env var override
    - GD-002: Partial face occlusion handling (glasses, masks)
    - GD-003: Multi-face handling with primary face tracking
    """

    def __init__(
        self,
        camera_id: int,
        attention_threshold: float,
        message_bus: MessageBus,
        shutdown_event: Event,
        max_faces: int = 5  # GD-003
    ):
        self.camera_id = camera_id
        self.attention_threshold = attention_threshold
        self.message_bus = message_bus
        self.shutdown_event = shutdown_event
        self.max_faces = max_faces  # GD-003

        # State
        self.has_attention = False
        self.attention_start_time = None
        self.last_attention_time = None
        self._enabled = True
        self._camera_available = False

        # Smoothing
        self.attention_history = []
        self.history_size = 10

        # Attention timeout
        self.attention_timeout = 2.0

        # Face tracking (GD-003)
        self._tracked_faces: Dict[int, FaceInfo] = {}
        self._next_face_id = 0
        self._primary_face_id: Optional[int] = None
        self._face_timeout = 1.0  # seconds before face is considered lost

        # Components
        self._face_mesh = None
        self._camera = None

        # Message queue
        self._queue = None

    def get_status(self) -> Dict[str, Any]:
        """Get detector status for CLI (GD-001)"""
        return {
            "enabled": self._enabled,
            "camera_available": self._camera_available,
            "has_attention": self.has_attention,
            "attention_threshold": self.attention_threshold,
            "camera_id": self.camera_id,
            "tracked_faces": len(self._tracked_faces),
            "primary_face_id": self._primary_face_id,
            "using_mediapipe": self._face_mesh is not None
        }

    def get_tracked_faces(self) -> List[FaceInfo]:
        """Get list of currently tracked faces (GD-003)"""
        return list(self._tracked_faces.values())

    def run(self):
        """Main detection loop"""
        logger.info("Gaze detector starting...")

        # Check if disabled via environment variable (GD-001)
        if is_gaze_disabled():
            logger.warning("Gaze detection disabled via SENTER_DISABLE_GAZE environment variable")
            self._enabled = False
            self._wait_for_shutdown()
            return

        # Check dependencies
        if not CV2_AVAILABLE:
            logger.error("opencv-python required for gaze detection - disabling")
            self._enabled = False
            self._wait_for_shutdown()
            return

        if not MEDIAPIPE_AVAILABLE:
            logger.warning("mediapipe not available - using basic face detection")

        if not NUMPY_AVAILABLE:
            logger.error("numpy required for gaze detection - disabling")
            self._enabled = False
            self._wait_for_shutdown()
            return

        # Initialize camera with graceful fallback (GD-001)
        try:
            self._init_camera()
            self._camera_available = True
        except Exception as e:
            logger.warning(f"Camera initialization failed: {e} - gaze detection disabled")
            self._camera_available = False
            self._enabled = False
            self._wait_for_shutdown()
            return

        # Initialize face mesh with multi-face support (GD-003)
        if MEDIAPIPE_AVAILABLE:
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=self.max_faces,  # GD-003: Track multiple faces
                    refine_landmarks=True,  # GD-002: Better landmark detection
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                logger.warning(f"Face mesh init failed: {e}")

        # Register with message bus
        self._queue = self.message_bus.register("gaze")

        logger.info(f"Gaze detector started (max_faces={self.max_faces})")

    def _wait_for_shutdown(self):
        """Wait for shutdown when disabled"""
        while not self.shutdown_event.is_set():
            time.sleep(0.5)

    def run_loop(self):
        """Main detection loop (called after run() setup)"""
        try:
            while not self.shutdown_event.is_set():
                self._process_frame()
                self._cleanup_stale_faces()  # GD-003
                time.sleep(0.066)  # ~15 FPS
        except Exception as e:
            logger.error(f"Gaze detector error: {e}")
        finally:
            self._cleanup()

        logger.info("Gaze detector stopped")

    def _cleanup_stale_faces(self):
        """Remove faces not seen recently (GD-003)"""
        now = time.time()
        stale_ids = [
            fid for fid, face in self._tracked_faces.items()
            if now - face.last_seen > self._face_timeout
        ]
        for fid in stale_ids:
            del self._tracked_faces[fid]
            if self._primary_face_id == fid:
                self._primary_face_id = None
                # Select new primary if other faces exist
                self._select_primary_face()

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

    def _select_primary_face(self):
        """Select the primary face (largest/closest) (GD-003)"""
        if not self._tracked_faces:
            self._primary_face_id = None
            return

        # Select face with largest area
        primary = max(self._tracked_faces.values(), key=lambda f: f.area)
        self._primary_face_id = primary.id

        # Mark faces
        for face in self._tracked_faces.values():
            face.is_primary = (face.id == self._primary_face_id)

    def _calculate_attention_mediapipe(self, frame) -> float:
        """Calculate attention using MediaPipe Face Mesh (GD-002, GD-003)"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        now = time.time()

        # Process
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # Log face count for debugging (GD-003)
            if self._tracked_faces:
                logger.debug(f"Faces lost: {len(self._tracked_faces)} -> 0")
            return 0.0

        # Log face count changes (GD-003)
        num_faces = len(results.multi_face_landmarks)
        if num_faces != len(self._tracked_faces):
            logger.debug(f"Face count: {len(self._tracked_faces)} -> {num_faces}")

        # Update tracked faces (GD-003)
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = face_landmarks.landmark

            # Calculate face bounding box
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x1, x2 = int(min(xs) * w), int(max(xs) * w)
            y1, y2 = int(min(ys) * h), int(max(ys) * h)
            bbox = (x1, y1, x2 - x1, y2 - y1)
            area = (x2 - x1) * (y2 - y1)
            center = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h)

            # Calculate occlusion score (GD-002)
            occlusion_score = self._calculate_occlusion_score(landmarks)

            # Find or create face tracking
            face_id = self._match_or_create_face(bbox, center)
            self._tracked_faces[face_id] = FaceInfo(
                id=face_id,
                bbox=bbox,
                area=area,
                center=center,
                landmarks=landmarks,
                occlusion_score=occlusion_score,
                last_seen=now
            )

        # Select primary face (GD-003)
        self._select_primary_face()

        # Calculate attention for primary face only
        if self._primary_face_id is None:
            return 0.0

        primary_face = self._tracked_faces.get(self._primary_face_id)
        if not primary_face or not primary_face.landmarks:
            return 0.0

        landmarks = primary_face.landmarks

        # Calculate face orientation
        nose_tip = landmarks[1]
        nose_x = nose_tip.x
        face_center_score = 1.0 - abs(nose_x - 0.5) * 2
        face_center_score = max(0, face_center_score)

        # Calculate eye openness (GD-002: handles glasses)
        left_ear = self._eye_aspect_ratio(landmarks, "left")
        right_ear = self._eye_aspect_ratio(landmarks, "right")
        avg_ear = (left_ear + right_ear) / 2
        eye_open_score = min(1.0, avg_ear / 0.25)

        # Calculate gaze score
        gaze_score = self._estimate_gaze_score(landmarks)

        # Combine scores with occlusion penalty (GD-002)
        base_score = (
            face_center_score * 0.3 +
            eye_open_score * 0.3 +
            gaze_score * 0.4
        )

        # Apply occlusion factor - reduce confidence when face is occluded
        attention_score = base_score * primary_face.occlusion_score

        return attention_score

    def _match_or_create_face(self, bbox: Tuple, center: Tuple) -> int:
        """Match existing face or create new one (GD-003)"""
        # Try to match by center proximity
        for fid, face in self._tracked_faces.items():
            dx = abs(face.center[0] - center[0])
            dy = abs(face.center[1] - center[1])
            if dx < 0.15 and dy < 0.15:  # Within ~15% of frame
                return fid

        # Create new face
        face_id = self._next_face_id
        self._next_face_id += 1
        logger.debug(f"New face detected: ID {face_id}")
        return face_id

    def _calculate_occlusion_score(self, landmarks) -> float:
        """Calculate face occlusion score (GD-002)

        1.0 = fully visible, 0.0 = fully occluded
        Handles glasses, masks, partial visibility
        """
        try:
            # Check eye landmark visibility - glasses detection
            left_eye_landmarks = [159, 145, 33, 133]  # Top, bottom, left, right
            right_eye_landmarks = [386, 374, 362, 263]

            left_visibility = np.mean([landmarks[i].visibility for i in left_eye_landmarks
                                       if hasattr(landmarks[i], 'visibility')])
            right_visibility = np.mean([landmarks[i].visibility for i in right_eye_landmarks
                                        if hasattr(landmarks[i], 'visibility')])
            eye_visibility = (left_visibility + right_visibility) / 2

            # Check nose and mouth visibility - mask detection
            nose_landmarks = [1, 2, 4, 5, 6]
            mouth_landmarks = [13, 14, 78, 308]

            nose_visibility = np.mean([landmarks[i].visibility for i in nose_landmarks
                                       if hasattr(landmarks[i], 'visibility')])
            mouth_visibility = np.mean([landmarks[i].visibility for i in mouth_landmarks
                                        if hasattr(landmarks[i], 'visibility')])

            # Combine: eyes are most important for attention
            occlusion_score = (
                eye_visibility * 0.6 +
                nose_visibility * 0.2 +
                mouth_visibility * 0.2
            )

            # Clamp and return
            return max(0.3, min(1.0, occlusion_score))  # Min 0.3 to allow reduced confidence detection

        except (AttributeError, IndexError):
            # Landmarks don't have visibility - assume visible
            return 1.0

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
