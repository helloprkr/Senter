"""
Gaze detection for wake-word-free activation.

Look at camera -> Senter starts listening.
"""

from __future__ import annotations
import asyncio
import time
from typing import Callable, Optional


class GazeDetector:
    """
    Detects when user is looking at camera.

    Uses MediaPipe for face mesh and gaze estimation.
    """

    def __init__(self):
        self.running = False
        self.cap = None
        self.face_mesh = None
        self.mp_face_mesh = None

        # Gaze tracking state
        self.looking_at_camera = False
        self.gaze_start_time: Optional[float] = None
        self.activation_threshold = 0.5  # Seconds of sustained gaze
        self._activated = False

    def load(self) -> None:
        """Load face detection model."""
        try:
            import mediapipe as mp

            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("[GAZE] MediaPipe loaded")
        except ImportError:
            print("[GAZE] MediaPipe not available. Install: pip install mediapipe")
            print("[GAZE] Gaze detection will be disabled")

    def is_available(self) -> bool:
        """Check if gaze detection is available."""
        return self.face_mesh is not None

    async def start(
        self,
        on_gaze_start: Callable[[], None],
        on_gaze_end: Callable[[], None],
        camera_id: int = 0,
    ) -> None:
        """
        Start gaze detection.

        Calls on_gaze_start when user looks at camera,
        on_gaze_end when they look away.
        """
        if not self.face_mesh:
            print("[GAZE] Face mesh not loaded, cannot start")
            return

        try:
            import cv2
        except ImportError:
            print("[GAZE] OpenCV not available. Install: pip install opencv-python")
            return

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"[GAZE] Could not open camera {camera_id}")
            return

        self.running = True
        self._activated = False
        print(f"[GAZE] Started with camera {camera_id}")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            is_looking = False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get eye landmarks for gaze estimation
                    is_looking = self._estimate_gaze(face_landmarks, frame.shape)

            # State machine
            current_time = time.time()

            if is_looking and not self.looking_at_camera:
                # Started looking
                self.gaze_start_time = current_time

            elif is_looking and self.looking_at_camera:
                # Continue looking - check if past threshold
                if (
                    self.gaze_start_time
                    and current_time - self.gaze_start_time >= self.activation_threshold
                ):
                    if not self._activated:
                        self._activated = True
                        on_gaze_start()

            elif not is_looking and self.looking_at_camera:
                # Stopped looking
                if self._activated:
                    self._activated = False
                    on_gaze_end()
                self.gaze_start_time = None

            self.looking_at_camera = is_looking

            # Small delay to not hog CPU
            await asyncio.sleep(0.033)  # ~30 FPS

        # Cleanup
        if self.cap:
            self.cap.release()

    def _estimate_gaze(self, landmarks, frame_shape) -> bool:
        """
        Estimate if user is looking at camera.

        Uses iris position relative to eye corners.
        """
        try:
            # Left eye landmarks
            left_eye_inner = landmarks.landmark[133]
            left_eye_outer = landmarks.landmark[33]
            left_iris = landmarks.landmark[468]  # Iris center

            # Right eye landmarks
            right_eye_inner = landmarks.landmark[362]
            right_eye_outer = landmarks.landmark[263]
            right_iris = landmarks.landmark[473]  # Iris center

            # Calculate iris position relative to eye corners (0 = outer, 1 = inner)
            def iris_ratio(iris, inner, outer):
                eye_width = abs(inner.x - outer.x)
                if eye_width < 0.01:
                    return 0.5
                return (iris.x - outer.x) / (inner.x - outer.x)

            left_ratio = iris_ratio(left_iris, left_eye_inner, left_eye_outer)
            right_ratio = iris_ratio(right_iris, right_eye_inner, right_eye_outer)

            avg_ratio = (left_ratio + right_ratio) / 2

            # Looking at camera when iris is roughly centered (0.35-0.65)
            return 0.35 <= avg_ratio <= 0.65

        except (AttributeError, IndexError):
            return False

    def stop(self) -> None:
        """Stop detection."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[GAZE] Stopped")

    def is_user_looking(self) -> bool:
        """Check if user is currently looking at camera."""
        return self.looking_at_camera and self._activated
