#!/usr/bin/env python3
"""
Tests for Gaze Detection (GD-001, GD-002, GD-003)
Tests gaze detection features including safety checks, occlusion, and multi-face.
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== GD-001: Gaze Detection Enable/Disable Tests ==========

def test_gaze_disabled_env_var():
    """Test SENTER_DISABLE_GAZE environment variable (GD-001)"""
    from vision.gaze_detector import is_gaze_disabled

    # Save original value
    original = os.environ.get("SENTER_DISABLE_GAZE")

    try:
        # Test disabled states
        for val in ["1", "true", "yes", "TRUE", "Yes"]:
            os.environ["SENTER_DISABLE_GAZE"] = val
            assert is_gaze_disabled(), f"Should be disabled with {val}"

        # Test enabled states
        for val in ["0", "false", "no", ""]:
            os.environ["SENTER_DISABLE_GAZE"] = val
            assert not is_gaze_disabled(), f"Should be enabled with {val}"

        # Test unset
        if "SENTER_DISABLE_GAZE" in os.environ:
            del os.environ["SENTER_DISABLE_GAZE"]
        assert not is_gaze_disabled(), "Should be enabled when unset"

    finally:
        # Restore original value
        if original is not None:
            os.environ["SENTER_DISABLE_GAZE"] = original
        elif "SENTER_DISABLE_GAZE" in os.environ:
            del os.environ["SENTER_DISABLE_GAZE"]

    return True


def test_gaze_enabled_by_default():
    """Test that gaze_detection.enabled is true in default config (GD-001)"""
    import json

    config_path = Path(__file__).parent.parent / "config" / "daemon_config.json"
    if not config_path.exists():
        print("  (skipped - config not found)")
        return True

    with open(config_path) as f:
        config = json.load(f)

    assert config["components"]["gaze_detection"]["enabled"] is True

    return True


def test_gaze_detector_get_status():
    """Test GazeDetector.get_status() returns all fields (GD-001)"""
    from vision.gaze_detector import GazeDetector, CV2_AVAILABLE, NUMPY_AVAILABLE
    from multiprocessing import Event

    if not NUMPY_AVAILABLE or not CV2_AVAILABLE:
        print("  (skipped - dependencies not available)")
        return True

    mock_bus = MagicMock()
    mock_bus.register.return_value = MagicMock()

    detector = GazeDetector(
        camera_id=0,
        attention_threshold=0.7,
        message_bus=mock_bus,
        shutdown_event=Event()
    )

    status = detector.get_status()

    # Check all required fields
    required_fields = [
        "enabled", "camera_available", "has_attention",
        "attention_threshold", "camera_id", "tracked_faces",
        "primary_face_id", "using_mediapipe"
    ]
    for field in required_fields:
        assert field in status, f"Missing field: {field}"

    return True


def test_camera_availability_check():
    """Test check_camera_available function (GD-001)"""
    from vision.gaze_detector import check_camera_available, CV2_AVAILABLE

    # Function should return bool
    result = check_camera_available(0)
    assert isinstance(result, bool)

    # If cv2 not available, should return False
    if not CV2_AVAILABLE:
        assert result is False

    return True


# ========== GD-002: Occlusion Handling Tests ==========

def test_face_info_has_occlusion_score():
    """Test FaceInfo has occlusion_score field (GD-002)"""
    from vision.gaze_detector import FaceInfo

    face = FaceInfo(
        id=1,
        bbox=(100, 100, 200, 200),
        area=40000,
        center=(0.5, 0.5),
        occlusion_score=0.8
    )

    assert face.occlusion_score == 0.8
    assert 0.0 <= face.occlusion_score <= 1.0

    return True


def test_occlusion_score_in_face_dict():
    """Test occlusion_score is in FaceInfo.to_dict() (GD-002)"""
    from vision.gaze_detector import FaceInfo

    face = FaceInfo(
        id=1,
        bbox=(100, 100, 200, 200),
        area=40000,
        center=(0.5, 0.5),
        occlusion_score=0.7
    )

    d = face.to_dict()

    assert "occlusion_score" in d
    assert d["occlusion_score"] == 0.7

    return True


def test_occlusion_score_default():
    """Test default occlusion_score is 1.0 (fully visible) (GD-002)"""
    from vision.gaze_detector import FaceInfo

    face = FaceInfo(
        id=1,
        bbox=(100, 100, 200, 200),
        area=40000,
        center=(0.5, 0.5)
    )

    # Default should be 1.0 (fully visible)
    assert face.occlusion_score == 1.0

    return True


def test_gaze_detector_has_occlusion_method():
    """Test GazeDetector has _calculate_occlusion_score method (GD-002)"""
    from vision.gaze_detector import GazeDetector, CV2_AVAILABLE, NUMPY_AVAILABLE
    from multiprocessing import Event

    if not NUMPY_AVAILABLE:
        print("  (skipped - numpy not available)")
        return True

    mock_bus = MagicMock()

    detector = GazeDetector(
        camera_id=0,
        attention_threshold=0.7,
        message_bus=mock_bus,
        shutdown_event=Event()
    )

    # Method should exist
    assert hasattr(detector, '_calculate_occlusion_score')
    assert callable(detector._calculate_occlusion_score)

    return True


# ========== GD-003: Multi-Face Handling Tests ==========

def test_face_info_dataclass():
    """Test FaceInfo dataclass structure (GD-003)"""
    from vision.gaze_detector import FaceInfo

    face = FaceInfo(
        id=1,
        bbox=(100, 100, 200, 200),
        area=40000,
        center=(0.5, 0.5),
        is_primary=True,
        last_seen=time.time()
    )

    assert face.id == 1
    assert face.bbox == (100, 100, 200, 200)
    assert face.area == 40000
    assert face.center == (0.5, 0.5)
    assert face.is_primary is True

    return True


def test_face_info_to_dict():
    """Test FaceInfo.to_dict() (GD-003)"""
    from vision.gaze_detector import FaceInfo

    now = time.time()
    face = FaceInfo(
        id=2,
        bbox=(50, 50, 150, 150),
        area=22500,
        center=(0.4, 0.4),
        is_primary=False,
        last_seen=now
    )

    d = face.to_dict()

    assert d["id"] == 2
    assert d["bbox"] == (50, 50, 150, 150)
    assert d["area"] == 22500
    assert d["center"] == (0.4, 0.4)
    assert d["is_primary"] is False
    assert d["last_seen"] == now

    return True


def test_gaze_detector_max_faces():
    """Test GazeDetector max_faces parameter (GD-003)"""
    from vision.gaze_detector import GazeDetector
    from multiprocessing import Event

    mock_bus = MagicMock()

    detector = GazeDetector(
        camera_id=0,
        attention_threshold=0.7,
        message_bus=mock_bus,
        shutdown_event=Event(),
        max_faces=3
    )

    assert detector.max_faces == 3

    return True


def test_gaze_detector_tracks_faces():
    """Test GazeDetector has face tracking structures (GD-003)"""
    from vision.gaze_detector import GazeDetector
    from multiprocessing import Event

    mock_bus = MagicMock()

    detector = GazeDetector(
        camera_id=0,
        attention_threshold=0.7,
        message_bus=mock_bus,
        shutdown_event=Event()
    )

    # Should have tracking structures
    assert hasattr(detector, '_tracked_faces')
    assert isinstance(detector._tracked_faces, dict)
    assert hasattr(detector, '_primary_face_id')

    return True


def test_get_tracked_faces():
    """Test get_tracked_faces returns list (GD-003)"""
    from vision.gaze_detector import GazeDetector, FaceInfo
    from multiprocessing import Event

    mock_bus = MagicMock()

    detector = GazeDetector(
        camera_id=0,
        attention_threshold=0.7,
        message_bus=mock_bus,
        shutdown_event=Event()
    )

    # Initially empty
    faces = detector.get_tracked_faces()
    assert isinstance(faces, list)
    assert len(faces) == 0

    # Add a face manually
    detector._tracked_faces[0] = FaceInfo(
        id=0,
        bbox=(0, 0, 100, 100),
        area=10000,
        center=(0.5, 0.5),
        last_seen=time.time()
    )

    faces = detector.get_tracked_faces()
    assert len(faces) == 1
    assert faces[0].id == 0

    return True


def test_select_primary_face():
    """Test _select_primary_face selects largest face (GD-003)"""
    from vision.gaze_detector import GazeDetector, FaceInfo
    from multiprocessing import Event

    mock_bus = MagicMock()

    detector = GazeDetector(
        camera_id=0,
        attention_threshold=0.7,
        message_bus=mock_bus,
        shutdown_event=Event()
    )

    now = time.time()

    # Add faces with different sizes
    detector._tracked_faces[0] = FaceInfo(
        id=0, bbox=(0, 0, 100, 100), area=10000, center=(0.3, 0.5), last_seen=now
    )
    detector._tracked_faces[1] = FaceInfo(
        id=1, bbox=(200, 0, 200, 200), area=40000, center=(0.7, 0.5), last_seen=now
    )
    detector._tracked_faces[2] = FaceInfo(
        id=2, bbox=(100, 100, 50, 50), area=2500, center=(0.5, 0.5), last_seen=now
    )

    # Select primary
    detector._select_primary_face()

    # Largest face (id=1 with area=40000) should be primary
    assert detector._primary_face_id == 1
    assert detector._tracked_faces[1].is_primary is True
    assert detector._tracked_faces[0].is_primary is False
    assert detector._tracked_faces[2].is_primary is False

    return True


def test_match_or_create_face():
    """Test _match_or_create_face for face persistence (GD-003)"""
    from vision.gaze_detector import GazeDetector, FaceInfo
    from multiprocessing import Event

    mock_bus = MagicMock()

    detector = GazeDetector(
        camera_id=0,
        attention_threshold=0.7,
        message_bus=mock_bus,
        shutdown_event=Event()
    )

    now = time.time()

    # Add existing face and update counter
    detector._tracked_faces[0] = FaceInfo(
        id=0, bbox=(100, 100, 200, 200), area=40000, center=(0.5, 0.5), last_seen=now
    )
    detector._next_face_id = 1  # Simulate proper counter state

    # Match nearby position - should return same ID
    face_id = detector._match_or_create_face(
        bbox=(105, 105, 195, 195),
        center=(0.52, 0.52)
    )
    assert face_id == 0

    # Far away position - should create new face
    face_id = detector._match_or_create_face(
        bbox=(400, 400, 100, 100),
        center=(0.9, 0.9)
    )
    assert face_id == 1  # New face ID

    return True


if __name__ == "__main__":
    tests = [
        # GD-001
        test_gaze_disabled_env_var,
        test_gaze_enabled_by_default,
        test_gaze_detector_get_status,
        test_camera_availability_check,
        # GD-002
        test_face_info_has_occlusion_score,
        test_occlusion_score_in_face_dict,
        test_occlusion_score_default,
        test_gaze_detector_has_occlusion_method,
        # GD-003
        test_face_info_dataclass,
        test_face_info_to_dict,
        test_gaze_detector_max_faces,
        test_gaze_detector_tracks_faces,
        test_get_tracked_faces,
        test_select_primary_face,
        test_match_or_create_face,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
