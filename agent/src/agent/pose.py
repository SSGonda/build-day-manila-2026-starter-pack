"""YOLO-Pose wrapper for extracting skeleton keypoints from frames.

Uses ultralytics YOLO26n-pose model. Returns COCO-format 17-keypoint arrays.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from ultralytics import YOLO

# COCO 17 keypoints: nose, left_eye, right_eye, left_ear, right_ear,
# left_shoulder, right_shoulder, left_elbow, right_elbow,
# left_wrist, right_wrist, left_hip, right_hip,
# left_knee, right_knee, left_ankle, right_ankle
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolo26n-pose.pt")
    return _model


def run_yolo_pose(image: Image.Image) -> np.ndarray:
    """Run YOLO-Pose on a single frame and return keypoints.

    Args:
        image: PIL Image in RGB format.

    Returns:
        np.ndarray of shape (17, 3) — [x, y, confidence] per joint.
        Returns zeros if no person detected.
    """
    model = _get_model()
    results = model(image, verbose=False)

    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        # Pick the most prominent person (first detection, highest confidence)
        kps = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
        return kps

    return np.zeros((17, 3), dtype=np.float32)
