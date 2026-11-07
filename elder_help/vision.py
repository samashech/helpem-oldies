"""MediaPipe wrappers. Frames are analysed in memory and never written to disk."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# MediaPipe imports sounddevice for audio features we don't use. Loading PortAudio under PipeWire's
# JACK layer sets RLIMIT_RTTIME to 0, and the kernel then SIGKILLs the process when models load.
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELS = Path(__file__).resolve().parent.parent / "models"
NVIDIA_EGL = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
OBJECT_INTERVAL_S = 0.5
OBJECT_MEMORY_S = 2.0
# COCO classes someone would handle. Furniture is left out so sitting near a chair does not count.
USABLE_OBJECTS = {
    "bottle", "cup", "bowl", "fork", "knife", "spoon", "cell phone", "remote", "book",
    "laptop", "keyboard", "mouse", "scissors", "toothbrush", "umbrella", "handbag",
}
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

Detection = tuple[str, tuple[float, float, float, float]]


def prefer_nvidia_gpu() -> bool:
    """MediaPipe's GPU mode picks the integrated GPU on dual-GPU laptops unless told otherwise."""
    if not NVIDIA_EGL.exists():
        return False
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(NVIDIA_EGL))
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    return True


class VisionPipeline:
    def __init__(self, detect_objects: bool = True, delegate: str = "GPU", pose_model: str = "heavy"):
        accelerator = getattr(BaseOptions.Delegate, delegate)
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODELS / f"pose_landmarker_{pose_model}.task"), delegate=accelerator),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._objects = None
        if detect_objects:
            self._objects = vision.ObjectDetector.create_from_options(
                vision.ObjectDetectorOptions(
                    # This quantized model only runs on the CPU delegate.
                    base_options=BaseOptions(model_asset_path=str(MODELS / "efficientdet_lite0.tflite"), delegate=BaseOptions.Delegate.CPU),
                    score_threshold=0.4,
                    max_results=8,
                )
            )
        self._last_ms = -1
        self._next_objects = 0.0
        self._detections: list[Detection] = []
        self._detections_at = 0.0

    def process(self, frame_bgr: np.ndarray, t: float) -> tuple[np.ndarray | None, list[Detection]]:
        """Returns pixel landmarks shaped (33, 3), or None, plus recent object detections."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = timestamp
        result = self._pose.detect_for_video(image, timestamp)
        points = None
        if result.pose_landmarks:
            height, width = frame_bgr.shape[:2]
            points = np.array(
                [[lm.x * width, lm.y * height, lm.visibility if lm.visibility is not None else 0.0] for lm in result.pose_landmarks[0]],
                dtype=np.float64,
            )
        if self._objects is not None and t >= self._next_objects:
            self._next_objects = t + OBJECT_INTERVAL_S
            self._detections = [
                (d.categories[0].category_name, (d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.origin_x + d.bounding_box.width, d.bounding_box.origin_y + d.bounding_box.height))
                for d in self._objects.detect(image).detections
                if d.categories[0].category_name in USABLE_OBJECTS
            ]
            self._detections_at = t
        return points, self._detections if t - self._detections_at <= OBJECT_MEMORY_S else []

    def close(self) -> None:
        self._pose.close()
        if self._objects is not None:
            self._objects.close()


def draw_overlay(frame: np.ndarray, points: np.ndarray | None) -> np.ndarray:
    output = frame.copy()
    if points is not None:
        for a, b in SKELETON:
            if points[a, 2] >= 0.5 and points[b, 2] >= 0.5:
                cv2.line(output, (int(points[a, 0]), int(points[a, 1])), (int(points[b, 0]), int(points[b, 1])), (80, 220, 120), 3)
    return output
