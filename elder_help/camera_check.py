"""Guided check of what the camera can detect in a real room. It keeps numbers, never images."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .activity import Activity, ActivityTracker, classify_posture, describe_view
from .config import Config
from .vision import VisionPipeline

MOVE_S = 10.0  # Long enough to walk away from the computer.
# Instruction, and the activities that count as right.
POSES = {
    "sit": ("Sit down normally and stay still.", {Activity.SITTING}),
    "stand": ("Stand up, facing the camera, and stay still.", {Activity.STANDING, Activity.UPRIGHT}),
    "walk": ("Walk slowly from side to side.", {Activity.WALKING}),
    "bend": ("Stand and bend down as if picking something up. Keep doing it.", {Activity.STANDING, Activity.WALKING, Activity.UPRIGHT}),
    "lie": ("Carefully lie down on your side, across the camera's view, and stay still.", {Activity.LYING}),
}


def run_check(
    config: Config,
    vision: VisionPipeline,
    speak: Callable[[str, str], None] | None,
    seconds: float,
    output: Path,
    only: list[str] | None = None,
) -> int:
    def say(text: str) -> None:
        print(f"\n>> {text}", flush=True)
        if speak is not None:
            speak("en", text)

    capture = cv2.VideoCapture(config.camera_index)
    if not capture.isOpened():
        print("Could not open the camera.")
        return 1
    results = {}
    tracker = ActivityTracker(bed_zones=config.bed_zones)
    try:
        for name, (instruction, accepted) in POSES.items():
            if only and name not in only:
                continue
            say(instruction)
            _drain(capture, vision, tracker, MOVE_S if speak is not None or only is None else 0)
            results[name] = _measure(capture, vision, tracker, seconds, accepted)
            print(_summary(name, results[name]), flush=True)
    finally:
        capture.release()
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved numbers, with no images, to {output}.")
    _advise(results)
    return 0


def _drain(capture, vision, tracker, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        ok, frame = capture.read()
        if ok:
            now = time.monotonic()
            points, _ = vision.process(frame, now)
            tracker.update(points, now, (frame.shape[1], frame.shape[0]))


def _measure(capture, vision, tracker, seconds: float, accepted: set[Activity]) -> dict:
    labels: Counter[str] = Counter()
    raw: Counter[str] = Counter()
    views: Counter[str] = Counter()
    joints: dict[str, list[float]] = {"torso_deg": [], "shoulder_width": [], "shoulder_y": [], "hip_visibility": [], "ankle_visibility": []}
    deadline = time.monotonic() + seconds
    frames = 0
    while time.monotonic() < deadline:
        ok, frame = capture.read()
        if not ok:
            continue
        now = time.monotonic()
        height, width = frame.shape[:2]
        points, _ = vision.process(frame, now)
        obs = tracker.update(points, now, (width, height))
        frames += 1
        labels[obs.activity.value] += 1
        views[describe_view(points)] += 1
        if points is None:
            raw["no_person"] += 1
            continue
        posture, torso = classify_posture(points, tracker.th, (width, height))
        raw[posture.value] += 1
        if torso is not None:
            joints["torso_deg"].append(torso)
        joints["shoulder_width"].append(float(np.linalg.norm(points[11, :2] - points[12, :2]) / width))
        joints["shoulder_y"].append(float((points[11, 1] + points[12, 1]) / 2 / height))
        joints["hip_visibility"].append(float((points[23, 2] + points[24, 2]) / 2))
        joints["ankle_visibility"].append(float((points[27, 2] + points[28, 2]) / 2))
    right = sum(labels[activity.value] for activity in accepted)
    return {
        "frames": frames,
        "correct_fraction": round(right / max(frames, 1), 3),
        "labels": dict(labels),
        "single_frame_postures": dict(raw),
        "views": dict(views),
        "medians": {key: round(float(np.median(values)), 3) for key, values in joints.items() if values},
    }


def _summary(name: str, result: dict) -> str:
    labels = ", ".join(f"{label} {count}" for label, count in Counter(result["labels"]).most_common())
    view = Counter(result["views"]).most_common(1)[0][0] if result["views"] else "nothing"
    return f"   {name}: {result['correct_fraction'] * 100:.0f}% right | saw: {labels} | view: {view}"


def _advise(results: dict) -> None:
    views = Counter()
    for result in results.values():
        views.update(result["views"])
    total = sum(views.values()) or 1
    if views["full_body"] / total < 0.5:
        print(
            "\nThe camera could not see the whole body for most of this. Walking and lying on the floor\n"
            "can't be detected from there. Put it further back and higher up, then run this again."
        )
