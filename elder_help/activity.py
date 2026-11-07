"""Turns pose landmarks into activities. No camera or model code lives here, so it can be tested."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

MIN_VISIBILITY = 0.5
FALL_CLEAR_S = 3.0


class Activity(str, Enum):
    ABSENT = "absent"
    UPRIGHT = "upright"  # Only the upper body is in view and it can't be told apart any further.
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    RUNNING = "running"
    LYING = "lying"


UPRIGHT_ACTIVITIES = {Activity.UPRIGHT, Activity.STANDING, Activity.WALKING, Activity.RUNNING}


@dataclass(frozen=True)
class Thresholds:
    """Starting values chosen by hand. They need checking against real footage of the person."""

    lying_torso_deg: float = 60.0
    sitting_thigh_deg: float = 55.0
    sitting_thigh_ratio: float = 0.45
    bent_leg_drop: float = 0.9
    walk_speed: float = 0.5
    run_speed: float = 2.5
    walk_leg_swing: float = 0.08
    run_leg_swing: float = 0.35
    fall_drop: float = 1.0
    absent_after_s: float = 3.0
    rise_attempt: float = 0.35
    attempts_for_struggle: int = 3
    # A person at a laptop fills the frame with their shoulders low in it. Measured at a real desk:
    # shoulders 40% of the frame width and 87% of the way down.
    desk_shoulder_width: float = 0.2
    desk_shoulder_height: float = 0.5
    exit_drop: float = 0.12
    change_after_s: float = 0.7


@dataclass(frozen=True)
class BedZone:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class Observation:
    activity: Activity
    fall_at: float | None = None
    struggling: bool = False
    in_bed: bool = False
    held_object: str | None = None
    torso_deg: float | None = None
    came_back: bool = False  # Seen again after dropping out of frame, so it wasn't a fall.


@dataclass(frozen=True)
class _Sample:
    t: float
    posture: Activity
    hip: tuple[float, float]
    shoulder_y: float
    torso: float
    leg_gap: float | None
    full_torso: bool = True


def _visible(points: np.ndarray, *indices: int) -> bool:
    return all(points[i, 2] >= MIN_VISIBILITY for i in indices)


def _mid(points: np.ndarray, a: int, b: int) -> np.ndarray:
    return (points[a, :2] + points[b, :2]) / 2


def _deg_from_vertical(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def describe_view(points: np.ndarray | None) -> str:
    """Says how much of the body the camera can see, which limits what can be detected."""
    if points is None or not _visible(points, L_SHOULDER, R_SHOULDER):
        return "nobody"
    if not _visible(points, L_HIP, R_HIP):
        return "head_and_shoulders"
    if not (_visible(points, L_KNEE, R_KNEE) or _visible(points, L_ANKLE, R_ANKLE)):
        return "upper_body"
    return "full_body"


def classify_posture(
    points: np.ndarray, th: Thresholds = Thresholds(), frame_size: tuple[int, int] | None = None
) -> tuple[Activity, float | None]:
    """Classifies one frame from pixel landmarks shaped (33, 3): x, y, visibility."""
    if not _visible(points, L_SHOULDER, R_SHOULDER):
        return Activity.ABSENT, None
    shoulders = _mid(points, L_SHOULDER, R_SHOULDER)

    if not _visible(points, L_HIP, R_HIP):
        # A desk webcam often sees only head and shoulders. A sideways head over a steep
        # shoulder line is the remaining evidence that the person is lying down.
        if _visible(points, NOSE):
            head = points[NOSE, :2] - shoulders
            line = points[L_SHOULDER, :2] - points[R_SHOULDER, :2]
            head_deg = _deg_from_vertical(head[0], head[1])
            line_deg = 90 - _deg_from_vertical(line[0], line[1])
            if head_deg >= th.lying_torso_deg and line_deg >= 45:
                return Activity.LYING, head_deg
        if frame_size is not None:
            width, height = frame_size
            shoulder_width = float(np.linalg.norm(points[L_SHOULDER, :2] - points[R_SHOULDER, :2]))
            if shoulder_width / width >= th.desk_shoulder_width and shoulders[1] / height >= th.desk_shoulder_height:
                return Activity.SITTING, None
        return Activity.UPRIGHT, None

    hips = _mid(points, L_HIP, R_HIP)
    torso_vec = shoulders - hips
    torso = float(np.linalg.norm(torso_vec))
    if torso < 1e-3:
        return Activity.ABSENT, None
    torso_deg = _deg_from_vertical(torso_vec[0], torso_vec[1])

    if torso_deg >= th.lying_torso_deg:
        # Bending or mopping also tilts the torso, but both feet stay well below the hips.
        # One low foot is not enough, because someone who fell can have a leg sprawled out.
        if _visible(points, L_ANKLE, R_ANKLE):
            feet_drop = min(points[L_ANKLE, 1], points[R_ANKLE, 1]) - hips[1]
            if feet_drop > th.bent_leg_drop * torso:
                return Activity.STANDING, torso_deg
        return Activity.LYING, torso_deg

    if _visible(points, L_KNEE, R_KNEE):
        thigh = _mid(points, L_KNEE, R_KNEE) - hips
        thigh_len = float(np.linalg.norm(thigh))
        if (
            _deg_from_vertical(thigh[0], thigh[1]) >= th.sitting_thigh_deg
            or thigh_len < th.sitting_thigh_ratio * torso
            or thigh[1] < 0
        ):
            return Activity.SITTING, torso_deg
        return Activity.STANDING, torso_deg
    return Activity.UPRIGHT, torso_deg


def find_held_object(points: np.ndarray, detections: list[tuple[str, tuple[float, float, float, float]]]) -> str | None:
    """Returns the label of a detected object with a visible wrist on or next to it."""
    for label, (x0, y0, x1, y1) in detections:
        margin = 0.25 * max(x1 - x0, y1 - y0)
        for wrist in (L_WRIST, R_WRIST):
            x, y, visibility = points[wrist]
            if visibility >= MIN_VISIBILITY and x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
                return label
    return None


class ActivityTracker:
    """Smooths per-frame postures and spots movement, sudden drops and failed attempts to rise."""

    def __init__(self, thresholds: Thresholds = Thresholds(), bed_zones: tuple[BedZone, ...] = ()):
        self.th = thresholds
        self.bed_zones = bed_zones
        self._samples: deque[_Sample] = deque()
        self._last_seen: float | None = None
        self._activity = Activity.ABSENT
        self._in_bed = False
        self._torso_deg: float | None = None
        self._fall_at: float | None = None
        self._rising = False
        self._attempts: deque[float] = deque()
        self._candidate: Activity | None = None
        self._candidate_since = 0.0
        self._fell_out_of_view = False

    def update(
        self,
        points: np.ndarray | None,
        t: float,
        frame_size: tuple[int, int] = (1, 1),
        held_object: str | None = None,
    ) -> Observation:
        posture, torso_deg = (Activity.ABSENT, None) if points is None else classify_posture(
            points, self.th, frame_size if frame_size != (1, 1) else None
        )
        struggling = False
        came_back = False
        if posture is not Activity.ABSENT:
            self._last_seen = t
            self._torso_deg = torso_deg
            self._record(points, posture, t, frame_size)
            self._activity = self._settled(self._smoothed_activity(t), t)
            if self._fell_out_of_view and posture is not Activity.LYING:
                self._fell_out_of_view = False
                self._fall_at = None
                came_back = True
            self._detect_fall(t)
            struggling = self._detect_struggle(t)
            # The label lags behind a fall, so only being upright well after it counts as recovery.
            if self._activity in UPRIGHT_ACTIVITIES and self._fall_at is not None and t - self._fall_at >= FALL_CLEAR_S:
                self._fall_at = None
        elif self._last_seen is None or t - self._last_seen >= self.th.absent_after_s:
            self._detect_drop_out_of_view(frame_size)
            self._activity = Activity.ABSENT
            self._candidate = None
            self._samples.clear()
        return Observation(
            activity=self._activity,
            fall_at=self._fall_at,
            struggling=struggling,
            in_bed=self._in_bed,
            held_object=held_object if self._activity is not Activity.ABSENT else None,
            torso_deg=self._torso_deg,
            came_back=came_back,
        )

    def _settled(self, activity: Activity, t: float) -> Activity:
        """Holds the current label until a new one lasts, so it doesn't flicker every second."""
        if activity is self._activity or self._activity is Activity.ABSENT:
            self._candidate = None
            return activity
        if activity is not self._candidate:
            self._candidate = activity
            self._candidate_since = t
        if t - self._candidate_since >= self.th.change_after_s:
            self._candidate = None
            return activity
        return self._activity

    def _detect_drop_out_of_view(self, frame_size: tuple[int, int]) -> None:
        """A desk camera can't see the floor, so sliding out of the bottom of the frame is the signal."""
        height = frame_size[1]
        if height == 1 or self._last_seen is None or not self._samples or self._fall_at is not None:
            return
        recent = [s for s in self._samples if self._last_seen - s.t <= 1.5]
        lowest = recent[-1].shoulder_y
        if lowest / height >= 0.6 and (lowest - min(s.shoulder_y for s in recent)) / height >= self.th.exit_drop:
            self._fall_at = self._last_seen
            self._fell_out_of_view = True

    def _record(self, points: np.ndarray, posture: Activity, t: float, frame_size: tuple[int, int]) -> None:
        shoulders = _mid(points, L_SHOULDER, R_SHOULDER)
        hips_visible = _visible(points, L_HIP, R_HIP)
        hips = _mid(points, L_HIP, R_HIP) if hips_visible else shoulders
        if hips_visible:
            torso = float(np.linalg.norm(shoulders - hips))
        else:
            torso = 1.5 * float(np.linalg.norm(points[L_SHOULDER, :2] - points[R_SHOULDER, :2]))
        leg_gap = None
        if _visible(points, L_ANKLE, R_ANKLE):
            leg_gap = float(points[L_ANKLE, 1] - points[R_ANKLE, 1]) / max(torso, 1e-3)
        self._samples.append(
            _Sample(t, posture, (float(hips[0]), float(hips[1])), float(shoulders[1]), max(torso, 1e-3), leg_gap, hips_visible)
        )
        while self._samples and t - self._samples[0].t > 10.0:
            self._samples.popleft()
        width, height = frame_size
        self._in_bed = any(zone.contains(hips[0] / width, hips[1] / height) for zone in self.bed_zones)

    def _window(self, t: float, seconds: float) -> list[_Sample]:
        return [s for s in self._samples if t - s.t <= seconds]

    def _smoothed_activity(self, t: float) -> Activity:
        recent = self._window(t, 1.5)
        counts: dict[Activity, int] = {}
        for sample in recent:
            counts[sample.posture] = counts.get(sample.posture, 0) + 1
        posture = max(counts, key=counts.__getitem__)
        if posture not in (Activity.STANDING, Activity.UPRIGHT):
            return posture
        return self._with_movement(posture, t)

    def _with_movement(self, posture: Activity, t: float) -> Activity:
        recent = self._window(t, 2.0)
        span = recent[-1].t - recent[0].t
        # Leaning in a chair moves the shoulders too, so walking needs the hips in view.
        if span < 0.8 or not all(s.full_torso for s in recent):
            return posture
        torso = float(np.median([s.torso for s in recent]))
        travelled = math.dist(recent[0].hip, recent[-1].hip)
        speed = travelled / torso / span
        gaps = [s.leg_gap for s in recent if s.leg_gap is not None]
        swing = float(np.std(gaps)) if len(gaps) >= 5 else 0.0
        if speed >= self.th.run_speed or swing >= self.th.run_leg_swing:
            return Activity.RUNNING
        if speed >= self.th.walk_speed or swing >= self.th.walk_leg_swing:
            return Activity.WALKING
        return posture

    def _detect_fall(self, t: float) -> None:
        # Slow collapses never trigger this. They are caught by the lying-down timer instead.
        if self._fall_at is not None:
            return
        before = [s for s in self._window(t, 1.2) if t - s.t >= 0.4 and s.posture is not Activity.LYING]
        now = self._window(t, 0.25)
        if not before or not now:
            return
        highest = min(s.shoulder_y for s in before)
        torso = float(np.median([s.torso for s in before]))
        # Every recent sample has to be low, so one bad detection cannot raise an alarm.
        if min(s.shoulder_y for s in now) - highest >= self.th.fall_drop * torso:
            self._fall_at = t

    def _detect_struggle(self, t: float) -> bool:
        if self._activity in UPRIGHT_ACTIVITIES:
            self._attempts.clear()
            self._rising = False
            return False
        if self._activity not in (Activity.SITTING, Activity.LYING):
            return False
        recent = self._window(t, 8.0)
        baseline = max(s.shoulder_y for s in recent)
        torso = float(np.median([s.torso for s in recent]))
        rise = (baseline - recent[-1].shoulder_y) / torso
        if not self._rising and rise >= self.th.rise_attempt:
            self._rising = True
        elif self._rising and rise <= 0.1:
            self._rising = False
            self._attempts.append(t)
        while self._attempts and t - self._attempts[0] > 60.0:
            self._attempts.popleft()
        if len(self._attempts) >= self.th.attempts_for_struggle:
            self._attempts.clear()
            return True
        return False
