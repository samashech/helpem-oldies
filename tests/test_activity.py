import numpy as np

from elder_help.activity import Activity, ActivityTracker, BedZone, classify_posture, find_held_object


def body(joints, visibility=0.95):
    points = np.zeros((33, 3))
    for index, (x, y) in joints.items():
        points[index] = (x, y, visibility)
    return points


def standing(x=320.0, y=0.0):
    return body({
        0: (x, 80 + y), 11: (x - 40, 130 + y), 12: (x + 40, 130 + y),
        15: (x - 50, 260 + y), 16: (x + 50, 260 + y),
        23: (x - 25, 280 + y), 24: (x + 25, 280 + y),
        25: (x - 25, 370 + y), 26: (x + 25, 370 + y),
        27: (x - 25, 460 + y), 28: (x + 25, 460 + y),
    })


def lying():
    return body({
        0: (90, 400), 11: (150, 370), 12: (150, 430),
        23: (300, 375), 24: (300, 425),
        25: (400, 380), 26: (400, 420), 27: (500, 385), 28: (500, 415),
    })


def bending():
    # Torso is flat from reaching for the floor, but the legs are straight beneath the hips.
    return body({
        0: (120, 300), 11: (170, 280), 12: (170, 320),
        23: (320, 270), 24: (320, 310),
        25: (320, 370), 26: (320, 400), 27: (320, 470), 28: (320, 490),
    })


def sitting():
    return body({
        0: (320, 100), 11: (280, 150), 12: (360, 150),
        23: (295, 300), 24: (345, 300),
        25: (200, 310), 26: (250, 310), 27: (200, 420), 28: (250, 420),
    })


def desk_view_upper_body():
    points = body({0: (320, 150), 11: (240, 300), 12: (400, 300)})
    return points


def desk_view_lying():
    return body({0: (150, 240), 11: (300, 160), 12: (300, 320)})


def test_static_postures():
    assert classify_posture(standing())[0] is Activity.STANDING
    assert classify_posture(lying())[0] is Activity.LYING
    assert classify_posture(sitting())[0] is Activity.SITTING


def test_bending_over_is_not_lying():
    assert classify_posture(bending())[0] is Activity.STANDING


def test_fallen_person_with_a_sprawled_leg_is_lying():
    points = lying()
    points[27] = (400, 700, 0.95)  # One foot far below the hips, the other level with them.
    assert classify_posture(points)[0] is Activity.LYING


def test_upper_body_only_camera():
    assert classify_posture(desk_view_upper_body())[0] is Activity.UPRIGHT
    assert classify_posture(desk_view_lying())[0] is Activity.LYING


def test_low_visibility_counts_as_absent():
    assert classify_posture(standing() * np.array([1, 1, 0.1]))[0] is Activity.ABSENT


def run_frames(tracker, frames, start=0.0, fps=15):
    obs = None
    for i, frame in enumerate(frames):
        obs = tracker.update(frame, start + i / fps)
    return obs


def test_walking_and_running_by_body_speed():
    # The torso is 150 px tall here, so 150 px/s is one torso length per second.
    walker = ActivityTracker()
    obs = run_frames(walker, [standing(x=100 + i * 10) for i in range(30)])
    assert obs.activity is Activity.WALKING
    runner = ActivityTracker()
    obs = run_frames(runner, [standing(x=100 + i * 30) for i in range(30)])
    assert obs.activity is Activity.RUNNING
    still = ActivityTracker()
    assert run_frames(still, [standing() for _ in range(30)]).activity is Activity.STANDING


def test_sudden_drop_is_flagged_as_possible_fall():
    tracker = ActivityTracker()
    run_frames(tracker, [standing() for _ in range(30)])
    obs = None
    for i, frame in enumerate([standing(y=40 * k) for k in range(1, 7)] + [lying()] * 15):
        obs = tracker.update(frame, 2.0 + i / 15)
    assert obs.activity is Activity.LYING
    assert obs.fall_at is not None


def test_sudden_drop_from_a_chair_is_flagged():
    tracker = ActivityTracker()
    run_frames(tracker, [sitting() for _ in range(30)])
    obs = None
    for i, frame in enumerate([lying()] * 15):
        obs = tracker.update(frame, 2.0 + i / 15)
    assert obs.activity is Activity.LYING
    assert obs.fall_at is not None


def test_slow_lying_down_is_not_a_fall_but_is_lying():
    tracker = ActivityTracker()
    frames = [standing() for _ in range(30)] + [standing(y=3 * k) for k in range(60)] + [lying()] * 30
    obs = run_frames(tracker, frames)
    assert obs.activity is Activity.LYING
    assert obs.fall_at is None


def test_one_bad_detection_is_not_a_fall():
    tracker = ActivityTracker()
    frames = [standing() for _ in range(30)] + [standing(y=300)] + [standing() for _ in range(20)]
    assert run_frames(tracker, frames).fall_at is None


def test_person_leaving_frame_becomes_absent_only_after_a_delay():
    tracker = ActivityTracker()
    run_frames(tracker, [standing() for _ in range(15)])
    assert tracker.update(None, 2.0).activity is Activity.STANDING
    assert tracker.update(None, 5.0).activity is Activity.ABSENT


def test_repeated_failed_attempts_to_rise_are_a_struggle():
    tracker = ActivityTracker()
    results = []
    t = 0.0

    def sit_shifted(lift):
        points = sitting()
        for index in (0, 11, 12):
            points[index, 1] -= lift
        return points

    for _ in range(4):
        for lift in [0] * 15 + [20, 40, 60, 75, 75, 60, 40, 20] + [0] * 15:
            results.append(tracker.update(sit_shifted(lift), t).struggling)
            t += 1 / 15
    assert any(results)


def test_sitting_still_is_not_a_struggle():
    tracker = ActivityTracker()
    assert not any(tracker.update(sitting(), i / 15).struggling for i in range(300))


def test_bed_zone_uses_hip_position():
    tracker = ActivityTracker(bed_zones=(BedZone(0.3, 0.5, 1.0, 1.0),))
    assert tracker.update(lying(), 0.0, (640, 480)).in_bed
    tracker = ActivityTracker(bed_zones=(BedZone(0.8, 0.0, 1.0, 0.2),))
    assert not tracker.update(lying(), 0.0, (640, 480)).in_bed


def test_held_object_requires_a_wrist_near_it():
    points = standing()
    assert find_held_object(points, [("cup", (250, 240, 290, 280))]) == "cup"
    assert find_held_object(points, [("cup", (560, 20, 600, 60))]) is None
