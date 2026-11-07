from elder_help.activity import Activity, Observation
from elder_help.safety import Decision, Phase, Reason, Reply, SafetyConfig, SafetyMonitor

CFG = SafetyConfig(lying_grace_s=20, fall_grace_s=5, fall_absent_grace_s=15, bed_grace_s=1800, recheck_s=600, wellness_interval_s=10000, recovered_s=5)
LYING = Observation(Activity.LYING)
WALKING = Observation(Activity.WALKING)


def feed(monitor, obs, start, end, step=1.0):
    t = start
    while t <= end:
        reason = monitor.update(obs, t)
        if reason is not None:
            return reason, t
        t += step
    return None, end


def test_lying_on_the_floor_triggers_check_after_grace():
    reason, t = feed(SafetyMonitor(CFG), LYING, 0, 60)
    assert reason is Reason.LYING and t == 20


def test_silence_after_lying_escalates_and_unclear_speech_is_not_okay():
    for reply in (Reply.SILENCE, Reply.UNCLEAR, Reply.HELP):
        monitor = SafetyMonitor(CFG)
        feed(monitor, LYING, 0, 60)
        assert monitor.finish_checkin(reply, 21) is Decision.ESCALATE
        assert monitor.phase is Phase.ESCALATING


def test_okay_stands_down_then_asks_again_while_still_lying():
    monitor = SafetyMonitor(CFG)
    feed(monitor, LYING, 0, 60)
    assert monitor.finish_checkin(Reply.OK, 21) is Decision.STAND_DOWN
    assert feed(monitor, LYING, 22, 620)[0] is None
    reason, t = feed(monitor, LYING, 621, 700)
    assert reason is Reason.LYING and t == 621


def test_new_fall_ignores_cooldown_from_earlier_okay():
    monitor = SafetyMonitor(CFG)
    feed(monitor, LYING, 0, 60)
    monitor.finish_checkin(Reply.OK, 21)
    reason, _ = feed(monitor, Observation(Activity.LYING, fall_at=30.0), 30, 40)
    assert reason is Reason.FALL


def test_getting_up_clears_cooldown_so_a_slow_collapse_is_not_delayed():
    monitor = SafetyMonitor(CFG)
    feed(monitor, LYING, 0, 60)
    monitor.finish_checkin(Reply.OK, 21)
    feed(monitor, WALKING, 22, 40)
    reason, t = feed(monitor, LYING, 41, 100)
    assert reason is Reason.LYING and t == 61


def test_fall_uses_short_grace_and_is_only_handled_once():
    monitor = SafetyMonitor(CFG)
    obs = Observation(Activity.LYING, fall_at=0.0)
    reason, t = feed(monitor, obs, 0, 30)
    assert reason is Reason.FALL and t == 5
    monitor.finish_checkin(Reply.OK, 6)
    assert feed(monitor, obs, 7, 300)[0] is None


def test_fall_then_out_of_view_still_gets_a_check():
    monitor = SafetyMonitor(CFG)
    reason, t = feed(monitor, Observation(Activity.ABSENT, fall_at=0.0), 0, 30)
    assert reason is Reason.FALL_OUT_OF_VIEW and t == 15


def test_person_vanishing_mid_lying_keeps_the_timer_running():
    monitor = SafetyMonitor(CFG)
    feed(monitor, LYING, 0, 10)
    reason, t = feed(monitor, Observation(Activity.ABSENT), 11, 30)
    assert reason is Reason.LYING and t == 20


def test_brief_misdetection_does_not_reset_the_lying_timer():
    monitor = SafetyMonitor(CFG)
    feed(monitor, LYING, 0, 10)
    feed(monitor, Observation(Activity.SITTING), 11, 12)
    reason, t = feed(monitor, LYING, 13, 30)
    assert reason is Reason.LYING and t == 20


def test_getting_up_cancels_the_timer():
    monitor = SafetyMonitor(CFG)
    feed(monitor, LYING, 0, 10)
    feed(monitor, WALKING, 11, 17)
    assert feed(monitor, WALKING, 18, 60)[0] is None


def test_bed_at_night_never_prompts_without_a_fall():
    monitor = SafetyMonitor(CFG, is_sleep_time=lambda: True)
    assert feed(monitor, Observation(Activity.LYING, in_bed=True), 0, 28800, 60)[0] is None


def test_daytime_bed_rest_gets_long_grace():
    monitor = SafetyMonitor(CFG)
    reason, t = feed(monitor, Observation(Activity.LYING, in_bed=True), 0, 4000, 10)
    assert reason is Reason.LYING and t == 1800


def test_fall_in_bed_zone_at_night_still_checks():
    monitor = SafetyMonitor(CFG, is_sleep_time=lambda: True)
    reason, _ = feed(monitor, Observation(Activity.LYING, in_bed=True, fall_at=0.0), 0, 30)
    assert reason is Reason.FALL


def test_silence_on_routine_check_only_notifies():
    monitor = SafetyMonitor(CFG)
    reason, t = feed(monitor, Observation(Activity.SITTING), 0, 20000, 100)
    assert reason is Reason.WELLNESS and t == 10000
    assert monitor.finish_checkin(Reply.SILENCE, t) is Decision.NOTIFY


def test_no_routine_checks_during_sleep_hours_or_when_nobody_is_there():
    assert feed(SafetyMonitor(CFG, is_sleep_time=lambda: True), Observation(Activity.SITTING), 0, 20000, 100)[0] is None
    assert feed(SafetyMonitor(CFG), Observation(Activity.ABSENT), 0, 20000, 100)[0] is None


def test_struggle_silence_notifies_but_saying_help_escalates():
    monitor = SafetyMonitor(CFG)
    assert monitor.update(Observation(Activity.SITTING, struggling=True), 1) is Reason.STRUGGLE
    assert monitor.finish_checkin(Reply.SILENCE, 2) is Decision.NOTIFY
    monitor = SafetyMonitor(CFG)
    monitor.update(Observation(Activity.SITTING, struggling=True), 1)
    assert monitor.finish_checkin(Reply.HELP, 2) is Decision.ESCALATE


def test_help_word_ignores_cooldown_and_silence_escalates():
    monitor = SafetyMonitor(CFG)
    feed(monitor, LYING, 0, 60)
    monitor.finish_checkin(Reply.OK, 21)
    monitor.request_check(Reason.HELP_WORD)
    assert monitor.update(LYING, 22) is Reason.HELP_WORD
    assert monitor.finish_checkin(Reply.SILENCE, 23) is Decision.ESCALATE


def test_retries_after_nobody_was_reached_sooner_than_after_someone_was():
    monitor = SafetyMonitor(CFG)
    feed(monitor, LYING, 0, 60)
    monitor.finish_checkin(Reply.SILENCE, 21)
    monitor.finish_escalation(False, 100)
    assert feed(monitor, LYING, 101, 219)[0] is None
    assert feed(monitor, LYING, 220, 230)[0] is Reason.LYING
    monitor.finish_checkin(Reply.SILENCE, 231)
    monitor.finish_escalation(True, 300)
    assert feed(monitor, LYING, 301, 1199)[0] is None


def test_coming_back_into_view_cancels_a_possible_fall():
    monitor = SafetyMonitor(CFG)
    assert feed(monitor, Observation(Activity.ABSENT, fall_at=0.0), 0, 10)[0] is None
    assert monitor.update(Observation(Activity.SITTING, came_back=True), 11) is None
    assert feed(monitor, Observation(Activity.SITTING), 12, 100)[0] is None
