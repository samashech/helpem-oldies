"""Decides when to check on the person and what their answer, or silence, means."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .activity import UPRIGHT_ACTIVITIES, Activity, Observation


class Reason(str, Enum):
    LYING = "lying_down"
    FALL = "possible_fall"
    FALL_OUT_OF_VIEW = "possible_fall_out_of_view"
    STRUGGLE = "struggling_to_get_up"
    HELP_WORD = "asked_for_help"
    WELLNESS = "routine_check"
    MANUAL = "manual_check"


class Reply(str, Enum):
    OK = "ok"
    HELP = "help"
    EMERGENCY = "emergency"  # Asked for 112 or an ambulance. Always confirmed before dialling.
    UNCLEAR = "unclear"  # Speech we could not understand. Never treated as "okay".
    SILENCE = "silence"


class Phase(str, Enum):
    WATCHING = "watching"
    CHECKING = "checking"
    ESCALATING = "escalating"


class Decision(str, Enum):
    STAND_DOWN = "stand_down"
    NOTIFY = "notify"
    ESCALATE = "escalate"


# Silence after any of these can mean an emergency, so it leads to phone calls.
# Silence after a routine check usually means they did not hear, so contacts get a text.
EMERGENCY_REASONS = {Reason.LYING, Reason.FALL, Reason.FALL_OUT_OF_VIEW, Reason.HELP_WORD, Reason.MANUAL}
# A request for help stands even if the person is seen upright afterwards.
CANCEL_ON_RECOVERY = {Reason.LYING, Reason.FALL, Reason.FALL_OUT_OF_VIEW, Reason.WELLNESS}


@dataclass(frozen=True)
class SafetyConfig:
    lying_grace_s: float = 20
    fall_grace_s: float = 5
    fall_absent_grace_s: float = 15
    bed_grace_s: float = 1800
    recheck_s: float = 600
    wellness_interval_s: float = 10800
    recovered_s: float = 5
    reached_cooldown_s: float = 900
    unreached_cooldown_s: float = 120


class SafetyMonitor:
    def __init__(self, config: SafetyConfig, is_sleep_time: Callable[[], bool] = lambda: False, start: float = 0.0):
        self.cfg = config
        self.is_sleep_time = is_sleep_time
        self.phase = Phase.WATCHING
        self.reason: Reason | None = None
        self._down_since: float | None = None
        self._pending_fall: float | None = None
        self._handled_fall: float | None = None
        self._upright_since: float | None = None
        self._not_down_since: float | None = None
        self._cooldown_until = 0.0
        self._last_contact = start
        self._requested: Reason | None = None

    def begin_escalation(self, reason: Reason) -> bool:
        """Starts alerting because the person asked for it. Returns False if already alerting."""
        if self.phase is Phase.ESCALATING:
            return False
        self.phase = Phase.ESCALATING
        self.reason = reason
        return True

    def note_conversation(self, now: float) -> None:
        """A chat shows they are responsive, so the routine check-in can wait."""
        self._last_contact = now

    def request_check(self, reason: Reason) -> None:
        self._requested = reason

    def recovered(self, now: float) -> bool:
        return self._upright_since is not None and now - self._upright_since >= self.cfg.recovered_s

    def update(self, obs: Observation, now: float) -> Reason | None:
        """Takes the latest observation and returns a reason when a check-in should start."""
        self._track(obs, now)
        if self.phase is not Phase.WATCHING:
            return None
        # New emergencies come first and ignore the cooldown from an earlier "I'm okay".
        if self._requested is not None:
            reason, self._requested = self._requested, None
            return self._begin(reason, now)
        if self._pending_fall is not None:
            absent = obs.activity is Activity.ABSENT
            grace = self.cfg.fall_absent_grace_s if absent else self.cfg.fall_grace_s
            if now - self._pending_fall >= grace:
                self._handled_fall, self._pending_fall = self._pending_fall, None
                return self._begin(Reason.FALL_OUT_OF_VIEW if absent else Reason.FALL, now)
            return None
        if obs.struggling:
            return self._begin(Reason.STRUGGLE, now)
        if now < self._cooldown_until:
            return None
        if self._down_since is not None and obs.activity in (Activity.LYING, Activity.ABSENT):
            grace = self._down_grace(obs)
            if grace is not None and now - self._down_since >= grace:
                return self._begin(Reason.LYING, now)
            return None
        if (
            obs.activity is not Activity.ABSENT
            and not self.is_sleep_time()
            and now - self._last_contact >= self.cfg.wellness_interval_s
        ):
            return self._begin(Reason.WELLNESS, now)
        return None

    def finish_checkin(self, reply: Reply, now: float) -> Decision:
        reason = self.reason
        self._last_contact = now
        if reply is Reply.OK:
            self._rest(now, self.cfg.recheck_s)
            return Decision.STAND_DOWN
        if reply in (Reply.HELP, Reply.EMERGENCY) or reason in EMERGENCY_REASONS:
            self.phase = Phase.ESCALATING
            return Decision.ESCALATE
        self._rest(now, self.cfg.recheck_s)
        return Decision.NOTIFY

    def cancel_checkin(self, now: float) -> None:
        """The person was seen back on their feet during the check-in."""
        self._last_contact = now
        self._rest(now, 0)

    def finish_escalation(self, reached: bool, now: float) -> None:
        self._rest(now, self.cfg.reached_cooldown_s if reached else self.cfg.unreached_cooldown_s)

    def _track(self, obs: Observation, now: float) -> None:
        if obs.fall_at is not None and obs.fall_at not in (self._pending_fall, self._handled_fall):
            self._pending_fall = obs.fall_at
            self._down_since = self._down_since if self._down_since is not None else obs.fall_at
        if obs.activity is Activity.LYING and self._down_since is None:
            self._down_since = now

        if obs.activity in UPRIGHT_ACTIVITIES:
            self._upright_since = self._upright_since if self._upright_since is not None else now
            if self.recovered(now):
                self._down_since = None
                self._pending_fall = None
                self._cooldown_until = 0.0  # They are fine again, so a new fall must not wait.
        else:
            self._upright_since = None

        if obs.activity in (Activity.LYING, Activity.ABSENT):
            self._not_down_since = None
        else:
            self._not_down_since = self._not_down_since if self._not_down_since is not None else now
            if now - self._not_down_since >= self.cfg.recovered_s:
                self._down_since = None

    def _down_grace(self, obs: Observation) -> float | None:
        if not obs.in_bed:
            return self.cfg.lying_grace_s
        return None if self.is_sleep_time() else self.cfg.bed_grace_s

    def _begin(self, reason: Reason, now: float) -> Reason:
        self.phase = Phase.CHECKING
        self.reason = reason
        self._last_contact = now
        return reason

    def _rest(self, now: float, seconds: float) -> None:
        self.phase = Phase.WATCHING
        self.reason = None
        self._cooldown_until = now + seconds
