"""Connects the camera, safety logic, voice and alerts."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .activity import Activity, ActivityTracker, Observation, find_held_object
from .alerts import Gateway, call_emergency_services, escalate, notify
from .assistant import Assistant, ChatClient, Confirmation, PlaceMemory, item_name
from .config import Config
from .safety import CANCEL_ON_RECOVERY, EMERGENCY_REASONS, Decision, Phase, Reason, Reply, SafetyMonitor
from .vision import VisionPipeline, draw_overlay
from .voice import PROMPTS, Listener, Voice, microphone_blocks, prompt_key

log = logging.getLogger(__name__)
ANSWERED = (Reply.OK, Reply.HELP, Reply.EMERGENCY)
MEDICINE_REMINDER_WINDOW_MIN = 30


class EventLog:
    """Keeps a text-only history. No images or audio are ever recorded."""

    def __init__(self, path: Path | None):
        self.path = path
        self.recent: deque[dict] = deque(maxlen=30)
        self._lock = threading.Lock()

    def write(self, event: str, **details) -> None:
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **details}
        with self._lock:
            self.recent.appendleft(entry)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("%s %s", event, details or "")


class App:
    def __init__(
        self,
        config: Config,
        voice: Voice,
        gateway: Gateway,
        events: EventLog,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] = time.sleep,
        memory: PlaceMemory | None = None,
        wall_clock: Callable[[], datetime] = datetime.now,
    ):
        self.cfg = config
        self.voice = voice
        self.gateway = gateway
        self.events = events
        self.clock = clock
        self.wait = wait
        self.tracker = ActivityTracker(bed_zones=config.bed_zones)
        self.monitor = SafetyMonitor(
            config.safety, lambda: config.in_sleep_hours(datetime.now().hour), start=clock()
        )
        self.observation = Observation(Activity.ABSENT)
        self.latest_frame: np.ndarray | None = None
        self.stream_enabled = False
        self.camera_ok = True
        self.mic_paused = threading.Event()
        self.stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_activity: Activity | None = None
        self._stand_down = threading.Event()
        self.wall_clock = wall_clock
        self.memory = memory or PlaceMemory(None, config.places)
        # Only one thing can use the speaker and mic at a time, and safety checks always win.
        self._voice_lock = threading.Lock()
        self._preempt = threading.Event()
        self._audio_users = 0
        self._audio_lock = threading.Lock()
        self._reminders_done: set[str] = set()
        chat = ChatClient(config.chat, config.person_name) if config.chat.enabled else None
        self.assistant = Assistant(
            voice,
            self.memory,
            config.contacts,
            config.person_name,
            self._send_text,
            gateway.simulated,
            self._emergency_requested,
            self._family_alert_requested,
            lambda: self.monitor.request_check(Reason.HELP_WORD),
            chat,
            events.write,
            wall_clock,
        )

    def state(self) -> dict:
        return {
            "activity": self.observation.activity.value,
            "object": self.observation.held_object,
            "phase": self.monitor.phase.value,
            "reason": self.monitor.reason.value if self.monitor.reason else None,
            "camera_ok": self.camera_ok,
            "alerts_live": not self.gateway.simulated,
            "events": list(self.events.recent),
        }

    def on_frame(self, points: np.ndarray | None, detections: list, frame_size: tuple[int, int], now: float) -> Observation:
        held = find_held_object(points, detections) if points is not None else None
        obs = self.tracker.update(points, now, frame_size, held)
        self.observation = obs
        if obs.activity is not self._last_activity:
            self.events.write("activity", activity=obs.activity.value)
            self._last_activity = obs.activity
        reason = self.monitor.update(obs, now)
        if reason is not None:
            self.begin_checkin(reason)
        else:
            self.check_medicine_reminder(obs)
        return obs

    def _hold_audio(self) -> None:
        with self._audio_lock:
            self._audio_users += 1
            self.mic_paused.set()

    def _release_audio(self) -> None:
        with self._audio_lock:
            self._audio_users -= 1
            if self._audio_users <= 0:
                self.mic_paused.clear()

    def _send_text(self, phone: str, body: str) -> bool:
        try:
            self.gateway.send_sms(phone, body)
            return True
        except Exception as error:
            log.error("Text failed: %s", error)
            return False

    def start_conversation(self, first_text: str = "") -> bool:
        """Starts talking unless something else has the voice. Safety checks can interrupt it."""
        if self.monitor.phase is not Phase.WATCHING or not self._voice_lock.acquire(blocking=False):
            return False
        self._hold_audio()
        self.monitor.note_conversation(self.clock())
        self._worker = threading.Thread(target=self._conversation, args=(first_text,), daemon=True)
        self._worker.start()
        return True

    def _conversation(self, first_text: str) -> None:
        try:
            self.events.write("conversation_started")
            self.assistant.converse(self._preempt.is_set, first_text or None)
        except Exception:
            log.exception("Conversation failed")
            self.events.write("conversation_error")
        finally:
            self.monitor.note_conversation(self.clock())
            self._voice_lock.release()
            self._release_audio()

    def _say_in(self, language: str, key: str) -> None:
        text = PROMPTS[key][language].format(name=self.cfg.person_name, emergency=self.cfg.alerts.emergency_number)
        try:
            self.voice.say_text(language, text)
        except Exception:
            log.exception("Could not speak %s", key)

    def _emergency_requested(self, language: str) -> None:
        self._emergency_flow(Reason.HELP_WORD, language)

    def _family_alert_requested(self, language: str) -> None:
        if self.monitor.begin_escalation(Reason.HELP_WORD):
            self.events.write("family_alert_requested")
            self._escalate(Reason.HELP_WORD)

    def _emergency_flow(self, reason: Reason, language: str = "hi") -> None:
        """They asked for 112. One spoken confirmation, because a false call ties up a real line."""
        answer = Confirmation.UNCLEAR
        for _ in range(2):
            self._say_in(language, "confirm_emergency")
            answer = self.voice.confirm(self.cfg.voice.listen_s)
            if answer is not Confirmation.UNCLEAR:
                break
        self.events.write("emergency_call_confirmation", answer=answer.value)
        if answer is Confirmation.NO:
            self.assistant.say_line("cancelled", language)
            self.monitor.finish_checkin(Reply.OK, self.clock())
            return
        self.monitor.begin_escalation(reason)
        if answer is Confirmation.UNCLEAR:
            # They asked but can't confirm, so a person gets to make the call.
            self._escalate(reason, note=f"They asked for {self.cfg.alerts.emergency_number} but could not confirm. It has NOT been called.")
            return
        call = call_emergency_services(self.cfg.person_name, self.cfg.address, self.cfg.alerts, self.gateway)
        self.events.write("emergency_call", connected=call.connected, error=call.error, simulated=call.simulated)
        if call.simulated:
            self._say_in(language, "emergency_practice")
        else:
            self._say_in(language, "emergency_connected" if call.connected else "emergency_failed")
        number = self.cfg.alerts.emergency_number
        status = f"They asked for {number}, and the call connected." if call.connected else f"They asked for {number}, but the call FAILED."
        if call.simulated:
            status = f"They asked for {number}. Practice mode: it was not really called."
        self._escalate(reason, note=status)

    def check_medicine_reminder(self, obs: Observation) -> None:
        """Says when it is time and where the medicine is kept. It never mentions doses."""
        if not self.cfg.medicine_times or obs.activity is Activity.ABSENT:
            return
        now = self.wall_clock()
        for slot in self.cfg.medicine_times:
            hour, minute = map(int, slot.split(":"))
            due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            key = f"{now.date()} {slot}"
            late_minutes = (now - due).total_seconds() / 60
            if key in self._reminders_done or not 0 <= late_minutes <= MEDICINE_REMINDER_WINDOW_MIN:
                continue
            if self.monitor.phase is not Phase.WATCHING or not self._voice_lock.acquire(blocking=False):
                return
            self._reminders_done.add(key)
            self._hold_audio()
            self._worker = threading.Thread(target=self._medicine_reminder, args=(slot,), daemon=True)
            self._worker.start()
            return

    def _medicine_reminder(self, slot: str) -> None:
        try:
            self.voice.say("medicine_time")
            for language in self.cfg.languages:
                found = self.memory.recall("medicine", language)
                if found is not None and found[1] == language:
                    self.assistant.say_line("place_known", language, item=item_name("medicine", language), place=found[0])
            self.events.write("medicine_reminder", time=slot)
        except Exception:
            log.exception("Medicine reminder failed")
            self.events.write("medicine_reminder_failed", time=slot)
        finally:
            self._voice_lock.release()
            self._release_audio()

    def stand_down(self) -> None:
        """A person pressed "I'm okay" on the dashboard."""
        self._stand_down.set()

    def begin_checkin(self, reason: Reason) -> None:
        self.events.write("check_in_started", reason=reason.value)
        self._preempt.set()
        self._hold_audio()
        self._worker = threading.Thread(target=self._checkin, args=(reason,), daemon=True)
        self._worker.start()

    def wait_for_worker(self, timeout: float | None = None) -> None:
        if self._worker is not None:
            self._worker.join(timeout)

    def _checkin(self, reason: Reason) -> None:
        self._voice_lock.acquire()
        self._preempt.clear()
        try:
            self._run_checkin(reason)
        except Exception:
            # A broken speaker or mic must not hide a possible emergency.
            log.exception("Check-in failed, alerting contacts")
            self.events.write("check_in_error", reason=reason.value)
            self.monitor.phase = Phase.ESCALATING
            self._escalate(reason)
        finally:
            self._stand_down.clear()
            self.mic_paused.clear()

    def _run_checkin(self, reason: Reason) -> None:
        self._stand_down.clear()

        def recovered() -> bool:
            return self._stand_down.is_set() or (reason in CANCEL_ON_RECOVERY and self.monitor.recovered(self.clock()))

        reply = Reply.SILENCE
        for attempt in range(self.cfg.voice.ask_attempts):
            if recovered():
                break
            self.voice.say(prompt_key(reason, attempt))
            if recovered():
                break
            heard = self.voice.listen(self.cfg.voice.listen_s, recovered)
            self.events.write("reply_heard", attempt=attempt + 1, reply=heard.value)
            reply = heard if heard is not Reply.SILENCE or reply is not Reply.UNCLEAR else reply
            if reply in ANSWERED:
                break

        if reply not in ANSWERED and reason in EMERGENCY_REASONS and not recovered():
            self.voice.say("final")
            if not recovered():
                last = self.voice.listen(self.cfg.voice.listen_s / 2, recovered)
                self.events.write("reply_heard", attempt="final", reply=last.value)
                if last in ANSWERED:
                    reply = last

        if self._stand_down.is_set():
            self.events.write("check_in_cleared", how="dashboard")
            self.monitor.finish_checkin(Reply.OK, self.clock())
            return
        if reply not in ANSWERED and reason in CANCEL_ON_RECOVERY and self.monitor.recovered(self.clock()):
            self.events.write("check_in_cleared", how="seen_upright")
            self.monitor.cancel_checkin(self.clock())
            return

        if reply is Reply.EMERGENCY:
            self.events.write("check_in_finished", reason=reason.value, reply=reply.value, decision="confirm_emergency_call")
            self._emergency_flow(reason)
            return

        decision = self.monitor.finish_checkin(reply, self.clock())
        self.events.write("check_in_finished", reason=reason.value, reply=reply.value, decision=decision.value)
        if decision is Decision.STAND_DOWN:
            self.voice.say("thanks")
        elif decision is Decision.NOTIFY:
            errors = notify(self.cfg.person_name, reason, self.cfg.contacts, self.gateway)
            self.events.write("contacts_notified", simulated=self.gateway.simulated, errors=errors)
        else:
            self._escalate(reason)

    def _escalate(self, reason: Reason, note: str = "") -> None:
        reached = False
        try:
            self._say_safely("calling")
            result = escalate(self.cfg.person_name, reason, self.cfg.contacts, self.gateway, self.cfg.alerts, self.wait, note)
            reached = result.reached is not None
            self.events.write(
                "escalation_finished",
                reason=reason.value,
                reached=result.reached.name if result.reached else None,
                texted=result.texted,
                errors=result.errors,
                simulated=result.simulated,
            )
            self._say_safely("reached" if reached else "unreached")
        finally:
            self.monitor.finish_escalation(reached, self.clock())

    def _say_safely(self, key: str) -> None:
        try:
            self.voice.say(key)
        except Exception:
            log.exception("Could not speak %s", key)

    def report_camera_fault(self) -> None:
        self.camera_ok = False
        self.events.write("camera_fault")
        errors: list[str] = []
        for contact in self.cfg.contacts:
            try:
                self.gateway.send_sms(contact.phone, f"Elder-Help: the camera watching {self.cfg.person_name} has stopped working. They are not being monitored.")
            except Exception as error:
                errors.append(str(error))
                log.error("Camera fault SMS to %s failed: %s", contact.name, error)
        self.events.write("camera_fault_notified", simulated=self.gateway.simulated, errors=errors)
        self._say_safely("camera_fault")

    def listen_for_triggers(self, listener: Listener) -> None:
        """Waits for a help word or the assistant's name whenever the mic is free."""
        while not self.stop.is_set():
            if self.mic_paused.is_set():
                self.wait(0.5)
                continue
            try:
                trigger = listener.wait_for_trigger(microphone_blocks(None, self.mic_paused))
            except Exception:
                log.exception("Microphone listening failed, retrying in 30 seconds")
                self.events.write("microphone_fault")
                self.wait(30)
                continue
            if trigger is None or self.mic_paused.is_set():
                continue
            kind, rest = trigger
            if kind == "help":
                self.events.write("help_word_heard")
                self.monitor.request_check(Reason.HELP_WORD)
                self.wait(1)
            else:
                self.events.write("assistant_called")
                self.start_conversation(rest)

    def run(self, source: int | str, vision: VisionPipeline) -> None:
        is_file = isinstance(source, str)
        capture = cv2.VideoCapture(source)
        frame_gap = 0.0
        if is_file:
            fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
            frame_gap = 1.0 / fps
        last_good = self.clock()
        fault_sent = False
        try:
            while not self.stop.is_set():
                started = self.clock()
                ok, frame = capture.read()
                if not ok:
                    if is_file:
                        break
                    if not fault_sent and started - last_good >= self.cfg.camera_fault_s:
                        fault_sent = True
                        threading.Thread(target=self.report_camera_fault, daemon=True).start()
                    self.wait(1)
                    capture.release()
                    capture = cv2.VideoCapture(source)
                    continue
                last_good = started
                if fault_sent:
                    fault_sent = False
                    self.events.write("camera_recovered")
                self.camera_ok = True
                points, detections = vision.process(frame, started)
                height, width = frame.shape[:2]
                self.on_frame(points, detections, (width, height), started)
                if self.stream_enabled:
                    self.latest_frame = draw_overlay(frame, points)
                if frame_gap:
                    self.wait(max(0.0, frame_gap - (self.clock() - started)))
        finally:
            capture.release()
