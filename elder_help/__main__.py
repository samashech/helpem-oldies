"""Command line: python -m elder_help run | doctor | say | hear"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .activity import Activity, Observation
from .alerts import make_gateway
from .app import App, EventLog
from .assistant import Confirmation, PlaceMemory, Transcript, classify_confirmation
from .config import Config, load_config
from .safety import Reply
from .voice import MODELS, PIPER_VOICES, VOSK_MODELS, LocalVoice, Speaker, microphone_blocks, play_audio

log = logging.getLogger("elder_help")


class ScriptedVoice:
    """Stands in for the speaker and mic when replaying a video."""

    def __init__(self, reply: Reply, listen_s: float | None = None):
        self.reply = reply
        self.listen_s = listen_s

    def say(self, key: str, **_values: str) -> None:
        log.info("Would say: %s", key)

    def listen(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> Reply:
        deadline = time.monotonic() + (self.listen_s if self.listen_s is not None else seconds)
        while time.monotonic() < deadline and not should_abort():
            time.sleep(0.05)
        return self.reply

    def say_text(self, language: str, text: str) -> None:
        log.info("Would say (%s): %s", language, text)

    def hear(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> list:
        return []

    def confirm(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> Confirmation:
        return Confirmation.UNCLEAR


class TextVoice:
    """Lets you try the assistant by typing, with no speaker or mic."""

    def say(self, key: str, **_values: str) -> None:
        print(f"[{key}]")

    def say_text(self, language: str, text: str) -> None:
        print(f"assistant ({language}): {text}")

    def hear(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> list:
        line = input("you: ").strip()
        language = "hi" if any("\u0900" <= c <= "\u097f" for c in line) else "en"
        return [Transcript(language, line, 1.0)] if line else []

    def confirm(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> Confirmation:
        return classify_confirmation([(word, 1.0) for word in input("you (confirm): ").split()])

    def listen(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> Reply:
        return Reply.SILENCE


class ShowHeard:
    """Prints what each speech model heard, for troubleshooting. Nothing is saved."""

    def __init__(self, voice: LocalVoice):
        self.voice = voice

    def __getattr__(self, name: str):
        return getattr(self.voice, name)

    def hear(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> list:
        transcripts = self.voice.hear(seconds, should_abort)
        for transcript in transcripts:
            print(f"heard ({transcript.language}, confidence {transcript.confidence:.2f}): {transcript.text}")
        if not transcripts:
            print("heard: nothing")
        return transcripts


def _talk(args: argparse.Namespace, config: Config) -> int:
    voice = TextVoice() if args.text else LocalVoice(config.languages, config.person_name, config.alerts.emergency_number)
    if args.show_heard and not args.text:
        voice = ShowHeard(voice)
    memory = PlaceMemory(Path(args.memory), config.places)
    app = App(config, voice, make_gateway(config.alerts), EventLog(None), memory=memory, wait=lambda _s: None)
    if not app.start_conversation():
        return 1
    app.wait_for_worker()
    # Without a camera loop, nothing else would pick up a check-in the conversation asked for.
    reason = app.monitor.update(Observation(Activity.UPRIGHT), app.clock())
    if reason is not None:
        app.begin_checkin(reason)
        app.wait_for_worker()
    return 0


def _run(args: argparse.Namespace, config: Config) -> int:
    from .dashboard import serve
    from .vision import VisionPipeline, prefer_nvidia_gpu

    if config.prefer_nvidia and prefer_nvidia_gpu():
        log.info("Pose detection will use the NVIDIA GPU")
    live_camera = args.video is None
    if live_camera and not config.monitored_person_consented:
        print(
            "Refusing to start the camera: set monitored_person_consented = true in the config\n"
            "after the person being monitored has agreed to this.",
            file=sys.stderr,
        )
        return 2
    if config.alerts.live and not config.contacts:
        print("alerts.live is true but no contacts are configured.", file=sys.stderr)
        return 2
    gateway = make_gateway(config.alerts)
    if gateway.simulated:
        log.warning("PRACTICE MODE: alerts.live is false, so nobody will really be called or texted.")

    voice = ScriptedVoice(Reply(args.reply), args.listen_s) if args.no_audio else LocalVoice(
        config.languages, config.person_name, config.alerts.emergency_number
    )
    app = App(config, voice, gateway, EventLog(Path(args.events)), memory=PlaceMemory(Path(args.memory), config.places))
    server = None
    if args.port:
        app.stream_enabled = True
        server = serve(app, args.port)
        log.info("Dashboard: http://127.0.0.1:%d", args.port)
    if config.listen_for_help_words and not args.no_audio:
        threading.Thread(target=app.listen_for_triggers, args=(voice.listener,), daemon=True).start()

    vision = VisionPipeline(delegate=config.vision_delegate)
    try:
        app.run(config.camera_index if live_camera else args.video, vision)
        app.wait_for_worker()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        vision.close()
        if server is not None:
            server.shutdown()
    return 0


def _doctor(config: Config) -> int:
    import cv2

    failures = 0

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not passed
        print(f"{'ok  ' if passed else 'FAIL'} {name}{': ' + detail if detail else ''}")

    for path in ["pose_landmarker_lite.task", "efficientdet_lite0.tflite"]:
        check(path, (MODELS / path).exists())
    for language in config.languages:
        check(f"Vosk {language}", (MODELS / VOSK_MODELS[language]).exists())
        check(f"Piper {language}", (MODELS / "piper" / f"{PIPER_VOICES[language]}.onnx").exists())
    capture = cv2.VideoCapture(config.camera_index)
    ok, frame = capture.read()
    capture.release()
    check("camera", ok, f"{frame.shape[1]}x{frame.shape[0]}" if ok else f"index {config.camera_index}")
    try:
        samples = np.frombuffer(b"".join(microphone_blocks(1.0)), dtype=np.int16)
        check("microphone", samples.size > 0, f"{samples.size} samples, level {int(np.abs(samples).max()) if samples.size else 0}")
    except Exception as error:
        check("microphone", False, str(error))
    try:
        play_audio(np.zeros(1600, dtype=np.int16), 16000)
        check("speaker", True, "device opened, run the say command to hear it")
    except Exception as error:
        check("speaker", False, str(error))
    check("consent recorded", config.monitored_person_consented)
    check("contacts configured", bool(config.contacts))
    check("real alerts enabled", config.alerts.live, "" if config.alerts.live else "practice mode")
    return 1 if failures else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(prog="elder_help", description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--memory", default="memory.json", help="Where remembered places are stored")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Start monitoring")
    run.add_argument("--video", help="Replay a video file instead of using the camera")
    run.add_argument("--port", type=int, default=8765, help="Dashboard port, 0 to disable")
    run.add_argument("--events", default="events.jsonl")
    run.add_argument("--no-audio", action="store_true", help="Do not use the speaker or mic")
    run.add_argument("--reply", default="silence", choices=[r.value for r in Reply], help="Reply to pretend to hear with --no-audio")
    run.add_argument("--listen-s", type=float, help="Listening time with --no-audio")
    commands.add_parser("doctor", help="Check models, camera, audio and settings")
    commands.add_parser("say", help="Play the check-in question out loud")
    talk = commands.add_parser("talk", help="Talk to the assistant once")
    talk.add_argument("--text", action="store_true", help="Type instead of using the speaker and mic")
    talk.add_argument("--show-heard", action="store_true", help="Print what the speech models heard, without saving it")
    hear = commands.add_parser("hear", help="Listen once and print how the reply was understood")
    hear.add_argument("--seconds", type=float, default=8)
    args = parser.parse_args()

    path = Path(args.config)
    config = load_config(path) if path.exists() else Config()
    if not path.exists():
        log.warning("%s not found, using defaults. Copy config.example.toml to get started.", path)

    if args.command == "run":
        return _run(args, config)
    if args.command == "doctor":
        return _doctor(config)
    if args.command == "talk":
        return _talk(args, config)
    if args.command == "say":
        Speaker(config.languages, config.person_name, config.alerts.emergency_number).say("ask")
        return 0
    voice = LocalVoice(config.languages, config.person_name, config.alerts.emergency_number)
    print(voice.listen(args.seconds).value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
