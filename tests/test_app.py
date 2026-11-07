import json
from pathlib import Path

from elder_help.activity import Activity, Observation
from datetime import datetime
from threading import Event

from elder_help.alerts import DryRunGateway
from elder_help.app import App, EventLog
from elder_help.assistant import Confirmation, Transcript
from elder_help.config import Config, Contact, load_config
from elder_help.safety import Phase, Reason, Reply, SafetyConfig


class FakeVoice:
    def __init__(self, replies=(), confirmations=()):
        self.replies = list(replies)
        self.confirmations = list(confirmations)
        self.said = []
        self.texts = []

    def say_text(self, language, text):
        self.texts.append((language, text))

    def hear(self, _seconds, _abort=lambda: False):
        return []

    def confirm(self, _seconds, _abort=lambda: False):
        return self.confirmations.pop(0) if self.confirmations else Confirmation.UNCLEAR

    def say(self, key, **_):
        self.said.append(key)

    def listen(self, _seconds, _abort=lambda: False):
        return self.replies.pop(0) if self.replies else Reply.SILENCE


class BrokenVoice(FakeVoice):
    def say(self, key, **_):
        raise OSError("no speaker")


class RecordingGateway(DryRunGateway):
    simulated = False

    def __init__(self, connect=True):
        self.sms, self.calls, self.spoken = [], [], []
        self.connect = connect

    def send_sms(self, to, body):
        self.sms.append((to, body))

    def call(self, to, english, hindi, require_human=True):
        self.calls.append(to)
        self.spoken.append(english)
        return self.connect


def make_app(voice, gateway=None, **config_values):
    config = Config(person_name="Dadi", contacts=(Contact("A", "+911"),), address="12 MG Road", **config_values)
    gateway = RecordingGateway() if gateway is None else gateway
    return App(config, voice, gateway, EventLog(None), wait=lambda _s: None), gateway


def checkin(app, reason=Reason.LYING):
    app.monitor.request_check(reason)
    assert app.monitor.update(Observation(Activity.LYING), app.clock()) is reason
    app.begin_checkin(reason)
    app.wait_for_worker(5)


def test_silence_retries_warns_and_then_calls():
    voice = FakeVoice([])
    app, gateway = make_app(voice)
    checkin(app)
    assert voice.said == ["ask", "ask_again", "ask_again", "final", "calling", "reached"]
    assert gateway.calls == ["+911"] and len(gateway.sms) == 1
    assert app.monitor.phase is Phase.WATCHING


def test_answering_okay_on_second_try_stops_everything():
    voice = FakeVoice([Reply.SILENCE, Reply.OK])
    app, gateway = make_app(voice)
    checkin(app)
    assert voice.said == ["ask", "ask_again", "thanks"]
    assert not gateway.calls and not gateway.sms


def test_saying_okay_to_the_final_warning_cancels_the_call():
    voice = FakeVoice([Reply.SILENCE, Reply.SILENCE, Reply.SILENCE, Reply.OK])
    app, gateway = make_app(voice)
    checkin(app)
    assert not gateway.calls
    assert voice.said[-1] == "thanks"


def test_help_calls_immediately_without_more_questions():
    voice = FakeVoice([Reply.HELP])
    app, gateway = make_app(voice)
    checkin(app)
    assert voice.said == ["ask", "calling", "reached"]
    assert gateway.calls == ["+911"]


def test_unclear_speech_never_stands_down():
    voice = FakeVoice([Reply.UNCLEAR, Reply.SILENCE, Reply.UNCLEAR, Reply.UNCLEAR])
    app, gateway = make_app(voice)
    checkin(app)
    assert gateway.calls == ["+911"]


def test_dashboard_stand_down_cancels():
    voice = FakeVoice([])
    app, gateway = make_app(voice)
    original_listen = voice.listen

    def listen(seconds, abort=lambda: False):
        app.stand_down()
        return original_listen(seconds, abort)

    voice.listen = listen
    checkin(app)
    assert not gateway.calls and not gateway.sms
    assert app.monitor.phase is Phase.WATCHING


def test_broken_speaker_still_alerts_contacts():
    app, gateway = make_app(BrokenVoice([]))
    checkin(app)
    assert gateway.calls == ["+911"]
    assert app.monitor.phase is Phase.WATCHING


def test_practice_mode_tells_the_person_nobody_was_reached():
    voice = FakeVoice([Reply.HELP])
    app, _ = make_app(voice, DryRunGateway())
    checkin(app)
    assert voice.said[-1] == "unreached"
    entry = next(e for e in app.events.recent if e["event"] == "escalation_finished")
    assert entry["simulated"] is True


def test_event_log_is_json_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    EventLog(path).write("thing", note="हाँ")
    assert json.loads(path.read_text(encoding="utf-8"))["note"] == "हाँ"


def test_example_config_loads():
    config = load_config(Path(__file__).parent.parent / "config.example.toml")
    assert config.person_name and config.contacts and not config.alerts.live
    assert not config.monitored_person_consented
    assert config.safety == SafetyConfig()
    assert config.in_sleep_hours(23) and config.in_sleep_hours(3) and not config.in_sleep_hours(12)


def test_saying_112_during_a_check_in_confirms_then_calls_112_and_family():
    voice = FakeVoice([Reply.EMERGENCY], [Confirmation.YES])
    app, gateway = make_app(voice)
    checkin(app)
    assert gateway.calls[0] == "112" and "+911" in gateway.calls
    assert "12 MG Road" in gateway.spoken[0]
    assert "call connected" in gateway.sms[0][1]
    assert app.monitor.phase is Phase.WATCHING


def test_112_is_never_dialled_without_a_yes():
    voice = FakeVoice([Reply.EMERGENCY], [Confirmation.UNCLEAR, Confirmation.UNCLEAR])
    app, gateway = make_app(voice)
    checkin(app)
    assert "112" not in gateway.calls
    assert gateway.calls == ["+911"]
    assert "NOT been called" in gateway.sms[0][1]


def test_saying_no_to_112_calls_nobody():
    voice = FakeVoice([Reply.EMERGENCY], [Confirmation.NO])
    app, gateway = make_app(voice)
    checkin(app)
    assert not gateway.calls and not gateway.sms
    assert app.monitor.phase is Phase.WATCHING


def test_silence_alone_never_dials_112():
    app, gateway = make_app(FakeVoice())
    checkin(app)
    assert "112" not in gateway.calls


def test_failed_112_call_tells_the_person_and_family():
    voice = FakeVoice([Reply.EMERGENCY], [Confirmation.YES])
    app, gateway = make_app(voice, RecordingGateway(connect=False))
    checkin(app)
    assert any("कॉल नहीं लग पाई" in text for _, text in voice.texts)
    assert "FAILED" in gateway.sms[0][1]


def test_practice_mode_says_112_was_not_really_called():
    voice = FakeVoice([Reply.EMERGENCY], [Confirmation.YES])
    app, _ = make_app(voice, DryRunGateway())
    checkin(app)
    assert any("अभ्यास मोड" in text for _, text in voice.texts)


class TalkingVoice(FakeVoice):
    def __init__(self, sentences=(), confirmations=()):
        super().__init__([], confirmations)
        self.sentences = list(sentences)

    def hear(self, _seconds, _abort=lambda: False):
        if not self.sentences:
            return []
        language = "hi" if any("\u0900" <= c <= "\u097f" for c in self.sentences[0]) else "en"
        return [Transcript(language, self.sentences.pop(0), 0.9)]


def talk(app):
    assert app.start_conversation()
    app.wait_for_worker(5)


def test_conversation_requesting_112_uses_the_same_confirmation():
    voice = TalkingVoice(["एम्बुलेंस बुलाओ"], [Confirmation.YES])
    app, gateway = make_app(voice)
    talk(app)
    assert gateway.calls[0] == "112" and "+911" in gateway.calls
    assert app.monitor.phase is Phase.WATCHING


def test_conversation_alerting_family_calls_contacts_but_not_112():
    voice = TalkingVoice(["मेरे परिवार को बताओ"])
    app, gateway = make_app(voice)
    talk(app)
    assert gateway.calls == ["+911"]


def test_safety_check_interrupts_a_conversation():
    entered, finished = Event(), Event()

    class Waiting(TalkingVoice):
        def hear(self, _seconds, abort=lambda: False):
            entered.set()
            while not abort():
                finished.wait(0.01)
            return []

    voice = Waiting()
    app, gateway = make_app(voice)
    assert app.start_conversation()
    assert entered.wait(2)
    assert app.monitor.phase is Phase.WATCHING
    assert not app.start_conversation()  # The voice is busy.
    conversation = app._worker
    checkin(app)
    conversation.join(5)
    assert "ask" in voice.said
    assert gateway.calls == ["+911"]


def test_no_conversation_during_a_check_in():
    app, _ = make_app(FakeVoice())
    app.monitor.request_check(Reason.LYING)
    app.monitor.update(Observation(Activity.LYING), app.clock())
    assert not app.start_conversation()


def reminder_app(now, activity=Activity.SITTING):
    voice = FakeVoice()
    config = dict(medicine_times=("09:00",), places={"medicine": {"hi": "रसोई की अलमारी में"}}, languages=("hi", "en"))
    app, _ = make_app(voice, **config)
    app.wall_clock = lambda: now
    app.check_medicine_reminder(Observation(activity))
    app.wait_for_worker(2)
    return app, voice


def test_medicine_reminder_says_where_it_is_once():
    app, voice = reminder_app(datetime(2026, 1, 1, 9, 5))
    assert voice.said == ["medicine_time"]
    assert any("रसोई की अलमारी में" in text for _, text in voice.texts)
    app.check_medicine_reminder(Observation(Activity.SITTING))
    app.wait_for_worker(2)
    assert voice.said == ["medicine_time"]


def test_medicine_reminder_waits_for_the_right_time_and_for_someone_to_be_there():
    assert reminder_app(datetime(2026, 1, 1, 8, 59))[1].said == []
    assert reminder_app(datetime(2026, 1, 1, 9, 45))[1].said == []
    assert reminder_app(datetime(2026, 1, 1, 9, 5), Activity.ABSENT)[1].said == []


def test_saying_you_are_hurt_leads_to_a_check_in_and_then_contacts():
    voice = TalkingVoice(["मेरे सीने में दर्द है"])
    app, gateway = make_app(voice)
    talk(app)
    reason = app.monitor.update(Observation(Activity.SITTING), app.clock())
    assert reason is Reason.HELP_WORD
    app.begin_checkin(reason)
    app.wait_for_worker(5)
    assert "ask" in voice.said and gateway.calls == ["+911"]
