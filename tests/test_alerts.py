import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from elder_help.alerts import DryRunGateway, TwilioGateway, call_emergency_services, escalate, notify
from elder_help.config import AlertConfig, Contact
from elder_help.safety import Reason

CONFIG = AlertConfig(live=True, twilio_account_sid="AC1", twilio_auth_token="tok", twilio_from="+15550001111", call_rounds=2)
A, B = Contact("A", "+911"), Contact("B", "+912")


@pytest.fixture
def twilio():
    """A local server that behaves like the parts of the Twilio API we use."""
    state = {"messages": [], "calls": [], "outcomes": {}, "auth": [], "fail_sms": False}

    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            state["auth"].append(self.headers["Authorization"])
            form = {k: v[0] for k, v in urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode()).items()}
            if self.path.endswith("/Messages.json"):
                if state["fail_sms"]:
                    return self._json({"message": "blocked"}, 400)
                state["messages"].append(form)
                return self._json({"sid": "SM1"}, 201)
            state["calls"].append(form)
            return self._json({"sid": form["To"].replace("+", "CA")}, 201)

        def do_GET(self):
            sid = self.path.rsplit("/", 1)[1].removesuffix(".json")
            status, answered_by = state["outcomes"].get("+" + sid.removeprefix("CA"), ("no-answer", None))
            self._json({"status": status, "answered_by": answered_by})

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["gateway"] = TwilioGateway(CONFIG, base_url=f"http://127.0.0.1:{server.server_port}", poll_s=0.01, call_timeout_s=2)
    yield state
    server.shutdown()


def run(gateway, contacts=(A, B), config=CONFIG):
    return escalate("Dadi ji", Reason.FALL, contacts, gateway, config, wait=lambda _s: None)


def test_texts_everyone_and_stops_calling_once_a_person_answers(twilio):
    twilio["outcomes"] = {"+911": ("completed", "human"), "+912": ("completed", "human")}
    result = run(twilio["gateway"])
    assert result.reached == A and not result.simulated
    assert [m["To"] for m in twilio["messages"]] == ["+911", "+912"]
    assert [c["To"] for c in twilio["calls"]] == ["+911"]
    assert "Dadi ji" in twilio["messages"][0]["Body"] and "112" in twilio["messages"][0]["Body"]
    assert twilio["auth"][0].startswith("Basic ")


def test_moves_to_the_next_contact_when_nobody_answers(twilio):
    twilio["outcomes"] = {"+911": ("no-answer", None), "+912": ("completed", "human")}
    assert run(twilio["gateway"]).reached == B


def test_voicemail_does_not_count_as_reaching_someone(twilio):
    twilio["outcomes"] = {"+911": ("completed", "machine_start"), "+912": ("busy", None)}
    result = run(twilio["gateway"])
    assert result.reached is None
    assert len(twilio["calls"]) == 4  # Two contacts, two rounds.


def test_failed_texts_are_reported_and_calls_still_go_out(twilio):
    twilio["fail_sms"] = True
    twilio["outcomes"] = {"+911": ("completed", "human")}
    result = run(twilio["gateway"])
    assert result.reached == A
    assert len(result.errors) == 2 and "400" in result.errors[0]


def test_call_message_escapes_xml(twilio):
    twilio["outcomes"] = {"+911": ("completed", "human")}
    escalate("A & <B>", Reason.FALL, (A,), twilio["gateway"], CONFIG, wait=lambda _s: None)
    assert "A &amp; &lt;B&gt;" in twilio["calls"][0]["Twiml"]


def test_practice_mode_never_claims_someone_was_reached():
    result = run(DryRunGateway())
    assert result.simulated and result.reached is None


def test_live_mode_requires_credentials():
    with pytest.raises(ValueError):
        TwilioGateway(AlertConfig(live=True))


def test_notify_returns_errors(twilio):
    twilio["fail_sms"] = True
    assert len(notify("X", Reason.WELLNESS, (A,), twilio["gateway"])) == 1


def test_emergency_call_goes_to_the_configured_number_with_the_address(twilio):
    config = AlertConfig(emergency_number="+919")
    twilio["outcomes"] = {"+919": ("in-progress", None)}
    call = call_emergency_services("Dadi", "12 MG Road", config, twilio["gateway"])
    assert call.connected and not call.simulated
    placed = twilio["calls"][0]
    assert placed["To"] == "+919" and "12 MG Road" in placed["Twiml"]
    # A recorded menu on an emergency line must not be mistaken for a failure.
    assert "MachineDetection" not in placed


def test_failed_emergency_call_reports_failure(twilio):
    twilio["outcomes"] = {"+919": ("failed", None)}
    assert not call_emergency_services("D", "", AlertConfig(emergency_number="+919"), twilio["gateway"]).connected


def test_practice_mode_never_reports_an_emergency_call_as_connected():
    call = call_emergency_services("D", "", AlertConfig(), DryRunGateway())
    assert call.simulated and not call.connected
