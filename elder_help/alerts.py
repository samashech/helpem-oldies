"""Texts and calls emergency contacts."""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol
from xml.sax.saxutils import escape

from .config import AlertConfig, Contact
from .safety import Reason

log = logging.getLogger(__name__)

REASON_TEXT = {
    Reason.LYING: ("is lying down and did not answer", "लेटे हुए हैं और जवाब नहीं दे रहे"),
    Reason.FALL: ("may have fallen and did not answer", "शायद गिर गए हैं और जवाब नहीं दे रहे"),
    Reason.FALL_OUT_OF_VIEW: ("may have fallen out of the camera's view", "शायद गिर गए हैं और कैमरे में नहीं दिख रहे"),
    Reason.STRUGGLE: ("is having trouble getting up", "उठने में परेशानी हो रही है"),
    Reason.HELP_WORD: ("asked for help", "मदद माँग रहे हैं"),
    Reason.WELLNESS: ("did not answer a routine check-in", "नियमित जाँच का जवाब नहीं दिया"),
    Reason.MANUAL: ("did not answer a check-in", "जाँच का जवाब नहीं दिया"),
}


class Gateway(Protocol):
    simulated: bool

    def send_sms(self, to: str, body: str) -> None: ...

    def call(self, to: str, english: str, hindi: str, require_human: bool = True) -> bool:
        """Returns True only if the call connected, and to a person when require_human is set."""
        ...


@dataclass
class EscalationResult:
    reached: Contact | None = None
    texted: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    simulated: bool = False


class DryRunGateway:
    """Logs what would be sent. It never reports a contact as reached."""

    simulated = True

    def send_sms(self, to: str, body: str) -> None:
        log.warning("SIMULATED SMS to %s: %s", to, body)

    def call(self, to: str, english: str, hindi: str, require_human: bool = True) -> bool:
        log.warning("SIMULATED CALL to %s: %s", to, english)
        return False


class TwilioGateway:
    simulated = False

    def __init__(
        self,
        config: AlertConfig,
        base_url: str = "https://api.twilio.com",
        poll_s: float = 3.0,
        call_timeout_s: float = 90.0,
    ):
        if not (config.twilio_account_sid and config.twilio_auth_token and config.twilio_from):
            raise ValueError("alerts.live is true but the Twilio account SID, auth token or from number is missing")
        self.cfg = config
        self.base = f"{base_url}/2010-04-01/Accounts/{config.twilio_account_sid}"
        self.poll_s = poll_s
        self.call_timeout_s = call_timeout_s
        credentials = f"{config.twilio_account_sid}:{config.twilio_auth_token}".encode()
        self._auth = "Basic " + base64.b64encode(credentials).decode()

    def _request(self, path: str, data: dict | None = None) -> dict:
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        request = urllib.request.Request(self.base + path, data=body, headers={"Authorization": self._auth})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Twilio returned {error.code}: {error.read()[:300].decode(errors='replace')}") from error

    def send_sms(self, to: str, body: str) -> None:
        self._request("/Messages.json", {"To": to, "From": self.cfg.twilio_from, "Body": body})

    def call(self, to: str, english: str, hindi: str, require_human: bool = True) -> bool:
        twiml = (
            '<Response><Say language="hi-IN" loop="2">'
            f"{escape(hindi)}</Say><Say language=\"en-IN\" loop=\"2\">{escape(english)}</Say></Response>"
        )
        request = {"To": to, "From": self.cfg.twilio_from, "Twiml": twiml, "Timeout": "30"}
        if require_human:
            request["MachineDetection"] = "Enable"
        created = self._request("/Calls.json", request)
        deadline = time.monotonic() + self.call_timeout_s
        while time.monotonic() < deadline:
            call = self._request(f"/Calls/{created['sid']}.json")
            if not require_human and call["status"] in ("completed", "in-progress"):
                return True
            if call["status"] in ("completed", "in-progress"):
                answered_by = call.get("answered_by") or "human"
                if call["status"] == "completed" or answered_by != "unknown":
                    return not answered_by.startswith(("machine", "fax"))
            if call["status"] in ("busy", "failed", "no-answer", "canceled"):
                return False
            time.sleep(self.poll_s)
        return False


def make_gateway(config: AlertConfig) -> Gateway:
    return TwilioGateway(config) if config.live else DryRunGateway()


def escalate(
    person: str,
    reason: Reason,
    contacts: tuple[Contact, ...],
    gateway: Gateway,
    config: AlertConfig,
    wait: Callable[[float], None] = time.sleep,
    note: str = "",
) -> EscalationResult:
    """Texts every contact, then calls them in order until someone answers."""
    result = EscalationResult(simulated=gateway.simulated)
    english, hindi = REASON_TEXT[reason]
    fallback = f"If you cannot reach them, call {config.emergency_number}."
    note = f" {note}" if note else ""
    text = f"Elder-Help: {person} {english}. Please check on them now.{note} {fallback}"
    spoken_en = f"This is an urgent alert. {person} {english}. Please check on them now. {fallback}"
    spoken_hi = f"यह एक ज़रूरी सूचना है। {person} {hindi}। कृपया तुरंत देखें। संपर्क न हो तो {config.emergency_number} पर कॉल करें।"

    for contact in contacts:
        try:
            gateway.send_sms(contact.phone, text)
            result.texted.append(contact.name)
        except Exception as error:
            result.errors.append(f"SMS to {contact.name}: {error}")

    for round_number in range(config.call_rounds):
        if round_number:
            wait(30)
        for contact in contacts:
            try:
                if gateway.call(contact.phone, spoken_en, spoken_hi):
                    result.reached = contact
                    break
            except Exception as error:
                result.errors.append(f"Call to {contact.name}: {error}")
        if result.reached:
            break
    for error in result.errors:
        log.error("%s", error)
    return result


@dataclass
class EmergencyCall:
    connected: bool = False
    error: str = ""
    simulated: bool = False


def call_emergency_services(person: str, address: str, config: AlertConfig, gateway: Gateway) -> EmergencyCall:
    """Places the call the person asked for. Connected only means the line picked up."""
    place = address or "an address not on file"
    english = f"Emergency. An elderly person, {person}, at {place}, has asked for urgent help. This is an automated call. They may not be able to speak."
    hindi = f"आपातकाल। {person}, पता {place}, को तुरंत मदद चाहिए। यह एक स्वचालित कॉल है। वे शायद बोल न पाएँ।"
    try:
        connected = gateway.call(config.emergency_number, english, hindi, require_human=False)
        return EmergencyCall(connected=connected, simulated=gateway.simulated)
    except Exception as error:
        log.error("Emergency call failed: %s", error)
        return EmergencyCall(error=str(error), simulated=gateway.simulated)


def notify(person: str, reason: Reason, contacts: tuple[Contact, ...], gateway: Gateway) -> list[str]:
    """Sends a low-urgency text and returns any delivery errors."""
    errors = []
    for contact in contacts:
        try:
            gateway.send_sms(contact.phone, f"Elder-Help: {person} {REASON_TEXT[reason][0]}. Please check in when you can.")
        except Exception as error:
            errors.append(f"SMS to {contact.name}: {error}")
            log.error("%s", errors[-1])
    return errors
