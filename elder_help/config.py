"""Loads config.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .activity import BedZone
from .safety import SafetyConfig


@dataclass(frozen=True)
class Contact:
    name: str
    phone: str
    aliases: tuple[str, ...] = ()  # Other ways they get called, like "बेटा", "son" or "राहुल".


@dataclass(frozen=True)
class ChatConfig:
    enabled: bool = True
    url: str = "http://127.0.0.1:11434"
    model: str = "gemma3:4b"
    timeout_s: float = 40


@dataclass(frozen=True)
class AlertConfig:
    live: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from: str = ""
    emergency_number: str = "112"
    call_rounds: int = 2


@dataclass(frozen=True)
class VoiceConfig:
    ask_attempts: int = 3
    listen_s: float = 12


@dataclass(frozen=True)
class Config:
    person_name: str = "them"
    address: str = ""
    languages: tuple[str, ...] = ("hi", "en")
    monitored_person_consented: bool = False
    camera_index: int = 0
    vision_delegate: str = "GPU"
    prefer_nvidia: bool = True
    listen_for_help_words: bool = True
    sleep_hours: tuple[int, int] = (22, 7)
    camera_fault_s: float = 60
    bed_zones: tuple[BedZone, ...] = ()
    contacts: tuple[Contact, ...] = ()
    alerts: AlertConfig = field(default_factory=AlertConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    places: dict[str, dict[str, str]] = field(default_factory=dict)
    medicine_times: tuple[str, ...] = ()

    def in_sleep_hours(self, hour: int) -> bool:
        start, end = self.sleep_hours
        return start <= hour < end if start <= end else hour >= start or hour < end


def _build(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {', '.join(sorted(unknown))}")
    return cls(**values)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    safety = dict(raw.pop("safety", {}))
    camera_fault_s = safety.pop("camera_fault_s", 60)
    config = Config(
        person_name=raw.pop("person_name", "them"),
        address=raw.pop("address", ""),
        languages=tuple(raw.pop("languages", ("hi", "en"))),
        monitored_person_consented=raw.pop("monitored_person_consented", False),
        camera_index=raw.pop("camera_index", 0),
        vision_delegate=raw.pop("vision_delegate", "GPU"),
        prefer_nvidia=raw.pop("prefer_nvidia", True),
        listen_for_help_words=raw.pop("listen_for_help_words", True),
        sleep_hours=tuple(raw.pop("sleep_hours", (22, 7))),
        camera_fault_s=camera_fault_s,
        bed_zones=tuple(BedZone(**zone) for zone in raw.pop("bed_zones", [])),
        contacts=tuple(
            Contact(**{**contact, "aliases": tuple(contact.get("aliases", ()))}) for contact in raw.pop("contacts", [])
        ),
        alerts=_build(AlertConfig, raw.pop("alerts", {})),
        safety=_build(SafetyConfig, safety),
        voice=_build(VoiceConfig, raw.pop("voice", {})),
        chat=_build(ChatConfig, raw.pop("chat", {})),
        places=raw.pop("places", {}),
        medicine_times=tuple(raw.pop("medicine_times", ())),
    )
    if raw:
        raise ValueError(f"Unknown settings: {', '.join(sorted(raw))}")
    return config
