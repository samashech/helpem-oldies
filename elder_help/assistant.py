"""Spoken assistant: remembers where things are, passes on messages and keeps people company.

Everything that acts on the world is parsed by the rules here and confirmed aloud. The chat model
only produces conversation, so nothing it says or is told can send a message or place a call.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .config import ChatConfig, Contact

log = logging.getLogger(__name__)

MIN_UTTERANCE_CONFIDENCE = 0.55


class IntentKind(str, Enum):
    CALL_EMERGENCY = "call_emergency"
    ALERT_FAMILY = "alert_family"
    SEND_MESSAGE = "send_message"
    ASK_PLACE = "ask_place"
    REMEMBER_PLACE = "remember_place"
    HELP = "help"
    DISTRESS = "distress"  # Sounds unwell or hurt, without asking for anyone.
    TIME = "time"
    GOODBYE = "goodbye"
    CHAT = "chat"


class Confirmation(str, Enum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class Intent:
    kind: IntentKind
    item: str = ""
    place: str = ""
    who: str = ""
    body: str = ""


@dataclass(frozen=True)
class Transcript:
    language: str
    text: str
    confidence: float


def normalize(text: str) -> str:
    """Evens out Vosk's spelling, so हाँ/हां and मेज़/मेज compare equal."""
    text = unicodedata.normalize("NFD", text.lower()).replace("़", "").replace("ँ", "ं")
    return re.sub(r"\s+", " ", re.sub(r"[।,.?!]", " ", unicodedata.normalize("NFC", text))).strip()


ITEM_ALIASES = {
    "medicine": ["दवाई", "दवा", "दवाई", "दबाई", "दवाइयां", "दवाएं", "गोली", "गोलियां", "medicine", "medicines", "tablet", "tablets", "pills"],
    "glasses": ["चश्मा", "चश्मे", "glasses", "spectacles"],
    "keys": ["चाबी", "चाबियां", "key", "keys"],
    "phone": ["फोन", "मोबाइल", "phone", "mobile"],
}
_ITEM_LOOKUP = {normalize(alias): item for item, aliases in ITEM_ALIASES.items() for alias in aliases}
_STOP_WORDS = {"मेरी", "मेरा", "मेरे", "मैंने", "मैं", "ने", "याद", "रखो", "कि", "वो", "वह", "my", "the", "i", "that", "remember"}
EMERGENCY_PHRASES = [normalize(p) for p in (
    "एक सौ बारह", "एक एक दो", "एम्बुलेंस", "एंबुलेंस", "112",
    "one one two", "one twelve", "ambulance", "emergency",
)]
_FAMILY_WORDS = {normalize(w) for w in ("परिवार", "सबको", "सब", "घरवालों", "घर", "family", "everyone", "everybody", "contacts")}
_HI_MESSAGE = re.compile(
    r"^(?P<who>.+?) (?:को|से) (?:मैसेज|मेसेज|संदेश|बोलो|बोल दो|कहो|कह दो|बताओ|बता दो|लिखो)"
    r"(?: करो| कर दो| भेजो| भेज दो)?(?: कि| की)?(?: (?P<body>.*))?$"
)
_EN_MESSAGE = re.compile(
    r"^(?:please )?(?:text|message|tell|ask|inform|send (?:a )?message to) (?:my )?(?P<who>\S+)(?: to| that)?(?: (?P<body>.*))?$"
)
_HI_ASK = re.compile(r"(?:^| )(?:कहां|कहा|किधर)(?: |$)")
_EN_ASK = re.compile(r"^(?:please )?(?:where|we|were) (?:is|are|did i (?:keep|put|leave)|have i (?:kept|put|left)|do i keep) (?P<item>.+)$")
_HI_PLACED = re.compile(r"^(?P<lead>.+?) (?P<prep>में|पर|के ऊपर|के नीचे|के पास|के अंदर) (?:रखी|रखा|रखे|रखीं)(?: हुई| हुआ| हुए)?(?: है| हैं| थी| था)?$")
_EN_PLACED = (
    re.compile(r"^(?:remember )?(?:that )?i (?:kept|put|left) (?P<item>.+?) (?P<place>(?:in|on|under|near|behind|inside) .+)$"),
    re.compile(r"^(?:remember )?(?:that )?(?P<item>.+?) (?:is|are) (?:kept )?(?P<place>(?:in|on|under|near|behind|inside) .+)$"),
)
_TIME = ("कितने बजे", "समय क्या", "टाइम क्या", "क्या टाइम", "what time", "the time")
_GOODBYE = ("बस", "धन्यवाद", "शुक्रिया", "अलविदा", "thank you", "thanks", "bye", "goodbye", "that's all", "nothing")
_HELP_WORDS = {normalize(w) for w in ("मदद", "बचाओ", "बचाव", "help")}
_DISTRESS_WORDS = {normalize(w) for w in (
    "गिर", "गिरा", "गिरी", "चक्कर", "सांस", "सीने", "सीना", "दर्द", "बेहोश", "तकलीफ",
    "fell", "fallen", "pain", "dizzy", "breathe", "breathing", "chest", "hurt", "hurts", "bleeding",
)}


def _clean_item(words: Iterable[str]) -> str:
    kept = [word for word in words if word not in _STOP_WORDS]
    for word in kept:
        if word in _ITEM_LOOKUP:
            return _ITEM_LOOKUP[word]
    return " ".join(kept)


def parse_intent(text: str) -> Intent:
    """Reads one sentence, in either language. Unmatched speech is plain conversation."""
    text = normalize(text).replace("सबको", "सब को")
    words = text.split()
    if any(re.search(rf"(?:^| ){re.escape(phrase)}(?: |$)", text) for phrase in EMERGENCY_PHRASES):
        return Intent(IntentKind.CALL_EMERGENCY)

    match = _HI_MESSAGE.match(text) or _EN_MESSAGE.match(text)
    if match:
        who_words = [word for word in match["who"].split() if word not in _STOP_WORDS]
        if set(who_words) & _FAMILY_WORDS:
            return Intent(IntentKind.ALERT_FAMILY)
        # "Tell me about your day" and "मुझे बताओ" are conversation, not a message to someone.
        if who_words and not set(who_words) & {"me", "us", "you", "मुझे", "मुझ", "हमें", "आप", "तुम"}:
            return Intent(IntentKind.SEND_MESSAGE, who=" ".join(who_words), body=(match["body"] or "").strip())
    if re.match(r"^(?:please )?(?:alerts?|call|phone|inform) (?:my |all )?(?:family|everyone|contacts)", text):
        return Intent(IntentKind.ALERT_FAMILY)
    if _HI_ASK.search(text):
        before = _HI_ASK.split(text)[0].split()
        item = _clean_item(before)
        if item:
            return Intent(IntentKind.ASK_PLACE, item=item)
    match = _EN_ASK.match(text)
    if match:
        return Intent(IntentKind.ASK_PLACE, item=_clean_item(match["item"].split()))

    match = _HI_PLACED.match(text)
    if match:
        lead = [word for word in match["lead"].split() if word not in _STOP_WORDS]
        known = next((word for word in lead if word in _ITEM_LOOKUP), None)
        if known is not None:
            lead.remove(known)
            item = _ITEM_LOOKUP[known]
        elif len(lead) >= 2:
            item = lead.pop(0)
        else:
            item = ""
        if item and lead:
            return Intent(IntentKind.REMEMBER_PLACE, item=item, place=f"{' '.join(lead)} {match['prep']}")
    for pattern in _EN_PLACED:
        match = pattern.match(text)
        if match:
            item = _clean_item(match["item"].split())
            if item:
                return Intent(IntentKind.REMEMBER_PLACE, item=item, place=match["place"])

    if _HELP_WORDS & set(words):
        return Intent(IntentKind.HELP)
    if _DISTRESS_WORDS & set(words):
        return Intent(IntentKind.DISTRESS)
    if any(phrase in text for phrase in _TIME):
        return Intent(IntentKind.TIME)
    if len(words) <= 3 and any(re.search(rf"(?:^| ){re.escape(normalize(p))}(?: |$)", text) for p in _GOODBYE):
        return Intent(IntentKind.GOODBYE)
    return Intent(IntentKind.CHAT)


_YES_WORDS = {normalize(w) for w in ("yes", "okay", "ok", "send", "call", "ठीक", "हां", "जी", "भेज", "भेजो", "करो", "कर", "है", "दो", "it", "do")}
_YES_SIGNALS = {normalize(w) for w in ("yes", "okay", "ok", "send", "call", "ठीक", "हां", "जी", "भेज", "भेजो", "करो", "कर")}
_NO_WORDS = {normalize(w) for w in ("no", "cancel", "don't", "dont", "नहीं", "मत", "रहने", "रद्द")}


def classify_confirmation(words: Iterable[tuple[str, float]]) -> Confirmation:
    """A yes must be the whole sentence, so a stray "ठीक" on TV cannot send anything."""
    items = [(normalize(word), conf) for word, conf in words]
    spoken = {word for word, conf in items if conf >= 0.3}
    if spoken & _NO_WORDS:
        return Confirmation.NO
    only_yes = bool(items) and all(word in _YES_WORDS for word, _ in items) and len(items) <= 4
    if only_yes and any(word in _YES_SIGNALS and conf >= 0.6 for word, conf in items):
        return Confirmation.YES
    return Confirmation.UNCLEAR


class PlaceMemory:
    """Where things are kept. Stored only on this computer."""

    def __init__(self, path: Path | None, seeded: dict[str, dict[str, str]] | None = None):
        self.path = path
        self._lock = threading.Lock()
        self._places: dict[str, dict[str, str]] = dict(seeded or {})
        if path is not None and path.exists():
            self._places.update(json.loads(path.read_text(encoding="utf-8")))

    def remember(self, item: str, place: str, language: str) -> None:
        with self._lock:
            # A new location replaces the old one in both languages.
            self._places[item] = {language: place}
            if self.path is not None:
                self.path.write_text(json.dumps(self._places, ensure_ascii=False, indent=2), encoding="utf-8")

    def recall(self, item: str, language: str) -> tuple[str, str] | None:
        """Returns (place, language), preferring the language being spoken."""
        with self._lock:
            entry = self._places.get(item)
        if not entry:
            return None
        if language in entry:
            return entry[language], language
        other, place = next(iter(entry.items()))
        return place, other


def find_contact(who: str, contacts: Iterable[Contact]) -> Contact | None:
    wanted = normalize(who)
    if not wanted:
        return None
    for contact in contacts:
        for name in (contact.name, *contact.aliases):
            name = normalize(name)
            if wanted == name or name in wanted.split():
                return contact
    return None


LINES = {
    "greet": {"hi": "जी, बोलिए।", "en": "Yes, I am listening."},
    "repeat": {"hi": "माफ़ कीजिए, मैं समझ नहीं पाया। फिर से बोलिए।", "en": "Sorry, I did not catch that. Please say it again."},
    "bye": {"hi": "ठीक है। ज़रूरत हो तो सहायक बोलिए।", "en": "All right. Say helper whenever you need me."},
    "place_known": {"hi": "{item} {place} है।", "en": "Your {item} is {place}."},
    "place_unknown": {"hi": "मुझे नहीं पता {item} कहाँ है। रखने के बाद मुझे बता दीजिए।", "en": "I do not know where your {item} is. Tell me when you put it somewhere."},
    "place_saved": {"hi": "ठीक है, याद रखूँगा। {item} {place}।", "en": "All right, I will remember. Your {item} is {place}."},
    "who": {"hi": "मुझे {who} का नंबर नहीं मिला। यह नाम मेरी सूची में नहीं है।", "en": "I do not have a number for {who}. That name is not on my list."},
    "ask_body": {"hi": "{who} को क्या बताना है?", "en": "What should I tell {who}?"},
    "confirm_message": {"hi": "{who} को यह भेजूँ: {body}। ठीक है, या रहने दो?", "en": "Shall I send this to {who}: {body}. Say okay, or cancel."},
    "sent": {"hi": "संदेश भेज दिया।", "en": "Message sent."},
    "practice_sent": {"hi": "अभी अभ्यास मोड है, इसलिए संदेश सच में नहीं गया।", "en": "This is practice mode, so the message was not really sent."},
    "send_failed": {"hi": "संदेश नहीं जा पाया। कृपया किसी और तरीके से संपर्क कीजिए।", "en": "The message could not be sent. Please reach them another way."},
    "cancelled": {"hi": "ठीक है, रहने दिया।", "en": "All right, cancelled."},
    "time": {"hi": "अभी {hour} बजकर {minute} मिनट हुए हैं।", "en": "It is {hour} {minute:02d}."},
    "no_chat": {
        "hi": "मैं इसका जवाब नहीं दे पाऊँगा। मैं चीज़ों की जगह याद रख सकता हूँ और संदेश भेज सकता हूँ। ज़रूरत हो तो बोलिए, मदद।",
        "en": "I cannot answer that. I can remember where things are and send messages. If you need someone, say help.",
    },
    "thinking": {"hi": "एक मिनट।", "en": "One moment."},
}
ITEM_NAMES = {
    "medicine": {"hi": "आपकी दवाई", "en": "medicine"},
    "glasses": {"hi": "आपका चश्मा", "en": "glasses"},
    "keys": {"hi": "आपकी चाबी", "en": "keys"},
    "phone": {"hi": "आपका फ़ोन", "en": "phone"},
}


def item_name(item: str, language: str) -> str:
    return ITEM_NAMES.get(item, {}).get(language, item)


SYSTEM_PROMPT = (
    "You are a kind companion speaking aloud with an elderly person named {person}. Reply in {language}, "
    "in at most two short, simple sentences, with no lists, formatting or emoji. You cannot take any "
    "action: never say you have called, messaged, reminded or told anyone, and never promise to. Never "
    "give medical advice, diagnoses, or medicine doses, and for anything medical suggest asking their "
    "doctor or family. If they sound hurt or in danger, tell them to say the word help."
)
_LANGUAGE_NAMES = {"hi": "simple Hindi written in Devanagari", "en": "simple English"}
# The model cannot act, so any talk of calling or sending is at best misleading, and telling someone
# in trouble that help is on the way would be the worst thing this program could do. Wording varies
# too much to catch only false claims, so replies on these topics are dropped entirely.
_ACTION_TOPICS = re.compile(
    r"\b(?:call|calls|called|calling|phone|phoned|message|messaged|text|texted|sms|alert|alerted|notify|notified|"
    r"inform|informed|ambulance|emergency|police|112|try again|help|helping|assist|assistance|someone|somebody|coming|on (?:its|their|the) way|arrive|arriving)\b"
    r"|फोन|कॉल|मैसेज|मेसेज|संदेश|एम्बुलेंस|एंबुलेंस|पुलिस|आपातकाल|बुला|सूचित|खबर|कोशिश|मदद|रास्ते|आ रही|आ रहे|आ रहा|पहुंच|कर दूंगा|कर दूंगी|कर दिया|किसी को",
    re.IGNORECASE,
)
# Narrower than the check on replies, because "याद आ रही है" is just someone reminiscing.
_RESCUE_QUESTION = re.compile(
    r"\b(?:call|called|calling|phone|phoned|message|messaged|text|texted|ambulance|emergency|police|help|someone|somebody|anyone|coming)\b"
    r"|फोन|कॉल|मैसेज|मेसेज|संदेश|एम्बुलेंस|एंबुलेंस|पुलिस|बुला|मदद|कोई आ|किसी को",
    re.IGNORECASE,
)
_DOSE = re.compile(r"\d\s*(?:mg|ml|tablets?|pills?|गोली|गोलियां|टैबलेट)|\b(?:mg|milligram)", re.IGNORECASE)


class ChatClient:
    """Talks to a local Ollama model. Conversation stays on this computer."""

    def __init__(self, config: ChatConfig, person: str):
        self.cfg = config
        self.person = person

    def reply(self, text: str, language: str, history: list[dict]) -> str | None:
        system = SYSTEM_PROMPT.format(person=self.person, language=_LANGUAGE_NAMES[language])
        payload = {
            "model": self.cfg.model,
            "stream": False,
            "keep_alive": "30m",
            "options": {"num_predict": 90, "temperature": 0.6},
            "messages": [{"role": "system", "content": system}, *history, {"role": "user", "content": text}],
        }
        request = urllib.request.Request(
            f"{self.cfg.url}/api/chat", json.dumps(payload).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout_s) as response:
                answer = json.load(response)["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
            log.warning("Chat model unavailable: %s", error)
            return None
        return clean_chat_reply(answer)


def clean_chat_reply(answer: str) -> str | None:
    answer = re.sub(r"[*_#`]|[\U0001F300-\U0001FAFF☀-➿]", "", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    checked = normalize(answer)
    if not answer or _ACTION_TOPICS.search(checked) or _DOSE.search(checked):
        return None
    return answer[:300]


class AssistantVoice(Protocol):
    def say_text(self, language: str, text: str) -> None: ...

    def hear(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> list[Transcript]: ...

    def confirm(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> Confirmation: ...


def choose_transcript(transcripts: list[Transcript]) -> tuple[Transcript, Intent] | None:
    """Both models hear every sentence, and the wrong-language one produces nonsense."""
    heard = [t for t in transcripts if t.text]
    if not heard:
        return None
    parsed = [(t, parse_intent(t.text)) for t in heard]
    useful = [pair for pair in parsed if pair[1].kind is not IntentKind.CHAT and pair[0].confidence >= 0.4]
    return max(useful or parsed, key=lambda pair: pair[0].confidence)


class Assistant:
    MAX_TURNS = 12
    LISTEN_S = 8.0

    def __init__(
        self,
        voice: AssistantVoice,
        memory: PlaceMemory,
        contacts: tuple[Contact, ...],
        person: str,
        send_sms: Callable[[str, str], bool],
        is_simulated: bool,
        on_emergency: Callable[[str], None],
        on_alert_family: Callable[[str], None],
        on_distress: Callable[[], None],
        chat: ChatClient | None,
        record: Callable[..., None] = lambda *_a, **_k: None,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.voice = voice
        self.memory = memory
        self.contacts = contacts
        self.person = person
        self.send_sms = send_sms
        self.is_simulated = is_simulated
        self.on_emergency = on_emergency
        self.on_alert_family = on_alert_family
        self.on_distress = on_distress
        self.chat = chat
        self.record = record
        self.clock = clock
        self._history: list[dict] = []

    def say_line(self, key: str, language: str, **values) -> None:
        self.voice.say_text(language, LINES[key][language].format(**values))

    def converse(self, should_abort: Callable[[], bool] = lambda: False, first_text: str | None = None, language: str = "hi") -> None:
        self._history = []
        pending = Transcript(language, first_text, 1.0) if first_text else None
        if pending is None:
            self.say_line("greet", language)
        misses = 0
        for _ in range(self.MAX_TURNS):
            if should_abort():
                return
            choice = (pending, parse_intent(pending.text)) if pending else choose_transcript(self.voice.hear(self.LISTEN_S, should_abort))
            pending = None
            if should_abort():
                return
            if choice is None:
                return
            transcript, intent = choice
            language = transcript.language
            self.record("assistant_turn", intent=intent.kind.value, language=language)
            if intent.kind is IntentKind.CHAT and transcript.confidence < MIN_UTTERANCE_CONFIDENCE:
                misses += 1
                if misses >= 2:
                    self.say_line("bye", language)
                    return
                self.say_line("repeat", language)
                continue
            misses = 0
            if not self._act(intent, transcript, should_abort):
                return

    def _act(self, intent: Intent, transcript: Transcript, should_abort: Callable[[], bool]) -> bool:
        """Carries out one request and returns False when the conversation should end."""
        language = transcript.language
        kind = intent.kind
        if kind is IntentKind.CALL_EMERGENCY:
            self.on_emergency(language)
            return False
        if kind in (IntentKind.ALERT_FAMILY, IntentKind.HELP):
            self.on_alert_family(language)
            return False
        if kind is IntentKind.DISTRESS:
            self.on_distress()
            return False
        if kind is IntentKind.GOODBYE:
            self.say_line("bye", language)
            return False
        if kind is IntentKind.ASK_PLACE:
            found = self.memory.recall(intent.item, language)
            if found is None:
                self.say_line("place_unknown", language, item=item_name(intent.item, language))
            else:
                place, place_language = found
                self.say_line("place_known", place_language, item=item_name(intent.item, place_language), place=place)
        elif kind is IntentKind.REMEMBER_PLACE:
            self.memory.remember(intent.item, intent.place, language)
            self.say_line("place_saved", language, item=item_name(intent.item, language), place=intent.place)
        elif kind is IntentKind.SEND_MESSAGE:
            self._send_message(intent, language, should_abort)
        elif kind is IntentKind.TIME:
            now = self.clock()
            self.say_line("time", language, hour=now.hour % 12 or 12, minute=now.minute)
        else:
            self._chat(transcript, should_abort)
        return True

    def _send_message(self, intent: Intent, language: str, should_abort: Callable[[], bool]) -> None:
        contact = find_contact(intent.who, self.contacts)
        if contact is None:
            self.say_line("who", language, who=intent.who)
            return
        body = intent.body
        if not body:
            self.say_line("ask_body", language, who=contact.name)
            choice = choose_transcript(self.voice.hear(self.LISTEN_S, should_abort))
            if choice is None:
                self.say_line("cancelled", language)
                return
            body = choice[0].text
        self.say_line("confirm_message", language, who=contact.name, body=body)
        if self.voice.confirm(self.LISTEN_S, should_abort) is not Confirmation.YES:
            self.record("message_not_sent", contact=contact.name)
            self.say_line("cancelled", language)
            return
        sent = self.send_sms(contact.phone, f"Elder-Help: {self.person} asks: {body}")
        self.record("message_sent", contact=contact.name, delivered=sent, simulated=self.is_simulated)
        if self.is_simulated:
            self.say_line("practice_sent", language)
        else:
            self.say_line("sent" if sent else "send_failed", language)

    def _chat(self, transcript: Transcript, should_abort: Callable[[], bool]) -> None:
        language = transcript.language
        answer = None
        # Questions like "did you call him" or "is someone coming" get a fixed answer. A model must
        # never be the one reassuring somebody about whether help has been sent.
        if self.chat is not None and not _RESCUE_QUESTION.search(normalize(transcript.text)):
            # Runs aside so a safety check can interrupt while the model is thinking.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.chat.reply, transcript.text, language, self._history[-6:])
                while not future.done():
                    if should_abort():
                        pool.shutdown(wait=False, cancel_futures=True)
                        return
                    threading.Event().wait(0.1)
                answer = future.result()
        if answer is None:
            self.say_line("no_chat", language)
            return
        self._history += [{"role": "user", "content": transcript.text}, {"role": "assistant", "content": answer}]
        self.voice.say_text(language, answer)
