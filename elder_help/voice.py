"""Hindi and English speech, running locally with Piper and Vosk."""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np

from .assistant import Confirmation, Transcript, classify_confirmation
from .safety import Reason, Reply

log = logging.getLogger(__name__)

MODELS = Path(__file__).resolve().parent.parent / "models"
PIPER_VOICES = {"hi": "hi_IN-pratham-medium", "en": "en_GB-alba-medium"}
ESPEAK_VOICES = {"hi": "hi", "en": "en-gb"}
VOSK_MODELS = {"hi": "vosk-model-small-hi-0.22", "en": "vosk-model-small-en-us-0.15"}
SAMPLE_RATE = 16000
BLOCK_BYTES = 4000
MIN_CONFIDENCE = 0.6
# Vosk gives short words low confidence, and mishearing help only costs a phone call.
MIN_HELP_CONFIDENCE = 0.3
VOICE_RMS = 500

# Every question is worded so that "yes" means okay and "no" means help.
PROMPTS = {
    "ask": {
        "hi": "{name}, क्या आप ठीक हैं? कृपया बोलिए, ठीक हूँ, या मदद चाहिए।",
        "en": "{name}, are you okay? Please say, yes I am okay, or help.",
    },
    "ask_again": {
        "hi": "{name}, मुझे आपका जवाब नहीं मिला। क्या आप ठीक हैं? ज़ोर से बोलिए, ठीक हूँ, या मदद।",
        "en": "{name}, I did not hear you. Are you okay? Please say loudly, okay, or help.",
    },
    "ask_struggle": {
        "hi": "{name}, लगता है आपको उठने में परेशानी हो रही है। क्या आप ठीक हैं? बोलिए, ठीक हूँ, या मदद।",
        "en": "{name}, it looks like getting up is difficult. Are you okay? Say okay, or help.",
    },
    "final": {
        "hi": "जवाब नहीं मिला। अब मैं आपके परिवार को फ़ोन कर रहा हूँ। ठीक हैं तो अभी बोलिए, ठीक हूँ।",
        "en": "No answer. I am now calling your family. If you are fine, say okay now.",
    },
    "thanks": {
        "hi": "धन्यवाद। मैं थोड़ी देर बाद फिर पूछूँगा।",
        "en": "Thank you. I will check on you again soon.",
    },
    "calling": {
        "hi": "मैं आपके परिवार को फ़ोन कर रहा हूँ। घबराइए मत।",
        "en": "I am calling your family now. Please stay calm.",
    },
    "reached": {
        "hi": "आपके परिवार से बात हो गई है। मदद आ रही है।",
        "en": "Your family has been told. Help is on the way.",
    },
    "unreached": {
        "hi": "कोई फ़ोन नहीं उठा रहा है। हो सके तो {emergency} पर कॉल कीजिए। मैं कोशिश करता रहूँगा।",
        "en": "Nobody is answering. If you can, call {emergency}. I will keep trying.",
    },
    "confirm_emergency": {
        "hi": "क्या मैं {emergency} पर फ़ोन करूँ? बोलिए, ठीक है, या रहने दो।",
        "en": "Should I call {emergency}? Say okay, or cancel.",
    },
    "emergency_connected": {
        "hi": "{emergency} पर कॉल लग गई है और आपके परिवार को बता रहा हूँ। हिम्मत रखिए।",
        "en": "The call to {emergency} went through, and I am telling your family. Stay calm.",
    },
    "emergency_failed": {
        "hi": "{emergency} पर कॉल नहीं लग पाई। हो सके तो फ़ोन से खुद कॉल कीजिए। मैं आपके परिवार को बता रहा हूँ।",
        "en": "The call to {emergency} did not go through. If you can, call from a phone. I am telling your family.",
    },
    "emergency_practice": {
        "hi": "अभी अभ्यास मोड है, इसलिए {emergency} पर सच में कॉल नहीं हुई। असली ज़रूरत हो तो फ़ोन से खुद कॉल कीजिए।",
        "en": "This is practice mode, so {emergency} was not really called. In a real emergency, call from a phone.",
    },
    "medicine_time": {
        "hi": "आपकी दवाई का समय हो गया है।",
        "en": "It is time for your medicine.",
    },
    "camera_fault": {
        "hi": "कैमरा काम नहीं कर रहा है। कृपया किसी से इसे ठीक करवाइए।",
        "en": "The camera has stopped working. Please ask someone to fix it.",
    },
}

# Words that signal trouble anywhere in a sentence. A false match here costs one extra phone call.
HELP_WORDS = {
    "help", "no", "not", "pain", "fell", "doctor",
    "मदद", "बचाओ", "बचाव", "नहीं", "दर्द", "गिर", "गिरा", "गिरी", "डॉक्टर",
}
# "Okay" must fill the whole sentence. A false match here could cancel a real emergency, and
# ordinary TV dialogue often contains these words.
OK_WORDS = {"yes", "okay", "fine", "ठीक", "हां", "हाँ", "जी"}
OK_FILLERS = {"i", "am", "i'm", "all", "good", "मैं", "हूं", "हूँ", "है", "सब"}
# English uses a fixed word list, which reliably rejects unrelated speech as [unk]. Hindi is decoded
# freely, because a fixed Hindi list forced unrelated sentences onto words like "ठीक".
ENGLISH_GRAMMAR = sorted(
    {"yes", "okay", "fine", "i am fine", "i am okay", "yes i am okay", "i'm okay", "i'm fine", "all good",
     "help", "help me", "no", "not okay", "pain", "i fell", "doctor", "ambulance", "call an ambulance",
     "one one two", "call one one two", "emergency"}
)
# "ok" is left out of these lists: it sounds identical to "okay" and splits its confidence in half.
ENGLISH_CONFIRM_GRAMMAR = ["yes", "okay", "send it", "call", "do it", "no", "cancel", "don't"]
# "No" and "नहीं" are left out. They only mean help when answering a question.
WAKE_WORDS = {"help", "मदद", "बचाओ", "बचाव"}
ENGLISH_WAKE_GRAMMAR = ["help", "help me", "helper"]
WAKE_NAMES = {"helper", "सहायक"}
EMERGENCY_WORDS = ("एक सौ बारह", "एक एक दो", "एम्बुलेंस", "एंबुलेंस", "one one two", "ambulance", "emergency")


def _normalize(word: str) -> str:
    return word.replace("\u0901", "\u0902")  # Treats हाँ and हां as the same word.


_HELP = {_normalize(word) for word in HELP_WORDS}
_OK = {_normalize(word) for word in OK_WORDS}
_FILLERS = {_normalize(word) for word in OK_FILLERS}


def classify_transcript(words: Iterable[tuple[str, float]]) -> Reply:
    """Maps one utterance to a reply. Anything short of a clear "okay" is not treated as okay."""
    items = [(_normalize(word), conf) for word, conf in words]
    spoken = [(word, conf) for word, conf in items if word != "[unk]"]
    if not items:
        return Reply.SILENCE
    joined = " ".join(word for word, _ in spoken)
    if any(phrase in joined for phrase in EMERGENCY_WORDS):
        return Reply.EMERGENCY
    if any(word in _HELP and conf >= MIN_HELP_CONFIDENCE for word, conf in spoken):
        return Reply.HELP
    only_reply_words = len(spoken) == len(items) and all(word in _OK | _FILLERS for word, _ in spoken)
    has_ok_word = any(word in _OK and conf >= MIN_CONFIDENCE for word, conf in spoken)
    return Reply.OK if only_reply_words and has_ok_word else Reply.UNCLEAR


def prompt_key(reason: Reason, attempt: int) -> str:
    if attempt:
        return "ask_again"
    return "ask_struggle" if reason is Reason.STRUGGLE else "ask"


class Voice(Protocol):
    def say(self, key: str, **values: str) -> None: ...

    def listen(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> Reply: ...


def split_by_script(text: str, default: str) -> list[tuple[str, str]]:
    """Splits text so each stretch goes to the voice that can pronounce it.

    The English voice can't read Devanagari and the Hindi voice can't read Latin, and a line can
    hold both, like "Rahul को यह भेजूँ" or a chat reply that ends in "दादी जी".
    """
    parts: list[list] = []
    for word in text.split():
        if any("\u0900" <= char <= "\u097f" for char in word):
            language = "hi"
        elif any(char.isascii() and char.isalpha() for char in word):
            language = "en"
        else:
            language = None  # Digits and punctuation stay with whatever surrounds them.
        if parts and (language is None or parts[-1][0] is None or parts[-1][0] == language):
            parts[-1][0] = parts[-1][0] or language
            parts[-1][1].append(word)
        else:
            parts.append([language, [word]])
    return [(language or default, " ".join(words)) for language, words in parts]


def play_audio(audio: np.ndarray, rate: int) -> None:
    # arecord and aplay run as separate processes because PortAudio breaks MediaPipe. See vision.py.
    command = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(rate), "-"]
    subprocess.run(command, input=audio.astype(np.int16).tobytes(), check=True)


class Speaker:
    """Speaks with Piper, and falls back to espeak-ng if a Piper voice fails to load."""

    def __init__(self, languages: Iterable[str], person: str, emergency: str):
        self.languages = tuple(languages)
        self.values = {"name": person, "emergency": emergency}
        self._voices = {}
        self._cache: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
        self._fixed_prompts = {
            text.format(**self.values) for lines in PROMPTS.values() for text in lines.values()
        }
        for language in self.languages:
            path = MODELS / "piper" / f"{PIPER_VOICES[language]}.onnx"
            try:
                from piper import PiperVoice

                self._voices[language] = PiperVoice.load(str(path))
            except Exception as error:
                log.warning("Piper voice %s unavailable, using espeak-ng: %s", language, error)

    def synthesize(self, language: str, text: str) -> tuple[np.ndarray, int]:
        cached = self._cache.get((language, text))
        if cached is not None:
            return cached
        chunks = list(self._voices[language].synthesize(text))
        audio = np.concatenate([chunk.audio_int16_array for chunk in chunks])
        # Only fixed prompts are worth keeping. Conversation is never held on to.
        if text in self._fixed_prompts:
            self._cache[(language, text)] = (audio, chunks[0].sample_rate)
        return audio, chunks[0].sample_rate

    def say(self, key: str, **values: str) -> None:
        for language in self.languages:
            self.say_text(language, PROMPTS[key][language].format(**{**self.values, **values}))

    def say_text(self, language: str, text: str) -> None:
        for part_language, part in split_by_script(text, language):
            # With only one language configured, it has to read everything.
            if part_language not in self.languages:
                part_language = language
            log.info("Saying (%s)", part_language)
            if part_language in self._voices:
                audio, rate = self.synthesize(part_language, part)
                play_audio(audio, rate)
            else:
                subprocess.run(["espeak-ng", "-v", ESPEAK_VOICES[part_language], "-s", "130", part], check=True)



class Listener:
    """Recognises replies in both languages at once from one microphone stream."""

    def __init__(self, languages: Iterable[str]):
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        self._recognizer = KaldiRecognizer
        self._models = {lang: Model(str(MODELS / VOSK_MODELS[lang])) for lang in languages}

    def _recognizers(self, english_grammar: list[str]) -> list:
        recognizers = []
        for language, model in self._models.items():
            if language == "en":
                grammar = json.dumps(english_grammar + ["[unk]"])
                recognizer = self._recognizer(model, SAMPLE_RATE, grammar)
            else:
                recognizer = self._recognizer(model, SAMPLE_RATE)
            recognizer.SetWords(True)
            recognizers.append(recognizer)
        return recognizers

    @staticmethod
    def _words(result_json: str) -> list[tuple[str, float]]:
        result = json.loads(result_json)
        return [(item["word"], item.get("conf", 0.0)) for item in result.get("result", [])]

    def transcribe_reply(self, blocks: Iterable[bytes], should_abort: Callable[[], bool] = lambda: False) -> Reply:
        """Waits for a whole utterance so that "okay" cannot be picked out of a longer sentence."""
        recognizers = self._recognizers(ENGLISH_GRAMMAR)
        voiced = False
        reply = Reply.SILENCE
        for block in blocks:
            if should_abort():
                return Reply.SILENCE
            samples = np.frombuffer(block, dtype=np.int16)
            voiced = voiced or (samples.size > 0 and float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) > VOICE_RMS)
            for recognizer in recognizers:
                if recognizer.AcceptWaveform(block):
                    reply = self._merge(reply, classify_transcript(self._words(recognizer.Result())))
            if reply in (Reply.HELP, Reply.EMERGENCY):
                return reply
        for recognizer in recognizers:
            reply = self._merge(reply, classify_transcript(self._words(recognizer.FinalResult())))
        if reply is Reply.SILENCE and voiced:
            return Reply.UNCLEAR
        return reply

    @staticmethod
    def _merge(current: Reply, new: Reply) -> Reply:
        """Help outranks okay, because the other language's model may mishear either one."""
        order = [Reply.EMERGENCY, Reply.HELP, Reply.OK, Reply.UNCLEAR, Reply.SILENCE]
        return min(current, new, key=order.index)

    def wait_for_trigger(self, blocks: Iterable[bytes]) -> tuple[str, str] | None:
        """Waits for a help word or the assistant's name, and returns ("help" or "wake", rest of the sentence)."""
        recognizers = self._recognizers(ENGLISH_WAKE_GRAMMAR)
        for block in blocks:
            for recognizer in recognizers:
                if not recognizer.AcceptWaveform(block):
                    continue
                words = self._words(recognizer.Result())
                confident = [word for word, conf in words if conf >= MIN_HELP_CONFIDENCE]
                if any(word in WAKE_WORDS for word in confident):
                    return "help", ""
                for index, (word, conf) in enumerate(words):
                    if word in WAKE_NAMES and conf >= MIN_CONFIDENCE:
                        return "wake", " ".join(w for w, _ in words[index + 1:] if w != "[unk]")
        return None

    def transcribe_utterance(self, blocks: Iterable[bytes], should_abort: Callable[[], bool] = lambda: False) -> list[Transcript]:
        """Decodes one sentence freely in every language, for the assistant."""
        recognizers = {language: self._recognizer(model, SAMPLE_RATE) for language, model in self._models.items()}
        for recognizer in recognizers.values():
            recognizer.SetWords(True)
        finished: dict[str, str] = {}
        grace = 0
        for block in blocks:
            if should_abort():
                return []
            for language, recognizer in recognizers.items():
                if language not in finished and recognizer.AcceptWaveform(block):
                    result = recognizer.Result()
                    if json.loads(result).get("text"):
                        finished[language] = result
            if finished:
                grace += 1
                if len(finished) == len(recognizers) or grace > 12:
                    break
        output = []
        for language, recognizer in recognizers.items():
            result = json.loads(finished.get(language) or recognizer.FinalResult())
            words = result.get("result", [])
            if result.get("text"):
                confidence = sum(word.get("conf", 0.0) for word in words) / max(len(words), 1)
                output.append(Transcript(language, result["text"], confidence))
        return output

    def transcribe_confirmation(self, blocks: Iterable[bytes], should_abort: Callable[[], bool] = lambda: False) -> Confirmation:
        recognizers = self._recognizers(ENGLISH_CONFIRM_GRAMMAR)
        outcome = Confirmation.UNCLEAR
        for block in blocks:
            if should_abort():
                return Confirmation.UNCLEAR
            for recognizer in recognizers:
                if recognizer.AcceptWaveform(block):
                    outcome = self._stronger(outcome, classify_confirmation(self._words(recognizer.Result())))
            if outcome is not Confirmation.UNCLEAR:
                return outcome
        for recognizer in recognizers:
            outcome = self._stronger(outcome, classify_confirmation(self._words(recognizer.FinalResult())))
        return outcome

    @staticmethod
    def _stronger(current: Confirmation, new: Confirmation) -> Confirmation:
        """No outranks yes, so that a mishearing leads to nothing being sent."""
        return min(current, new, key=[Confirmation.NO, Confirmation.YES, Confirmation.UNCLEAR].index)


def microphone_blocks(seconds: float | None, stop: threading.Event | None = None) -> Iterable[bytes]:
    command = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(SAMPLE_RATE), "-"]
    deadline = None if seconds is None else time.monotonic() + seconds
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    received = False
    try:
        while (deadline is None or time.monotonic() < deadline) and not (stop is not None and stop.is_set()):
            block = process.stdout.read(BLOCK_BYTES)
            if not block:
                raise OSError(f"Microphone stopped: {process.stderr.read(300).decode(errors='replace').strip()}")
            received = True
            yield block
    finally:
        process.terminate()
        process.wait(timeout=5)
    if not received:
        raise OSError("Microphone produced no audio")


class LocalVoice:
    """The real speaker and microphone."""

    def __init__(self, languages: Iterable[str], person: str, emergency: str):
        self.speaker = Speaker(languages, person, emergency)
        self.listener = Listener(languages)

    def say(self, key: str, **values: str) -> None:
        self.speaker.say(key, **values)

    def say_text(self, language: str, text: str) -> None:
        self.speaker.say_text(language, text)

    def listen(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> Reply:
        return self.listener.transcribe_reply(microphone_blocks(seconds), should_abort)

    def hear(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> list[Transcript]:
        return self.listener.transcribe_utterance(microphone_blocks(seconds), should_abort)

    def confirm(self, seconds: float, should_abort: Callable[[], bool] = lambda: False) -> Confirmation:
        return self.listener.transcribe_confirmation(microphone_blocks(seconds), should_abort)
