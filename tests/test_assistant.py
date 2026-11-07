import json

import pytest

from elder_help.assistant import (
    Assistant, ChatClient, Confirmation, IntentKind, PlaceMemory, Transcript,
    choose_transcript, classify_confirmation, clean_chat_reply, find_contact, parse_intent,
)
from elder_help.config import ChatConfig, Contact

K = IntentKind
RAHUL = Contact("Rahul", "+911", aliases=("राहुल", "बेटा", "son"))


# These sentences are what Vosk actually returned in testing, including its misspellings.
@pytest.mark.parametrize("text,item", [
    ("मेरी दवाई कहा है", "medicine"),
    ("मेरा चश्मा कहां रखा है", "glasses"),
    ("where are my medicines", "medicine"),
    ("where is my medicine", "medicine"),
    ("where did i keep my glasses", "glasses"),
    ("we did i keep my glasses", "glasses"),
])
def test_asking_where_something_is(text, item):
    intent = parse_intent(text)
    assert intent.kind is K.ASK_PLACE and intent.item == item


@pytest.mark.parametrize("text,item,place", [
    ("मेरी दबाई रसोई की अलमारी में रखी है", "medicine", "रसोई की अलमारी में"),
    ("मैंने चाबी मेज पर रखी है", "keys", "मेज पर"),
    ("my medicines are in the kitchen cupboard", "medicine", "in the kitchen cupboard"),
    ("i kept my glasses on the table", "glasses", "on the table"),
])
def test_saying_where_something_is(text, item, place):
    intent = parse_intent(text)
    assert (intent.kind, intent.item, intent.place) == (K.REMEMBER_PLACE, item, place)


@pytest.mark.parametrize("text,who,body", [
    ("राहुल को मैसेज करो की पानी ले आओ", "राहुल", "पानी ले आओ"),
    ("राहुल को बोलो कि मेरी दवाई ले आए", "राहुल", "मेरी दवाई ले आए"),
    ("text rahul to bring me some water", "rahul", "bring me some water"),
    ("message my son to come home", "son", "come home"),
    ("राहुल को मैसेज करो", "राहुल", ""),
])
def test_sending_a_message(text, who, body):
    intent = parse_intent(text)
    assert (intent.kind, intent.who, intent.body) == (K.SEND_MESSAGE, who, body)


@pytest.mark.parametrize("text", [
    "एक सौ बारह पर फोन करो", "एक एक दो को कॉल करो", "एम्बुलेंस बुलाओ",
    "call one one two", "call an ambulance", "राहुल को बोलो कि एम्बुलेंस बुलाए",
])
def test_asking_for_emergency_services(text):
    assert parse_intent(text).kind is K.CALL_EMERGENCY


@pytest.mark.parametrize("text", ["मेरे परिवार को बताओ", "सबको बता दो", "alert my family", "alerts my family", "call my family"])
def test_alerting_family(text):
    assert parse_intent(text).kind is K.ALERT_FAMILY


@pytest.mark.parametrize("text,kind", [
    ("मदद", K.HELP), ("मुझे मदद चाहिए", K.HELP), ("help me", K.HELP),
    ("कितने बजे हैं", K.TIME), ("what time is it", K.TIME),
    ("धन्यवाद", K.GOODBYE), ("thank you", K.GOODBYE),
    ("आप कैसे हैं", K.CHAT), ("how are you", K.CHAT),
    ("आज बाजार में बहुत भीड़ थी", K.CHAT), ("the match starts at seven tonight", K.CHAT),
])
def test_other_requests_and_plain_conversation(text, kind):
    assert parse_intent(text).kind is kind


@pytest.mark.parametrize("text", [
    "मैं गिर गई हूं", "मेरे सीने में दर्द है", "मुझे चक्कर आ रहे हैं", "मुझे सांस नहीं आ रही",
    "i fell", "my chest hurts", "i feel dizzy", "i cannot breathe",
])
def test_sounding_unwell_is_never_treated_as_chat(text):
    assert parse_intent(text).kind is K.DISTRESS


def test_distress_starts_a_safety_check_without_asking_the_chat_model():
    chat = FakeChat()
    assistant, _, _, alerts = make(Voice(["मेरे सीने में दर्द है"]), chat)
    assistant.converse()
    assert alerts == ["distress"] and not chat.calls


@pytest.mark.parametrize("text", ["tell me about your day", "tell me a story", "मुझे कहानी बताओ", "मुझको बताओ"])
def test_asking_to_be_told_something_is_conversation(text):
    assert parse_intent(text).kind is K.CHAT


def test_reminiscing_reaches_the_chat_model():
    chat = FakeChat()
    make(Voice(["मुझे अपने बचपन की याद आ रही है"]), chat)[0].converse()
    assert len(chat.calls) == 1


def test_a_message_asking_for_help_is_a_message_not_an_alert():
    assert parse_intent("राहुल को बोलो कि मेरी मदद करे").kind is K.SEND_MESSAGE


def words(*items, conf=0.95):
    return [(word, conf) for word in items]


def test_confirmation_yes_must_be_the_whole_sentence():
    for reply in (["ठीक", "है"], ["okay"], ["yes"], ["भेज", "दो"], ["send", "it"], ["कर", "दो"]):
        assert classify_confirmation(words(*reply)) is Confirmation.YES
    for chatter in (["मौसम", "ठीक", "है"], ["[unk]", "yes"], ["[unk]"], []):
        assert classify_confirmation(words(*chatter)) is Confirmation.UNCLEAR


def test_confirmation_no_wins_and_low_confidence_is_not_yes():
    for reply in (["रहने", "दो"], ["नहीं"], ["cancel"], ["no"], ["नहीं", "ठीक", "है"]):
        assert classify_confirmation(words(*reply)) is Confirmation.NO
    assert classify_confirmation(words("okay", conf=0.4)) is Confirmation.UNCLEAR


def test_memory_survives_a_restart_and_a_new_place_replaces_the_old(tmp_path):
    path = tmp_path / "memory.json"
    memory = PlaceMemory(path, {"medicine": {"hi": "मेज पर", "en": "on the table"}})
    assert memory.recall("medicine", "en") == ("on the table", "en")
    memory.remember("medicine", "रसोई में", "hi")
    again = PlaceMemory(path, {"medicine": {"hi": "मेज पर", "en": "on the table"}})
    assert again.recall("medicine", "hi") == ("रसोई में", "hi")
    assert again.recall("medicine", "en") == ("रसोई में", "hi")  # The stale English place is gone.
    assert again.recall("keys", "hi") is None


def test_find_contact_uses_names_and_aliases():
    for who in ("rahul", "राहुल", "बेटा", "son"):
        assert find_contact(who, (RAHUL,)) == RAHUL
    assert find_contact("priya", (RAHUL,)) is None
    assert find_contact("", (RAHUL,)) is None


def test_choose_transcript_prefers_the_language_that_made_sense():
    hindi = Transcript("hi", "अमर ने", 0.9)
    english = Transcript("en", "where is my medicine", 0.7)
    chosen, intent = choose_transcript([hindi, english])
    assert chosen is english and intent.kind is K.ASK_PLACE
    assert choose_transcript([]) is None


def test_chat_replies_are_cleaned_and_false_action_claims_are_dropped():
    assert clean_chat_reply("**नमस्ते!** 😊\n\nआप कैसे हैं?") == "नमस्ते! आप कैसे हैं?"
    assert clean_chat_reply("I have called your son.") is None
    assert clean_chat_reply("मैंने राहुल को फोन कर दिया है।") is None
    # These came from the real model when it was pushed to claim things it had not done.
    assert clean_chat_reply("नमस्ते दादी जी, मैंने एम्बुलेंस बुला ली है। आप आराम करें।") is None
    assert clean_chat_reply("नहीं, अभी तक नहीं किया। चलो, हम फिर से कोशिश करते हैं।") is None
    assert clean_chat_reply("I haven't called your son, did you want me to?") is None
    assert clean_chat_reply("Help is on the way, the ambulance is coming.") is None
    assert clean_chat_reply("Yes, my dear, help is on its way.") is None
    assert clean_chat_reply("Let's see if someone can arrive soon.") is None
    assert clean_chat_reply("चिंता मत कीजिए, वे आ रहे हैं।") is None
    assert clean_chat_reply("Let's see if we can find someone to assist you.") is None
    assert clean_chat_reply("नहीं, अभी तक नहीं किया। मैं जल्द ही कर दूंगा।") is None


def test_chat_replies_giving_doses_are_dropped():
    assert clean_chat_reply("You can take 500 mg of it.") is None
    assert clean_chat_reply("आप 2 गोली ले लीजिए।") is None


def test_ordinary_kind_replies_are_kept():
    for reply in ("कृपया डॉक्टर से बात करें, दादी जी।", "It is lovely to be talking with you.", "मैं यहीं हूँ।"):
        assert clean_chat_reply(reply) == reply
    assert clean_chat_reply("  ") is None


class Voice:
    def __init__(self, sentences=(), confirmations=()):
        self.sentences = list(sentences)
        self.confirmations = list(confirmations)
        self.spoken = []

    def say_text(self, language, text):
        self.spoken.append((language, text))

    def hear(self, _seconds, _abort=lambda: False):
        if not self.sentences:
            return []
        item = self.sentences.pop(0)
        if isinstance(item, Transcript):
            return [item]
        language = "hi" if any("ऀ" <= c <= "ॿ" for c in item) else "en"
        return [Transcript(language, item, 0.9)]

    def confirm(self, _seconds, _abort=lambda: False):
        return self.confirmations.pop(0) if self.confirmations else Confirmation.UNCLEAR


class FakeChat:
    def __init__(self, answer="बहुत अच्छा।"):
        self.answer = answer
        self.calls = []

    def reply(self, text, language, history):
        self.calls.append((text, language, list(history)))
        return self.answer


def make(voice, chat=None, simulated=False, send_ok=True, places=None):
    sent, emergencies, alerts = [], [], []

    def send(phone, body):
        sent.append((phone, body))
        return send_ok

    assistant = Assistant(
        voice, PlaceMemory(None, places), (RAHUL,), "Dadi", send, simulated,
        emergencies.append, alerts.append, lambda: alerts.append("distress"), chat,
    )
    return assistant, sent, emergencies, alerts


def said(voice):
    return " ".join(text for _, text in voice.spoken)


def test_tells_where_the_medicine_is_kept():
    voice = Voice(["मेरी दवाई कहाँ है"])
    assistant, *_ = make(voice, places={"medicine": {"hi": "रसोई की अलमारी में"}})
    assistant.converse()
    assert "आपकी दवाई रसोई की अलमारी में है" in said(voice)


def test_remembers_then_recalls_in_one_conversation():
    voice = Voice(["मेरी दवाई रसोई की अलमारी में रखी है", "मेरी दवाई कहाँ है"])
    assistant, *_ = make(voice)
    assistant.converse()
    assert voice.spoken[-1][1] == "आपकी दवाई रसोई की अलमारी में है।"


def test_admits_when_it_does_not_know_and_never_guesses():
    chat = FakeChat("शायद मेज पर होगी।")
    voice = Voice(["मेरी दवाई कहाँ है"])
    assistant, *_ = make(voice, chat)
    assistant.converse()
    assert "मुझे नहीं पता" in said(voice)
    assert not chat.calls


def test_message_is_read_back_and_sent_only_after_yes():
    voice = Voice(["राहुल को बोलो कि पानी ले आओ"], [Confirmation.YES])
    assistant, sent, *_ = make(voice)
    assistant.converse()
    assert "पानी ले आओ" in said(voice)
    assert sent == [("+911", "Elder-Help: Dadi asks: पानी ले आओ")]
    assert "संदेश भेज दिया" in said(voice)


@pytest.mark.parametrize("answer", [Confirmation.NO, Confirmation.UNCLEAR])
def test_message_is_not_sent_without_a_clear_yes(answer):
    voice = Voice(["text rahul to bring water"], [answer])
    assistant, sent, *_ = make(voice)
    assistant.converse()
    assert not sent
    assert "cancelled" in said(voice)


def test_asks_what_to_say_when_the_message_is_missing():
    voice = Voice(["राहुल को मैसेज करो", "दूध ले आओ"], [Confirmation.YES])
    assistant, sent, *_ = make(voice)
    assistant.converse()
    assert sent[0][1].endswith("दूध ले आओ")


def test_unknown_contact_is_not_guessed():
    voice = Voice(["text priya to bring water"])
    assistant, sent, *_ = make(voice)
    assistant.converse()
    assert not sent and "do not have a number" in said(voice)


def test_practice_mode_and_failed_sends_are_said_honestly():
    voice = Voice(["text rahul to bring water"], [Confirmation.YES])
    assistant, *_ = make(voice, simulated=True)
    assistant.converse()
    assert "not really sent" in said(voice) and "Message sent" not in said(voice)
    voice = Voice(["text rahul to bring water"], [Confirmation.YES])
    assistant, *_ = make(voice, send_ok=False)
    assistant.converse()
    assert "could not be sent" in said(voice)


def test_chat_model_is_never_asked_about_actions():
    chat = FakeChat()
    for sentence in ["एम्बुलेंस बुलाओ", "मेरे परिवार को बताओ", "मदद"]:
        assistant, _, emergencies, alerts = make(Voice([sentence]), chat)
        assistant.converse()
        assert emergencies or alerts
    assert not chat.calls


@pytest.mark.parametrize("sentence", [
    "did you call my son", "is someone coming", "क्या आपने राहुल को फोन कर दिया", "क्या कोई आ रहा है",
])
def test_questions_about_calls_or_rescue_never_reach_the_model(sentence):
    chat = FakeChat("Yes, they are on the way.")
    voice = Voice([sentence])
    make(voice, chat)[0].converse()
    assert not chat.calls
    assert "on the way" not in said(voice)


def test_chat_keeps_short_context_and_falls_back_when_unavailable():
    chat = FakeChat()
    voice = Voice(["आप कैसे हैं", "आज मौसम अच्छा है"])
    make(voice, chat)[0].converse()
    assert len(chat.calls) == 2 and chat.calls[1][2][0]["content"] == "आप कैसे हैं"
    voice = Voice(["आप कैसे हैं"])
    make(voice, FakeChat(None))[0].converse()
    assert "संदेश भेज सकता हूँ" in said(voice)


def test_mumbling_is_asked_again_then_the_conversation_ends():
    voice = Voice([Transcript("hi", "था", 0.3), Transcript("hi", "था", 0.3), "मदद"])
    assistant, _, _, alerts = make(voice)
    assistant.converse()
    assert not alerts and len(voice.sentences) == 1


def test_silence_ends_the_conversation():
    voice = Voice()
    make(voice)[0].converse()
    assert len(voice.spoken) == 1


def test_chat_client_sends_rules_and_handles_being_offline():
    client = ChatClient(ChatConfig(url="http://127.0.0.1:9", timeout_s=1), "Dadi")
    assert client.reply("hello", "en", []) is None
