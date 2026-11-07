from elder_help.safety import Reason, Reply
from elder_help.voice import PROMPTS, classify_transcript, prompt_key, split_by_script


def words(*items, conf=0.95):
    return [(word, conf) for word in items]


def test_okay_replies_in_both_languages():
    for reply in (["yes"], ["i", "am", "okay"], ["ठीक", "हूँ"], ["हाँ"]):
        assert classify_transcript(words(*reply)) is Reply.OK


def test_help_replies_in_both_languages():
    for reply in (["help"], ["no"], ["मदद"], ["बचाओ"], ["दर्द"], ["नहीं"]):
        assert classify_transcript(words(*reply)) is Reply.HELP


def test_negated_okay_means_help():
    assert classify_transcript(words("not", "okay")) is Reply.HELP
    assert classify_transcript(words("नहीं", "ठीक")) is Reply.HELP


def test_low_confidence_is_never_taken_as_okay():
    assert classify_transcript(words("okay", conf=0.4)) is Reply.UNCLEAR


def test_low_confidence_help_still_counts():
    assert classify_transcript(words("बचाओ", conf=0.4)) is Reply.HELP


def test_okay_words_inside_longer_speech_do_not_count():
    assert classify_transcript(words("मौसम", "ठीक", "है", "और", "बारिश")) is Reply.UNCLEAR
    assert classify_transcript(words("[unk]", "okay")) is Reply.UNCLEAR


def test_help_words_count_anywhere_in_a_sentence():
    assert classify_transcript(words("मुझे", "मदद", "चाहिए")) is Reply.HELP
    assert classify_transcript(words("[unk]", "help", "me")) is Reply.HELP


def test_unknown_only_and_empty():
    assert classify_transcript(words("[unk]")) is Reply.UNCLEAR
    assert classify_transcript([]) is Reply.SILENCE


def test_asking_for_112_or_an_ambulance_in_a_reply():
    for reply in (["एक", "सौ", "बारह"], ["एम्बुलेंस"], ["one", "one", "two"], ["call", "ambulance"]):
        assert classify_transcript(words(*reply)) is Reply.EMERGENCY


def test_every_prompt_exists_in_both_languages():
    assert all(set(texts) == {"hi", "en"} for texts in PROMPTS.values())


def test_prompt_selection():
    assert prompt_key(Reason.LYING, 0) == "ask"
    assert prompt_key(Reason.STRUGGLE, 0) == "ask_struggle"
    assert prompt_key(Reason.LYING, 1) == "ask_again"


def test_mixed_text_goes_to_the_voice_that_can_read_it():
    assert split_by_script("I am doing well, दादी जी. How are you?", "en") == [
        ("en", "I am doing well,"), ("hi", "दादी जी."), ("en", "How are you?"),
    ]
    assert split_by_script("Rahul को यह भेजूँ", "hi") == [("en", "Rahul"), ("hi", "को यह भेजूँ")]


def test_numbers_stay_with_the_surrounding_words():
    assert split_by_script("क्या मैं 112 पर फ़ोन करूँ", "hi") == [("hi", "क्या मैं 112 पर फ़ोन करूँ")]
    assert split_by_script("112", "hi") == [("hi", "112")]
    assert split_by_script("It is 9 05.", "en") == [("en", "It is 9 05.")]
