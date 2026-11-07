# Elder Help

A local camera companion for older adults. It watches posture through a laptop or PC camera. If someone is lying down or may have fallen, it asks in Hindi and English whether they are okay. If they don't answer, it texts and calls their emergency contacts.

**This is a prototype, not a medical device or a certified alarm.** Don't rely on it as anyone's only safety measure.

## What it does

| Situation | Response |
| --- | --- |
| Sudden drop in body height | Asks after 5 seconds |
| Sudden drop, then out of view | Asks after 15 seconds |
| Lying down outside a bed zone, including after a slow collapse | Asks after 20 seconds |
| Lying in a bed zone | Asks after 30 minutes in the daytime, not at all during sleep hours |
| Several failed attempts to get up | Asks whether they need help |
| Says "help", "मदद" or "बचाओ" | Asks to confirm, then escalates |
| Awake and in view | Routine check-in every 3 hours |
| Camera stops working | Texts contacts that monitoring has stopped |

It also labels walking, running, standing and sitting, and notes objects in use, such as a cup or phone.

## Talking to it

Say **"सहायक"** or **"helper"**, wait for the reply, then ask. You can also say it in one breath in Hindi: "सहायक, मेरी दवाई कहाँ है".

| Say | What happens |
| --- | --- |
| "मेरी दवाई रसोई की अलमारी में रखी है" / "my medicines are in the kitchen cupboard" | Remembers the place |
| "मेरी दवाई कहाँ है" / "where are my medicines" | Says the place, or admits it doesn't know |
| "राहुल को बोलो कि पानी ले आओ" / "text Rahul to bring me some water" | Reads the text back, and sends it only after "ठीक है" or "okay" |
| "मेरे परिवार को बताओ" / "alert my family" | Texts and calls contacts straight away |
| "एक सौ बारह पर फ़ोन करो" / "call an ambulance" | Asks to confirm, calls 112, then alerts contacts |
| "मेरे सीने में दर्द है" / "my chest hurts" | Starts a safety check, and alerts contacts if there's no clear "okay" |
| "कितने बजे हैं" / "what time is it" | Tells the time |
| Anything else | Chats, using a local model |

With `medicine_times` set, it also says when it's time and where the medicine is kept. It never suggests doses.

**How 112 works**

- It's only dialled when the person asks and then confirms aloud. Silence alone alerts contacts and never dials 112.
- If they ask but can't confirm, contacts are told that 112 has **not** been called.
- The call plays a recorded message with `address`, so it can't hold a two-way conversation. It tells the person if the call failed.
- **Never test this against the real 112.** Put your own number in `emergency_number` while testing.

**The chat model cannot do anything.** Texts, alerts and calls only come from fixed rules. Replies that mention help, calling, messaging, emergency services or doses are thrown away, because a chat model saying "I've called an ambulance" could stop someone from getting real help. Safety check-ins interrupt any conversation.

**How replies are handled**

- **"Okay":** it stands down, then asks again every 10 minutes while they are still lying down.
- **"Help" or "no":** it texts and calls contacts straight away.
- **Silence or unclear speech:** it asks again, warns that it is about to call, then calls. After a routine or getting-up check, contacts get a text instead.
- **Nobody answers the phone:** it tells the person to call 112, and tries again 2 minutes later.

Answering machines don't count as reaching someone.

## Setup

Requires Linux with `arecord` and `aplay`, plus Python 3.11 or 3.12.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
./scripts/download_models.sh        # About 280 MB
ollama pull gemma3:4b               # Optional, for free conversation (3.4 GB). Needs `ollama serve` running.
cp config.example.toml config.toml  # Then fill in names and contacts
.venv/bin/python -m elder_help doctor
```

## Running it

```bash
.venv/bin/python -m elder_help say           # Plays the question out loud
.venv/bin/python -m elder_help hear          # Listens once and prints how it understood you
.venv/bin/python -m elder_help talk          # Has one conversation, using the speaker and mic
.venv/bin/python -m elder_help talk --text   # Same conversation, typed
.venv/bin/python -m elder_help run           # Starts monitoring
```

The dashboard is at <http://127.0.0.1:8765>. It is only reachable from this computer.

To replay a video without using the speaker, mic or camera:

```bash
.venv/bin/python -m elder_help run --video clip.mp4 --no-audio --reply silence
```

## Turning on real alerts

Alerts start in practice mode, which logs messages without sending anything. To send them for real, set `alerts.live = true` and add Twilio credentials.

- **112 may not work through Twilio.** Twilio limits emergency calling and it may not connect in India, so check this before relying on it, and keep a phone within reach. Contacts are always told to call 112 themselves if needed.
- **India:** SMS to Indian numbers can require sender registration, so check Twilio's current rules. Trial accounts can only reach verified numbers.

## Testing the camera

Live mode needs the consent setting, so create a config first if you don't have one. Alerts stay in practice mode, so nothing is really sent.

```bash
cp config.example.toml config.toml
sed -i 's/monitored_person_consented = false/monitored_person_consented = true/' config.toml
.venv/bin/python -m elder_help run --no-audio
```

Open <http://127.0.0.1:8765> and stand far enough back that your whole body is in frame. Remove `--no-audio` to hear the check-ins too.

## Privacy and consent

- Live camera mode won't start until `monitored_person_consented = true`. Get the person's agreement first.
- Video and audio are processed on this machine and are not saved, and chat runs locally through Ollama. `events.jsonl` keeps a text-only history without what was said, and `memory.json` keeps where things are.
- The microphone stays open while help-word listening is on. Set `listen_for_help_words = false` to turn that off.
- Phone numbers and messages are only sent to Twilio, and only in live mode.

## Known limits

- **Thresholds are unvalidated.** Timings and angles are hand-chosen starting values. They live in `Thresholds` and `config.toml`, and need testing with the actual person and room.
- **The camera must see them.** A dark room, another room, or a blanket can hide a fall. Lying with feet pointing at the camera can look upright.
- **Desk webcams see less.** With only head and shoulders in view, it reports "upright" rather than standing, walking or running, and leg-based checks don't work.
- **One person at a time.** A visitor in frame can be tracked instead.
- **Some English is harder to catch.** Hindi commands were recognised far more reliably in testing, and "alert my family" was often missed in English. Saying "help" works instead. Names may need adding as `aliases`.
- **Chat can be wrong.** It is a small local model. It is company, not a source of facts or health advice.
- **Speech was tested on synthetic voices.** Older voices, dialects and room noise will do worse. A bare "हाँ" is unreliable, so prompts ask for "ठीक हूँ". TV dialogue could occasionally pass as "okay", which delays the next question by up to 10 minutes.
- **The computer must stay on, awake and online.** Calls also depend on the internet connection.

## Tests

```bash
.venv/bin/python -m pytest
```

## GPU

Chat runs on the NVIDIA GPU through Ollama whenever one is available. Pose detection also uses it by default, because MediaPipe otherwise picks the Intel chip on dual-GPU laptops. On the test laptop that took 2.5 ms per frame on NVIDIA, 8.8 ms on Intel and 7.8 ms on CPU. Any of these is fast enough, so set `prefer_nvidia = false` or `vision_delegate = "CPU"` to save power on a machine that stays on all day.

## Troubleshooting

**The process dies with exit code 137.** Something has loaded PortAudio, for example `import sounddevice`. Under PipeWire's JACK layer this sets `RLIMIT_RTTIME` to 0, and the process is then killed when models load. `vision.py` stops MediaPipe from importing it, so keep it out of this project.
