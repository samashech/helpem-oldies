#!/usr/bin/env bash
# Downloads the vision, speech recognition and voice models into models/.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models/piper
cd models

fetch() {
  [[ -s "$1" ]] && return 0
  echo "Downloading $1"
  curl -fL --retry 3 -o "$1.part" "$2"
  mv "$1.part" "$1"
}

fetch_zip() {
  [[ -d "$1" ]] && return 0
  fetch "$1.zip" "https://alphacephei.com/vosk/models/$1.zip"
  unzip -q "$1.zip"
  rm "$1.zip"
}

MEDIAPIPE=https://storage.googleapis.com/mediapipe-models
fetch pose_landmarker_lite.task "$MEDIAPIPE/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
fetch pose_landmarker_full.task "$MEDIAPIPE/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
fetch pose_landmarker_heavy.task "$MEDIAPIPE/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
fetch efficientdet_lite0.tflite "$MEDIAPIPE/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"

fetch_zip vosk-model-small-hi-0.22
fetch_zip vosk-model-small-en-us-0.15

PIPER=https://huggingface.co/rhasspy/piper-voices/resolve/main
for voice in hi/hi_IN/pratham/medium/hi_IN-pratham-medium en/en_GB/alba/medium/en_GB-alba-medium; do
  name=$(basename "$voice")
  fetch "piper/$name.onnx" "$PIPER/$voice.onnx"
  fetch "piper/$name.onnx.json" "$PIPER/$voice.onnx.json"
done

echo "Models are ready."
