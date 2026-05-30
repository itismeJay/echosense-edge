# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup (first-time only)

Run once from the repo root to install dependencies and download all model files:

```bash
# System package for audio (already installed on Pi OS)
sudo apt-get install -y python3-pyaudio

# Virtual environment — inherits numpy and requests from system Python
python3 -m venv --system-site-packages echosense-env
echosense-env/bin/pip install vosk ai-edge-litert

# YAMNet model and class map (fast, ~4 MB total)
wget -O yamnet_class_map.csv \
  "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
wget -O yamnet.tflite \
  "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"

# Vosk speech models (Filipino 329 MB, English 41 MB — slow on first run)
wget "https://alphacephei.com/vosk/models/vosk-model-tl-ph-generic-0.6.zip"
unzip vosk-model-tl-ph-generic-0.6.zip && rm vosk-model-tl-ph-generic-0.6.zip

wget "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
unzip vosk-model-small-en-us-0.15.zip && rm vosk-model-small-en-us-0.15.zip
```

All model files are `.gitignore`'d — they must be re-downloaded on a fresh clone.

## Running the system

- Entry point: `echosense-env/bin/python3 main.py`. Run from the repo root — `config.py` uses absolute paths pinned to `/home/echosense/echosense-edge/...` and the imports expect the CWD to be the repo root.
- Backend connectivity smoke-test (no hardware needed): `echosense-env/bin/python3 test_backend.py` — sends a synthetic `high` alert and checks the `/health` endpoint.
- No build system, no tests, no linter. There are no other commands.
- Runtime libs listed in `requirements.txt`: `pyaudio`, `numpy`, `ai_edge_litert`, `vosk`, `requests`.
- Hardware target: Raspberry Pi with a USB microphone. `audio/capture.py` auto-selects by name priority: **ReSpeaker/Seeed** → **EMEET/M0** → **any USB input device** → system default.

## Pipeline architecture

`main.py` runs a single infinite loop. Every iteration captures a 1-second, 16 kHz mono int16 chunk and pushes it through this pipeline:

```
capture_audio ─┬─► run_yamnet               ─► (class, score)
               └─► process_audio_chunk      ─► (has_profanity, detected_words)
                         │                           │
                         └───────────┬───────────────┘
                                     ▼
                       AggressionDetector.process  (duration gate + cooldown)
                                     │
                                     ▼
                             send_alert (HTTP POST)
```

Both branches run **sequentially** on the same chunk. `capture_audio` returns both `audio_np` (float-compatible int16 array, used by YAMNet) and `audio_bytes` (raw PCM, used by Vosk).

Non-obvious behaviors that span multiple files:

- **Aggression gating** (`detection/aggression.py`): a sound must be classified aggressive *continuously* for `DURATION_THRESHOLD` (2.0 s) before an alert fires, and alerts are further rate-limited by a 10-second `alert_cooldown`. After an alert fires, `aggressive_start_time` resets to `None` — the next alert requires a fresh 2.0 s run from scratch.
- **Profanity force-promotion** (`detection/aggression.py`): profanity detection force-sets `is_aggressive=True` and floors the YAMNet score to 0.65, bypassing the YAMNet class check entirely.
- **YAMNet aggressive-class match** (`model/yamnet_infer.py`): the `AGGRESSIVE_CLASSES` list is matched by case-insensitive substring against the top-1 class. No soft voting across classes.
- **Vosk profanity is utterance-level** (`model/vosk_detect.py`): `AcceptWaveform` must return `True` (complete utterance recognized) for profanity to be checked. Partial/in-progress speech is silently ignored each iteration.
- **Profanity lists** (`model/blacklist.py`): English, Filipino, and Bisaya lists are merged into one flat `ALL_BLACKLIST` and matched by case-insensitive substring.
- **Vosk models load at import time** (`model/vosk_detect.py`, top-level): importing this module eagerly constructs both the Filipino and English recognizers — slow and memory-hungry. Do not import it from scripts that don't need ASR.
- **Severity buckets** (`detection/thresholds.py`): profanity adds `PROFANITY_BOOST` (0.15) to confidence. Severity is `high` ≥ 0.80, `medium` ≥ 0.60, else `low`.
- **Alert payload** (`sender/http_client.py`): POSTs `{severity, confidence, duration, location}` to `{API_URL}/alerts` with 3 retries and a 3-second backoff between attempts.

## Configuration caveats (`config.py`)

- Paths are absolute and hard-coded to `/home/echosense/echosense-edge/...`. The project is not location-portable; moving it requires editing `config.py`.
- `LOCATION` (`"Grade 6 Classroom"`) is read at runtime and included in every alert payload — edit it to change the reported sensor location.
- Backend URL `https://echosense-backend-75h3.onrender.com` lives in a separate repo (not vendored here).
- `YAMNET_CONFIDENCE_THRESHOLD` and `DURATION_THRESHOLD` in `config.py` are **duplicates that the runtime does not read** — the authoritative copies are `YAMNET_THRESHOLD` and `DURATION_THRESHOLD` in `detection/thresholds.py`. If asked to tune thresholds, edit the ones in `detection/thresholds.py` (and flag the duplication).

## Known landmines

- `yamnet_class_map.csv` is **0 bytes** on disk. `load_class_names` will silently return an empty list, and the first YAMNet inference will `IndexError` on `class_names[top_index]`. To bring the system up, populate it from the official YAMNet class map — the file must be a 3-column CSV (`index,mid,display_name`); `load_class_names` reads column index 2 (the display name).
- The repo contains both `vosk-english.zip` / `vosk-filipino.zip` and their extracted directories. The zips are install artifacts, not used at runtime.
- There are no tests; correctness can only be validated by running `main.py` on the target hardware or `test_backend.py` for the sender path only.
