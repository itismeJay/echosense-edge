# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the system

- Entry point: `python3 main.py`. Run from the repo root — `config.py` uses absolute paths pinned to `/home/echosense/echosense-edge/...` and the imports expect the CWD to be the repo root.
- No build system, no tests, no linter. There are no other commands.
- No dependency manifest exists. Runtime libs: `pyaudio`, `numpy`, `ai_edge_litert`, `vosk`, `requests`.
- Hardware target: Raspberry Pi with a Seeed ReSpeaker mic array. `audio/capture.py` auto-selects any input device whose name contains `respeaker` or `seeed`, falling back to the system default.

## Pipeline architecture

`main.py` runs a single infinite loop. Every iteration captures a 1-second, 16 kHz mono int16 chunk and pushes it through this pipeline:

```
capture_audio ─┬─► run_yamnet               ─► (class, score)
               └─► process_audio_chunk      ─► (has_profanity, detected_words)
                                │
                                ▼
                  AggressionDetector.process  (duration gate + cooldown)
                                │
                                ▼
                        send_alert (HTTP POST)
```

Non-obvious behaviors that span multiple files:

- **Aggression gating** (`detection/aggression.py`): a sound must be classified aggressive *continuously* for `DURATION_THRESHOLD` (2.0 s) before an alert fires, and alerts are further rate-limited by a 10-second `alert_cooldown`. Profanity detection force-sets `is_aggressive=True` and floors the YAMNet score to 0.65.
- **YAMNet aggressive-class match** (`model/yamnet_infer.py`): the `AGGRESSIVE_CLASSES` list is matched by case-insensitive substring against the top-1 class. No soft voting across classes.
- **Profanity lists** (`model/blacklist.py`): English, Filipino, and Bisaya lists are merged into one flat `ALL_BLACKLIST` and matched by case-insensitive substring.
- **Vosk models load at import time** (`model/vosk_detect.py`, top-level): importing this module eagerly constructs both the Filipino and English recognizers — slow and memory-hungry. Do not import it from scripts that don't need ASR.
- **Severity buckets** (`detection/thresholds.py`): profanity adds `PROFANITY_BOOST` (0.15) to confidence. Severity is `high` ≥ 0.80, `medium` ≥ 0.60, else `low`.
- **Alert payload** (`sender/http_client.py`): POSTs `{severity, confidence, duration, location}` to `{API_URL}/alerts` with 3 retries and a 3-second backoff between attempts.

## Configuration caveats (`config.py`)

- Paths are absolute and hard-coded to `/home/echosense/echosense-edge/...`. The project is not location-portable; moving it requires editing `config.py`.
- Backend URL `https://echosense-backend-75h3.onrender.com` lives in a separate repo (not vendored here).
- `YAMNET_CONFIDENCE_THRESHOLD` and `DURATION_THRESHOLD` in `config.py` are **duplicates that the runtime does not read** — the authoritative copies are `YAMNET_THRESHOLD` and `DURATION_THRESHOLD` in `detection/thresholds.py`. If asked to tune thresholds, edit the ones in `detection/thresholds.py` (and flag the duplication).

## Known landmines

- `yamnet_class_map.csv` is **0 bytes** on disk. `load_class_names` will silently return an empty list, and the first YAMNet inference will `IndexError` on `class_names[top_index]`. To bring the system up, populate the file from the official YAMNet class map.
- The repo contains both `vosk-english.zip` / `vosk-filipino.zip` and their extracted directories. The zips are install artifacts, not used at runtime.
- The repo is not a git repository (no `.git`). "What changed recently?" cannot be answered from history.
- There are no tests; correctness can only be validated by running `main.py` on the target hardware.
