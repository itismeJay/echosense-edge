# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## One codebase for Mac and Pi

The same commit runs on a Mac (development and demos) and on the Raspberry Pi (classroom). Never keep a separate copy per machine and never hard-code machine paths. Everything machine-specific (backend URL, device ID and key, microphone, model paths, data and log folders, network interface, hostname) comes from `edge.env` in the repo root or from the environment. `edge.env.example` documents each setting for both machines.

- `config.py` loads `edge.env` at import (`ECHOSENSE_ENV_FILE` picks another file, an empty value disables loading). Variables already exported by the shell, the demo scripts or systemd's `EnvironmentFile` always win.
- Relative paths resolve against the repo root, so the repo can live anywhere.
- `PLATFORM` is auto-detected (`macos`, `raspberrypi`, `linux`; `ECHOSENSE_PLATFORM` overrides). Pi-only features degrade to no-ops elsewhere: the ACT LED (`audio/led_indicator.py`) and the SSID lookup.
- `edge.env` holds the device key: it is gitignored, must never be committed or printed.

## Environment setup (first-time only)

From the repo root:

```bash
# Mac: Homebrew python@3.11 and portaudio. Venv folder: venv/
python3.11 -m venv venv && venv/bin/pip install -r requirements.txt

# Pi: sudo apt-get install -y python3-pyaudio. Venv folder: echosense-env/
python3 -m venv --system-site-packages echosense-env
echosense-env/bin/pip install -r requirements.txt

# YAMNet model and class map (~4 MB total, both machines)
wget -O yamnet_class_map.csv \
  "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
wget -O yamnet.tflite \
  "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"

cp edge.env.example edge.env && chmod 600 edge.env   # then fill in the device ID/key
```

The Whisper model (`ECHOSENSE_WHISPER_MODEL`, default `base`) is fetched by faster-whisper into the Hugging Face cache on first run. Model files, venvs, `edge.env`, `data/` and `logs/` are gitignored.

## Running the system

- Entry point: `venv/bin/python main.py` on the Mac, `echosense-env/bin/python3 main.py` on the Pi (systemd `echosense.service`, drop-ins in `deploy/`). Run from the repo root; the imports expect it as the CWD.
- Mac demo wrappers live outside the repo in `~/echosense-demo`: `edge-local.sh` (backend `http://localhost:8000`) and `edge-live.sh` (Render). Options: `--mic builtin|emeet`, `--device <code> --key-file <path>`, `--test-alert`.
- Tests: `venv/bin/python -m unittest discover -s tests -t .` (unittest, no pytest). There is no build step and no linter.
- `test_backend.py` contacts the configured backend; `test_tone.py` is a standalone synthetic-audio check.

## Pipeline architecture

Detection is the event-driven "AFTER" runtime started by `runtime.py` (details in `docs/after_detection_architecture.md`):

```
PyAudioSharedSource (the only mic reader, float32 16 kHz mono)
   ├─► RealtimeSTT + faster-whisper (en/tl) ─► TextDetector ─────┐
   ├─► YAMNet + tone (AcousticDetector, noise floor) ────────────┤
   └─► incident context audio ───────────────────────────────────┤
                                                                 ▼
                    EventCoordinator (incident window, fusion, severity)
                                                                 ▼
                    SQLite outbox ─► background delivery to {API_URL}/alerts/
```

Non-obvious behaviors that span multiple files:

- **Speech recognition is Whisper, not Vosk.** `model/whisper_stt.py` feeds RealtimeSTT (`use_microphone=False`) from the shared source. `model/language_restricted_engine.py` restricts language detection to English and Tagalog (Bisaya decodes as Tagalog). Whisper runs with `initial_prompt=None`: prompting with the monitored terms made Whisper hallucinate them. `model/vosk_detect.py` is legacy, unused at runtime, and loads both Vosk models at import, so never import it.
- **Word-gated alerts:** only a monitored word opens an incident. Sound alone is logged locally and never sent. Acoustic evidence (YAMNet exact-label allowlist plus the tone rule, judged against an adaptive noise floor) can only change severity when it is time-linked to the text.
- **Monitored terms:** `model/blacklist.py` holds the local English/Filipino/Bisaya lists. `model/monitored_terms.py` syncs the backend dictionary at startup and every 300 s (`[DICTIONARY] Synchronized N monitored terms`).
- **False-alarm filters** (`detection/false_alarm_filters.py`): decode-quality gate, hallucination and repeated-word checks, word-probability filter and a media-audio gate, each tunable or switchable through `ECHOSENSE_FILTER_*`.
- **Incident timing** (`config.py` `DetectionRuntimeConfig`): 2 s pre-trigger, 1.5 s incident window, 6 s hard maximum. There is no global cooldown, only a 10 s same-terms duplicate guard.
- **Severity** (`detection/severity.py`, rules in `docs/SEVERITY_RULES.md`): LOW/MEDIUM/HIGH internally, lowercased only when building the payload.
- **Alert payload** (`sender/payload_v2.py`, `sender/delivery.py`): a strict schema-v2 body is committed to the SQLite outbox before background delivery. Retries reuse its UUID as `Idempotency-Key`. The backend derives classroom and school from the authenticated device headers.

## Configuration caveats (`config.py`)

- `ECHOSENSE_BACKEND_BASE_URL` is required and has no implicit production fallback.
- Microphone selection is a case-insensitive exact-then-substring match on `ECHOSENSE_PREFERRED_MICROPHONE` (default `EMEET M0 Plus`). Strict mode (default) refuses to start if it is missing.
- `LOCATION` is a deprecated compatibility constant and is not sent in schema-v2.
- Detection thresholds live in `detection/thresholds.py` and `DetectionRuntimeConfig`. The old duplicates in `config.py` were removed.

## Known landmines

- YAMNet needs a non-empty 3-column class map (`index,mid,display_name`, ~14 KB). If the model or map is missing or empty, `load_yamnet_bundle` logs `yamnet_available=false` and the runtime continues text-only (degraded mode) instead of crashing.
- After SIGTERM the runtime logs `[RUNTIME] stopped cleanup_errors=0`, but the process can linger because of RealtimeSTT's child processes. Check for leftover `main.py` and `multiprocessing` processes before starting another instance.
- Vosk zips or model folders may still sit in an old checkout. They are install artifacts and are never committed.
