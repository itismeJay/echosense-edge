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

The Whisper fallback model (`ECHOSENSE_WHISPER_MODEL`, default `base`) is fetched by faster-whisper into the Hugging Face cache on first run. The emotion model (`models/emotion2vec_plus_base.onnx`, made by `tools/export_emotion2vec.py`) is copied by hand; `docs/PI_FILES.md` lists every untracked file. Model files, venvs, `edge.env`, `data/` and `logs/` are gitignored.

## Running the system

- Pi-only files that are not in git (models, `edge.env` keys) are listed in `docs/PI_FILES.md`.
- Entry point: `venv/bin/python main.py` on the Mac, `echosense-env/bin/python3 main.py` on the Pi (systemd `echosense.service`, drop-ins in `deploy/`). Run from the repo root; the imports expect it as the CWD.
- Mac demo wrappers live outside the repo in `~/echosense-demo`: `edge-local.sh` (backend `http://localhost:8000`) and `edge-live.sh` (Render). Options: `--mic builtin|emeet`, `--device <code> --key-file <path>`, `--identity <path>` (a QR-set-up box), `--test-alert`.
- QR setup: `provisioning/identity.py` resolves credentials (identity.json + the key saved at activation, else `ECHOSENSE_DEVICE_ID/KEY`); `main.py` runs `provisioning/activation.py` (polls `/devices/activate`, speaks status) before the runtime and restarts itself once the key is saved. `tools/provision_box.py` (super admin) makes boxes, identity files and QR labels (needs `tools/requirements-provision.txt`).
- Tests: `venv/bin/python -m unittest discover -s tests -t .` (unittest, no pytest). There is no build step and no linter.
- `test_backend.py` contacts the configured backend; `test_tone.py` is a standalone synthetic-audio check.

## Pipeline architecture

Detection is the event-driven "AFTER" runtime started by `runtime.py` (details in `docs/after_detection_architecture.md`):

```
PyAudioSharedSource (the only mic reader, float32 16 kHz mono)
   ├─► RealtimeSTT (VAD) ─► HybridTranscriptionExecutor
   │      ElevenLabs scribe_v3 (ceb + keyterms) primary, Whisper fallback
   │      in parallel: tone = emotion2vec + YAMNet shout/laughter + RMS
   │                         ─► TextDetector (lexicon) ─► Haiku classifier ─┐
   ├─► YAMNet + tone (AcousticDetector, noise floor) ─────────────────────┤
   └─► incident context audio ────────────────────────────────────────────┤
                                                                          ▼
           EventCoordinator (incident window, fusion, tone vote, semantic rules, severity)
                                                                          ▼
           SQLite outbox ─► background delivery to {API_URL}/alerts/
```

Non-obvious behaviors that span multiple files:

- **Bisaya first.** The deployment school speaks Bisaya. Choose models, settings and word-list work by Bisaya recall (`tools/evaluate_testset.py --report` ranks that way); English/Tagalog terms stay in the lists but are not optimized.
- **Speech recognition: ElevenLabs primary, Whisper fallback, never Vosk.** `model/stt_provider.py` sends every utterance to ElevenLabs Scribe (`scribe_v3`, language `ceb`, keyterms from `lexicon/stt_keyterms.txt`, `ELEVENLABS_API_KEY`). A failure queues the clip in RAM (`RetryQueue`, retried every 30 s, lost on restart); Whisper takes over only after 30 min of continuous failure (`ECHOSENSE_ELEVENLABS_OUTAGE_MINUTES`) or for a clip ElevenLabs rejects 3 times. Without the key the box runs Whisper only. `model/whisper_stt.py` feeds RealtimeSTT (`use_microphone=False`) from the shared source. `ECHOSENSE_WHISPER_LANGUAGE` defaults to `tl` (Whisper has no Cebuano; forced Tagalog beat `auto` on Bisaya). The recommended model is the Cebuano fine-tune in `models/sapinsapin_whisper-small-pld-ceb-ct2-int8` (gitignored, copy it to the Pi). For the Whisper fallback, `detection/soundalike.py` repairs its Bisaya mishearings and re-joins "ka ayo" into "kaayo" before matching. Whisper runs with `initial_prompt=None`: prompting with the monitored terms made Whisper hallucinate them. `model/vosk_detect.py` is legacy, unused at runtime, and loads both Vosk models at import, so never import it.
- **Tone vote** (`model/emotion_recognizer.py` + `detection/tone_vote.py`): emotion2vec_plus_base (onnxruntime) runs on each utterance clip in parallel with the STT request. `angry_tone` = emotion2vec angry ≥ threshold AND a loud cue (YAMNet shout/yell, speech RMS ≥ ratio × rolling baseline, or an ElevenLabs shout tag); emotion alone never counts. Laughter = YAMNet laughter classes or an ElevenLabs laugh tag. Tone never opens an alert; it only raises the severity of a word that did (`ECHOSENSE_TONE_*`).
- **Claude semantic layer** (`detection/llm_semantic.py`, `ECHOSENSE_SEMANTIC_MODE=off|on_match|all`, default all): Haiku 5.5 labels each utterance (insult/threat/exclusion/shaming/banter/friendly_slang/...) via `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`, 4 s deadline, else lexicon-only. Fusion (`apply_semantic_rules` in `fusion.py`): lexicon threat stays HIGH (quoting/negation ≥0.9 → MEDIUM), word + banter ≥0.8 → LOW (never dropped) unless angry tone + targeted (stays HIGH, "angry insult (AI suggested joking — check)"), no word + targeted harm ≥0.85 → alert "detected by meaning", distress/YAMNet crying after the words → +1, 3 similar insults in 10 min → +1. Flags travel in `severity_evidence.supporting_evidence` (`semantic_*`). Eval: `tools/evaluate_semantic_llm.py`.
- **Word-gated alerts:** apart from "detected by meaning" above, only a monitored word opens an incident. Sound alone is logged locally and never sent. Acoustic evidence (YAMNet exact-label allowlist plus the tone rule, judged against an adaptive noise floor) can only change severity when it is time-linked to the text.
- **Bisaya Grade 6 lexicon:** `lexicon/bisaya_bullying.yaml` (category → root → patterns, hand-written idioms, and everyday sentences that must never alert) is expanded by `detection/lexicon.py` into ~780 phrases and merged into the lists below. Pattern variants match exactly (spelling folded, never fuzzy). `tests/test_lexicon.py` checks every phrase and every everyday sentence; `tools/export_lexicon.py` writes the teacher review CSV.
- **Monitored terms:** `model/blacklist.py` holds the local English/Filipino/Bisaya lists. `model/monitored_terms.py` holds the synchronized list.
- **School config:** `model/school_config.py` fetches `/devices/me/config` at startup and every 300 s (`[SCHOOL_CONFIG] applied ...`) and caches it in `data/school_config.<hash>.json` (one file per backend URL and device), so an offline start uses the last known config. It sets the word list (global words minus the school's turned-off ones, plus the school's own words, sent as text without a `term_id`) and `detection/school_policy.py`: turned-off words are dropped in `check_transcript` even when the built-in lists hold them (with any shorter built-in word inside a turned-off phrase that was said), and Strict sensitivity lifts the descriptive-untargeted gate (`fusion.py`), the bare-ambiguous-word rule (`text_context.py`) and allows a second differing sound on single words of 7+ letters (`soundalike.py`; phrases keep the normal rule). A 404 (older backend) or 409 (device has no school) falls back to the `/dictionary/` sync with normal rules. Strict alerts on 35 of the lexicon's 94 everyday sentences (normal: 0).
- **False-alarm filters** (`detection/false_alarm_filters.py`): decode-quality gate, hallucination and repeated-word checks, word-probability filter and a media-audio gate, each tunable or switchable through `ECHOSENSE_FILTER_*`.
- **Incident timing** (`config.py` `DetectionRuntimeConfig`): 2 s pre-trigger, 1.5 s incident window, 6 s hard maximum. There is no global cooldown, only a 10 s same-terms duplicate guard.
- **Severity** (`detection/severity.py`, rules in `docs/SEVERITY_RULES.md`): LOW/MEDIUM/HIGH internally, lowercased only when building the payload.
- **Alert payload** (`sender/payload_v2.py`, `sender/delivery.py`): a strict schema-v2 body is committed to the SQLite outbox before background delivery. Building never drops an alert for size: evidence lists over 50 entries are summarized (flags and the first value per key kept, plus an `omitted_evidence:N (...)` entry), long entries are truncated; validating a finished body stays strict. Retries reuse its UUID as `Idempotency-Key`. The backend derives classroom and school from the authenticated device headers.

## Configuration caveats (`config.py`)

- `ECHOSENSE_BACKEND_BASE_URL` is required and has no implicit production fallback.
- Microphone selection is a case-insensitive exact-then-substring match on `ECHOSENSE_PREFERRED_MICROPHONE` (default `EMEET M0 Plus`). Strict mode (default) refuses to start if it is missing.
- `LOCATION` is a deprecated compatibility constant and is not sent in schema-v2.
- Detection thresholds live in `detection/thresholds.py` and `DetectionRuntimeConfig`. The old duplicates in `config.py` were removed.

## Known landmines

- YAMNet needs a non-empty 3-column class map (`index,mid,display_name`, ~14 KB). If the model or map is missing or empty, `load_yamnet_bundle` logs `yamnet_available=false` and the runtime continues text-only (degraded mode) instead of crashing.
- After SIGTERM the runtime logs `[RUNTIME] stopped cleanup_errors=0`, but the process can linger because of RealtimeSTT's child processes. Check for leftover `main.py` and `multiprocessing` processes before starting another instance.
- Vosk zips or model folders may still sit in an old checkout. They are install artifacts and are never committed.
