# Files the Pi needs that are not in git

Everything below is gitignored (or lives outside the repo) and must be copied to the
Pi by hand. Paths are relative to the repo root. Sizes were measured on the Mac on
2026-11-13.

## Model files

| Path | Size | Used by | If missing |
|---|---|---|---|
| `models/emotion2vec_plus_base.onnx` | 356 MB | `model/emotion_recognizer.py` (tone vote, fp32, preferred) | falls back to the int8 file below |
| `models/emotion2vec_plus_base.int8.onnx` | 112 MB | same, used only when the fp32 file is absent (angry score shifts up to 0.2) | `[EMOTION] available=false`, `angry_tone` always false |
| `yamnet.tflite` | 3.9 MB | `model/yamnet_infer.py` (`ECHOSENSE_YAMNET_MODEL_PATH`) | `yamnet_available=false`, text-only degraded mode |
| `yamnet_class_map.csv` | 16 KB | same (`ECHOSENSE_YAMNET_CLASSES_PATH`), 3 columns, non-empty | same as above |
| `models/sapinsapin_whisper-small-pld-ceb-ct2-int8/` | 241 MB | Whisper fallback STT (set `ECHOSENSE_WHISPER_MODEL` to this path) | Whisper uses the `base` model from the Hugging Face cache |
| `~/.cache/huggingface/hub/models--Systran--faster-whisper-base/` | ~145 MB | Whisper fallback when `ECHOSENSE_WHISPER_MODEL` is unset (`base`) | downloaded on first run; needs internet |

Not needed on the Pi: `models/emotion2vec_plus_base_src/` (1.0 GB, PyTorch source for
`tools/export_emotion2vec.py`), `models/multilingual-MiniLMv2-L12-mnli-xnli/` (470 MB,
experimental NLI layer, off by default), `models/eemberda_phcodeswitch-ceb-dvo-ct2-int8/`
and `models/troxyz1268_whisper-small-bisaya-ct2-int8/` (241 MB each, evaluated and not
chosen), Vosk models (legacy). The Silero VAD model ships inside the `silero-vad` pip
package. `lexicon/stt_keyterms.txt` and `lexicon/bisaya_bullying.yaml` are tracked in git.

## `edge.env` (copy from `edge.env.example`, `chmod 600`)

Never commit or print this file. Keys the Pi needs filled in (names only):

| Key | Needed for |
|---|---|
| `ECHOSENSE_BACKEND_BASE_URL` | required, no default |
| `ECHOSENSE_DEVICE_ID`, `ECHOSENSE_DEVICE_KEY` | device auth, unless the box was QR-provisioned (`data/identity.json` + `data/device_credentials.json`) |
| `ELEVENLABS_API_KEY` | primary STT; without it the Pi runs Whisper only |
| `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` | Haiku semantic layer; without both it is lexicon-only |
| `ECHOSENSE_PREFERRED_MICROPHONE` | default `EMEET M0 Plus`; strict mode refuses to start without it |
| `ECHOSENSE_WHISPER_MODEL` | set to the Cebuano fine-tune path above |

## Runtime data (created by the Pi, do not copy from the Mac)

`data/*.sqlite3` (alert outbox), `data/school_config.<hash>.json` (cached school
config), `logs/`. QR-provisioned boxes also hold `data/identity.json` and
`data/device_credentials.json`, which are made for that box only.
