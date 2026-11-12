# Experimental semantic and context upgrade

## Current state

The production decision path is unchanged. Sentence semantics are disabled by
default and, when explicitly enabled, run in shadow-only mode. A semantic result
cannot open, escalate, suppress, or deliver an incident. The analyzer keeps at
most six prior turns, 90 seconds, and 4,000 characters in RAM by default and
clears them on worker shutdown. It does not write conversational context or
audio to disk.

No sentence-semantic model is installed in the project environment. The three
local 241 MB CTranslate2 assets are Cebuano-adapted Whisper speech-recognition
models, not text-understanding models. The local 3.9 MB YAMNet TFLite asset
recognizes acoustic events, not sentence meaning. `transformers` and
`sentence-transformers` are not installed; `onnxruntime` is available.

Therefore, genuine English/Cebuano/Tagalog/code-switched semantic
classification is **not operational**. Enabling
`ECHOSENSE_EXPERIMENTAL_SEMANTICS=true` without an approved adapter produces an
explicit `unavailable` status, never a silent rules-based substitute.

`model.semantic_nli.TransformerNliClassifier` is now a prepared reference
adapter, not an installed model. It accepts an existing local Transformers NLI
directory, forces `local_files_only=True`, requires safetensors, uses CPU
inference, and batches six fixed hypotheses. It has no default model path and
runtime startup does not construct it automatically. Missing dependencies or
assets fail explicitly; nothing is fetched.

The fixed English hypotheses are a reproducible zero-shot baseline, not a
validated prompt design. Prompt or threshold selection must use a separate
development set; the final held-out set must stay frozen. English hypotheses
may particularly disadvantage Cebuano and code-switched inputs, so errors must
be reported by language rather than hidden in an overall score.

## Implemented interfaces and safety policy

`detection.semantic.SemanticClassifier` accepts the current utterance, ASR
language metadata, and bounded preceding turns. It returns one calibrated label:

- `neutral`
- `academic_discussion`
- `playful_teasing`
- `targeted_belittling`
- `threat`
- `repeated_harassment`
- `uncertain`

Low-confidence output is recorded as `uncertain` with the original candidate
label retained separately. Model errors are bounded and explicit. Original ASR
text, correction provenance, baseline decisions, semantic output, acoustic
features, and decision reasons remain distinct evidence.

YAMNet's exact `Laughter` class is now retained as acoustic evidence. It cannot
qualify an incident or imply playfulness, harmlessness, emotion, or intent.
Likewise, loudness and the existing multi-feature tone proxy remain supporting
acoustic observations only. Existing high-priority text threats retain their
current behavior.

## Candidate assessment

No candidate can be called Cebuano-capable for this use case until it passes a
held-out Cebuano and code-switching evaluation. “Multilingual” alone is not
evidence of that capability.

| Candidate | What the published artifact provides | License / artifact size | Cebuano evidence | Raspberry Pi 5 assessment |
|---|---|---|---|---|
| `MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli` | Genuine zero-shot NLI classifier; XNLI supervision covers 15 languages | MIT; 471 MB weights plus about 22 MB tokenizer assets | Cebuano and Tagalog are not XNLI evaluation languages; unverified | Best first classifier candidate. Establish a local FP32 reference result, then separately evaluate ONNX/int8 only if latency or RAM requires it. |
| `intfloat/multilingual-e5-small` | 12-layer, 384-dimensional, roughly 0.1B-parameter sentence encoder | MIT; 471 MB weights | Claims XLM-R language breadth and warns low-resource performance may degrade; no Cebuano safety-classification result | Plausible encoder after quantization, but it is not a classifier; requires a trained, independently held-out classification head |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Genuine zero-shot NLI classifier, about 0.3B parameters | MIT; 558 MB weights | Fine-tuned/evaluated on XNLI languages, not Cebuano; cross-lingual Cebuano accuracy unverified | Higher accuracy potential than MiniLM, but greater latency/RAM risk; secondary Pi candidate |
| `sentence-transformers/LaBSE` | 109-language sentence encoder, about 0.5B parameters | Apache-2.0 | No project-specific Cebuano harassment evidence | Too large to prioritize for the 5–10 second end-to-end budget |
| `BAAI/bge-m3` | General multilingual retrieval encoder, 1024 dimensions | MIT; 2.27 GB weights | No project-specific Cebuano classification evidence | Poor Pi 5 fit relative to smaller candidates |
| `jfernandez/cebuano-filipino-sentence-transformer` | Cebuano/Filipino name only | CC0-1.0 | Empty model card, no training or evaluation evidence | Rejected pending reproducible documentation |

Published model-card facts used in this assessment: [multilingual MiniLM NLI](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli), [multilingual E5 small](https://huggingface.co/intfloat/multilingual-e5-small), [mDeBERTa NLI](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli), [LaBSE](https://huggingface.co/sentence-transformers/LaBSE), [BGE-M3](https://huggingface.co/BAAI/bge-m3), and the undocumented [Cebuano/Filipino artifact](https://huggingface.co/jfernandez/cebuano-filipino-sentence-transformer). FLORES-200 explicitly includes `ceb_Latn`, so it can check Cebuano representation/translation alignment, but it is not a harassment or classroom-context benchmark: [FLORES-200 language list](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).

The fuller primary-source comparison, including XLM-R, Gemma 3 270M,
EmbeddingGemma 2, LASER3, SONAR, licensing, estimated memory ranges, and the
download approval boundary, is in `docs/MULTILINGUAL_MODEL_RESEARCH.md`.

The memory and latency values required for a release decision are deliberately
not estimated. They must be measured on the target Raspberry Pi 5 while Whisper,
YAMNet, queues, and the outbox are also active.

## Independent evaluation gate

Use consenting-adult data that was not used to design prompts, labels, rules, or
fine-tuning. Keep separate strata for English, Cebuano/Bisaya, Tagalog, and mixed
language, crossed with near/far microphone distance, background noise, and
overlapping speakers. Include neutral conversation, academic discussion,
possible teasing, targeted belittling, threats, repeated harassment, uncertainty,
unfamiliar wording, spelling variation, colloquial grammar, indirect statements,
and expressions without profanity.

Each JSONL line accepted by `tools/evaluate_semantic_upgrade.py` is one bounded
conversation:

```json
{"id":"held-out-001","language":"ceb","condition":"far_noise","category":"indirect_belittling","expected_alert":true,"expected_semantic_label":"targeted_belittling","duration_seconds":18.4,"acoustic_context":"none","data_origin":"consenting_adult_transcript","utterances":["redacted example"]}
```

`data_origin` values containing `child` or `minor` are rejected. The tool reads
local JSONL only and does not upload or persist model inputs. Acoustic context is
stratification metadata: `none`, `aggression_observed`, `laughter_observed`, or
`aggression_and_laughter_observed`. It never changes the semantic prediction or
acts as proof of emotion or intent. The adult-authored fixture at
`tests/fixtures/semantic_evaluation_smoke.jsonl` checks coverage and plumbing;
it is visible development data, not a held-out accuracy set.

The runner applies AI-off and AI-on candidate paths to identical utterances and
reports accuracy, precision, recall, false positives, missed detections,
semantic-label coverage/accuracy, median/p95 decision latency, process CPU time,
and peak RSS high-water marks overall and by language, condition, category, and
acoustic context. The AI-on policy exists only inside this offline evaluator.
Runtime shadow decisions are compared with baseline and must have zero
mismatches. Existing urgent directives remain alerts even in the offline
candidate. False alerts per hour is omitted until the data contains at least one
hour of negative exposure. With no adapter, semantic status is explicitly
`unavailable`; the runner falls back to baseline and does not fabricate AI
metrics.

```bash
venv/bin/python3 tools/evaluate_semantic_upgrade.py \
  tests/fixtures/semantic_evaluation_smoke.jsonl
```

After an approved local adapter exists:

```bash
venv/bin/python3 tools/evaluate_semantic_upgrade.py \
  /path/to/frozen-held-out.jsonl \
  --dataset-role held_out \
  --adapter model.semantic_nli:create_classifier
```

The adapter command also requires
`ECHOSENSE_SEMANTIC_MODEL_DIR=/approved/local/snapshot`. Optional
`ECHOSENSE_SEMANTIC_MODEL_ID` controls the non-path identifier recorded in
evidence and `ECHOSENSE_SEMANTIC_MAX_LENGTH` defaults to 256. Do not install the
model or missing Python dependencies until the approval list in the research
note is accepted.

Release remains blocked until a candidate shows a measurable held-out improvement
with no unexplained threat regression, passes consenting-adult microphone tests,
and meets CPU, RAM, inference-time, queue-backlog, recovery, offline/privacy, and
5–10 second end-to-end latency gates on Raspberry Pi 5. Any future decision-path
change requires a separate explicit approval and must remain reversible.
