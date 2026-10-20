# AFTER detection architecture

The production runtime has one physical audio owner: `PyAudioSharedSource`.
It opens one input stream, converts device PCM at the boundary to normalized
mono `float32` at 16 kHz, and publishes immutable frames to three bounded
consumer queues:

- RealtimeSTT manual input (`use_microphone=False`)
- continuous overlapping YAMNet/tone analysis
- temporary active-incident audio context

Every subscriber uses a fixed-capacity drop-oldest queue. RealtimeSTT's manual
input is additionally guarded by a fixed internal-queue ceiling and drops the
newest frame when that ceiling is reached. Drops are counted and logged. STT
results use a bounded FIFO with drop-oldest overflow. Acoustic windows and hops
default to 15,600 and 7,800 samples respectively.

Finalized STT PCM has a separate safety/resource limit configured by
`ECHOSENSE_MAX_STT_UTTERANCE_SECONDS`. The development default is 15 seconds,
matching the incident hard maximum: at 16 kHz mono `float32`, each `AudioEvent`
retains at most 240,000 of the most recent samples (960,000 bytes). The 32-item
result FIFO therefore has a default worst-case retained-result PCM bound of
30,720,000 bytes (29.30 MiB), excluding model/dependency memory. Tail retention
preserves the final words/context and never shares memory with the full callback
array. Values must be finite and greater than zero, and are capped at 30 seconds.
This is a memory-safety boundary, not an accuracy threshold.

RealtimeSTT 1.0.2 has no constructor setting for a maximum active recording.
EchoSense uses its public recording callbacks and `stop()` method as a
15-second watchdog, while the sample cap above remains the strict bound on all
EchoSense-owned finalized copies and queued results. Scheduling and any
temporary arrays internal to RealtimeSTT remain third-party limitations that
require runtime observation on the Pi.

`TextDetector` and `AcousticDetector` only produce typed evidence. Either may
open the one `EventCoordinator` incident. Qualifying evidence arriving while
it is active joins that identity and may extend its close deadline, never past
the monotonic hard deadline. Exact timestamp ties use `KEYWORD` as a stable
tie-break; evidence with a later timestamp never changes the opener.

Only qualifying monitored/aggressive `TextEvidence` is retained as incident
content and serialized. Non-qualifying classroom speech is reduced to a
bounded activity count while an incident is active; its transcript, terms,
language, and matches cannot enter the finalized incident, payload, or outbox.
Serialization filters qualifying evidence again, orders it by observation,
deduplicates normalized content, and caps the one transcript field at 10,000
characters. Acoustic-only incidents use schema-v2's established neutral state:
`transcription_status="unavailable"` with no fabricated `transcribed_text`.

Initial timing values are 2 seconds pre-trigger, 5 seconds post-trigger,
3 seconds extension, 15 seconds hard maximum, and 15 seconds cooldown. These
are configurable development starting values, not scientifically validated
classroom thresholds.

Fusion is rule-based and emits reasons. Acoustic-only incidents are initially
limited to LOW or MEDIUM. A single loud window is neither qualifying evidence
nor a HIGH alert. HIGH from combined evidence requires text severity or strong
acoustic corroboration; acoustic-only HIGH is intentionally disabled pending
real classroom evaluation.

The acoustic opener asks whether a window contains meaningful possible-
aggression/distress evidence, not whether it is merely loud or unusual. It may
qualify through an explicitly allowed human distress YAMNet label at the
provisional 0.45 threshold or the existing multi-feature tone rule (RMS plus
energy variance plus zero-crossing rate). High RMS, high peak, duration, and a
neutral loud class do not qualify by themselves. Door slams, bangs, impacts,
and other generic anomaly categories have not been added. YAMNet labels use an
exact normalized allowlist rather than substrings, so `Whimper` can qualify but
`Whimper (dog)` and `Children shouting` do not. The intended map labels are
`Screaming`, `Yell`, `Shout`, `Crying, sobbing`, `Whimper`, and `Wail, moan`,
with controlled exact aliases for compatible maps.

Payload selection prefers qualifying aggression/distress acoustic evidence
over a higher-scoring neutral label. When ACOUSTIC opened the incident, the
existing structured evidence `role` is `primary_trigger`; acoustic evidence
joining a KEYWORD-opened incident remains `supporting_only`.

YAMNet assets default to repository-relative ignored files and may be replaced
with `ECHOSENSE_YAMNET_MODEL_PATH` and `ECHOSENSE_YAMNET_CLASSES_PATH`. The
runtime validates both files. With `ECHOSENSE_ACOUSTIC_DEGRADED_MODE=true`
(the default), a missing/failed YAMNet model is reported explicitly while the
existing multi-feature tone detector remains active. Setting it false makes
model unavailability fatal.

Raw PCM exists only in bounded RAM frames, the rolling pre-trigger buffer, and
the active incident buffer. Finalization derives a bounded integer waveform,
clears the incident PCM, and passes only schema-v2 fields to the SQLite-first
sender. No WAV, base64 audio, raw samples, or audio paths are persisted or sent.

The external-feed API is pinned and verified against RealtimeSTT 1.0.2. Real
EMEET enumeration/selection, ALSA behavior, Pi throughput, contention,
temperature, dropouts, and classroom accuracy still require physical hardware
validation. Deterministic synthetic policy tests do not establish real-world
classification accuracy.
