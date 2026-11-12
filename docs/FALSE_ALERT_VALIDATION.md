# False-alert validation

These checks validate policy and runtime behavior without treating an alert as
proof of bullying. Use consenting adults only. Do not use students, enable a
production deployment, upload audio, or retain voice recordings.

## Controlled EMEET microphone session

1. Run the unit suite before connecting the microphone:

   ```bash
   venv/bin/python -m unittest discover -s tests -p 'test_*.py'
   ```

2. Connect the EMEET OfficeCore M0 Plus directly and confirm that it is the
   selected input. Keep the normal 16 kHz mono runtime path and the deployed
   model/language settings unchanged for the first baseline.
3. In a non-production, network-isolated session, enable exact transcript
   display only if every adult consents and the console is not being captured.
   Restore the privacy-safe default immediately after the session. Do not write
   WAV files or redirect transcript output to a file.
4. Make a labeled worksheet with the intended phrase and these observed fields:
   original Whisper transcript, corrected matching text, correction provenance,
   qualification decision/reason, and whether a final incident was delivered.
   Record categorical results on the worksheet, not voice or unnecessary names.
5. At near (about 0.5 m) and far (about 2-3 m) distances, repeat at least three
   sessions containing:

   - neutral English, Tagalog, Cebuano, and code-switched classroom discussion;
   - exact targeted insults and exact serious threats in each applicable language;
   - `buwan`, `pangita`, `kadiyot`, `iyot`, and approved spelling aliases;
   - explicit vocabulary discussion such as “the word idiot is offensive”;
   - silence, fan/air-conditioner noise, chair movement, and ordinary media audio;
   - one speaker, overlapping adult speakers, quiet speech, and normal speech.

6. For every mismatch, classify the first failing boundary: microphone/device
   selection, VAD/segmentation, preprocessing, Whisper decode, transcript-quality
   filter, correction/provenance, context, qualification, fusion, or delivery.
   A fuzzy-only correction must remain non-qualifying on its first occurrence;
   standalone exact `iyot` must not be delivered without independent support.
7. Compare false positives and false negatives separately. Do not tune a
   threshold from one phrase or claim classroom accuracy from this session.

## Raspberry Pi 5 checklist

- Boot and USB storage: confirm the expected root device with `findmnt /` and
  inspect devices read-only with `lsblk -o NAME,MODEL,SIZE,FSTYPE,MOUNTPOINTS`.
- Microphone: connect the EMEET, run `arecord -l`, and verify the configured
  preferred-device selection in the startup log. Do not fall back silently to
  a built-in or HDMI device.
- Canonical audio: inspect ALSA parameters with `arecord --dump-hw-params` and
  test 16 kHz mono S16_LE delivery to `/dev/null`; do not save a voice sample.
- Dependencies: run the unit command above with the same venv and service user
  used by the runtime. Confirm the pinned RealtimeSTT and model dependencies load.
- Models: confirm the configured local Whisper and YAMNet asset paths exist and
  that startup reports their actual load/degraded status. Do not download or
  replace a production model during validation.
- Sustained processing: run repeated consenting-adult sessions for at least 30
  minutes; watch frame/result overflow counters, recording-duration limits,
  worker errors, and incident latency.
- CPU/RAM: observe the service with `systemd-cgtop` or `top` and record steady
  state plus peak RSS. Confirm queues stay bounded under overlapping speech.
- Temperature/throttling: sample `vcgencmd measure_temp` and
  `vcgencmd get_throttled` before, during, and after the sustained run.
- Real-time latency: record the existing privacy-safe `[LATENCY]` stage timings;
  compare median and worst case without adding transcript logging.
- Offline queue: disconnect only the test network, create authorized synthetic
  test incidents, reconnect, and verify retry ordering. Never reset or delete a
  queue that contains pending production alerts.
- Backend delivery: use only a test backend/device identity. Verify one payload
  per incident, retry behavior, schema validation, and the required “unverified,
  human review” wording.
- Recovery: stop through the service manager, confirm bounded shutdown and queue
  durability, reboot once, and verify microphone/model recovery without manual
  database edits.

Passing this checklist does not establish production readiness. Representative
adult speech, classroom-like acoustics, hardware load, and human review are still
required before any deployment decision.
