# Dashboard Test Mode

Dashboard Test Mode runs the normal microphone, STT, text detector, false-positive
filters, severity policy, incident fusion, duplicate guard, and SQLite-first
delivery pipeline. It changes only the backend label at the final payload boundary:
genuine detector incidents use schema-v2 `trigger_type=TEST` and `test_mode=true`.
The backend and dashboard already render these as “Test alert, not a real
incident.”

The mode is disabled by default. `tools/live_validation.py` remains isolated and
explicitly forces the mode, backend delivery, and the outbox off.

## Safety boundary

- Use only consenting adults for spoken phrases.
- A non-local backend, including production, requires an exact
  `--approved-test-origin` plus both live-capture and test-delivery approvals.
- Device credentials come from a private file; they are never command-line
  arguments, payload fields, SQLite fields, or logs.
- The configured backend must assign that device to a dedicated test room.
  Classroom and school cannot be supplied by the edge payload.
- A dedicated test-named outbox is mandatory, so production pending alerts are
  never reset, rewritten, or mixed with test records.
- Raw audio is kept only in bounded memory. The outbox and HTTP payload retain
  the existing transcript/evidence contract and reject raw-audio-shaped fields.
- Dashboard verification uses a reviewer login only to obtain an in-memory JWT.
  The token and password are never logged or stored by the edge.

## Local end-to-end command

Start the isolated local backend and seed its dedicated `EchoSense Development /
Test Classroom / classroom-pi-01` assignment first. The frontend should point to
`http://localhost:8000`.

From the edge repository, run:

```bash
venv/bin/python tools/dashboard_test_mode.py start \
  --backend-url http://127.0.0.1:8000 \
  --credentials-file ../echosense-backend/.env.local.device \
  --reviewer-env-file ../echosense-backend/.env.local \
  --microphone "EMEET" \
  --i-approve-live-capture \
  --i-approve-dashboard-test-delivery
```

Press `Ctrl+C` once to stop cleanly. The same command starts a later session.

The runner keeps exact transcript console logging off. A successful spoken test
produces privacy-safe lines like:

```text
[OUTBOX] queued event=<uuid>
[OUTBOX] delivery attempted event=<uuid> attempt=1
[OUTBOX] delivered event=<uuid> http=201
[LATENCY] ... speech_end_to_delivered_ms=<milliseconds>
[DASHBOARD_TEST] visible=true event=<uuid> alert_id=<id> backend_ack_to_dashboard_feed_ms=<milliseconds> event_end_to_dashboard_feed_ms=<milliseconds>
```

`visible=true` is stronger than a local queue result: an authenticated GET of the
same alert feed polled by the dashboard returned the matching UUID with both TEST
labels. The browser polls that feed every three seconds while visible, so actual UI
rendering can trail the feed observation by up to one polling interval. Confirm the
TEST banner in the open dashboard as the final human-visible check.

Failed requests, authentication failures, retry scheduling, permanent failures,
outbox duplicates, and health counts remain visible through the existing
privacy-safe outbox logs and `sender.outbox_admin` commands.
