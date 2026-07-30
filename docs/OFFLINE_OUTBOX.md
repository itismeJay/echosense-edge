# Persistent Offline Alert Outbox

## Purpose and architecture

The edge service writes each complete backend-compatible alert payload to a
local SQLite outbox before background delivery begins. This removes repeated
HTTP waits from the detection loop and allows pending alerts to survive a
process or device restart.

The flow is:

1. Detection preserves the `AudioEvent.event_id`.
2. `sender/http_client.py` builds the existing backend payload.
3. `sender/outbox.py` transactionally inserts that payload.
4. The detection loop continues after the local enqueue result.
5. One `echosense-outbox-worker` thread claims due records.
6. The worker makes one HTTP attempt and records the acknowledgement or failure.
7. Retryable records retain the same serialized payload and `event_id`.

The worker does not open a microphone, hold an `AudioEvent`, regenerate a
transcript, recalculate severity, or store raw audio.

## Database and schema

The default database is:

`/home/echosense/echosense-edge/data/alert_outbox.sqlite3`

The `data/*.sqlite3*` pattern is ignored by Git. SQLite WAL and shared-memory
sidecars are ignored as well.

`alert_outbox` contains:

- `id`
- unique `event_id`
- `payload_json`
- `status`
- `attempt_count`
- `next_attempt_at`
- `created_at`
- `updated_at`
- `last_attempt_at`
- `delivered_at`
- `last_error`
- `last_error_type`
- `last_http_status`
- `lease_expires_at`

All application timestamps are UTC ISO-8601 values. Initialization creates the
schema and safe additive migrations automatically. SQLite uses WAL,
`synchronous=FULL`, foreign-key checks, and a configurable busy timeout.

Statuses are:

- `PENDING`: transactionally stored and waiting for its first attempt.
- `SENDING`: atomically leased by the worker.
- `FAILED_RETRYABLE`: retained until `next_attempt_at`.
- `FAILED_PERMANENT`: retained for operator review.
- `DELIVERED`: a valid 2xx or confirmed idempotent-duplicate acknowledgement was
  received.
- `RESOLVED`: an operator reviewed a permanent failure; the row remains stored.

Pending, retryable, permanent, and resolved rows are never removed by retention
cleanup. Only old `DELIVERED` rows are eligible.

## HTTP classification

Delivered:

- any 2xx acknowledgement
- HTTP 409 only when the response explicitly identifies a duplicate and returns
  the same `event_id`

Retryable:

- timeout, DNS failure, connection refusal, and other request/network errors
- HTTP 408, 425, 429, 500, 502, 503, and 504
- other server errors and unexpected non-client statuses

Permanent:

- unconfirmed HTTP 409
- HTTP 400 and other non-retryable 4xx responses, including authentication,
  authorization, invalid endpoint, and unsupported-format responses
- corrupt stored JSON or a stored `event_id` mismatch

Response bodies and complete payloads are not written to routine delivery logs.
`Retry-After` is respected for 429 and 503, up to the configured maximum delay.

## Retry schedule and startup recovery

Defaults produce delays of approximately 5, 15, and 30 seconds, followed by
delays capped at 300 seconds. Jitter is applied to avoid synchronized retries.
There is no small global attempt limit for retryable records.

Every claim increments `attempt_count` transactionally. A process interruption
can leave a row in `SENDING`; its lease expires and startup/periodic recovery
moves it to `FAILED_RETRYABLE`. Neither restart nor retry changes `event_id`.

The deployment drop-in at
`deploy/echosense.service.d/offline-outbox.conf` removes the installed
connectivity-gated startup command while preserving LED permission setup. This
allows local detection and enqueueing to start while connectivity is absent.

## Configuration

All settings are environment variables:

| Variable | Default |
|---|---:|
| `ECHOSENSE_OUTBOX_ENABLED` | `true` |
| `ECHOSENSE_OUTBOX_DATABASE_PATH` | `data/alert_outbox.sqlite3` |
| `ECHOSENSE_OUTBOX_POLL_INTERVAL_SECONDS` | `1` |
| `ECHOSENSE_OUTBOX_REQUEST_TIMEOUT_SECONDS` | `10` |
| `ECHOSENSE_OUTBOX_INITIAL_RETRY_DELAY_SECONDS` | `5` |
| `ECHOSENSE_OUTBOX_BACKOFF_MULTIPLIER` | `3` |
| `ECHOSENSE_OUTBOX_MAX_RETRY_DELAY_SECONDS` | `300` |
| `ECHOSENSE_OUTBOX_JITTER_PERCENT` | `0.20` |
| `ECHOSENSE_OUTBOX_STALE_SENDING_SECONDS` | `60` |
| `ECHOSENSE_OUTBOX_DELIVERED_RETENTION_DAYS` | `30` |
| `ECHOSENSE_OUTBOX_MAX_RECORDS_PER_CYCLE` | `10` |
| `ECHOSENSE_OUTBOX_BUSY_TIMEOUT_MS` | `5000` |
| `ECHOSENSE_OUTBOX_HEALTH_LOG_INTERVAL_SECONDS` | `60` |

Disabling the outbox also disables alert delivery; it does not restore the old
blocking sender.

## Health and safe inspection

Routine health output contains only:

- `database_status`
- pending, sending, retryable-failed, permanent-failed, delivered, and resolved
  counts
- oldest pending age
- last successful delivery time
- last attempt time
- last delivery error type
- worker-running state
- recovered-stale-record count

It does not contain transcript text, monitored terms, payload JSON, tokens, or
credentials.

Use the privacy-safe operator commands:

```bash
echosense-env/bin/python3 -m sender.outbox_admin health
echosense-env/bin/python3 -m sender.outbox_admin list
echosense-env/bin/python3 -m sender.outbox_admin list --status FAILED_PERMANENT
```

These commands intentionally omit `payload_json` and transcript fields.

After correcting the configuration or payload problem, explicitly retry a
reviewed permanent record:

```bash
echosense-env/bin/python3 -m sender.outbox_admin retry EVENT_ID
```

If review determines that no retry should occur, retain the row as resolved:

```bash
echosense-env/bin/python3 -m sender.outbox_admin resolve EVENT_ID --note "reviewed"
```

Do not put transcript content in the note.

## Retention, backup, and test reset

Delivered cleanup runs at worker startup using the configured retention period.
It never deletes pending or failed rows.

For a consistent backup, stop the test worker and use SQLite's backup command:

```bash
sqlite3 data/alert_outbox.sqlite3 ".backup '/tmp/echosense-outbox-backup.sqlite3'"
```

To reset only a known test outbox, stop its worker, verify the configured path,
and move that explicit test database and its sidecars to a dated backup
directory. Never reset the production outbox while pending records exist.

## Troubleshooting and limitations

- `database_status` other than `OK` means the alert was not safely queued.
- `database_locked` is normally temporary; check for an abandoned maintenance
  connection.
- `storage_full` requires freeing local space without deleting pending records.
- A persistent `FAILED_PERMANENT` row needs endpoint, credentials, or payload
  review.
- A worker stop timeout can leave `SENDING`; lease recovery handles it later.
- SQLite corruption, storage-device failure, power loss, transcription errors,
  and backend failures remain possible operational limitations.

The persistent outbox improves resilience during connectivity outages. Remote
backend synchronization and mobile push notification still require
connectivity, and immediate delivery is not guaranteed.

## Rollback

1. Stop `echosense.service`.
2. Back up the outbox database and sidecars without deleting pending rows.
3. Remove the installed offline-start drop-in and reload systemd.
4. Revert only the outbox-specific edge files through the project's normal
   reviewed deployment process.
5. Do not restore the old blocking sender while pending records still require
   delivery; export or resolve them first.
