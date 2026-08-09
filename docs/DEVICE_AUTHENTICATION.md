# Raspberry Pi Device Authentication

Required safety wording: “Unverified possible-aggression alert. Human review required.”
Device authentication identifies the approved Raspberry Pi and its classroom
assignment. It does not identify a speaker or establish intent or guilt.

## Provision the one-time key

An approved backend administrator creates the device through `POST /devices`.
The response returns `device_key` once. Transfer it directly to the Raspberry
Pi; do not paste it into source control, tickets, chat, screenshots, or logs.

Create a root-owned environment file:

```bash
sudo install -d -m 750 -o root -g root /etc/echosense
sudo install -m 600 -o root -g root /dev/null \
  /etc/echosense/echosense-edge.env
sudoedit /etc/echosense/echosense-edge.env
```

Store placeholder-shaped values:

```dotenv
ECHOSENSE_BACKEND_BASE_URL=http://192.168.1.93:8000
ECHOSENSE_DEVICE_ID=classroom-101-pi
ECHOSENSE_DEVICE_KEY=<DEVICE_KEY>
```

The backend target is required and has no implicit fallback. To switch back to
production, change only that line to:

```dotenv
ECHOSENSE_BACKEND_BASE_URL=https://echosense-backend-75h3.onrender.com
```

Then restart `echosense.service`. Use the same registered device ID and key for
both targets because both FastAPI deployments use the same Neon database.

Install the service drop-ins without copying an environment file into the
repository:

```bash
sudo install -d -m 755 /etc/systemd/system/echosense.service.d
sudo install -m 644 deploy/echosense.service.d/offline-outbox.conf \
  /etc/systemd/system/echosense.service.d/offline-outbox.conf
sudo install -m 644 deploy/echosense.service.d/device-auth.conf \
  /etc/systemd/system/echosense.service.d/device-auth.conf
sudo systemctl daemon-reload
```

Do not restart the service until the backend migration/code and registered
device are ready.

## Delivery behavior

Every outbox delivery sends:

```text
X-EchoSense-Device-Id: classroom-101-pi
X-EchoSense-Device-Key: <DEVICE_KEY>
Idempotency-Key: <EVENT_ID_UUID>
```

`Idempotency-Key` is exactly the canonical body `event_id` on every attempt.
The backend returns HTTP 200 and the original alert for an identical replay;
HTTP 409 means the device or body conflicts and is not treated as success.
Credentials are not stored in SQLite or alert payloads. Missing credentials and
HTTP 401/403 responses retain the event as `FAILED_RETRYABLE` and use the
existing capped exponential backoff. They never print the key or headers.
HTTP 408/429/5xx and network errors remain retryable. Invalid payload responses
remain retained as `FAILED_PERMANENT`.

The schema-v2 body includes `device_identifier` only so the backend can verify
it matches the authenticated device. It sends no authoritative classroom,
school, or location claim; the backend derives those assignments from its
device record.

After correcting credentials, the worker retries due authentication failures
automatically. To inspect only privacy-safe state:

```bash
echosense-env/bin/python3 -m sender.outbox_admin health
echosense-env/bin/python3 -m sender.outbox_admin list \
  --status FAILED_RETRYABLE
```

## Rotation, disablement, and rollback

Rotate a compromised key through the backend admin route
`POST /devices/<DEVICE_UUID>/rotate-key`, replace only
`ECHOSENSE_DEVICE_KEY`, and restart the edge service. The old key stops working
immediately. Disable a lost device through `PATCH /devices/<DEVICE_UUID>` with
`{"is_active": false}`; its retained events will receive HTTP 403 with backoff.

Rollback:

1. Stop `echosense.service`.
2. Back up the SQLite database and sidecars without deleting pending rows.
3. Restore the previous edge code, but retain the outbox files.
4. Remove `device-auth.conf` only when rolling back to a backend that does not
   require device authentication.
5. Reload systemd and start the service only after backend compatibility is
   confirmed.

Never reset or delete an outbox containing pending or failed alerts.
