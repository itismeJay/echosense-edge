"""Transactional SQLite storage for pending backend alert payloads.

The outbox stores backend-compatible JSON only. It does not store AudioEvent
objects or raw audio samples.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Callable, Iterator


PENDING = "PENDING"
SENDING = "SENDING"
DELIVERED = "DELIVERED"
FAILED_RETRYABLE = "FAILED_RETRYABLE"
FAILED_PERMANENT = "FAILED_PERMANENT"
RESOLVED = "RESOLVED"
OUTBOX_STATUSES = (
    PENDING,
    SENDING,
    DELIVERED,
    FAILED_RETRYABLE,
    FAILED_PERMANENT,
    RESOLVED,
)
SCHEMA_VERSION = 1

_FORBIDDEN_AUDIO_KEYS = frozenset({
    "audio",
    "audio_bytes",
    "audio_data",
    "audio_samples",
    "pcm",
    "raw_audio",
    "raw_samples",
    "wav",
    "wave_file",
})


class OutboxError(RuntimeError):
    """Base error for durable outbox operations."""


class OutboxStorageError(OutboxError):
    """SQLite or filesystem operation failed."""

    def __init__(self, code: str):
        self.code = str(code or "storage_error")
        super().__init__(self.code)


class OutboxSerializationError(OutboxError):
    """Payload could not be safely serialized."""


class OutboxSchemaError(OutboxError):
    """Existing database schema cannot be migrated safely."""


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    record_id: int
    event_id: str
    inserted: bool


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    id: int
    event_id: str
    payload_json: str
    status: str
    attempt_count: int
    next_attempt_at: str
    created_at: str
    updated_at: str
    last_attempt_at: str | None
    delivered_at: str | None
    last_error: str | None
    last_error_type: str | None
    last_http_status: int | None
    lease_expires_at: str | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(value: datetime | None = None) -> str:
    current = value or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _contains_raw_audio(value, *, key="") -> bool:
    normalized_key = str(key).strip().lower()
    if normalized_key in _FORBIDDEN_AUDIO_KEYS:
        return True
    if isinstance(value, dict):
        return any(
            _contains_raw_audio(item, key=item_key)
            for item_key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_audio(item) for item in value)
    return False


def _safe_error(value: object, *, maximum=240) -> str:
    text = " ".join(str(value or "").split())
    return text[:maximum] or "unspecified delivery error"


def _storage_error_code(exc: BaseException) -> str:
    message = str(exc).lower()
    if "database or disk is full" in message or "disk full" in message:
        return "storage_full"
    if "locked" in message or "busy" in message:
        return "database_locked"
    if "unable to open" in message or "not a directory" in message:
        return "database_unavailable"
    if isinstance(exc, OSError):
        return "filesystem_error"
    return type(exc).__name__.lower()


class AlertOutbox:
    """Short-connection SQLite outbox safe for worker/main thread use."""

    _ADDITIVE_COLUMNS = {
        "status": "TEXT NOT NULL DEFAULT 'PENDING'",
        "attempt_count": "INTEGER NOT NULL DEFAULT 0",
        "next_attempt_at": "TEXT",
        "created_at": "TEXT",
        "updated_at": "TEXT",
        "last_attempt_at": "TEXT",
        "delivered_at": "TEXT",
        "last_error": "TEXT",
        "last_error_type": "TEXT",
        "last_http_status": "INTEGER",
        "lease_expires_at": "TEXT",
    }

    def __init__(
        self,
        database_path,
        *,
        busy_timeout_ms=5000,
        clock: Callable[[], datetime] = utc_now,
        logger: Callable[[str], None] = print,
    ):
        self.database_path = Path(database_path)
        self.busy_timeout_ms = max(100, int(busy_timeout_ms))
        self._clock = clock
        self._logger = logger
        self._initialized = False

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            connection = sqlite3.connect(
                str(self.database_path),
                timeout=self.busy_timeout_ms / 1000.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        except (OSError, sqlite3.Error) as exc:
            raise OutboxStorageError(_storage_error_code(exc)) from exc
        finally:
            if connection is not None:
                connection.close()

    def initialize(self) -> None:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OutboxStorageError(_storage_error_code(exc)) from exc

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alert_outbox (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL UNIQUE,
                        payload_json TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'PENDING',
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        next_attempt_at TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_attempt_at TEXT,
                        delivered_at TEXT,
                        last_error TEXT,
                        last_error_type TEXT,
                        last_http_status INTEGER,
                        lease_expires_at TEXT
                    )
                    """
                )
                self._migrate_additive_columns(connection)
                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_alert_outbox_event_id ON alert_outbox(event_id)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_alert_outbox_due
                    ON alert_outbox(status, next_attempt_at, id)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS outbox_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        self._initialized = True
        self._logger("[OUTBOX] database initialized")

    def _migrate_additive_columns(self, connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(alert_outbox)"
            ).fetchall()
        }
        core = {"id", "event_id", "payload_json"}
        missing_core = core - columns
        if missing_core:
            raise OutboxSchemaError(
                "incompatible alert_outbox core schema"
            )
        now = utc_timestamp(self._clock())
        for name, definition in self._ADDITIVE_COLUMNS.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE alert_outbox ADD COLUMN {name} {definition}"
                )
        connection.execute(
            """
            UPDATE alert_outbox
            SET next_attempt_at = COALESCE(next_attempt_at, ?),
                created_at = COALESCE(created_at, ?),
                updated_at = COALESCE(updated_at, ?)
            """,
            (now, now, now),
        )

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _serialize_payload(self, payload: dict) -> tuple[str, str]:
        if not isinstance(payload, dict):
            raise OutboxSerializationError("payload must be a dictionary")
        event_id = str(payload.get("event_id") or "").strip()
        if not event_id:
            raise OutboxSerializationError("event_id is required")
        if _contains_raw_audio(payload):
            raise OutboxSerializationError(
                "raw audio fields are not allowed in the outbox"
            )
        try:
            payload_json = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise OutboxSerializationError(type(exc).__name__) from exc
        return event_id, payload_json

    def enqueue(self, payload: dict) -> EnqueueResult:
        event_id, payload_json = self._serialize_payload(payload)
        now = utc_timestamp(self._clock())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO alert_outbox (
                        event_id, payload_json, status, attempt_count,
                        next_attempt_at, created_at, updated_at
                    ) VALUES (?, ?, ?, 0, ?, ?, ?)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    (event_id, payload_json, PENDING, now, now, now),
                )
                inserted = cursor.rowcount == 1
                row = connection.execute(
                    "SELECT id FROM alert_outbox WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        if row is None:
            raise OutboxStorageError("record missing after enqueue")
        action = "queued" if inserted else "duplicate retained"
        self._logger(f"[OUTBOX] {action} event={event_id}")
        return EnqueueResult(int(row["id"]), event_id, inserted)

    def recover_stale_sending(self, *, now=None) -> int:
        current = now or self._clock()
        timestamp = utc_timestamp(current)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE alert_outbox
                    SET status = ?,
                        next_attempt_at = ?,
                        updated_at = ?,
                        lease_expires_at = NULL,
                        last_error = ?,
                        last_error_type = ?
                    WHERE status = ?
                      AND (
                          lease_expires_at IS NULL
                          OR lease_expires_at <= ?
                      )
                    """,
                    (
                        FAILED_RETRYABLE,
                        timestamp,
                        timestamp,
                        "stale sending lease recovered",
                        "stale_sending_recovered",
                        SENDING,
                        timestamp,
                    ),
                )
                recovered = int(cursor.rowcount)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        if recovered:
            self._logger(
                f"[OUTBOX] stale records recovered count={recovered}"
            )
        return recovered

    def claim_next(self, *, lease_seconds, now=None) -> OutboxRecord | None:
        current = now or self._clock()
        timestamp = utc_timestamp(current)
        lease_expires = utc_timestamp(
            current + timedelta(seconds=max(1.0, float(lease_seconds)))
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT *
                    FROM alert_outbox
                    WHERE status IN (?, ?)
                      AND next_attempt_at <= ?
                    ORDER BY next_attempt_at ASC, id ASC
                    LIMIT 1
                    """,
                    (PENDING, FAILED_RETRYABLE, timestamp),
                ).fetchone()
                if row is None:
                    connection.execute("COMMIT")
                    return None
                cursor = connection.execute(
                    """
                    UPDATE alert_outbox
                    SET status = ?,
                        attempt_count = attempt_count + 1,
                        last_attempt_at = ?,
                        updated_at = ?,
                        lease_expires_at = ?
                    WHERE id = ?
                      AND status = ?
                    """,
                    (
                        SENDING,
                        timestamp,
                        timestamp,
                        lease_expires,
                        row["id"],
                        row["status"],
                    ),
                )
                if cursor.rowcount != 1:
                    connection.execute("ROLLBACK")
                    return None
                self._set_meta(
                    connection,
                    "last_delivery_attempt_at",
                    timestamp,
                )
                claimed = connection.execute(
                    "SELECT * FROM alert_outbox WHERE id = ?",
                    (row["id"],),
                ).fetchone()
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        record = self._record_from_row(claimed)
        self._logger(
            f"[OUTBOX] claimed event={record.event_id} "
            f"attempt={record.attempt_count}"
        )
        return record

    def mark_delivered(
        self,
        record_id: int,
        *,
        http_status=None,
        now=None,
    ) -> bool:
        timestamp = utc_timestamp(now or self._clock())
        updated = self._update_sending(
            record_id,
            status=DELIVERED,
            next_attempt_at=timestamp,
            delivered_at=timestamp,
            last_error=None,
            last_error_type=None,
            last_http_status=http_status,
            lease_expires_at=None,
            meta={
                "last_successful_delivery_at": timestamp,
            },
        )
        return updated

    def mark_retryable_failure(
        self,
        record_id: int,
        *,
        next_attempt_at,
        error_type,
        error,
        http_status=None,
        now=None,
    ) -> bool:
        return self._update_sending(
            record_id,
            status=FAILED_RETRYABLE,
            next_attempt_at=utc_timestamp(next_attempt_at),
            delivered_at=None,
            last_error=_safe_error(error),
            last_error_type=_safe_error(error_type, maximum=80),
            last_http_status=http_status,
            lease_expires_at=None,
            updated_at=utc_timestamp(now or self._clock()),
            meta={
                "last_delivery_error_type": _safe_error(
                    error_type,
                    maximum=80,
                ),
            },
        )

    def mark_permanent_failure(
        self,
        record_id: int,
        *,
        error_type,
        error,
        http_status=None,
        now=None,
    ) -> bool:
        timestamp = utc_timestamp(now or self._clock())
        return self._update_sending(
            record_id,
            status=FAILED_PERMANENT,
            next_attempt_at=timestamp,
            delivered_at=None,
            last_error=_safe_error(error),
            last_error_type=_safe_error(error_type, maximum=80),
            last_http_status=http_status,
            lease_expires_at=None,
            updated_at=timestamp,
            meta={
                "last_delivery_error_type": _safe_error(
                    error_type,
                    maximum=80,
                ),
            },
        )

    def _update_sending(
        self,
        record_id,
        *,
        status,
        next_attempt_at,
        delivered_at,
        last_error,
        last_error_type,
        last_http_status,
        lease_expires_at,
        updated_at=None,
        meta=None,
    ) -> bool:
        timestamp = updated_at or utc_timestamp(self._clock())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE alert_outbox
                    SET status = ?,
                        next_attempt_at = ?,
                        updated_at = ?,
                        delivered_at = ?,
                        last_error = ?,
                        last_error_type = ?,
                        last_http_status = ?,
                        lease_expires_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        status,
                        next_attempt_at,
                        timestamp,
                        delivered_at,
                        last_error,
                        last_error_type,
                        last_http_status,
                        lease_expires_at,
                        record_id,
                        SENDING,
                    ),
                )
                for key, value in (meta or {}).items():
                    self._set_meta(connection, key, value)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return cursor.rowcount == 1

    def retry_permanent(self, event_id: str, *, now=None) -> bool:
        """Explicitly return one reviewed permanent failure to retryable state."""

        timestamp = utc_timestamp(now or self._clock())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE alert_outbox
                    SET status = ?,
                        next_attempt_at = ?,
                        updated_at = ?,
                        lease_expires_at = NULL,
                        last_error = ?,
                        last_error_type = ?
                    WHERE event_id = ? AND status = ?
                    """,
                    (
                        FAILED_RETRYABLE,
                        timestamp,
                        timestamp,
                        "manual retry requested after review",
                        "manual_retry",
                        event_id,
                        FAILED_PERMANENT,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return cursor.rowcount == 1

    def resolve_permanent(
        self,
        event_id: str,
        *,
        note="operator reviewed",
        now=None,
    ) -> bool:
        """Retain a reviewed permanent failure while removing it from retries."""

        timestamp = utc_timestamp(now or self._clock())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE alert_outbox
                    SET status = ?,
                        updated_at = ?,
                        lease_expires_at = NULL,
                        last_error = ?,
                        last_error_type = ?
                    WHERE event_id = ? AND status = ?
                    """,
                    (
                        RESOLVED,
                        timestamp,
                        _safe_error(note),
                        "operator_resolved",
                        event_id,
                        FAILED_PERMANENT,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return cursor.rowcount == 1

    def cleanup_delivered(self, retention_seconds: float, *, now=None) -> int:
        cutoff = utc_timestamp(
            (now or self._clock())
            - timedelta(seconds=max(0.0, float(retention_seconds)))
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    DELETE FROM alert_outbox
                    WHERE status = ?
                      AND delivered_at IS NOT NULL
                      AND delivered_at <= ?
                    """,
                    (DELIVERED, cutoff),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        deleted = int(cursor.rowcount)
        if deleted:
            self._logger(
                f"[OUTBOX] delivered retention cleanup count={deleted}"
            )
        return deleted

    def get_record(self, event_id: str) -> OutboxRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM alert_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._record_from_row(row) if row else None

    def list_records(self, *, status=None) -> list[OutboxRecord]:
        with self._connection() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT * FROM alert_outbox ORDER BY id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM alert_outbox
                    WHERE status = ?
                    ORDER BY id
                    """,
                    (status,),
                ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def health_summary(self, *, now=None) -> dict:
        current = now or self._clock()
        empty = {
            "database_status": "ERROR",
            "pending_count": 0,
            "sending_count": 0,
            "retryable_failed_count": 0,
            "permanent_failed_count": 0,
            "resolved_count": 0,
            "delivered_count": 0,
            "oldest_pending_age_seconds": None,
            "last_successful_delivery_at": None,
            "last_delivery_attempt_at": None,
            "last_delivery_error_type": None,
        }
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM alert_outbox
                    GROUP BY status
                    """
                ).fetchall()
                counts = {row["status"]: int(row["count"]) for row in rows}
                oldest = connection.execute(
                    """
                    SELECT MIN(created_at) AS created_at
                    FROM alert_outbox
                    WHERE status IN (?, ?, ?)
                    """,
                    (PENDING, SENDING, FAILED_RETRYABLE),
                ).fetchone()["created_at"]
                meta = {
                    row["key"]: row["value"]
                    for row in connection.execute(
                        "SELECT key, value FROM outbox_meta"
                    ).fetchall()
                }
        except OutboxError as exc:
            error_code = getattr(exc, "code", type(exc).__name__)
            empty["database_status"] = f"ERROR:{error_code}"
            return empty

        oldest_at = parse_utc_timestamp(oldest)
        oldest_age = None
        if oldest_at is not None:
            oldest_age = max(
                0.0,
                (current.astimezone(timezone.utc) - oldest_at).total_seconds(),
            )
        return {
            "database_status": "OK",
            "pending_count": counts.get(PENDING, 0),
            "sending_count": counts.get(SENDING, 0),
            "retryable_failed_count": counts.get(FAILED_RETRYABLE, 0),
            "permanent_failed_count": counts.get(FAILED_PERMANENT, 0),
            "resolved_count": counts.get(RESOLVED, 0),
            "delivered_count": counts.get(DELIVERED, 0),
            "oldest_pending_age_seconds": (
                round(oldest_age, 3) if oldest_age is not None else None
            ),
            "last_successful_delivery_at": meta.get(
                "last_successful_delivery_at"
            ),
            "last_delivery_attempt_at": meta.get(
                "last_delivery_attempt_at"
            ),
            "last_delivery_error_type": (
                meta.get("last_delivery_error_type") or None
            ),
        }

    @staticmethod
    def _set_meta(connection, key: str, value: object) -> None:
        connection.execute(
            """
            INSERT INTO outbox_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, str(value or "")),
        )

    @staticmethod
    def _record_from_row(row) -> OutboxRecord:
        return OutboxRecord(
            id=int(row["id"]),
            event_id=str(row["event_id"]),
            payload_json=str(row["payload_json"]),
            status=str(row["status"]),
            attempt_count=int(row["attempt_count"]),
            next_attempt_at=str(row["next_attempt_at"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_attempt_at=row["last_attempt_at"],
            delivered_at=row["delivered_at"],
            last_error=row["last_error"],
            last_error_type=row["last_error_type"],
            last_http_status=row["last_http_status"],
            lease_expires_at=row["lease_expires_at"],
        )

    def close(self) -> None:
        """Compatibility lifecycle hook; all actual connections are short-lived."""

        return None
