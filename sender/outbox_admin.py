"""Privacy-safe operator commands for the local alert outbox."""

from __future__ import annotations

import argparse
import json

from config import OUTBOX_BUSY_TIMEOUT_MS, OUTBOX_DATABASE_PATH
from sender.outbox import AlertOutbox, OUTBOX_STATUSES


def _outbox():
    outbox = AlertOutbox(
        OUTBOX_DATABASE_PATH,
        busy_timeout_ms=OUTBOX_BUSY_TIMEOUT_MS,
    )
    outbox.initialize()
    return outbox


def _safe_record(record):
    return {
        "id": record.id,
        "event_id": record.event_id,
        "status": record.status,
        "attempt_count": record.attempt_count,
        "next_attempt_at": record.next_attempt_at,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "last_attempt_at": record.last_attempt_at,
        "delivered_at": record.delivered_at,
        "last_error_type": record.last_error_type,
        "last_http_status": record.last_http_status,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect or review EchoSense outbox records safely."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("health")
    list_command = commands.add_parser("list")
    list_command.add_argument("--status", choices=OUTBOX_STATUSES)
    retry_command = commands.add_parser("retry")
    retry_command.add_argument("event_id")
    resolve_command = commands.add_parser("resolve")
    resolve_command.add_argument("event_id")
    resolve_command.add_argument(
        "--note",
        default="operator reviewed",
        help="Short operational note; do not include transcript text.",
    )
    args = parser.parse_args(argv)
    outbox = _outbox()

    if args.command == "health":
        output = outbox.health_summary()
    elif args.command == "list":
        output = [
            _safe_record(record)
            for record in outbox.list_records(status=args.status)
        ]
    elif args.command == "retry":
        output = {
            "event_id": args.event_id,
            "retry_scheduled": outbox.retry_permanent(args.event_id),
        }
    else:
        output = {
            "event_id": args.event_id,
            "resolved": outbox.resolve_permanent(
                args.event_id,
                note=args.note,
            ),
        }
    print(json.dumps(output, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
