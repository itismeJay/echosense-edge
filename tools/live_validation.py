#!/usr/bin/env python3
"""Local, no-persistence microphone validation for consenting adults only.

This runner deliberately bypasses backend synchronization, log shipping, and
the SQLite outbox. Audio exists only in the runtime's bounded in-memory queues
and buffers. It does not write WAV files or serialize waveform snapshots.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import threading


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Running ``python tools/live_validation.py`` otherwise exposes only the
    # tools directory as sys.path[0], so spawned workers cannot import project
    # packages unless the operator manually exports PYTHONPATH.
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MICROPHONE = "EMEET OfficeCore M0 Plus"
DEFAULT_WHISPER_MODEL = "models/sapinsapin_whisper-small-pld-ceb-ct2-int8"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated EchoSense live validation with no backend or "
            "persistent audio evidence."
        )
    )
    parser.add_argument(
        "--i-approve-live-capture",
        action="store_true",
        help="required acknowledgement that live microphone capture is approved",
    )
    parser.add_argument(
        "--show-transcripts",
        action="store_true",
        help="show exact and corrected transcripts on this terminal only",
    )
    parser.add_argument(
        "--all-adults-consent-to-transcript-display",
        action="store_true",
        help="required with --show-transcripts",
    )
    parser.add_argument("--microphone", default=DEFAULT_MICROPHONE)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    args = parser.parse_args(argv)
    if not args.i_approve_live_capture:
        parser.error("--i-approve-live-capture is required")
    if args.show_transcripts and not args.all_adults_consent_to_transcript_display:
        parser.error(
            "--all-adults-consent-to-transcript-display is required with "
            "--show-transcripts"
        )
    return args


def configure_isolated_environment(args):
    """Override sensitive settings before importing any EchoSense modules."""

    os.environ.update({
        "ECHOSENSE_ENV_FILE": "",
        "ECHOSENSE_BACKEND_BASE_URL": "http://127.0.0.1:9",
        "ECHOSENSE_DEVICE_ID": "local-validation-device",
        "ECHOSENSE_DEVICE_KEY": "local-validation-only",
        "ECHOSENSE_OUTBOX_ENABLED": "false",
        "ECHOSENSE_DASHBOARD_TEST_MODE": "false",
        "ECHOSENSE_DASHBOARD_TEST_REVIEWER_TOKEN": "",
        "ECHOSENSE_PREFERRED_MICROPHONE": args.microphone,
        "ECHOSENSE_MICROPHONE_STRICT": "true",
        "ECHOSENSE_SHOW_TRANSCRIPT_TEXT": (
            "true" if args.show_transcripts else "false"
        ),
        "ECHOSENSE_WHISPER_MODEL": args.whisper_model,
        "ECHOSENSE_WHISPER_LANGUAGE": "tl",
        "ECHOSENSE_WHISPER_GAIN": "off",
        "ECHOSENSE_WHISPER_PROMPT": "",
    })


def disable_realtimestt_log_file():
    """Force RealtimeSTT to keep its diagnostics off disk for this process."""

    import RealtimeSTT

    recorder_class = RealtimeSTT.AudioToTextRecorder

    def no_log_recorder(*args, **kwargs):
        kwargs["no_log_file"] = True
        return recorder_class(*args, **kwargs)

    RealtimeSTT.AudioToTextRecorder = no_log_recorder


def install_text_diagnostics(show_transcripts):
    """Print worksheet fields without changing text decisions."""

    from detection.text_detector import TextDetector

    original = TextDetector.analyze_stt_result

    def analyze_with_diagnostics(self, result, **kwargs):
        evidence = original(self, result, **kwargs)
        provenance = ",".join(
            f"{item.term}:{item.provenance}"
            for item in evidence.term_provenance
        ) or "none"
        corrections = ",".join(
            f"{item.heard}->{item.term}:{item.provenance}"
            for item in evidence.corrections
        ) or "none"
        print(
            "[VALIDATION_TEXT] "
            f"qualifies={str(evidence.qualifies_to_open).lower()} "
            f"provenance={provenance} corrections={corrections} "
            f"reasons={','.join(evidence.reasons) or 'none'}"
        )
        if show_transcripts:
            print(
                "[VALIDATION_TRANSCRIPT] "
                f"original={evidence.transcript!r} "
                f"corrected={evidence.decision_text!r}"
            )
        return evidence

    TextDetector.analyze_stt_result = analyze_with_diagnostics


def run_validation(args):
    configure_isolated_environment(args)
    disable_realtimestt_log_file()
    install_text_diagnostics(args.show_transcripts)

    import runtime as runtime_module
    from detection.latency import LATENCY

    # EdgeRuntime's audio/model/decision path stays intact. Only external and
    # persistent integrations are replaced with no-ops in this process.
    runtime_module.start_alert_delivery = lambda: True
    runtime_module.stop_alert_delivery = lambda: True
    runtime_module.check_backend_connection = lambda: True
    runtime_module.start_school_config_sync = lambda: None
    runtime_module.start_heartbeat = lambda **_kwargs: None

    class ValidationRuntime(runtime_module.EdgeRuntime):
        def _finalize(self, incident):
            print(
                "[VALIDATION_ALERT] would_alert=true "
                f"event={incident.event_id} severity={incident.severity} "
                f"trigger={incident.trigger_type} "
                f"reasons={','.join(incident.severity_reasons) or 'none'}"
            )
            # Reuse the existing timing line. In validation, this stage means
            # "decision emitted"; nothing is actually enqueued or delivered.
            LATENCY.mark_incident(incident.event_id, "incident_enqueued")
            return True

    stop_event = threading.Event()

    def stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    validation = ValidationRuntime(stop_event=stop_event)
    print(
        "[VALIDATION] isolated=true backend=false outbox=false "
        "audio_persistence=false"
    )
    print("[VALIDATION] Press Ctrl+C to stop cleanly.")
    try:
        validation.run()
    finally:
        validation.stop()


def main(argv=None):
    args = parse_args(argv)
    run_validation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
