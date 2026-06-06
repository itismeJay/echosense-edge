import sys
import time

from model.whisper_stt import get_recorder, transcribe_and_check
from detection.aggression import AggressionDetector
from sender.http_client import (
    send_alert,
    check_backend_connection,
    start_heartbeat,
)
from model.yamnet_infer import load_yamnet, load_class_names
from audio.led_indicator import LEDIndicator


def main():
    print("=" * 50)
    print("  EchoSense Edge AI System")
    print("  Acoustic Bullying Detection (5-layer)")
    print("  Davao del Norte State College")
    print("=" * 50)

    led = LEDIndicator()
    led.startup()

    print("\n[INIT] Checking backend connection...")
    check_backend_connection()
    start_heartbeat(interval=60)

    print("[INIT] Loading YAMNet model...")
    interpreter = load_yamnet()
    class_names = load_class_names()
    print("[INIT] YAMNet loaded!")

    print("[INIT] Loading RealtimeSTT...")
    get_recorder()

    detector = AggressionDetector(
        interpreter=interpreter,
        class_names=class_names,
    )

    led.listening_start()
    print("\n[HEARTBEAT] Started")
    print("\n[MAIN] Listening...")
    print("[INIT] Press Ctrl+C to stop\n")

    try:
        while True:
            result = transcribe_and_check()

            if not result or not result.get("has_profanity"):
                continue

            alert = detector.process_text(result)

            if alert and alert.get("should_alert"):
                led.alert()
                send_alert(
                    severity=alert["severity"],
                    confidence=alert["confidence"],
                    duration=alert["duration"],
                    transcribed_text=alert.get("transcribed_text", ""),
                    detected_words=alert.get("detected_words", []),
                    categories=alert.get("categories", []),
                    yamnet_class=alert.get("yamnet_class", "Speech"),
                    yamnet_score=alert.get("yamnet_score", 0.0),
                    emotion=alert.get("emotion", "neutral"),
                    tone_data=alert.get("tone_data", {}),
                    waveform_snapshot=alert.get("waveform_snapshot", []),
                    language=alert.get("language", "tl"),
                    hard_hits=alert.get("hard_hits", []),
                    soft_hits=alert.get("soft_hits", []),
                    duration_gate=alert.get("duration_gate", ""),
                    required_duration=alert.get("required_duration", 0),
                )

    except KeyboardInterrupt:
        print("\n[STOP] EchoSense stopped.")
    finally:
        led.cleanup()
        sys.exit(0)


if __name__ == "__main__":
    main()
