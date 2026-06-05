import time
import sys
from collections import deque

import numpy as np

from config import SAMPLE_RATE
from audio.capture import get_audio_device, capture_audio, terminate_audio_system
from audio.vad import is_voice_present
from audio.led_indicator import LEDIndicator
from model.yamnet_infer import load_yamnet, load_class_names
from model.whisper_stt import get_model
from detection.aggression import AggressionDetector
from sender.http_client import send_alert, check_backend_connection, start_heartbeat

# Rolling window of 1-second voiced chunks fed to STT/YAMNet (trailing context).
BUFFER_SECONDS = 5
# Start detecting once this many voiced seconds exist — enough for Whisper to
# transcribe a short utterance AND to satisfy the fastest tier (threat 1.5s).
MIN_VOICED_CHUNKS = 2
# This many consecutive silent seconds end the current utterance run.
MAX_SILENCE_TICKS = 2


def main():
    print("=" * 50)
    print("  EchoSense Edge AI System")
    print("  Acoustic Bullying Detection (5-layer)")
    print("  Davao del Norte State College")
    print("=" * 50)

    # LED indicator: 3 fast blinks while the system loads (no-op if LED is
    # unavailable — never affects the detection pipeline).
    led = LEDIndicator()
    led.startup()

    print("\n[INIT] Checking backend connection...")
    check_backend_connection()
    start_heartbeat(interval=60)

    print("[INIT] Loading YAMNet model...")
    interpreter = load_yamnet()
    class_names = load_class_names()
    print("[INIT] YAMNet loaded successfully!")

    print("[INIT] Loading Faster-Whisper model...")
    get_model()  # eager-load so the first detection isn't slow

    print("[INIT] Detecting audio device...")
    device_index = get_audio_device()
    if device_index is None:
        print("[INIT] No preferred mic found. Using system default.")
    else:
        print(f"[INIT] Using device index: {device_index}")

    detector = AggressionDetector(interpreter=interpreter, class_names=class_names)
    audio_buffer = deque(maxlen=BUFFER_SECONDS)   # trailing voiced chunks for STT/YAMNet
    voiced_samples = 0                            # total continuous voiced samples this run
    silence_ticks = 0                             # consecutive silent captures

    print("[HEARTBEAT] Started")
    print("\n[MAIN] Listening...")
    print("[INIT] Press Ctrl+C to stop\n")

    # LED indicator: slow heartbeat blink while the mic is active.
    led.listening_start()

    try:
        while True:
            try:
                # 1. Capture 1 second of audio (~15600 samples)
                audio_np, _ = capture_audio(duration=1.0, device_index=device_index)

                # 2. VAD — on silence, age out the current utterance run
                if not is_voice_present(audio_np):
                    silence_ticks += 1
                    if silence_ticks >= MAX_SILENCE_TICKS and audio_buffer:
                        audio_buffer.clear()
                        voiced_samples = 0
                    continue
                silence_ticks = 0

                # 3. Accumulate this voiced second
                audio_buffer.append(audio_np)
                voiced_samples += len(audio_np)

                # 4. Need a minimum of voiced audio before running the detector
                if len(audio_buffer) < MIN_VOICED_CHUNKS:
                    continue

                # 5. Trailing window (≤5s) for STT/YAMNet context; duration is the
                #    full continuous voiced length (uncapped, for tier + severity).
                combined = np.concatenate(list(audio_buffer))
                duration_seconds = voiced_samples / SAMPLE_RATE

                # 6. Run the 5-layer detector each second the run continues
                result = detector.process(combined, duration_seconds=duration_seconds)

                if result and result.get("should_alert"):
                    # LED indicator: 5 rapid blinks (non-blocking) on alert.
                    led.alert()
                    send_alert(
                        severity=result["severity"],
                        confidence=result["confidence"],
                        duration=result["duration"],
                        transcribed_text=result.get("transcribed_text", ""),
                        detected_words=result.get("detected_words", []),
                        categories=result.get("categories", []),
                        language=result.get("language", "unknown"),
                        hard_hits=result.get("hard_hits", []),
                        soft_hits=result.get("soft_hits", []),
                        required_duration=result.get("required_duration"),
                        duration_gate=result.get("duration_gate"),
                        yamnet_class=result.get("yamnet_class", "Unknown"),
                        yamnet_score=result.get("yamnet_score", 0.0),
                        emotion=result.get("emotion", "unknown"),
                        tone_data=result.get("tone_data", {}),
                        waveform_snapshot=result.get("waveform_snapshot", []),
                    )

            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[STOP] EchoSense stopped by user.")
    finally:
        print("[SHUTDOWN] Releasing microphone interface...")
        led.cleanup()
        terminate_audio_system()
        sys.exit(0)


if __name__ == "__main__":
    main()
