import time
import sys
from audio.capture import get_audio_device, capture_audio
from model.yamnet_infer import load_yamnet, load_class_names, run_yamnet
from model.vosk_detect import process_audio_chunk
from detection.aggression import AggressionDetector
from sender.http_client import send_alert, check_backend_connection

def main():
    print("=" * 50)
    print("  EchoSense Edge AI System")
    print("  Acoustic Aggression Detection")
    print("  Davao del Norte State College")
    print("=" * 50)

    print("\n[INIT] Checking backend connection...")
    check_backend_connection()

    print("[INIT] Loading YAMNet model...")
    interpreter = load_yamnet()
    class_names = load_class_names()
    print("[INIT] YAMNet loaded successfully!")

    print("[INIT] Detecting audio device...")
    device_index = get_audio_device()
    if device_index is None:
        print("[INIT] ReSpeaker not found! Using default microphone.")
    else:
        print(f"[INIT] Using device index: {device_index}")

    detector = AggressionDetector()

    print("\n[INIT] EchoSense is now listening...")
    print("[INIT] Press Ctrl+C to stop\n")

    while True:
        try:
            audio_np, audio_bytes = capture_audio(
                duration=1.0,
                device_index=device_index
            )

            yamnet_class, yamnet_score, _ = run_yamnet(
                interpreter,
                audio_np,
                class_names
            )

            print(f"[YAMNET] {yamnet_class}: {yamnet_score:.2f}")

            vosk_result = process_audio_chunk(audio_bytes)
            has_profanity = vosk_result["has_profanity"]
            detected_words = vosk_result["detected_words"]
            transcribed_text = vosk_result.get("transcribed_text", "")

            if has_profanity:
                print(f"[VOSK] Profanity detected: {detected_words}")

            result = detector.process(
                yamnet_class,
                yamnet_score,
                has_profanity,
                detected_words,
                audio_np,
                transcribed_text
            )

            if has_profanity:
                word = ", ".join(detected_words) if detected_words else "?"
                aggressive = result.get("tone_aggressive", False)
                if result["should_alert"]:
                    alert_str = "YES"
                elif aggressive:
                    alert_str = "NO (building duration)"
                else:
                    alert_str = "NO (casual tone)"
                print(f"[ECHOSENSE] Word detected: {word} | "
                      f"Tone: {'AGGRESSIVE' if aggressive else 'CASUAL'} | Alert: {alert_str}")

            if result["should_alert"]:
                send_alert(
                    severity=result["severity"],
                    confidence=result["confidence"],
                    duration=result["duration"],
                    transcribed_text=result.get("transcribed_text", ""),
                    detected_words=result.get("detected_words", []),
                    yamnet_class=result.get("yamnet_class", "Unknown"),
                    yamnet_score=result.get("yamnet_score", 0.0),
                    emotion=result.get("emotion", "unknown"),
                    tone_data=result.get("tone_data", {}),
                    waveform_snapshot=result.get("waveform_snapshot", [])
                )

        except KeyboardInterrupt:
            print("\n[STOP] EchoSense stopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
