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

    # Check backend connection
    print("\n[INIT] Checking backend connection...")
    check_backend_connection()

    # Load YAMNet
    print("[INIT] Loading YAMNet model...")
    interpreter = load_yamnet()
    class_names = load_class_names()
    print("[INIT] YAMNet loaded successfully!")

    # Get audio device
    print("[INIT] Detecting audio device...")
    device_index = get_audio_device()
    if device_index is None:
        print("[INIT] ReSpeaker not found! Using default microphone.")
    else:
        print(f"[INIT] Using device index: {device_index}")

    # Initialize aggression detector
    detector = AggressionDetector()

    print("\n[INIT] EchoSense is now listening...")
    print("[INIT] Press Ctrl+C to stop\n")

    # Main loop
    while True:
        try:
            # Capture audio
            audio_np, audio_bytes = capture_audio(
                duration=1.0,
                device_index=device_index
            )

            # Run YAMNet
            yamnet_class, yamnet_score, _ = run_yamnet(
                interpreter,
                audio_np,
                class_names
            )

            print(f"[YAMNET] {yamnet_class}: {yamnet_score:.2f}")

            # Run Vosk
            vosk_result = process_audio_chunk(audio_bytes)
            has_profanity = vosk_result["has_profanity"]
            detected_words = vosk_result["detected_words"]

            if has_profanity:
                print(f"[VOSK] Profanity detected: {detected_words}")

            # Run aggression detection
            result = detector.process(
                yamnet_class,
                yamnet_score,
                has_profanity,
                detected_words
            )

            # Send alert if needed
            if result["should_alert"]:
                send_alert(
                    severity=result["severity"],
                    confidence=result["confidence"],
                    duration=result["duration"]
                )

        except KeyboardInterrupt:
            print("\n[STOP] EchoSense stopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()