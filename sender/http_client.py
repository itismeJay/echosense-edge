import requests
import time
import threading
from config import API_URL, LOCATION

def send_alert(severity, confidence, duration,
               transcribed_text="", detected_words=None,
               categories=None, language="unknown",
               hard_hits=None, soft_hits=None,
               required_duration=None, duration_gate=None,
               yamnet_class="Unknown", yamnet_score=0.0,
               emotion="unknown", tone_data=None,
               waveform_snapshot=None, retries=3):
    payload = {
        "severity": str(severity),
        "confidence": float(round(float(confidence), 4)),
        "duration": float(round(float(duration), 2)),
        "required_duration": float(required_duration) if required_duration is not None else None,
        "duration_gate": duration_gate,
        "location": LOCATION,
        "transcribed_text": transcribed_text,
        "detected_words": detected_words or [],
        "categories": categories or [],
        "language": language,
        "hard_hits": hard_hits or [],
        "soft_hits": soft_hits or [],
        "yamnet_class": yamnet_class,
        "yamnet_score": float(round(float(yamnet_score), 4)),
        "emotion": emotion,
        "rms": round(float(tone_data.get("rms", 0)), 2) if tone_data else 0,
        "energy_variance": round(float(tone_data.get("energy_variance", 0)), 2) if tone_data else 0,
        "zero_crossing_rate": round(float(tone_data.get("zero_crossing_rate", 0)), 4) if tone_data else 0,
        "peak_to_average": round(float(tone_data.get("peak_to_average", 0)), 2) if tone_data else 0,
        "waveform_snapshot": waveform_snapshot or []
    }
    for attempt in range(retries):
        try:
            print(f"[SENDER] Sending alert... (attempt {attempt + 1})")
            response = requests.post(f"{API_URL}/alerts/", json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"[SENDER] Alert sent! ID: {data.get('id')} Severity: {data.get('severity')}")
                return True
            else:
                print(f"[SENDER] Failed! Status: {response.status_code}")
        except Exception as e:
            print(f"[SENDER] Error: {e}")
            time.sleep(3)
    return False

def _heartbeat_loop(interval):
    while True:
        time.sleep(interval)
        try:
            response = requests.post(
                f"{API_URL}/system-settings/heartbeat",
                timeout=10
            )
            print(f"[HEARTBEAT] {response.status_code}")
        except Exception as e:
            print(f"[HEARTBEAT] Error: {e}")

def start_heartbeat(interval=60):
    t = threading.Thread(target=_heartbeat_loop, args=(interval,), daemon=True)
    t.start()
    print(f"[HEARTBEAT] Started — posting every {interval}s")

def check_backend_connection():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"[SENDER] Backend connected!")
            return True
    except:
        print(f"[SENDER] Backend not reachable!")
        return False
