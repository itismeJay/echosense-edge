import requests
import time
from config import API_URL, LOCATION

def send_alert(severity: str, confidence: float, duration: float, retries: int = 3):
    payload = {"severity": severity, "confidence": round(confidence, 4), "duration": round(duration, 2), "location": LOCATION}
    for attempt in range(retries):
        try:
            print(f"[SENDER] Sending alert... (attempt {attempt + 1})")
            response = requests.post(f"{API_URL}/alerts", json=payload, timeout=10)
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

def check_backend_connection():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"[SENDER] Backend connected!")
            return True
    except:
        print(f"[SENDER] Backend not reachable!")
        return False
