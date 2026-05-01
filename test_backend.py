from sender.http_client import send_alert, check_backend_connection

print("Testing backend connection...")
check_backend_connection()

print("\nSending test alert...")
send_alert(
    severity="high",
    confidence=0.92,
    duration=3.5
)

print("\nDone! Check your dashboard!")
