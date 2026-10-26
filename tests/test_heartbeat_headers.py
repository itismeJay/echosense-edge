import unittest

from sender.http_client import heartbeat_headers


class HeartbeatHeaderTests(unittest.TestCase):
    def test_heartbeat_sends_device_credentials_when_configured(self):
        self.assertEqual(
            heartbeat_headers("grade6-pi", " secret-key "),
            {"X-EchoSense-Device-Id": "grade6-pi", "X-EchoSense-Device-Key": "secret-key"},
        )

    def test_heartbeat_sends_no_credentials_when_missing(self):
        self.assertEqual(heartbeat_headers("", "key"), {})
        self.assertEqual(heartbeat_headers("device", None), {})


if __name__ == "__main__":
    unittest.main()
