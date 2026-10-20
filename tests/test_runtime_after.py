import unittest
from unittest.mock import Mock, patch

from runtime import EdgeRuntime


class RuntimeAfterTests(unittest.TestCase):
    def test_stop_attempts_every_new_component_cleanup(self):
        source = Mock()
        led = Mock()
        runtime = EdgeRuntime(source=source, led=led)
        first = Mock()
        second = Mock()
        runtime.workers = [first, second]
        runtime.coordinator = Mock()
        with (
            patch("runtime.stop_recorder") as stop_recorder,
            patch("runtime.stop_alert_delivery") as stop_delivery,
        ):
            runtime.stop()
        source.stop.assert_called_once()
        first.stop.assert_called_once()
        second.stop.assert_called_once()
        stop_recorder.assert_called_once()
        runtime.coordinator.shutdown.assert_called_once_with(
            finalize_active=True
        )
        stop_delivery.assert_called_once()
        led.cleanup.assert_called_once()

    def test_enqueue_failure_never_flashes_success_led(self):
        led = Mock()
        runtime = EdgeRuntime(source=Mock(), led=led)
        with patch(
            "runtime.enqueue_finalized_incident",
            return_value=False,
        ):
            self.assertFalse(runtime._finalize(Mock()))
        led.alert.assert_not_called()

    def test_success_led_occurs_only_after_durable_enqueue(self):
        led = Mock()
        runtime = EdgeRuntime(source=Mock(), led=led)
        with patch(
            "runtime.enqueue_finalized_incident",
            return_value=True,
        ):
            self.assertTrue(runtime._finalize(Mock()))
        led.alert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
