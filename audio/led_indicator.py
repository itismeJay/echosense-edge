"""
LED status indicator for EchoSense on the Raspberry Pi 5 built-in green ACT LED.

No extra hardware or wiring — we drive /sys/class/leds/ACT directly.

States
------
    STARTUP     -> blink 3 times fast        (system loading)
    LISTENING   -> slow blink every 2 s       (mic active, background heartbeat)
    ALERT FIRED -> blink rapidly 5 times      (bullying detected!)
    ERROR / off -> LED stays OFF

Design notes
------------
* The ACT LED files are root-owned. The systemd unit grants the `gpio` group
  write access via an `ExecStartPre=+chmod` step (echosense is in `gpio`). If
  the files are not writable (e.g. a plain dev run without that grant), every
  method degrades to a silent no-op — the LED simply does nothing and the
  detection pipeline is never affected.
* The kernel's default ACT trigger is `mmc0` (SD-card activity). We switch the
  trigger to `none` so our manual brightness writes are not overwritten, and we
  restore the original trigger on cleanup().
* This module NEVER raises into the caller. Every public method is wrapped so a
  failing LED can never crash or slow the audio loop.
"""

import os
import threading

# Raspberry Pi 5 ACT LED sysfs paths
ACT_LED_PATH = "/sys/class/leds/ACT/brightness"
ACT_LED_TRIGGER = "/sys/class/leds/ACT/trigger"
ACT_LED_MAX = "/sys/class/leds/ACT/max_brightness"


def _write_led(value: str, path: str = ACT_LED_PATH) -> bool:
    """Write a raw value to an LED sysfs file. Returns True on success.

    All errors (missing file, permission denied) are swallowed and reported as
    False so callers can degrade gracefully.
    """
    try:
        with open(path, "w") as f:
            f.write(value)
        return True
    except OSError:
        return False


def _read_active_trigger() -> str:
    """Return the currently active LED trigger (the one in [brackets])."""
    try:
        with open(ACT_LED_TRIGGER) as f:
            content = f.read()
        for token in content.split():
            if token.startswith("[") and token.endswith("]"):
                return token[1:-1]
    except OSError:
        pass
    return "none"


def _read_on_value() -> str:
    """The brightness value that means 'fully on' (max_brightness, default 1)."""
    try:
        with open(ACT_LED_MAX) as f:
            return f.read().strip() or "1"
    except OSError:
        return "1"


class LEDIndicator:
    """Thread-safe controller for the ACT LED status indicator."""

    def __init__(self):
        self._on = _read_on_value()
        self._off = "0"
        self._io_lock = threading.Lock()          # serializes all LED writes
        self._listen_thread = None
        self._listen_stop = threading.Event()
        self._saved_trigger = None

        # Probe writability up front so the rest of the class can no-op cleanly.
        self.available = self._claim_led()

    # ------------------------------------------------------------------ setup
    def _claim_led(self) -> bool:
        """Take manual control of the LED: save + disable the kernel trigger.

        Returns False if the LED is unavailable / not writable, in which case
        every public method becomes a no-op.
        """
        if not os.path.exists(ACT_LED_PATH):
            return False
        # Remember the current trigger so cleanup() can restore it.
        self._saved_trigger = _read_active_trigger()
        # Setting the trigger to "none" hands brightness control to us.
        if not _write_led("none", ACT_LED_TRIGGER):
            return False
        # Confirm we can actually drive brightness.
        return _write_led(self._off)

    # --------------------------------------------------------------- low-level
    def _set(self, on: bool):
        _write_led(self._on if on else self._off)

    def _blink(self, times: int, on_s: float, off_s: float):
        """Blink `times`, holding the LED low between/after pulses."""
        if not self.available:
            return
        with self._io_lock:
            for _ in range(times):
                self._set(True)
                self._sleep(on_s)
                self._set(False)
                self._sleep(off_s)

    @staticmethod
    def _sleep(seconds: float):
        # Local import keeps the module import side-effect free.
        import time
        time.sleep(seconds)

    # ------------------------------------------------------------ public states
    def startup(self):
        """STARTUP: 3 fast blinks (blocking, ~0.6 s — runs once at boot)."""
        self._blink(3, 0.08, 0.12)

    def alert(self):
        """ALERT FIRED: 5 rapid blinks. Non-blocking — runs in its own thread so
        the detection loop is never delayed."""
        if not self.available:
            return
        threading.Thread(
            target=self._blink, args=(5, 0.07, 0.07), daemon=True
        ).start()

    def listening_start(self):
        """LISTENING: start the background heartbeat (brief flash every 2 s)."""
        if not self.available or (self._listen_thread and self._listen_thread.is_alive()):
            return
        self._listen_stop.clear()
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

    def listening_stop(self):
        """Stop the listening heartbeat and leave the LED off."""
        self._listen_stop.set()
        if self._listen_thread:
            self._listen_thread.join(timeout=2.5)
            self._listen_thread = None
        self._safe_off()

    def _listen_loop(self):
        # Flash on briefly, then stay dark ~2 s — a calm "I'm alive" heartbeat.
        while not self._listen_stop.is_set():
            with self._io_lock:
                self._set(True)
                self._sleep(0.05)
                self._set(False)
            # Wait ~2 s but stay responsive to stop().
            self._listen_stop.wait(timeout=2.0)

    def error(self):
        """ERROR: stop any activity and leave the LED OFF."""
        self.listening_stop()

    def off(self):
        self._safe_off()

    def _safe_off(self):
        if self.available:
            with self._io_lock:
                self._set(False)

    # ----------------------------------------------------------------- teardown
    def cleanup(self):
        """Stop threads, turn the LED off, and restore the original trigger."""
        self.listening_stop()
        if self.available and self._saved_trigger:
            _write_led(self._saved_trigger, ACT_LED_TRIGGER)
