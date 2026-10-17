"""
Simulated color bench (the "simulated bench" from HW2, rebuilt for this
instrument).

It behaves like the Uno running firmware/color_bench.ino: same commands,
same replies, same 1.5 s per measurement. The connector uses it when the
port is a URL starting with  sim://  , for example

    BENCH_PORT=sim://              real timing (1.5 s per reading)
    BENCH_PORT=sim://?speed=0.05   20x faster, for tests

What sits on the simulated bench is read from a small JSON state file on
EVERY measurement, so a person (or a test) can "insert the strip" or "pull a
wire" while the experiment runs, exactly like on the real bench:

    {"strip": "red", "fault": "none"}

    strip:  none | red | green | blue | yellow | cyan | magenta | white | black
    fault:  none       everything works
            zeros      sensor wire loose -> every reading is 0
            saturated  sensor shorted    -> every reading is 1023
            dead_red   red LED broken    -> red reading equals the dark reading
            no_reply   board hangs       -> measurement commands time out
            unplugged  USB cable pulled  -> the port cannot be used
            light_leak lid open          -> room light adds to every reading

Change it from the command line:
    python -m connector.sim_bench strip red
    python -m connector.sim_bench fault zeros
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import serial

def state_file() -> Path:
    """Where the simulated bench's state lives (env SIM_STATE_FILE overrides)."""
    return Path(os.environ.get(
        "SIM_STATE_FILE",
        Path(__file__).resolve().parent.parent / "logs" / "sim_bench.json"))

# Fraction of each LED's light that passes each strip (red, green, blue).
STRIPS = {
    "none":    (1.00, 1.00, 1.00),
    "white":   (0.90, 0.90, 0.88),
    "red":     (0.85, 0.12, 0.08),
    "green":   (0.10, 0.80, 0.15),
    "blue":    (0.08, 0.18, 0.78),
    "yellow":  (0.86, 0.78, 0.10),
    "cyan":    (0.10, 0.75, 0.74),
    "magenta": (0.80, 0.12, 0.72),
    "black":   (0.03, 0.03, 0.03),
}
FAULTS = ("none", "zeros", "saturated", "dead_red", "no_reply", "unplugged",
          "light_leak")

DARK_OFFSET = 4          # dark reading of a CLOSED bench (sensor offset)
LEAK_LIGHT = 220         # extra light when the lid is open (fault light_leak)
# The three LEDs are NOT equally bright on the sensor (real LEDs and light
# sensors aren't either). This is why the experiment needs a reference
# measurement without the strip.
LED_ON_SENSOR = {"R": 760, "G": 640, "B": 330}
MEASURE_S = 1.5                           # 500 ms settle + 50 x 20 ms


def read_state() -> dict:
    try:
        s = json.loads(state_file().read_text())
    except (OSError, ValueError):
        s = {}
    return {"strip": s.get("strip", "none"), "fault": s.get("fault", "none")}


def write_state(**changes) -> dict:
    s = read_state()
    s.update(changes)
    if s["strip"] not in STRIPS:
        raise ValueError(f"unknown strip {s['strip']!r}; use one of {list(STRIPS)}")
    if s["fault"] not in FAULTS:
        raise ValueError(f"unknown fault {s['fault']!r}; use one of {list(FAULTS)}")
    state_file().parent.mkdir(parents=True, exist_ok=True)
    state_file().write_text(json.dumps(s))
    return s


class SimFirmware:
    """The firmware's logic with no I/O: feed it bytes, it returns reply lines.

    Mirrors loop() / handleCommand() in color_bench.ino line by line.
    """

    def __init__(self, noise: int = 3):
        self.noise = noise
        self.cmd, self.n = "", 0
        self.received: list[str] = []     # every complete line, for tests

    def feed(self, data: str) -> list[tuple[str, float]]:
        """Return a list of (reply_line, seconds_the_board_needs)."""
        out = []
        for c in data:
            if c == "\r":
                continue
            if c == "\n":
                if self.n > 0:
                    self.received.append(self.cmd if self.n == 1 else self.cmd + "...")
                    out.append(self._handle(self.cmd, self.n))
                self.cmd, self.n = "", 0
            else:
                if self.n == 0:
                    self.cmd = c
                self.n += 1
        return [r for r in out if r is not None]

    def _handle(self, c: str, n: int):
        if n != 1:
            return ("ERR unknown_command", 0.0)
        if c == "I":
            return ("ID COLOR_BENCH", 0.0)
        if c in "DRGB":
            state = read_state()
            if state["fault"] == "no_reply":
                return None                       # board hangs, says nothing
            return (f"LIGHT {self._light(c, state)}", MEASURE_S)
        return ("ERR unknown_command", 0.0)

    def _light(self, c: str, state: dict) -> int:
        fault = state["fault"]
        if fault == "zeros":
            return 0
        if fault == "saturated":
            return 1023
        value = DARK_OFFSET + (LEAK_LIGHT if fault == "light_leak" else 0)
        if c != "D" and not (c == "R" and fault == "dead_red"):
            t = STRIPS[state["strip"]]["RGB".index(c)]
            value += LED_ON_SENSOR[c] * t
        value += random.uniform(-self.noise, self.noise)
        return max(0, min(1023, round(value)))


class SimulatedSerial:
    """Looks like a pyserial port to the connector (read/write/flush/...)."""

    def __init__(self, url: str, timeout: float = 0.1):
        q = parse_qs(urlparse(url).query)
        self.speed = float(q.get("speed", ["1.0"])[0])
        self.timeout = timeout
        self.port = url
        self.fw = SimFirmware()
        self._check_plugged()
        self._buf = b"READY\r\n"                  # setup() prints READY
        self._pending: list[tuple[bytes, float]] = []   # (reply, ready_time)
        self.is_open = True

    def _check_plugged(self):
        if read_state()["fault"] == "unplugged":
            raise serial.SerialException(f"could not open port {self.port}: "
                                         "device not connected (simulated)")

    def write(self, data: bytes) -> int:
        self._check_plugged()
        now = time.monotonic()
        start = max([t for _, t in self._pending], default=now)
        for line, secs in self.fw.feed(data.decode("ascii")):
            start = max(start, now) + secs * self.speed
            self._pending.append(((line + "\r\n").encode(), start))
        return len(data)

    def read(self, size: int = 1) -> bytes:
        deadline = time.monotonic() + self.timeout
        while True:
            now = time.monotonic()
            while self._pending and self._pending[0][1] <= now:
                self._buf += self._pending.pop(0)[0]
            if self._buf:
                chunk, self._buf = self._buf[:size], self._buf[size:]
                return chunk
            if now >= deadline:
                return b""
            nxt = self._pending[0][1] if self._pending else deadline
            time.sleep(max(0.0, min(nxt, deadline) - now))

    def flush(self):
        pass

    def reset_input_buffer(self):
        self._buf = b""

    def close(self):
        self.is_open = False


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("strip", "fault"):
        print(write_state(**{sys.argv[1]: sys.argv[2]}))
    else:
        print(__doc__.split("Change it")[1] if len(sys.argv) > 1 else read_state())
