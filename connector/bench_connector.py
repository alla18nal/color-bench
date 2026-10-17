"""
Connector for the COLOR_BENCH Arduino Uno (HW3, Task 1a).

This is the ONLY code that talks to the hardware. It uses the interface the
instrument documents: a USB serial port at 9600 baud with a line protocol
(see firmware/color_bench.ino):

    PC sends            board answers (exactly one line)
    "I\n"          ->   "ID COLOR_BENCH"
    "D\n"          ->   "LIGHT <0-1023>"   all LEDs off   (dark reading)
    "R\n"          ->   "LIGHT <0-1023>"   red LED on
    "G\n"          ->   "LIGHT <0-1023>"   green LED on
    "B\n"          ->   "LIGHT <0-1023>"   blue LED on
    anything else  ->   "ERR unknown_command"

Everything above this file (MCP server, LangGraph agent) only ever calls
ColorBench methods and never touches the serial port directly.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import serial                      # pyserial
from serial.tools import list_ports

BAUD_RATE = 9600                   # must match Serial.begin(9600) in the firmware
EXPECTED_ID = "COLOR_BENCH"

# One measurement on the board = 500 ms settle + 50 samples * 20 ms = 1.5 s.
# We give the board twice that before calling it a timeout.
MEASURE_TIMEOUT_S = 3.0
# Opening the port resets the Uno (DTR line). The bootloader runs ~1.5-2 s
# before setup() prints "READY".
BOOT_TIMEOUT_S = 5.0

# Letter sent for each channel. "dark" = all LEDs off.
CHANNELS = {"dark": "D", "red": "R", "green": "G", "blue": "B"}


class BenchError(RuntimeError):
    """Anything that went wrong talking to the bench (timeout, bad reply,
    unplugged cable, ...). The layers above only ever need to catch this."""


@dataclass
class Measurement:
    channel: str        # "dark", "red", "green" or "blue"
    value: int          # raw ADC average, 0..1023
    duration_s: float   # wall-clock time of the round trip
    raw_reply: str      # the exact line the board sent, kept for logging


def find_arduino_port() -> str | None:
    """Guess the Uno's serial port by its USB description / vendor id.

    Genuine Unos use vendor 0x2341 (Arduino); clones usually use a CH340
    (0x1A86) or FTDI (0x0403) chip. Returns e.g. "COM5" or "/dev/ttyACM0".
    """
    arduino_vids = {0x2341, 0x2A03, 0x1A86, 0x0403}
    for p in list_ports.comports():
        desc = (p.description or "").lower()
        if p.vid in arduino_vids or "arduino" in desc or "ch340" in desc:
            return p.device
    return None


class ColorBench:
    """One open connection to the color bench."""

    def __init__(self, port: str | None = None, baud: int = BAUD_RATE):
        # Port priority: argument -> BENCH_PORT env var -> auto-detect.
        self.port = port or os.environ.get("BENCH_PORT") or find_arduino_port()
        self.baud = baud
        self._ser: serial.SerialBase | None = None
        # The board handles one command at a time. If two callers sent
        # commands at once, the replies could be swapped, so we serialize.
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- connect
    def connect(self) -> str:
        """Open the port, wait for the board to boot, and verify its identity.

        Returns the ID line. Raises BenchError if this is not a COLOR_BENCH.
        """
        if self.port is None:
            raise BenchError(
                "No Arduino port found. Plug in the Uno or set BENCH_PORT "
                "(e.g. COM5 on Windows, /dev/ttyACM0 on Linux, "
                "/dev/cu.usbmodem* on macOS)."
            )
        try:
            if self.port.startswith("sim://"):
                # The simulated bench (see connector/sim_bench.py). Same
                # protocol, no hardware. Everything below is identical.
                from connector.sim_bench import SimulatedSerial
                self._ser = SimulatedSerial(self.port, timeout=0.1)
            else:
                # serial_for_url accepts normal ports ("COM5") and pyserial URLs.
                self._ser = serial.serial_for_url(self.port, self.baud, timeout=0.1)
        except serial.SerialException as e:
            raise BenchError(f"Cannot open {self.port}: {e}. "
                             "Is the Arduino IDE Serial Monitor still open?") from e

        # Opening the port reboots the Uno. Wait for the "READY" line that
        # setup() prints. If the board did not reboot (some clones / USB hubs),
        # READY never comes; that's fine, the identify() below still checks it.
        self._read_line(BOOT_TIMEOUT_S, expect_prefix="READY", required=False)
        self._ser.reset_input_buffer()

        ident = self.identify()
        return ident

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ low level
    def _read_line(self, timeout_s: float, expect_prefix: str | None = None,
                   required: bool = True) -> str | None:
        """Read one text line (without the newline) within timeout_s.

        If expect_prefix is given, lines that don't start with it are skipped
        (e.g. a stray "READY" after a reboot).
        """
        assert self._ser is not None
        deadline = time.monotonic() + timeout_s
        buf = b""
        while time.monotonic() < deadline:
            chunk = self._ser.read(1)          # returns b"" after 0.1 s idle
            if not chunk:
                continue
            if chunk == b"\n":
                line = buf.decode("ascii", errors="replace").strip()
                buf = b""
                if not line:
                    continue
                if expect_prefix is None or line.startswith(expect_prefix):
                    return line
                if line.startswith("ERR"):     # an error is always an answer
                    return line
                continue                       # unrelated line, keep reading
            buf += chunk
        if required:
            raise BenchError(f"Timed out after {timeout_s:.1f} s waiting for "
                             f"{expect_prefix or 'a reply'} from {self.port}")
        return None

    def _command(self, letter: str, expect_prefix: str, timeout_s: float) -> str:
        """Send one command letter and return the single reply line."""
        if not self.is_connected:
            raise BenchError("Bench is not connected. Call connect() first.")
        with self._lock:
            try:
                self._ser.reset_input_buffer()     # drop anything left over
                self._ser.write(f"{letter}\n".encode("ascii"))
                self._ser.flush()
                line = self._read_line(timeout_s, expect_prefix=expect_prefix)
            except (serial.SerialException, OSError) as e:
                # USB cable pulled, board reset, ... Close the port so the
                # next call opens it again instead of reusing a dead handle.
                self.close()
                raise BenchError(f"Lost the connection to {self.port}: {e}") from e
        if line.startswith("ERR"):
            raise BenchError(f"Board rejected command {letter!r}: {line}")
        return line

    # ----------------------------------------------------------- public API
    def identify(self) -> str:
        """Ask the board who it is. Returns "ID COLOR_BENCH"."""
        line = self._command("I", "ID", timeout_s=1.0)
        if line != f"ID {EXPECTED_ID}":
            raise BenchError(f"Unexpected device on {self.port}: {line!r}")
        return line

    def measure(self, channel: str) -> Measurement:
        """Light one LED (or none for "dark") and return the sensor reading."""
        channel = channel.lower()
        if channel not in CHANNELS:
            raise BenchError(f"Unknown channel {channel!r}; "
                             f"use one of {list(CHANNELS)}")
        t0 = time.monotonic()
        line = self._command(CHANNELS[channel], "LIGHT", MEASURE_TIMEOUT_S)
        try:
            value = int(line.split()[1])
        except (IndexError, ValueError) as e:
            raise BenchError(f"Could not parse reply {line!r}") from e
        if not 0 <= value <= 1023:
            raise BenchError(f"Reading {value} outside the ADC range 0..1023")
        return Measurement(channel, value, time.monotonic() - t0, line)


if __name__ == "__main__":
    # Quick manual check with the real board:
    #     python -m connector.bench_connector            (auto-detect port)
    #     python -m connector.bench_connector COM5
    import sys
    with ColorBench(sys.argv[1] if len(sys.argv) > 1 else None) as bench:
        print("port:", bench.port, "|", bench.identify())
        for ch in CHANNELS:
            m = bench.measure(ch)
            print(f"{ch:>5}: {m.value:4d}   ({m.duration_s:.2f} s)")
