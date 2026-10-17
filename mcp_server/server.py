"""
MCP server for the color bench (HW3, Task 1b).

Each tool = one action the instrument performs or one reading it returns.
Every tool call is carried out by the connector (connector/bench_connector.py);
this file never touches the serial port itself.

Run it (stdio transport, which is what the LangGraph agent will use):
    python -m mcp_server.server
Inspect it by hand in a browser:
    npx @modelcontextprotocol/inspector python -m mcp_server.server

IMPORTANT: with the stdio transport, stdout IS the protocol channel.
Never print() here.

Uses the official MCP Python SDK (pip install "mcp<2"). Its high-level server
class is FastMCP in 1.x and MCPServer in 2.x; the import below handles both.
Logging goes to stderr and to logs/tool_calls.jsonl.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

try:                                   # MCP Python SDK 1.x (what LangChain's
    from mcp.server.fastmcp import FastMCP as MCPServer      # adapter needs)
    from mcp.server.fastmcp.exceptions import ToolError
except ModuleNotFoundError:            # SDK 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

from connector.bench_connector import BenchError, ColorBench

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [bench-mcp] %(message)s")
log = logging.getLogger(__name__)

# Every tool call is appended here as one JSON line. This is the evidence for
# the use-case table in the PDF: request -> tool call -> what the hardware did.
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "tool_calls.jsonl"

mcp = MCPServer("color-bench")

# One shared connection, opened on the first tool call ("lazy"). That way the
# server starts even if the board is unplugged, and the agent gets a clear
# error message instead of the server crashing.
_bench: ColorBench | None = None


def _get_bench() -> ColorBench:
    global _bench
    if _bench is None or not _bench.is_connected:
        _bench = ColorBench()
        ident = _bench.connect()
        log.info("connected to %s on %s", ident, _bench.port)
    return _bench


def _record(tool: str, result: dict) -> dict:
    """Write one tool call to the JSONL log and return the result unchanged."""
    entry = {"time": datetime.now().isoformat(timespec="milliseconds"),
             "tool": tool, **result}
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("%s -> %s", tool, result)
    return result


def _measure(channel: str, led: str) -> dict:
    """Shared body of the four measurement tools."""
    tool = f"measure_{channel}"
    try:
        m = _get_bench().measure(channel)
    except BenchError as e:
        _record(tool, {"ok": False, "error": str(e)})
        # ToolError -> the agent receives an error result with this text,
        # so it can decide what to do (retry, check the bench, tell the human).
        raise ToolError(str(e)) from e
    return _record(tool, {
        "ok": True,
        "channel": m.channel,
        "led": led,
        "light": m.value,
        "scale": "raw ADC 0-1023 (higher = more light on the sensor)",
        "duration_s": round(m.duration_s, 2),
    })


# ---------------------------------------------------------------- the tools
# The docstring of each tool becomes its description for the model, so it is
# written for the agent: what it does, what it returns, when to use it.

@mcp.tool()
def identify_bench() -> dict:
    """Check that the color bench is connected and answering.

    Sends the identify command. Returns the port and the board's ID line.
    Use this first, or when a measurement fails, to see if the bench is alive.
    """
    try:
        bench = _get_bench()
        ident = bench.identify()
    except BenchError as e:
        _record("identify_bench", {"ok": False, "error": str(e)})
        raise ToolError(str(e)) from e
    return _record("identify_bench", {"ok": True, "port": bench.port, "id": ident})


@mcp.tool()
def measure_dark() -> dict:
    """Read the light sensor with ALL LEDs off (dark / ambient baseline).

    Takes about 1.5 s. Returns {"light": 0-1023}. Subtract this baseline from
    the red, green and blue readings to remove ambient light.
    """
    return _measure("dark", "none")


@mcp.tool()
def measure_red() -> dict:
    """Turn the RED LED on, read how much red light passes the strip, LED off.

    Takes about 1.5 s. Returns {"light": 0-1023}; higher means the strip lets
    more red light through.
    """
    return _measure("red", "red")


@mcp.tool()
def measure_green() -> dict:
    """Turn the GREEN LED on, read how much green light passes the strip, LED off.

    Takes about 1.5 s. Returns {"light": 0-1023}; higher means the strip lets
    more green light through.
    """
    return _measure("green", "green")


@mcp.tool()
def measure_blue() -> dict:
    """Turn the BLUE LED on, read how much blue light passes the strip, LED off.

    Takes about 1.5 s. Returns {"light": 0-1023}; higher means the strip lets
    more blue light through.
    """
    return _measure("blue", "blue")


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
