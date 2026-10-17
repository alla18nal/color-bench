"""
Run the color-bench agent system from a terminal.

    # real bench (auto-detects the Uno, or set BENCH_PORT=COM5)
    python -m agent.run "What color is the strip in the bench?"

    # simulated bench (HW2-style), no Arduino needed
    python -m agent.run --sim "What color is the strip in the bench?"

When the graph stops for the human, type your answer. On the simulated bench
you can also change what sits on it before answering:
    /strip red        put a red strip in the slot (none, green, blue, yellow, ...)
    /fault zeros      break something (none, zeros, saturated, dead_red, no_reply,
                      unplugged, light_leak)
    /state            show the simulated bench

Each run is saved in runs/<thread>/: dialogue.json, readings.json,
sequence.mmd (the sequence diagram) and the MCP tool-call log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.types import Command

from agent.graph import build_graph, initial_state
from agent.trace import to_mermaid

ROOT = Path(__file__).resolve().parent.parent
TOOL_LOG = ROOT / "logs" / "tool_calls.jsonl"

COLORS = {"user": "97", "orchestrator": "95", "general": "36", "scientist": "93",
          "executor": "92", "human": "96", "system": "90"}
if os.name == "nt":
    os.system("")                       # turn on ANSI colors in Windows terminals


def show(turn: dict) -> None:
    c = COLORS.get(turn["role"], "0")
    print(f"\n\033[1;{c}m[{turn['role'].upper()}]\033[0m \033[{c}m{turn['text']}\033[0m",
          flush=True)


def mcp_client() -> MultiServerMCPClient:
    """The executor's MCP connection: our server, started as a subprocess."""
    return MultiServerMCPClient({"bench": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "mcp_server.server"],
        "cwd": str(ROOT),
        "env": dict(os.environ),        # passes BENCH_PORT / SIM_STATE_FILE
    }})


def sim_command(reply: str) -> bool:
    """Handle /strip, /fault, /state on the simulated bench. True if handled."""
    from connector.sim_bench import read_state, write_state
    parts = reply.split()
    try:
        if parts[0] == "/state":
            print("  sim bench:", read_state())
        elif parts[0] in ("/strip", "/fault") and len(parts) == 2:
            print("  sim bench:", write_state(**{parts[0][1:]: parts[1]}))
        else:
            print("  usage: /strip <color> | /fault <name> | /state")
    except ValueError as e:
        print("  ", e)
    return True


async def run(app, request: str, thread_id: str, sim: bool, answers=None):
    """Run one request to the end, pausing for the human when asked.

    answers: optional list of scripted human replies (for tests); otherwise
    the replies are read from the keyboard.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inp = initial_state(request)
    show(inp["dialogue"][0])
    while True:
        pause = None
        async for chunk in app.astream(inp, config, stream_mode="updates"):
            for node, upd in chunk.items():
                if node == "__interrupt__":
                    pause = upd[0].value
                elif upd:
                    for turn in upd.get("dialogue", []):
                        if turn["role"] != "human":
                            show(turn)
        if pause is None:
            break
        print(f"\n\033[1;96m=== THE GRAPH ASKS YOU ({pause['asker']}) ===\033[0m\n"
              f"{pause['question']}")
        while True:
            if answers is not None:
                reply = answers.pop(0)
                print(f"your reply > {reply}")
            else:
                reply = input("your reply > ").strip()
            if sim and reply.startswith("/"):
                sim_command(reply)
                continue
            if reply:
                break
        inp = Command(resume=reply)
    return (await app.aget_state(config)).values


def save(state: dict, thread_id: str, request: str, log_start: int) -> Path:
    out = ROOT / "runs" / thread_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "dialogue.json").write_text(json.dumps(state["dialogue"], indent=2,
                                                  ensure_ascii=False), encoding="utf-8")
    (out / "readings.json").write_text(json.dumps(state["readings"], indent=2))
    (out / "sequence.mmd").write_text(to_mermaid(state["trace"], request), encoding="utf-8")
    if TOOL_LOG.exists():               # only this run's MCP tool calls
        lines = TOOL_LOG.read_text(encoding="utf-8").splitlines()[log_start:]
        (out / "tool_calls.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


async def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", nargs="?", default="What color is the strip in the bench?")
    ap.add_argument("--sim", action="store_true", help="use the simulated bench")
    ap.add_argument("--sim-speed", type=float, default=1.0,
                    help="simulated time per reading x this (default 1.0 = 1.5 s)")
    ap.add_argument("--port", help="serial port, e.g. COM5 (default: auto-detect)")
    ap.add_argument("--thread", default=datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    args = ap.parse_args(argv)

    if args.sim:
        from connector.sim_bench import write_state
        os.environ["BENCH_PORT"] = f"sim://?speed={args.sim_speed}"
        print("Simulated bench. Slot starts EMPTY:", write_state(strip="none", fault="none"))
        print("At a question you can type /strip <color>, /fault <name>, /state")
    elif args.port:
        os.environ["BENCH_PORT"] = args.port

    from agent.config import make_llm
    log_start = len(TOOL_LOG.read_text(encoding="utf-8").splitlines()) \
        if TOOL_LOG.exists() else 0

    # One MCP session for the whole run: one server process, one serial
    # connection, so the Uno is not reset before every measurement.
    async with mcp_client().session("bench") as session:
        tools = await load_mcp_tools(session)
        print("MCP tools given to the executor:", [t.name for t in tools])
        app = build_graph(make_llm(), tools)
        state = await run(app, args.request, args.thread, args.sim)

    out = save(state, args.thread, args.request, log_start)
    print(f"\nSaved dialogue, readings, sequence diagram and tool log to {out}")


if __name__ == "__main__":
    asyncio.run(main())
