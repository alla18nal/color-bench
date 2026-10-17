# Color Bench — Hardware in the Loop (HW3)

An Arduino Uno shines a red, green and blue LED through a light-blocking strip
onto a light sensor. A LangGraph team of agents decides which measurements to
take, calls the bench through MCP, and predicts the strip's color.

```
User -> Orchestrator -> Scientist <-> Human stop
                            |  ^
              hardware turn v  | readings in the dialogue
                         Executor --MCP (stdio)--> MCP server -> Connector --USB serial--> Arduino Uno
                                                                         \--sim://-----> simulated bench
```

Full diagram: `docs/architecture.mmd` (paste into https://mermaid.live).

## Layout

| Path | What it is | HW3 |
|---|---|---|
| `firmware/color_bench.ino` | Arduino sketch, line protocol `I/D/R/G/B` | instrument |
| `connector/bench_connector.py` | `ColorBench`: sends a command, reads the measurement back (pyserial) | (a) |
| `connector/sim_bench.py` | Simulated bench, same protocol, strips and faults you can change live | HW2 bench |
| `mcp_server/server.py` | MCP server, 5 tools: `identify_bench`, `measure_dark/red/green/blue` | (b) |
| `agent/graph.py` | Orchestrator, General, Scientist, Executor, Human stop | (c) (d) |
| `agent/analysis.py` | Sanity checks, transmittance, color classification | (d) |
| `agent/prompts.py` | System prompts of the four roles | (c) |
| `agent/trace.py` | Writes a Mermaid sequence diagram of each run | (e) |
| `agent/run.py` | Command-line runner | (f) |
| `runs/<thread>/` | Per run: `dialogue.json`, `readings.json`, `sequence.mmd`, `tool_calls.jsonl` | (e) (f) |

## The experiment protocol (Scientist)

1. Ask the human: is the bench connected, the slot empty, and the bench closed and dark?
2. Yes -> one **test** through the executor (identify + one red reading).
3. **Reference**: dark, red, green, blue with no strip.
4. Ask the human to insert the strip and close the bench again.
5. **Sample**: dark, red, green, blue with the strip.
6. Python computes transmittance `T = (sample - dark) / (reference - dark)` per
   color and a predicted color; the scientist judges it and reports.

Any tool error or unreasonable data (all zeros, saturated, LEDs not changing
the reading) revokes the human's "yes": the scientist must ask for a hardware
recheck and permission, then run the test again before re-measuring. This rule
is enforced in code (`hw_confirmed`, `test_ok` in the graph state), not only in
the prompt.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
set GOOGLE_API_KEY=...              # macOS/Linux: export GOOGLE_API_KEY=...
```

Optional: `LLM_PROVIDER=openrouter` (+ `OPENROUTER_API_KEY`), `LLM_MODEL`,
`LLM_RPM` (requests per minute, default 4).

## Run

```bash
# simulated bench (no Arduino needed)
python -m agent.run --sim "What color is the strip in the bench?"

# real bench: upload the firmware, CLOSE the Serial Monitor, then
python -m agent.run --port COM5 "What color is the strip in the bench?"
```

At a question from the graph, type your answer. On the simulated bench you can
first change the bench: `/strip red`, `/fault zeros`, `/fault light_leak`, `/fault none`, `/state`.

Other entry points:

```bash
python -m connector.bench_connector COM5                      # connector alone
npx @modelcontextprotocol/inspector python -m mcp_server.server   # MCP tools by hand
python -m pytest -q tests                                     # 18 tests (no key, no Arduino)
```

The tests run the real MCP server and the simulated bench; only the LLM is
replaced by a rule-based script (`tests/scripted_llm.py`), so they need no API key.
