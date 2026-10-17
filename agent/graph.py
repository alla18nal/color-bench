"""
The LangGraph system (HW1 roles + HW2 human stop + HW3 hardware turn).

    START -> orchestrator --route--> scientist | general | human | END
    general      -> orchestrator
    scientist    -> human      (ask: hardware check / insert strip / re-run?)
                 -> executor   (the HARDWARE TURN: measure)
                 -> orchestrator (final report)
    executor     -> scientist  (the result is in the dialogue; the scientist
                                reads it before it decides again)
    human        -> whoever asked (scientist or orchestrator), or END

The executor is the ONLY node that holds the MCP tools. Which tool it calls
is decided by its LLM from the dialogue (HW3 c).
"""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from agent.analysis import check_set, check_test, latest_set, notebook
from agent.prompts import EXEC_PROMPT, GEN_PROMPT, ORCH_PROMPT, SCI_PROMPT

MAX_ORCH_DECISIONS = 8      # orchestrator turns per request
MAX_SCI_TURNS = 25          # scientist turns per request (safety stop)
MAX_TOOL_ROUNDS = 6         # model <-> tool round trips inside one hardware turn

# For the sequence diagram: which firmware command each tool sends.
TOOL_COMMAND = {"identify_bench": "I", "measure_dark": "D", "measure_red": "R",
                "measure_green": "G", "measure_blue": "B"}


# ===================================================================== state
class State(TypedDict):
    dialogue: Annotated[list, operator.add]   # [{"role", "text"}], shared by all
    trace: Annotated[list, operator.add]      # [{"from","to","msg"}] -> sequence diagram
    readings: Annotated[list, operator.add]   # every hardware reading, tagged
    next_role: str
    decisions: int                            # orchestrator turns so far
    sci_turns: int                            # scientist turns so far
    # --- human stop
    question: str
    question_kind: str                        # hardware_check | strip_change | other
    asker: str                                # who asked the human
    pending_action: str                       # orchestrator: none | finish
    # --- hardware bookkeeping (written by code, never by an LLM)
    hw_confirmed: bool                        # human said the bench is ready
    test_ok: bool                             # a test passed after that "yes"
    last_error: str
    request: str                              # scientist -> executor instruction
    purpose: str                              # test | reference | sample
    run_count: int                            # hardware turns so far


def initial_state(user_request: str) -> dict:
    return {"dialogue": [{"role": "user", "text": user_request}],
            "trace": [{"from": "User", "to": "Orchestrator", "msg": "request"}],
            "readings": [], "next_role": "", "decisions": 0, "sci_turns": 0,
            "question": "", "question_kind": "other", "asker": "",
            "pending_action": "none", "hw_confirmed": False, "test_ok": False,
            "last_error": "", "request": "", "purpose": "", "run_count": 0}


def dialogue_text(dialogue) -> str:
    return "\n\n".join(f"[{t['role']}]: {t['text']}" for t in dialogue)


# ========================================================= structured outputs
class Decision(BaseModel):
    next_role: Literal["scientist", "general", "human", "finish"]
    instruction: str = Field(description="What the chosen role should do next")
    question: str = Field(default="", description="Only when next_role is human: "
                          "the exact question to ask the person")
    pending_action: Literal["none", "finish"] = Field(
        default="none", description="'finish' only when asking the human "
        "whether to stop; otherwise 'none'")
    reason: str = Field(description="One sentence explaining the choice")


class SciAction(BaseModel):
    thought: str = Field(description="1-3 sentences of reasoning")
    action: Literal["ask_human", "measure", "report"]
    question: str = Field(default="", description="ask_human: the exact question")
    question_kind: Literal["hardware_check", "strip_change", "other"] = Field(
        default="other", description="ask_human: what the question is about")
    purpose: Literal["test", "reference", "sample"] = Field(
        default="test", description="measure: which part of the protocol")
    request: str = Field(default="", description="measure: instruction to the executor")
    report: str = Field(default="", description="report: the final answer for the user")


ACCEPT = {"y", "yes", "ok", "okay", "sure", "yep", "yeah", "ready", "done",
          "proceed", "go", "accept", "confirmed", "inserted", "da", "ayo"}


def accepted(reply: str) -> bool:
    """True if the first word of the human's reply is a yes."""
    words = re.findall(r"[a-z]+", reply.lower())
    return bool(words) and words[0] in ACCEPT


def _structured(llm, schema):
    r = llm.with_structured_output(schema)
    return r.with_retry(stop_after_attempt=3) if hasattr(r, "with_retry") else r


def _text(content) -> str:
    """Tool/LLM content can be a string or a list of blocks; return plain text."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                   for b in content)


# ===================================================================== graph
def build_graph(llm, tools, checkpointer=None):
    """llm: a LangChain chat model. tools: the MCP tools (LangChain format)."""
    decider = _structured(llm, Decision)
    sci_llm = _structured(llm, SciAction)
    exec_llm = llm.bind_tools(tools)            # ONLY the executor gets the tools
    tools_by_name = {t.name: t for t in tools}

    # ------------------------------------------------------ orchestrator
    async def orchestrator(state):
        d = await decider.ainvoke([
            ("system", ORCH_PROMPT),
            ("human", "DIALOGUE SO FAR:\n\n" + dialogue_text(state["dialogue"]))])
        text = f"NEXT: {d.next_role}\nINSTRUCTION: {d.instruction}\nREASON: {d.reason}"
        if d.next_role == "human":
            text += f"\nQUESTION: {d.question}"
        target = {"finish": "User", "human": "Human"}.get(d.next_role,
                                                           d.next_role.capitalize())
        return {"dialogue": [{"role": "orchestrator", "text": text}],
                "trace": [{"from": "Orchestrator", "to": target,
                           "msg": d.question or d.instruction}],
                "next_role": d.next_role, "question": d.question,
                "question_kind": "other", "asker": "orchestrator",
                "pending_action": d.pending_action,
                "decisions": state["decisions"] + 1}

    def route(state):
        if state["next_role"] == "finish" or state["decisions"] >= MAX_ORCH_DECISIONS:
            return END
        return state["next_role"]

    # ------------------------------------------------------ general
    async def general(state):
        m = await llm.ainvoke([
            ("system", GEN_PROMPT),
            ("human", "DIALOGUE SO FAR:\n\n" + dialogue_text(state["dialogue"]))])
        return {"dialogue": [{"role": "general", "text": _text(m.content)}],
                "trace": [{"from": "General", "to": "Orchestrator", "msg": "answer"}]}

    # ------------------------------------------------------ scientist
    async def scientist(state) -> Command[Literal["human", "executor", "orchestrator"]]:
        nb = notebook(state["readings"], state["hw_confirmed"], state["test_ok"],
                      state["last_error"])
        turns = state["sci_turns"] + 1

        if turns > MAX_SCI_TURNS:
            a = SciAction(thought="Too many steps.", action="report",
                          report="Experiment stopped: step limit reached "
                                 "without a reliable result.")
        else:
            a = await sci_llm.ainvoke([
                ("system", SCI_PROMPT),
                ("human", f"LAB NOTEBOOK:\n{nb}\n\nDIALOGUE SO FAR:\n\n"
                          + dialogue_text(state["dialogue"]))])

        # ---- guards: the protocol rules the user asked for, enforced in code.
        # The LLM decides; these only stop it from skipping a safety step.
        notes = []
        if a.action == "measure" and not state["hw_confirmed"]:
            notes.append("guard: no measurement before the human confirms the "
                         "hardware; asking first")
            a = SciAction(
                thought=a.thought, action="ask_human", question_kind="hardware_check",
                question=(f"{state['last_error'] + ' ' if state['last_error'] else ''}"
                          "Before I use the bench: is the Arduino plugged in by USB, "
                          "the firmware uploaded, the Serial Monitor closed, the slot "
                          "set up and the bench CLOSED and DARK (no room light)? "
                          "May I run the measurement? (yes/no)"))
        elif a.action == "measure" and a.purpose != "test" and not state["test_ok"]:
            notes.append("guard: running the connection test before the experiment")
            a = SciAction(thought=a.thought, action="measure", purpose="test",
                          request="Connection test: identify the bench, then take "
                                  "one red reading.")

        upd = {"sci_turns": turns}
        text = f"(thinking) {a.thought}"
        if notes:
            text += "\n" + "\n".join(f"[{n}]" for n in notes)

        if a.action == "ask_human":
            text += f"\nQUESTION FOR THE HUMAN: {a.question}"
            upd.update(question=a.question, question_kind=a.question_kind,
                       asker="scientist")
            if a.question_kind == "hardware_check":
                # Asking for a (re)check means the hardware is not trusted
                # again until the human says yes and a test passes.
                upd.update(hw_confirmed=False, test_ok=False)
            upd["dialogue"] = [{"role": "scientist", "text": text}]
            upd["trace"] = [{"from": "Scientist", "to": "Human", "msg": a.question}]
            return Command(update=upd, goto="human")

        if a.action == "measure":
            text += f"\nMEASUREMENT REQUEST ({a.purpose}): {a.request}"
            upd.update(request=a.request, purpose=a.purpose)
            upd["dialogue"] = [{"role": "scientist", "text": text}]
            upd["trace"] = [{"from": "Scientist", "to": "Executor",
                             "msg": f"{a.purpose}: {a.request}"}]
            return Command(update=upd, goto="executor")

        text += f"\nREPORT: {a.report}"
        upd["dialogue"] = [{"role": "scientist", "text": text}]
        upd["trace"] = [{"from": "Scientist", "to": "Orchestrator", "msg": "report"}]
        return Command(update=upd, goto="orchestrator")

    # ------------------------------------------------------ executor = HARDWARE TURN
    async def executor(state):
        run = state["run_count"] + 1
        purpose = state["purpose"]
        msgs = [SystemMessage(EXEC_PROMPT),
                HumanMessage(f"DIALOGUE SO FAR:\n\n{dialogue_text(state['dialogue'])}"
                             f"\n\nSCIENTIST'S REQUEST NOW ({purpose}): "
                             f"{state['request']}")]
        trace, readings, lines = [], [], []
        error, final = "", ""

        for _ in range(MAX_TOOL_ROUNDS):
            ai = await exec_llm.ainvoke(msgs)           # the model decides the call
            msgs.append(ai)
            if not ai.tool_calls:
                final = _text(ai.content).strip()
                break
            for tc in ai.tool_calls:
                name = tc["name"]
                if error or name not in tools_by_name:
                    why = "skipped after an error" if error else f"unknown tool {name}"
                    msgs.append(ToolMessage(why, tool_call_id=tc["id"]))
                    continue
                trace.append({"from": "Executor", "to": "MCP", "msg": f"{name}()"})
                trace.append({"from": "MCP", "to": "Arduino",
                              "msg": f"'{TOOL_COMMAND.get(name, '?')}' over USB serial"})
                res = await tools_by_name[name].ainvoke(
                    {"type": "tool_call", "id": tc["id"], "name": name,
                     "args": tc.get("args", {})})
                out = _text(res.content)
                msgs.append(ToolMessage(out, tool_call_id=tc["id"]))
                try:
                    data = json.loads(out)
                except ValueError:
                    data = {"ok": False, "error": out}
                if not data.get("ok"):
                    error = data.get("error", out)
                    trace.append({"from": "Arduino", "to": "MCP", "msg": "no valid reply"})
                    trace.append({"from": "MCP", "to": "Executor", "msg": f"ERROR {error[:50]}"})
                    lines.append(f"{name}() -> ERROR: {error}")
                elif "light" in data:
                    readings.append({"run": run, "purpose": purpose,
                                     "channel": data["channel"], "light": data["light"]})
                    trace.append({"from": "Arduino", "to": "MCP", "msg": f"LIGHT {data['light']}"})
                    trace.append({"from": "MCP", "to": "Executor", "msg": f"light={data['light']}"})
                    lines.append(f"{name}() -> light={data['light']} "
                                 f"(LED {data['led']}, {data['duration_s']} s)")
                else:
                    trace.append({"from": "Arduino", "to": "MCP", "msg": data.get("id", "ok")})
                    trace.append({"from": "MCP", "to": "Executor", "msg": data.get("id", "ok")})
                    lines.append(f"{name}() -> {data.get('id', data)} on {data.get('port')}")

        if not lines:
            lines.append("no tool was called")
        text = (f"HARDWARE TURN #{run} ({purpose})\n" + "\n".join(lines)
                + (f"\n{final}" if final else ""))

        # ---- bookkeeping in code: errors or bad data revoke the "yes"
        upd = {"run_count": run, "readings": readings, "last_error": error}
        all_readings = state["readings"] + readings
        if purpose == "test":
            test_vals = latest_set(all_readings, "test")
            identified = any("ID COLOR_BENCH" in l for l in lines)
            upd["test_ok"] = (not error and identified and bool(test_vals)
                              and not check_test(test_vals))
        # judge only THIS turn's data (an old bad set is the scientist's to redo)
        this_set = latest_set(all_readings, purpose) if readings else {}
        if purpose == "test":
            problems = check_test(this_set) if this_set else []
        else:
            problems = (check_set(this_set, purpose, latest_set(all_readings, "reference"))
                        if this_set else [])
        hardware_problems = [p for p in problems if "missing channels" not in p]
        if problems:
            text += "\nDATA CHECK FAILED: " + "; ".join(problems)
        if error or hardware_problems or (purpose == "test" and not upd["test_ok"]):
            # the bench is no longer trusted: the scientist must ask again
            upd.update(hw_confirmed=False, test_ok=False)

        trace.append({"from": "Executor", "to": "Scientist",
                      "msg": "measurements written to dialogue" if not error else "error"})
        upd["dialogue"] = [{"role": "executor", "text": text}]
        upd["trace"] = trace
        return upd

    # ------------------------------------------------------ human stop
    def human(state) -> Command[Literal["orchestrator", "scientist", "__end__"]]:
        reply = interrupt({"question": state["question"], "asker": state["asker"],
                           "kind": state["question_kind"]})
        turns = [{"role": "human", "text": reply}]
        asker = state["asker"] or "orchestrator"
        trace = [{"from": "Human", "to": asker.capitalize(), "msg": reply}]

        if asker == "scientist":
            upd = {"dialogue": turns, "trace": trace}
            if state["question_kind"] == "hardware_check":
                yes = accepted(reply)
                upd.update(hw_confirmed=yes, test_ok=False,
                           last_error="" if yes else state["last_error"])
                turns.append({"role": "system", "text":
                              "The human CONFIRMED the hardware; a test must pass "
                              "before the experiment." if yes else
                              "The human did NOT confirm the hardware. No "
                              "measurement may run."})
            return Command(update=upd, goto="scientist")

        if state["pending_action"] == "finish" and accepted(reply):
            return Command(update={"dialogue": turns, "trace": trace}, goto=END)
        return Command(update={"dialogue": turns, "trace": trace,
                               "pending_action": "none"}, goto="orchestrator")

    # ------------------------------------------------------ wiring
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("general", general)
    g.add_node("scientist", scientist)
    g.add_node("executor", executor)
    g.add_node("human", human)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", route,
                            ["scientist", "general", "human", END])
    g.add_edge("general", "orchestrator")
    g.add_edge("executor", "scientist")        # hardware result -> scientist decides again
    # scientist and human choose their next node with Command(goto=...)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
