"""Turn a run's trace into a Mermaid sequence diagram (HW3 e).

Paste the .mmd file into https://mermaid.live to get a PNG/SVG for the PDF.
"""

PARTICIPANTS = ["User", "Orchestrator", "General", "Scientist", "Human",
                "Executor", "MCP", "Arduino"]
LABELS = {"MCP": "MCP server", "Arduino": "Arduino Uno",
          "Executor": "Executor (holds MCP tools)"}


def _clean(s: str, n: int = 70) -> str:
    s = " ".join(str(s).split())
    for bad, good in ((";", ","), ("#", ""), (":", " -"), ("<", "("), (">", ")")):
        s = s.replace(bad, good)
    return s if len(s) <= n else s[: n - 1] + "…"


def to_mermaid(trace: list[dict], title: str = "") -> str:
    used = {t["from"] for t in trace} | {t["to"] for t in trace}
    lines = ["sequenceDiagram"]
    if title:
        lines.append(f"    title {_clean(title, 90)}")
    for p in PARTICIPANTS:
        if p in used:
            kind = "actor" if p in ("User", "Human") else "participant"
            lines.append(f"    {kind} {p} as {LABELS.get(p, p)}")
    for t in trace:
        # dashed = a reply coming back from the hardware side
        reply = t["from"] == "Arduino" or (t["from"] == "MCP" and t["to"] == "Executor")
        arrow = "-->>" if reply else "->>"
        lines.append(f"    {t['from']}{arrow}{t['to']}: {_clean(t['msg'])}")
    return "\n".join(lines) + "\n"
