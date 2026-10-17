"""
Deterministic data analysis for the scientist.

The LLM does not do arithmetic. Python computes the numbers and the sanity
checks; the scientist (LLM) reads them in its "lab notebook", judges them,
and decides what to do next (report, re-measure, ask the human).

Physics in one paragraph
------------------------
The bench is used CLOSED and DARK, so room light is (ideally) zero:
    reading(LED) = offset + LED_brightness * transmittance(strip)
`offset` is what the sensor reads in the dark (an LDR divider rarely reads
exactly 0). The LEDs are not equally bright on the sensor, so raw R/G/B values
cannot be compared directly. We therefore measure twice:
    reference : no strip   -> offset + LED_brightness
    sample    : strip in   -> offset + LED_brightness * T
    T_c = (sample_c - sample_dark) / (reference_c - reference_dark)
T_c is the fraction of red / green / blue light the strip lets through.
In a truly dark bench the dark reading is ~0 and this is just sample/reference.
We still measure and subtract it: it costs 1.5 s, removes the sensor offset,
and it is the CHECK that the bench really is dark (see DARK_FRACTION).
"""

from __future__ import annotations

CHANNELS = ("dark", "red", "green", "blue")
LEDS = ("red", "green", "blue")
ADC_MAX = 1023
MIN_LED_SIGNAL = 2      # an LED must raise the reading by at least this much
# The bench must be dark: the dark reading may be at most this fraction of the
# weakest LED's reference signal. Above it, room light is leaking in.
DARK_FRACTION = 0.10


def latest_set(readings: list[dict], purpose: str) -> dict[str, int]:
    """Values from the LATEST hardware turn that measured this purpose.

    readings: [{"run": 3, "purpose": "reference", "channel": "red", "light": 757}, ...]
    Returns e.g. {"dark": 45, "red": 800, "green": 690, "blue": 380}
    (only the channels measured in that turn). A re-run therefore replaces
    the old set completely instead of mixing old and new channels.
    """
    runs = [r["run"] for r in readings if r["purpose"] == purpose]
    if not runs:
        return {}
    last = max(runs)
    return {r["channel"]: r["light"] for r in readings
            if r["purpose"] == purpose and r["run"] == last}


def check_test(values: dict[str, int]) -> list[str]:
    """Problems with the one-shot test measurement (empty list = OK)."""
    if not values:
        return ["the test produced no reading"]
    problems = []
    for ch, v in values.items():
        if v <= 0:
            problems.append(f"{ch} reads 0: sensor not connected or A0 wire loose")
        elif v >= ADC_MAX:
            problems.append(f"{ch} reads {ADC_MAX} (saturated): sensor shorted to 5V "
                            "or wired wrong")
    return problems


def check_set(values: dict[str, int], purpose: str,
              reference: dict[str, int] | None = None) -> list[str]:
    """Sanity checks on one full set (dark, red, green, blue).

    For a sample, pass the reference set too: its dark reading must match the
    reference's (the lid was closed again after inserting the strip).
    Returns a list of human-readable problems; empty list = data look sane.
    """
    missing = [c for c in CHANNELS if c not in values]
    if missing:
        return [f"{purpose}: missing channels {missing}"]

    problems = []
    if all(values[c] == 0 for c in CHANNELS):
        problems.append(f"{purpose}: every reading is 0 -> sensor disconnected "
                        "or A0 wire loose")
    elif any(values[c] >= ADC_MAX for c in CHANNELS):
        sat = [c for c in CHANNELS if values[c] >= ADC_MAX]
        problems.append(f"{purpose}: {sat} saturated at {ADC_MAX} -> sensor "
                        "shorted, wrong resistor, or strong ambient light")
    elif len({values[c] for c in CHANNELS}) == 1:
        problems.append(f"{purpose}: all four readings identical "
                        f"({values['dark']}) -> LEDs not reaching the sensor")

    if purpose == "reference" and not problems:
        for led in LEDS:
            signal = values[led] - values["dark"]
            if signal < MIN_LED_SIGNAL:
                problems.append(
                    f"reference: {led} LED adds only {signal} over dark "
                    f"(need >= {MIN_LED_SIGNAL}) -> {led} LED not lit, "
                    "pointing away, or sensor wired inverted")
        weakest = min(values[l] - values["dark"] for l in LEDS)
        if values["dark"] > DARK_FRACTION * max(weakest, 1):
            problems.append(
                f"reference: dark reading {values['dark']} is more than "
                f"{DARK_FRACTION:.0%} of the weakest LED signal ({weakest}) -> the "
                "bench is not dark: close the lid / cover it from room light")

    if purpose == "sample" and not problems and reference and "dark" in reference \
            and all(l in reference for l in LEDS):
        weakest = min(reference[l] - reference["dark"] for l in LEDS)
        allowed = reference["dark"] + max(5, DARK_FRACTION * weakest)
        if values["dark"] > allowed:
            problems.append(
                f"sample: dark reading {values['dark']} is higher than the "
                f"reference dark ({reference['dark']}) -> light got in after the "
                "strip was inserted: close the lid / cover the bench again")
    return problems


def transmittance(ref: dict[str, int], sample: dict[str, int]) -> dict[str, float]:
    """Fraction of each LED's light that the strip lets through (about 0..1)."""
    t = {}
    for led in LEDS:
        full = ref[led] - ref["dark"]
        passed = sample[led] - sample["dark"]
        t[led] = round(passed / full, 3) if full > 0 else float("nan")
    return t


def classify(t: dict[str, float]) -> dict:
    """Turn transmittances into a color name.

    The strip's color is the light it lets THROUGH. A red strip passes red
    and blocks green and blue.
    """
    r, g, b = t["red"], t["green"], t["blue"]
    top = max(r, g, b)
    hex_rgb = "#" + "".join(f"{round(255 * min(max(x, 0), 1)):02X}" for x in (r, g, b))

    if top < 0.15:
        name = "black / opaque"
    else:
        on = {k for k, v in t.items() if v >= 0.5 * top}
        name = {
            frozenset({"red"}): "red",
            frozenset({"green"}): "green",
            frozenset({"blue"}): "blue",
            frozenset({"red", "green"}): "yellow",
            frozenset({"green", "blue"}): "cyan",
            frozenset({"red", "blue"}): "magenta",
            frozenset({"red", "green", "blue"}): ("clear / white" if top >= 0.6
                                                  else "gray / neutral"),
        }[frozenset(on)]
        # orange: red passes, green partly
        if name == "red" and 0.25 * top <= g < 0.5 * top:
            name = "orange"

    # Confidence: how clearly the "on" channels stand out from the "off" ones.
    vals = sorted((r, g, b), reverse=True)
    spread = (vals[0] - vals[-1]) / vals[0] if vals[0] > 0 else 0
    confidence = "high" if spread > 0.6 or name.startswith(("clear", "black")) \
        else "medium" if spread > 0.3 else "low"
    if any(x > 1.15 for x in (r, g, b)):
        confidence = "low"          # strip passes MORE than no strip: suspicious
    return {"color": name, "approx_rgb": hex_rgb, "confidence": confidence}


def notebook(readings: list[dict], hw_confirmed: bool, test_ok: bool,
             last_error: str) -> str:
    """The scientist's lab notebook: everything computed, as plain text."""
    lines = [f"Human confirmed the hardware is set up: {'YES' if hw_confirmed else 'NO'}",
             f"Test measurement passed since that confirmation: "
             f"{'YES' if test_ok else 'NO'}"]
    if last_error:
        lines.append(f"LAST HARDWARE ERROR: {last_error}")

    test = latest_set(readings, "test")
    if test:
        p = check_test(test)
        lines.append(f"test: {test} -> {'OK' if not p else 'PROBLEMS: ' + '; '.join(p)}")

    sets = {}
    for purpose in ("reference", "sample"):
        v = latest_set(readings, purpose)
        if not v:
            lines.append(f"{purpose}: not measured yet")
            continue
        p = check_set(v, purpose, latest_set(readings, "reference"))
        sets[purpose] = (v, p)
        lines.append(f"{purpose}: {v} -> "
                     f"{'sanity checks OK' if not p else 'PROBLEMS: ' + '; '.join(p)}")

    if all(k in sets and not sets[k][1] for k in ("reference", "sample")):
        t = transmittance(sets["reference"][0], sets["sample"][0])
        c = classify(t)
        lines.append(f"transmittance (fraction passed): {t}")
        lines.append(f"COMPUTED PREDICTION: {c['color']} "
                     f"(approx {c['approx_rgb']}, confidence {c['confidence']})")
    return "\n".join(lines)


def data_problems(readings: list[dict]) -> list[str]:
    """All problems in the current test/reference/sample data."""
    out = []
    t = latest_set(readings, "test")
    if t:
        out += check_test(t)
    for purpose in ("reference", "sample"):
        v = latest_set(readings, purpose)
        if v and all(c in v for c in CHANNELS):
            out += check_set(v, purpose, latest_set(readings, "reference"))
    return out
