"""Turn physics evidence into named, typed telemetry fields with Claude.

The physics engine says *how* a field behaves (steps with sunlight, lags it by
20 minutes, counts up at 1 Hz...). Claude reads that evidence together with
the raw value ranges and proposes what the field is, its unit and scaling,
like a spacecraft engineer reverse-engineering a beacon by hand.

In blind mode the satellite's name and NORAD id are withheld, so the model
cannot recall a known decoder from memory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field as PField

# Sonnet 5.5 at medium effort: half the per-token price of Opus 5.5 and plenty for reading
# a dozen evidence rows. Override with SHADOWFIELD_MODEL / SHADOWFIELD_EFFORT if needed.
MODEL = os.environ.get("SHADOWFIELD_MODEL", "claude-sonnet-5-5")
EFFORT = os.environ.get("SHADOWFIELD_EFFORT", "medium")
PRICE_PER_MTOK = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.10, 0.50)}
CACHE_DIR = Path("data/llm_cache")

Category = Literal["thermal", "power_solar", "power_battery", "power_bus", "counter", "navigation",
                   "clock", "magnetometer", "attitude", "unknown"]


class FieldGuess(BaseModel):
    offset: int
    name: str = PField(description="snake_case field name, e.g. battery_voltage")
    category: Category
    unit: str
    scale: str = PField(description="formula from raw to unit, e.g. 'raw * 0.001' or 'raw - 273.15'")
    confidence: float = PField(ge=0, le=1)
    rationale: str = PField(description="one or two sentences citing the evidence")


class FrameGuess(BaseModel):
    summary: str = PField(description="two or three sentences on what this frame type appears to be")
    fields: list[FieldGuess]
    health: str = PField(description="two to four sentences: spacecraft health from the fields as named "
                                     "(reboots, battery and thermal trends), or 'not enough evidence'")


SYSTEM = """You are a spacecraft telemetry engineer reverse-engineering an unknown cubesat beacon.
You receive candidate fields that an automated engine located in the raw frame bytes, with evidence
of how each one behaves against orbital physics computed from the satellite's TLE:

- kind=sun: value steps when the spacecraft enters/leaves Earth's shadow (rho_sun = rank correlation
  with sunlight; sun_mean vs shadow_mean are the raw means in sun and in shadow).
- kind=lagged: value follows sunlight with a lag of tau_min minutes (thermal inertia or battery charge).
- kind=orbit: value repeats with the orbital phase (r2_orbit).
- kind=counter: only increases (per_second = rate, resets = wrap/reboot events).
- kind=clock: tracks wall-clock time; epoch_guess names the epoch.
- kind=magnet: three consecutive values whose vector length tracks the IGRF field model (rho_igrf).
- kind=position: the value is a linear function of one orbital coordinate computed from the TLE
  (component = pos/vel, eci/ecef, x/y/z, or lat / alt_km). matching_frames_fraction is the share of
  frames that agree with the orbit; the rest are usually stale fixes or "no fix" values. raw_per_km
  or raw_per_km_per_s gives the scale. This is a GNSS receiver output: name the coordinate and frame,
  and derive the unit from the scale (raw_per_km 1000 means metres, 1e5 means centimetres, etc.).
- rough: within-pass roughness, about 1.1 for pure noise, much lower for a smooth signal.

If every frame is in sunlight but fraction_of_frames_with_an_eclipse_in_the_previous_80_min is above
zero, the eclipse happened out of view of the ground stations (often over a polar cap). The lagged
correlations (kind=lagged, rank_corr_by_lag_tau_min) are then still real physics: they measure how
each value recovers after that eclipse. Judge such fields on them as usual, not as unknown.
Every field also carries its response shape. Ground stations often hear the satellite only at a few
orbital phases, so these cover whatever the data has:
- response_profile: median raw value in groups of frames by minutes since sunset (in shadow) or since
  sunrise (in sunlight).
- trend_in_sun_vs_minutes_since_sunrise / trend_in_shadow_vs_minutes_since_sunset: rank correlation
  inside each regime. A value that keeps rising long after sunrise, or keeps falling long after
  sunset, is integrating something (heat, charge). A plateau near 0 is set directly by illumination
  (a current).
- sun_minus_shadow_in_std: sun-vs-shadow contrast in standard deviations (sign gives the direction).
- rank_corr_by_lag_tau_min: correlation with sunlight history smoothed over tau minutes; the tau with
  the largest magnitude is the field's time constant.
- rank_corr_abs_beta: dependence on the sun angle to the orbit plane (season-like drift).
- moves_with_offsets: other fields that rise and fall with this one, often the same kind of sensor.
- relative_spread_p5_p95: (p95 - p5) / |median|. Voltages are well below about 0.3; currents and
  solar inputs that drop toward zero in shadow are far above 1.
- shadow_over_sun_median_ratio: about 0.85-1.0 for a battery voltage or state of charge, near 0
  (or negative) for a solar or charge current, and variable for temperatures.

How the physics separates field types:
- Solar-array or charge current: high plateau in sun, near zero or reversed in shadow, little trend
  within each regime, often rough. Battery current can be signed (charging in sun, discharging in shadow).
- Battery voltage: smooth and narrow-range (spread small relative to mean). It climbs during sunlight
  and falls during shadow, i.e. positive trend in sun and negative trend in shadow. State of charge
  behaves the same way, on a 0-100 or 0-255 scale.
- Temperature: also trends within each regime and has a clear time constant. Outer panels swing by
  tens of degrees and the different faces move together. Internal and battery temperatures swing a
  few degrees with long time constants. Signed bytes in -40..+80 are usually degC directly.
- Two quantities that move together (a battery voltage and its state of charge, four panel
  temperatures) form a group; name them consistently.
- Power consumption or bus load: weakly sun-correlated and rough.
- Power fields also carry a physics signature and power_class_hint, the engine's guess at the
  class: sign_flips_between_sun_and_shadow (a battery current charging in sun and discharging
  in shadow), near_zero_in_shadow (a solar input), charged_by_sunlight (a battery voltage or state
  of charge: smooth, clearly higher after sunlight, in a narrow band), and regulated_narrow_band
  with flat_between_sun_and_shadow, or higher_in_shadow (a bus or load). On the development
  satellites the hint was right about three times in four, and almost never wrong for battery
  or solar. Use it for power_battery / power_solar / power_bus unless the raw range or another
  clue clearly contradicts it.
- Raw ranges carry units. A Li-ion cell reads 3000-4200 mV (two cells 6000-8400, three
  9000-12600). A field confined to 0..100 that moves with such a voltage is a state of charge in
  percent, not a current. Currents in mA are usually signed or start near 0.
- A temperature that is warmer in shadow than in sun, with a narrow range, is a heated or
  internal component (a battery with a heater), not something unknown.
Use typical cubesat conventions to infer units and scaling from the raw ranges (ADC counts, mV, mA,
0.01 degC, Kelvin offsets, etc.).
Health evidence (when the window is long enough): resets_in_window and last_reset_days_before_end
for counters and clocks (a reset is a reboot or a wrap: a 1-2 byte sequence number wraps by design),
and drift_per_30_days_of_daily_low/high for the other fields (raw units per 30 days; a falling daily
low of a battery voltage suggests ageing, but a few weeks of data also include season effects, so
say how sure you are).
Be honest: use category "unknown" and low confidence when the evidence does not support a guess.
Return one entry per candidate field, keeping its offset."""


VOTES = int(os.environ.get("SHADOWFIELD_VOTES", "3"))


def name_fields(fields: list[dict], orbit: dict, frame_info: dict, sat_hint: str | None = None,
                votes: int | None = None) -> FrameGuess:
    """Ask Claude `votes` times (independent samples) and keep each field's majority category.

    One sample can flip between "battery" and "solar" from run to run; a majority of three
    is far steadier. The samples share one cached prompt, so extra votes cost mostly output
    tokens. Each field's confidence becomes mean confidence x agreement, and its rationale
    says how many samples agreed.
    """
    votes = VOTES if votes is None else votes
    payload = {
        "satellite": sat_hint or "withheld (blind test)",
        "orbit": orbit,
        "frame": frame_info,
        "candidate_fields": fields,
    }
    user = json.dumps(payload, indent=1, default=float)
    samples = [_ask(user, i) for i in range(votes)]
    return _vote(samples)


def _vote(samples: list[FrameGuess]) -> FrameGuess:
    if len(samples) == 1:
        return samples[0]
    from collections import Counter
    by_off: dict[int, list[FieldGuess]] = {}
    for smp in samples:
        for g in smp.fields:
            by_off.setdefault(g.offset, []).append(g)
    out = []
    for off, gs in sorted(by_off.items()):
        counts = Counter(g.category for g in gs)
        top = max(counts.values())
        tied = [c for c, k in counts.items() if k == top]
        # a tie goes to the more confident category; "unknown" never wins a tie
        win = max(tied, key=lambda c: (c != "unknown", sum(g.confidence for g in gs if g.category == c)))
        agree = [g for g in gs if g.category == win]
        rep = max(agree, key=lambda g: g.confidence)
        conf = sum(g.confidence for g in agree) / len(agree) * len(agree) / len(samples)
        out.append(rep.model_copy(update={"confidence": round(conf, 3),
                                          "rationale": f"{rep.rationale} [{len(agree)}/{len(samples)} samples agree]"}))
    return samples[0].model_copy(update={"fields": out})


def _ask(user: str, sample: int) -> FrameGuess:
    # identical evidence -> identical question: answer from disk instead of paying again
    key = hashlib.sha256(f"{MODEL}|{EFFORT}|{SYSTEM}|{user}|{sample}".encode()).hexdigest()[:24]
    if sample == 0:  # keep answers cached by earlier single-sample runs valid
        key = hashlib.sha256(f"{MODEL}|{EFFORT}|{SYSTEM}|{user}".encode()).hexdigest()[:24]
    cached = CACHE_DIR / f"{key}.json"
    if cached.exists():
        return FrameGuess.model_validate_json(cached.read_text(encoding="utf-8"))

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=8000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        cache_control={"type": "ephemeral"},  # later votes read the same prompt from cache
        system=SYSTEM,
        output_config={
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": _schema()},
        },
        messages=[{"role": "user", "content": user}],
    )
    u = resp.usage
    pin, pout = PRICE_PER_MTOK.get(resp.model, PRICE_PER_MTOK.get(MODEL, (0, 0)))
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    cost = (u.input_tokens * pin + cw * pin * 1.25 + cr * pin * 0.05 + u.output_tokens * pout) / 1e6
    print(f"[llm] {resp.model} vote {sample}: {u.input_tokens} in + {cw} cache-write + {cr} cache-read / "
          f"{u.output_tokens} out tokens, about ${cost:.4f}", flush=True)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {resp.stop_details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("response truncated at max_tokens")
    text = next(b.text for b in resp.content if b.type == "text")
    guess = FrameGuess.model_validate_json(text)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return guess


def _schema() -> dict:
    s = FrameGuess.model_json_schema()
    # structured outputs need closed objects with every property required
    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for k in ("minimum", "maximum"):
                node.pop(k, None)
            for v in node.values():
                close(v)
        elif isinstance(node, list):
            for v in node:
                close(v)
    close(s)
    return s
