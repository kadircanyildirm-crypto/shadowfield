"""A random fleet of synthetic satellites, to test how the engine's rules work *together*.

Each satellite gets a random byte order, a random selection and order of field types, random
gaps, and the traps that broke earlier versions:
- signed currents that reverse in shadow;
- counters whose top byte never changes;
- rolling 1-byte sequence numbers;
- status bytes and noise next to real fields;
- constant padding;
- GNSS fixes that are mostly stale or missing;
- float32 fields.

The physics is real: a real TLE, the real eclipse model and the real IGRF field. Only the
spacecraft is fake.

python -m tests.fleet [n_satellites]   prints recall / exact / precision over the fleet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shadowfield import physics
from shadowfield.frames import FrameType
from tests.synth import TLE1, TLE2

PHYSICAL = ("counter", "clock", "sun", "lagged", "position", "magnet")


def _frames(rng, n_pass, per_pass, days):
    t0 = pd.Timestamp("2026-10-20", tz="UTC").value / 1e9
    starts = np.sort(rng.uniform(t0, t0 + days * 86400, n_pass))
    unix = (starts[:, None] + rng.choice([5.0, 10.0, 30.0]) * np.arange(per_pass)[None, :]).ravel()
    obs = np.repeat(np.arange(n_pass), per_pass)
    df = pd.DataFrame({"ts": pd.to_datetime(unix, unit="s", utc=True), "obs_id": obs, "tle1": TLE1, "tle2": TLE2})
    return df, unix, t0


def make(seed: int, n_pass: int = 250, per_pass: int = 10, days: int = 30):
    """Returns (FrameType, phys, planted) with planted = [(offset, width, endian, signed, kind, name)]."""
    rng = np.random.default_rng(seed)
    df, unix, t0 = _frames(rng, n_pass, per_pass, days)
    ph = physics.compute(df)
    n = len(df)
    order = ">" if rng.random() < 0.5 else "<"
    endian = "be" if order == ">" else "le"
    menu = ["battery_mv", "battery_ma", "solar_ma", "temp_s1", "temp_centi", "soc", "uptime", "seq",
            "unix", "status", "noise", "const", "gnss", "float_temp"]
    picks = list(rng.choice(menu, size=int(rng.integers(7, 12)), replace=False))
    picks += ["noise", "status"]  # every frame has traps
    rng.shuffle(picks)
    cols, planted, off = [], [], 0

    def emit(name, values, dt, kind, signed):
        nonlocal off
        a = np.asarray(values)
        if np.dtype(dt).kind in "iu":
            info = np.iinfo(np.dtype(dt))
            a = np.clip(np.round(a), info.min, info.max)
        raw = a.astype(dt).view(np.uint8).reshape(n, -1)
        w = raw.shape[1]
        cols.append(raw)
        planted.append((off, w, endian if w > 1 else "be", signed, kind, name))
        off += w
        gap = int(rng.integers(0, 3))
        if gap:  # spare bytes between fields: zero or slowly varying junk
            cols.append(np.zeros((n, gap), np.uint8) if rng.random() < 0.5 else rng.integers(0, 3, (n, gap)).astype(np.uint8))
            off += gap

    lit = ph["sunlit"].to_numpy()
    for name in picks:
        if name == "battery_mv":
            emit(name, 7400 + 600 * (ph["ema_10"] - 0.6) + rng.normal(0, 6, n), order + "u2", "lagged", False)
        elif name == "battery_ma":  # charges in sun, discharges in shadow: sign reverses
            emit(name, np.where(lit > 0, 180, -120) + 40 * rng.standard_normal(n), order + "i2", "sun", True)
        elif name == "solar_ma":
            emit(name, np.clip(lit * (400 + 80 * rng.standard_normal(n)), 0, None), order + "u2", "sun", False)
        elif name == "temp_s1":
            emit(name, -5 + 30 * (ph["ema_20"] - 0.5) + rng.normal(0, 0.6, n), "i1", "lagged", True)
        elif name == "temp_centi":
            emit(name, 900 + 2200 * (ph["ema_40"] - 0.6) + rng.normal(0, 12, n), order + "i2", "lagged", True)
        elif name == "soc":
            emit(name, 70 + 25 * (ph["ema_40"] - 0.6) + rng.normal(0, 0.8, n), "u1", "lagged", False)
        elif name == "uptime":  # 4 bytes, top byte constant all month
            emit(name, 20_000_000 + (unix - t0), order + "u4", "counter", False)
        elif name == "seq":
            emit(name, np.arange(n) % 256, "u1", "counter", False)
        elif name == "unix":
            emit(name, unix.astype(np.int64), order + "u4", "clock", False)
        elif name == "status":
            emit(name, rng.choice([0x01, 0x03, 0x41, 0x43], n), "u1", None, False)
        elif name == "noise":
            emit(name, rng.integers(0, 65536, n), order + "u2", None, False)
        elif name == "const":
            emit(name, np.full(n, 0x5A), "u1", None, False)
        elif name == "gnss":  # ECEF x in metres, fresh on ~40% of frames, else stale or -1
            fresh = rng.random(n) < 0.4
            x = ph["pos_ecef_x"].to_numpy() * 1000
            last = np.where(fresh, x, np.nan)
            last = pd.Series(last).ffill().fillna(-1).to_numpy()
            emit(name, last, order + "i4", "position", True)
        elif name == "float_temp":
            emit(name, 12.5 + 18 * (ph["ema_20"] - 0.5) + rng.normal(0, 0.05, n), order + "f4", "lagged", "f")
    B = np.concatenate(cols, axis=1)
    df["payload"] = [bytes(r) for r in B]
    return FrameType(f"fleet{seed}", B.shape[1], df), ph, planted


def grade(found, planted, B):
    from shadowfield.score import _decode
    phys = [p for p in planted if p[4] in PHYSICAL]
    hit = exact = 0
    for o, w, en, sg, kind, name in phys:
        span = set(range(o, o + w))
        m = [f for f in found if set(f.span) & span]
        hit += bool(m)
        # exact = reproduces the planted values bit for bit (a sign-extension byte more or less is
        # the same number, as in the decoder grading)
        exact += any(np.array_equal(_decode(B, f.offset, f.width, f.endian, f.signed), _decode(B, o, w, en, sg))
                     for f in m if f.kind not in ("magnet", "attitude"))
    real = set().union(*(set(range(p[0], p[0] + p[1])) for p in phys)) if phys else set()
    false = [f for f in found if not set(f.span) & real]
    return {"physical": len(phys), "found": hit, "exact": exact, "reported": len(found), "false": len(false),
            "false_fields": [f"{f.offset}:{f.ctype}:{f.kind}" for f in false]}


def run(seeds) -> dict:
    from shadowfield import score
    tot = {"physical": 0, "found": 0, "exact": 0, "reported": 0, "false": 0}
    rows = []
    for s in seeds:
        ft, ph, planted = make(s)
        found = score.analyze(ft.bytes_, ft.df, ph)
        g = grade(found, planted, ft.bytes_)
        rows.append((s, g))
        for k in tot:
            tot[k] += g[k]
    return {"rows": rows, **tot, "recall": tot["found"] / tot["physical"], "exact_rate": tot["exact"] / tot["physical"],
            "precision": 1 - tot["false"] / max(tot["reported"], 1)}


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    res = run(range(100, 100 + k))
    for s, g in res["rows"]:
        print(f"seed {s}: {g['found']}/{g['physical']} found, {g['exact']} exact, {g['false']} false {g['false_fields']}")
    print(f"\nfleet of {k}: recall {res['recall']:.0%}, exact {res['exact_rate']:.0%}, precision {res['precision']:.0%}")
