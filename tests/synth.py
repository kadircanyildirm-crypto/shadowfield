"""Synthetic satellite: plant known fields in random frames and check Shadowfield recovers them.

Uses a real TLE and real physics (Skyfield eclipse model, IGRF field), so the
only thing that is fake is the spacecraft. Run:  python -m tests.synth
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadowfield import physics, score  # noqa: E402
from shadowfield.frames import FrameType  # noqa: E402

TLE1 = "1 43880U 18111E   26326.37627662  .00044965  00000-0  59503-3 0  9996"
TLE2 = "2 43880  97.5902 263.3375 0009751 100.8850 259.3503 15.58138980435300"

PLANTED = {  # offset: (name, type, expected kind)
    0: ("frame_counter", "u2be", "counter"),
    2: ("obc_unix_time", "u4be", "clock"),
    6: ("battery_mv", "u2be", "lagged"),
    8: ("obc_temp_centi_c", "s2le", "lagged"),
    10: ("solar_current_ma", "u2be", "sun"),
    12: ("mag_xyz", "3 x s2be", "magnet"),
    18: ("noise_byte", "u1", None),
    19: ("sat_id", "u1", None),
    20: ("panel_temp_centi_c", "s2be", "lagged"),
    22: ("noise_u16", "u2be", None),
}


def make(n_pass=350, per_pass=12, days=30, seed=1) -> tuple[FrameType, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-10-20", tz="UTC").value / 1e9
    starts = np.sort(rng.uniform(t0, t0 + days * 86400, n_pass))
    unix = (starts[:, None] + 10.0 * np.arange(per_pass)[None, :]).ravel()
    obs = np.repeat(np.arange(n_pass), per_pass)
    df = pd.DataFrame({"ts": pd.to_datetime(unix, unit="s", utc=True), "obs_id": obs, "tle1": TLE1, "tle2": TLE2})
    ph = physics.compute(df, Path("data/ephemeris"))
    n = len(df)
    L = 32
    B = np.zeros((n, L), np.uint8)

    def put(off, val, dt):
        B[:, off:off + np.dtype(dt).itemsize] = np.asarray(val).astype(dt).view(np.uint8).reshape(n, -1)

    put(0, ((unix - t0) / 10).astype(np.int64) % 65536, ">u2")
    put(2, unix.astype(np.int64), ">u4")
    put(6, 7400 + 600 * (ph["ema_10"] - 0.6) + rng.normal(0, 8, n), ">u2")
    put(8, 1200 + 2500 * (ph["ema_20"] - 0.6) + rng.normal(0, 15, n), "<i2")
    put(10, np.clip(ph["sunlit"] * (380 + 60 * rng.standard_normal(n)) + rng.normal(5, 3, n), 0, None), ">u2")
    # magnetometer: IGRF vector seen from a tumbling spacecraft, 10 nT per LSB
    be, bn, bu = _igrf_vec(ph, df)
    rot = Rotation.random(n, random_state=seed)
    body = rot.apply(np.column_stack([be, bn, bu]))
    for k in range(3):
        put(12 + 2 * k, body[:, k] / 10 + rng.normal(0, 20, n), ">i2")
    put(18, rng.integers(0, 256, n), "u1")
    put(19, np.full(n, 0x42), "u1")
    put(20, -500 + 4000 * (ph["ema_40"] - 0.6) + rng.normal(0, 25, n), ">i2")
    put(22, rng.integers(0, 65536, n), ">u2")
    B[:, 24:] = rng.integers(0, 4, (n, L - 24))  # near-constant padding
    df["payload"] = [bytes(r) for r in B]
    return FrameType("synthetic", L, df), ph


def _igrf_vec(ph, df):
    import ppigrf
    day = df["ts"].dt.floor("D")
    be, bn, bu = (np.zeros(len(df)) for _ in range(3))
    for d, idx in day.groupby(day).groups.items():
        r = df.index.get_indexer(idx)
        e, nn, u = ppigrf.igrf(ph["lon"].to_numpy()[r], ph["lat"].to_numpy()[r], ph["alt_km"].to_numpy()[r],
                               d.to_pydatetime().replace(tzinfo=None))
        be[r], bn[r], bu[r] = np.ravel(e), np.ravel(nn), np.ravel(u)
    return be, bn, bu


if __name__ == "__main__":
    ft, ph = make()
    found = score.analyze(ft.bytes_, ft.df, ph)
    got = {f.offset: f for f in found}
    print(f"{'off':>3} {'planted':<20} {'expect':<8} {'found':<8} {'type':<7} score")
    ok = 0
    expected = [(o, v) for o, v in PLANTED.items() if v[2]]
    for off, (name, typ, kind) in PLANTED.items():
        f = got.get(off)
        print(f"{off:>3} {name:<20} {str(kind):<8} {(f.kind if f else '-'):<8} {(f.ctype if f else ''):<7} "
              f"{f.score if f else '':.3}" if f else f"{off:>3} {name:<20} {str(kind):<8} -")
        ok += bool(kind and f and f.kind == kind)
    extra = [f for f in found if f.offset not in PLANTED or not PLANTED[f.offset][2]]
    print(f"\nrecovered {ok}/{len(expected)} planted physical fields; {len(extra)} false fields: "
          + ", ".join(f"{f.offset}:{f.ctype}:{f.kind}" for f in extra))
