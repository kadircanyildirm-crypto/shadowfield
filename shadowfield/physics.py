"""Orbital context for every frame: the physics used as ground truth.

For each frame time we compute, from the TLE of that pass:
  sunlit           is the spacecraft in sunlight right now (Earth-shadow model)
  ema_<tau>        exponentially weighted sunlight history (thermal lag proxies)
  since_change     minutes since the last sunrise/sunset (+ in sun, - in shadow)
  sin_u, cos_u     orbital phase (argument of latitude)
  beta             sun angle above the orbit plane, degrees
  lat, lon, alt_km sub-satellite point
  b_igrf           Earth magnetic field strength at the spacecraft (nT)
  unix             frame time, seconds

Speed matters: every frame needs ~200 minutes of sunlight history, i.e. about a
million orbit positions per satellite. Positions come straight from the vectorised
SGP4 propagator in its TEME frame, and the Sun from the Astronomical Almanac
low-precision formula (~0.01 deg). Skipping the full precession/nutation chain moves
shadow entry/exit by a few seconds at most, far below the one-minute resolution
used here (see tests/check_physics.py for the comparison with Skyfield).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec

HISTORY_MIN = 200           # about two orbits of sunlight history per frame
TAUS = (5, 10, 20, 40, 80)  # minutes
R_EARTH = 6378.137          # km
SHADOW_R = R_EARTH + 20.0   # include the dense lower atmosphere
JD_UNIX0 = 2440587.5


def sun_unit(jd: np.ndarray) -> np.ndarray:
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)


def propagate(sat: Satrec, unix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    jd = JD_UNIX0 + unix / 86400.0
    whole = np.floor(jd)
    e, r, v = sat.sgp4_array(whole, jd - whole)
    r[e != 0] = np.nan
    return r, v


def in_sun(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    along = (r * s).sum(-1)
    perp = np.linalg.norm(r - along[..., None] * s, axis=-1)
    return ~((along < 0) & (perp < SHADOW_R))


def gmst(unix: np.ndarray) -> np.ndarray:
    d = (JD_UNIX0 + unix / 86400.0) - 2451545.0
    return np.radians((280.46061837 + 360.98564736629 * d) % 360)


def geodetic(r: np.ndarray, unix: np.ndarray):
    th = gmst(unix)
    x = r[:, 0] * np.cos(th) + r[:, 1] * np.sin(th)
    y = -r[:, 0] * np.sin(th) + r[:, 1] * np.cos(th)
    z = r[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    p = np.hypot(x, y)
    a, f = R_EARTH, 1 / 298.257223563
    e2 = f * (2 - f)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(4):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
    return np.degrees(lat), lon, alt


def compute(df: pd.DataFrame, data_dir: Path | None = None) -> pd.DataFrame:
    # pandas 3 parses ISO strings at microsecond resolution, so never assume nanoseconds
    unix = df["ts"].astype("datetime64[ns, UTC]").astype("int64").to_numpy() / 1e9
    out = pd.DataFrame(index=df.index)
    out["unix"] = unix
    k = np.arange(HISTORY_MIN)
    cols = {c: np.full(len(df), np.nan) for c in
            ["sunlit", "since_change", "sin_u", "cos_u", "beta", "lat", "lon", "alt_km"]
            + [f"{q}_{fr}_{a}" for q in ("pos", "vel") for fr in ("eci", "ecef") for a in "xyz"]
            + [f"ema_{t}" for t in TAUS]}
    weights = {t: np.exp(-k / t) / np.exp(-k / t).sum() for t in TAUS}

    for (t1, t2), idx in df.groupby(["tle1", "tle2"]).groups.items():
        sat = Satrec.twoline2rv(t1, t2)
        rows = df.index.get_indexer(idx)
        u0 = unix[rows]
        grid = (u0[:, None] - 60.0 * k[None, :]).ravel()
        r, _ = propagate(sat, grid)
        s = sun_unit(JD_UNIX0 + grid / 86400.0)
        lit = in_sun(r, s).reshape(len(rows), HISTORY_MIN).astype(float)
        cols["sunlit"][rows] = lit[:, 0]
        for tau, w in weights.items():
            cols[f"ema_{tau}"][rows] = lit @ w
        changed = lit != lit[:, :1]
        first = np.where(changed.any(1), changed.argmax(1), HISTORY_MIN)
        cols["since_change"][rows] = np.where(lit[:, 0] > 0, first, -first)

        r0, v0 = propagate(sat, u0)
        n = np.cross(r0, v0)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        s0 = sun_unit(JD_UNIX0 + u0 / 86400.0)
        cols["beta"][rows] = np.degrees(np.arcsin(np.clip((n * s0).sum(1), -1, 1)))
        node = np.cross(np.array([0.0, 0.0, 1.0]), n)
        node /= np.linalg.norm(node, axis=1, keepdims=True)
        u = np.arctan2((r0 * np.cross(n, node)).sum(1), (r0 * node).sum(1))
        cols["sin_u"][rows], cols["cos_u"][rows] = np.sin(u), np.cos(u)
        lat, lon, alt = geodetic(r0, u0)
        cols["lat"][rows], cols["lon"][rows], cols["alt_km"][rows] = lat, lon, alt
        # GNSS receivers report these directly; TEME stands in for any inertial frame
        th = gmst(u0)
        c, s_ = np.cos(th), np.sin(th)
        pe = np.stack([c * r0[:, 0] + s_ * r0[:, 1], -s_ * r0[:, 0] + c * r0[:, 1], r0[:, 2]], 1)
        ve = np.stack([c * v0[:, 0] + s_ * v0[:, 1], -s_ * v0[:, 0] + c * v0[:, 1], v0[:, 2]], 1)
        ve -= np.cross(np.array([0.0, 0.0, 7.2921150e-5]), pe)
        for q, fr, arr in (("pos", "eci", r0), ("vel", "eci", v0), ("pos", "ecef", pe), ("vel", "ecef", ve)):
            for i, a in enumerate("xyz"):
                cols[f"{q}_{fr}_{a}"][rows] = arr[:, i]

    for c, a in cols.items():
        out[c] = a
    out["b_igrf"] = _igrf(out, df)
    return out


def _igrf(phys: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    try:
        import ppigrf
    except ImportError:
        return np.full(len(phys), np.nan)
    b = np.full(len(phys), np.nan)
    days = df["ts"].dt.floor("D")
    for day, idx in days.groupby(days).groups.items():
        rows = phys.index.get_indexer(idx)
        be, bn, bu = ppigrf.igrf(phys["lon"].to_numpy()[rows], phys["lat"].to_numpy()[rows],
                                 phys["alt_km"].to_numpy()[rows], day.to_pydatetime().replace(tzinfo=None))
        b[rows] = np.sqrt(np.asarray(be) ** 2 + np.asarray(bn) ** 2 + np.asarray(bu) ** 2).ravel()
    return b
