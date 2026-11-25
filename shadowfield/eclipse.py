"""Eclipse calendar: when did (or will) this satellite pass through Earth's shadow?

Shadowfield learns from the contrast between sunlight and shadow. Satellites in
dawn-dusk orbits can spend months in permanent sunlight, and then nothing changes
for the physics to explain. Their orbit plane still turns relative to the Sun over the
year, though, and SatNOGS keeps years of recordings. So we compute the calendar and
go back to the weeks when the satellite *did* see eclipses.

The beta angle (Sun above the orbit plane) is driven by the slow, well-modelled
precession of the orbit plane, so one TLE propagated over months gives the season
correctly even though its along-track position drifts. Eclipse fraction then follows
from beta and altitude.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sgp4.api import Satrec

from .physics import JD_UNIX0, in_sun, propagate, sun_unit


def calendar(tle1: str, tle2: str, start: float, end: float, step_days: float = 2.0) -> list[dict]:
    """Eclipse fraction and beta angle every step_days between two unix times."""
    sat = Satrec.twoline2rv(tle1, tle2)
    period = 2 * np.pi / sat.no_kozai * 60  # seconds (no_kozai is rad/min)
    rows = []
    for t0 in np.arange(start, end, step_days * 86400):
        u = t0 + np.arange(0, 3 * period, 20.0)  # three orbits, 20 s steps
        r, v = propagate(sat, u)
        if np.isnan(r).any():
            continue
        s = sun_unit(JD_UNIX0 + u / 86400)
        lit = in_sun(r, s)
        n = np.cross(r[0], v[0])
        n /= np.linalg.norm(n)
        beta = float(np.degrees(np.arcsin(np.clip(n @ s[0], -1, 1))))
        rows.append({"date": datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%d"), "unix": float(t0),
                     "eclipse_fraction": round(float(1 - lit.mean()), 3),
                     "eclipse_min_per_orbit": round(float((1 - lit.mean()) * period / 60), 1),
                     "beta_deg": round(beta, 1)})
    return rows


def eclipse_windows(cal: list[dict], min_fraction: float = 0.08, min_days: float = 6) -> list[tuple[float, float]]:
    """Contiguous stretches of the calendar with real eclipses (default: at least ~8% of each orbit)."""
    out, cur = [], None
    for row in cal:
        if row["eclipse_fraction"] >= min_fraction:
            cur = [row["unix"], row["unix"]] if cur is None else [cur[0], row["unix"]]
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return [w for w in out if (w[1] - w[0]) / 86400 >= min_days]
