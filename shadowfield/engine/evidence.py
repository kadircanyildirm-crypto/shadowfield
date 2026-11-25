"""Evidence for Claude: how each found field responds to sunrise and sunset, its drift and resets."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..physics import TAUS
from .field import Field

def describe(fields: list[Field], df: pd.DataFrame, phys: pd.DataFrame) -> None:
    """Add the *shape* of each field's response to sunrise/sunset to its evidence, in place.

    Correlation says a field follows the Sun; the response shape says how. A solar or
    charge current is a flat plateau in sunlight, a temperature keeps creeping up for
    tens of minutes, a battery voltage climbs across the whole sunlit arc. Ground stations
    often hear a satellite at the same few orbital phases, so the profile is built on
    whatever minutes-since-sunrise/sunset the data actually covers.
    """
    order = np.argsort(df["ts"].to_numpy())
    ph = phys.iloc[order].reset_index(drop=True)
    sc = ph["since_change"].to_numpy()
    obs = df["obs_id"].to_numpy()[order]
    unix = ph["unix"].to_numpy()
    same = (obs[1:] == obs[:-1]) & (np.diff(unix) <= 180)
    sun, shade = sc > 0, sc < 0
    ema = {t: stats.rankdata(ph[f"ema_{t}"].to_numpy()) for t in TAUS}
    abs_beta = stats.rankdata(np.abs(ph["beta"].to_numpy()))
    ranked = {id(f): stats.rankdata(f.values) for f in fields}

    def groups(mask, k):
        m = np.abs(sc[mask])
        if mask.sum() < 10:
            return []
        edges = np.unique(np.quantile(m, np.linspace(0, 1, k + 1)))
        return [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]

    sun_g, shade_g = groups(sun, 3), groups(shade, 3)

    def corr(a, b):
        return round(float(stats.spearmanr(a, b).statistic), 3) if len(a) > 10 and np.std(a) > 0 else None

    for f in fields:
        x = f.values.astype(float)
        prof = {}
        for label, mask, gs in (("shadow", shade, shade_g), ("sun", sun, sun_g)):
            for i, (lo, hi) in enumerate(gs):
                m = mask & (np.abs(sc) >= lo) & ((np.abs(sc) <= hi) if i == len(gs) - 1 else (np.abs(sc) < hi))
                if m.sum():
                    prof[f"{label} {lo:.0f}-{hi:.0f} min after {'sunset' if label == 'shadow' else 'sunrise'}"] =                         {"median": round(float(np.median(x[m])), 3), "n": int(m.sum())}
        d = {"response_profile": prof,
             "trend_in_sun_vs_minutes_since_sunrise": corr(x[sun], sc[sun]),
             "trend_in_shadow_vs_minutes_since_sunset": corr(x[shade], -sc[shade])}
        if sun.sum() and shade.sum() and x.std() > 0:
            d["sun_minus_shadow_in_std"] = round(float((np.median(x[sun]) - np.median(x[shade])) / x.std()), 2)
        rx = ranked[id(f)]
        d["rank_corr_by_lag_tau_min"] = {str(t): round(float(np.corrcoef(rx, v)[0, 1]), 3) for t, v in ema.items()}
        d["rank_corr_abs_beta"] = round(float(np.corrcoef(rx, abs_beta)[0, 1]), 3) if abs_beta.std() > 0 else None
        dx = np.abs(np.diff(x))[same]
        p5, p50, p95 = np.percentile(x, [5, 50, 95])
        if p50 != 0:
            # a voltage lives in a narrow band around its mean; a current spans zero to full
            d["relative_spread_p5_p95"] = round(float((p95 - p5) / abs(p50)), 3)
        if sun.sum() and shade.sum() and np.median(x[sun]) != 0:
            d["shadow_over_sun_median_ratio"] = round(float(np.median(x[shade]) / np.median(x[sun])), 3)
        d["distinct_values"] = int(np.unique(x).size)
        d["median_step_within_pass"] = float(np.median(dx)) if dx.size else None
        d["mean"] = round(float(x.mean()), 3)
        # health over the whole window: resets (reboots, counter wraps) and slow drift
        days = (unix - unix.min()) / 86400
        if days.max() >= 7:
            step = np.diff(x)
            if f.kind == "counter" or f.kind == "clock":
                big = (step < 0) & (np.abs(step) > 0.5 * (x.max() - x.min()))
                d["resets_in_window"] = int(big.sum())
                if big.any():
                    d["last_reset_days_before_end"] = round(float(days.max() - days[1:][big].max()), 1)
            else:
                day = np.floor(days)
                ud = np.unique(day)
                if ud.size >= 7:
                    lo = np.array([np.percentile(x[day == k], 10) for k in ud])
                    hi = np.array([np.percentile(x[day == k], 90) for k in ud])
                    d["drift_per_30_days_of_daily_low"] = round(float(stats.theilslopes(lo, ud)[0] * 30), 3)
                    d["drift_per_30_days_of_daily_high"] = round(float(stats.theilslopes(hi, ud)[0] * 30), 3)
                    d["window_days"] = round(float(days.max()), 1)
        d.update(_power_signature(x, sun, shade, {**f.detail, **d}))
        # fields that move together are often the same kind of sensor (four panel temperatures...)
        d["moves_with_offsets"] = [g.offset for g in fields if g is not f and
                                   abs(np.corrcoef(rx, ranked[id(g)])[0, 1]) > 0.85]
        f.detail.update(d)


def _power_signature(x: np.ndarray, sun: np.ndarray, shade: np.ndarray, d: dict) -> dict:
    """Physics that separates a battery, a solar input and a bus from one another.

    battery current   charges in sunlight, discharges in shadow: the sign flips
    solar input       falls to (almost) nothing in shadow
    battery voltage / state of charge   integrates sunlight: climbs through sun, sinks through shadow
    bus               regulated (a very narrow band), or drawn harder in shadow (heaters)
    """
    out = {}
    if not (sun.sum() >= 10 and shade.sum() >= 10):
        return out
    ms, mh = float(np.median(x[sun])), float(np.median(x[shade]))
    lo, hi = np.percentile(x, [2, 98])
    rng = hi - lo if hi > lo else 1.0
    out["sign_flips_between_sun_and_shadow"] = bool(ms * mh < 0 and min(abs(ms), abs(mh)) > 0.05 * rng)
    out["near_zero_in_shadow"] = bool(abs(mh - lo) < 0.1 * rng and abs(ms - mh) > 0.4 * rng and abs(mh) < 0.15 * max(abs(ms), 1e-9) + 0.1 * rng)
    ts, th = d.get("trend_in_sun_vs_minutes_since_sunrise"), d.get("trend_in_shadow_vs_minutes_since_sunset")
    out["integrates_sunlight"] = bool(ts is not None and th is not None and ts > 0.15 and th < -0.15)
    spread = d.get("relative_spread_p5_p95")
    out["regulated_narrow_band"] = bool(spread is not None and spread < 0.03)
    out["higher_in_shadow"] = bool(mh > ms and abs(ms - mh) > 0.1 * rng)
    rough = d.get("rough", 1.0)
    contrast = d.get("sun_minus_shadow_in_std") or 0.0
    soc_like = 0 <= lo and hi <= 100
    # A battery's voltage or charge is smooth and clearly higher after sunlight; its band is
    # narrow (a Li-ion cell spans ~4%), which a regulated bus shares, but a regulated bus
    # stays flat between sun and shadow.
    out["charged_by_sunlight"] = bool(contrast >= 1.0 and rough < 0.35 and spread is not None
                                      and (spread <= 0.35 or (soc_like and spread <= 1.0)))
    out["flat_between_sun_and_shadow"] = bool(abs(contrast) < 0.5)
    if out["sign_flips_between_sun_and_shadow"]:
        hint = "power_battery"   # charge / discharge current
    elif out["near_zero_in_shadow"]:
        hint = "power_solar"
    elif out["charged_by_sunlight"] or (out["integrates_sunlight"] and not out["regulated_narrow_band"]):
        hint = "power_battery"   # voltage or state of charge
    elif (out["regulated_narrow_band"] and out["flat_between_sun_and_shadow"]) or out["higher_in_shadow"]:
        hint = "power_bus"
    else:
        hint = None
    out["power_class_hint"] = hint
    return out
