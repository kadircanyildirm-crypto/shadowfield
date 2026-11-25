"""Find every telemetry field in one frame type, step by step.

  readings     every (offset, width, byte order, sign) reading, and plausible float32s
  motion       within-pass roughness and counting behaviour of each reading
  correlate    rank correlation with sunlight, sunlight history and orbit phase, within weeks
  classify     the best physical explanation of each reading, with its significance
  detectors    clocks, GNSS fixes, quaternions and magnetometers
  select       one reading per byte span, proven widths, a non-overlapping layout
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..physics import TAUS
from .detectors import _clock, _magnetometers, _positions, _quaternions, _slope
from .field import WIDTHS, Field, _decode
from .select import _canonical_widths, _greedy, _pick_encodings
from .stats import ALPHA, _demean, _p_from_rho, _ranks, _restd
from .structure import _width_ok


def analyze(B: np.ndarray, df: pd.DataFrame, phys: pd.DataFrame, alpha: float | None = None,
            quick: bool = False) -> list[Field]:
    """All fields in one frame type. quick=True (1-2 byte integers only, no GNSS search) is for
    comparing candidate decodings of the same bytes, where only the relative evidence matters."""
    alpha = ALPHA if alpha is None else alpha
    n, L = B.shape
    order = np.argsort(df["ts"].to_numpy())
    B, df, phys = B[order], df.iloc[order].reset_index(drop=True), phys.iloc[order].reset_index(drop=True)
    n_pass = int(df["obs_id"].nunique())
    unix = phys["unix"].to_numpy()

    cand, X = readings(B, quick)
    if not X:
        return []
    X = np.column_stack(X)
    mo = motion(X, cand, df, unix)
    co = correlate(X, df, phys, unix, n, n_pass)
    fields, phys_best = classify(cand, X, mo, co, unix, n, n_pass, alpha)

    fields += _magnetometers(cand, X, phys["b_igrf"].to_numpy(), n_pass, alpha)
    if not quick:
        fields += _positions(cand, X, phys, df["obs_id"].to_numpy())
        fields += _quaternions(cand, X)
    ctx = {"B": B, "same": mo.same, "phys_best": phys_best,
           "up": {c: int(mo.up[j]) for j, c in enumerate(cand)},
           "rough": {c: float(mo.rough[j]) for j, c in enumerate(cand)}}
    fields = _pick_encodings(fields, ctx["rough"], B)
    every = fields
    fields = [f for f in fields if _width_ok(f, ctx)]
    chosen = _greedy(fields, ctx["rough"])
    return _canonical_widths(chosen, every, B)


def readings(B: np.ndarray, quick: bool = False) -> tuple[list, list]:
    """Every candidate reading with at least four distinct values (time-ordered B)."""
    n, L = B.shape
    cand, X = [], []
    # a byte that never changes cannot be the least significant end of a field
    # (a constant top byte can: a clock's high byte barely moves in a month)
    const = np.array([np.unique(B[:, i]).size == 1 for i in range(L)])
    for w in ((1, 2) if quick else WIDTHS):
        for o in range(L - w + 1):
            for endian in (("be", "le") if w > 1 else ("be",)):
                if w > 1 and const[o + w - 1 if endian == "be" else o]:
                    continue
                for signed in (False, True):
                    x = _decode(B, o, w, endian, signed)
                    if np.unique(x[: min(n, 2000)]).size < 4:
                        continue
                    cand.append((o, w, endian, signed))
                    X.append(x)
    for o in range(0 if quick else L - 3):
        for endian in ("be", "le"):
            x = _decode(B, o, 4, endian, "f")
            ax = np.abs(x)
            sane = (x == 0) | ((ax > 1e-6) & (ax < 1e10))
            nz = x != 0
            # random bytes read as floats give wild exponents; real values stay in range
            # a real float has all four bytes moving and a busy lowest mantissa byte; an
            # integer read as a float (a temperature plus constant neighbours) does not
            bytes_move = all(np.unique(B[nz, o + i]).size > 1 for i in range(4)) if nz.any() else False
            lsb = o if endian == "le" else o + 3
            busy = np.unique(B[nz, lsb]).size >= min(32, max(8, nz.sum() // 8)) if nz.any() else False
            if (sane.mean() >= 0.98 and nz.sum() >= 30 and np.unique(x[: min(n, 2000)]).size >= 4
                    and bytes_move and busy):
                cand.append((o, 4, endian, "f"))
                X.append(x)
    return cand, X


@dataclass
class Motion:
    same: np.ndarray        # consecutive frames of one pass, at most 3 min apart
    rough: np.ndarray       # within-pass roughness (about 1.13 for noise)
    step: np.ndarray
    up: np.ndarray
    frac_up: np.ndarray
    frac_up_pass: np.ndarray
    up_p: np.ndarray
    wrap_p: np.ndarray
    sp: np.ndarray
    dt_same: np.ndarray
    rolling: dict


def motion(X: np.ndarray, cand: list, df: pd.DataFrame, unix: np.ndarray) -> Motion:
    # within-pass roughness: ~1.13 for pure noise, << 1 for a smooth signal
    same = (df["obs_id"].to_numpy()[1:] == df["obs_id"].to_numpy()[:-1]) & (np.diff(unix) <= 180)
    dX = np.abs(np.diff(X, axis=0))
    sd = X.std(0)
    sd[sd == 0] = np.inf
    rough = dX[same].mean(0) / sd if same.any() else np.ones(X.shape[1])
    # monotonic behaviour in global time order
    step = np.diff(X, axis=0)
    span = X.max(0) - X.min(0)
    # Several stations hear the same pass with clocks a few seconds apart, so frames that
    # close together may be out of order: monotonicity is judged only on pairs > 5 s apart.
    # Drops of more than half the range are resets (allowed if rare).
    dts = np.diff(unix)
    # ...and only between frames less than a week apart: across a gap of months a satellite
    # may have rebooted and restarted its counters, which says nothing about a counter
    apart = (dts > 5) & (dts < 7 * 86400)
    st = step[apart]
    up = (st > 0).sum(0)
    resets_all = ((st < 0) & (np.abs(st) > 0.5 * span)).sum(0)
    ok_resets = resets_all <= 0.05 * np.maximum(up, 1)
    down = (st < 0).sum(0) - np.where(ok_resets, resets_all, 0)
    frac_up = up / np.maximum(up + down, 1)
    # rolling counters (1-2 byte sequence numbers) wrap many times a day, so judge them
    # inside passes only, where a wrap shows up as one big drop
    m = same & apart
    sp = step[m]
    up_p, down_p = (sp > 0).sum(0), (sp < 0).sum(0)
    wrap_p = ((sp < 0) & (np.abs(sp) > 0.5 * span)).sum(0)
    frac_up_pass = up_p / np.maximum(up_p + down_p - wrap_p, 1)
    dt_same = dts[m]
    # in-pass "counts up (with wraps)" of every single byte, for alignment checks
    rolling = {c[0]: float(frac_up_pass[j]) if up_p[j] >= 10 else 0.0
               for j, c in enumerate(cand) if c[1] == 1 and not c[3]}

    return Motion(same, rough, step, up, frac_up, frac_up_pass, up_p, wrap_p, sp, dt_same, rolling)


@dataclass
class Correlation:
    C: np.ndarray           # (n_cand, n_feat) rank correlations
    ci: dict
    n_pass_sun: int
    r2_orbit: np.ndarray
    lit: np.ndarray


def correlate(X: np.ndarray, df: pd.DataFrame, phys: pd.DataFrame, unix: np.ndarray, n: int,
              n_pass: int) -> Correlation:
    feats = {"sunlit": phys["sunlit"].to_numpy()}
    for t in TAUS:
        feats[f"ema_{t}"] = phys[f"ema_{t}"].to_numpy()
    feats["since_change"] = phys["since_change"].to_numpy()
    feats["lat"], feats["alt_km"] = phys["lat"].to_numpy(), phys["alt_km"].to_numpy()
    F = np.column_stack(list(feats.values()))
    RX, RF = _ranks(X), _ranks(F)
    # Within-week comparison: remove each week's mean rank before correlating, so a slow drift
    # (season, ageing) that happens to coincide with a change in eclipse season cannot pose as
    # a response to sunlight. Orbit-to-orbit and pass-to-pass variation is untouched.
    week = np.floor((unix - unix.min()) / (7 * 86400)).astype(int)
    if week.max() >= 2:
        RX, RF = _demean(RX, week), _demean(RF, week)
    # Sunlight correlations only from weeks in which the sunlight history actually varies:
    # weeks of permanent sunshine carry no information about it, and would only dilute it.
    # (Counters, clocks, GNSS and orbit phase still use every frame.)
    wk_var = pd.Series(phys[f"ema_{TAUS[2]}"].to_numpy()).groupby(week).transform("std").fillna(0).to_numpy()
    E = wk_var > 0.02
    if E.sum() >= 30 and not E.all():
        RXe, RFe = _restd(RX[E]), _restd(RF[E])
        C = RXe.T @ RFe / E.sum()
        n_pass_sun = int(df["obs_id"].iloc[np.flatnonzero(E)].nunique())
    else:
        C = RX.T @ RF / n  # (n_cand, n_feat) Spearman
        n_pass_sun = n_pass
    names = list(feats)
    ci = {k: i for i, k in enumerate(names)}

    # orbital periodicity: R^2 of first two harmonics of the orbital phase
    su, cu = phys["sin_u"].to_numpy(), phys["cos_u"].to_numpy()
    H = np.column_stack([np.ones(n), su, cu, 2 * su * cu, cu ** 2 - su ** 2])
    beta, *_ = np.linalg.lstsq(H, RX, rcond=None)
    r2_orbit = 1 - ((RX - H @ beta) ** 2).mean(0)

    return Correlation(C, ci, n_pass_sun, r2_orbit, feats["sunlit"])


def classify(cand, X, mo: Motion, co: Correlation, unix, n, n_pass, alpha) -> tuple[list[Field], dict]:
    """The best physical explanation of each reading; keep the significant ones."""
    rough, step, up, frac_up = mo.rough, mo.step, mo.up, mo.frac_up
    frac_up_pass, up_p, wrap_p, sp, dt_same, rolling = mo.frac_up_pass, mo.up_p, mo.wrap_p, mo.sp, mo.dt_same, mo.rolling
    C, ci, n_pass_sun, r2_orbit = co.C, co.ci, co.n_pass_sun, co.r2_orbit
    fields: list[Field] = []
    phys_best: dict[tuple, float] = {}
    lit = co.lit
    for j, (o, w, endian, signed) in enumerate(cand):
        x = X[:, j]
        det = {"rough": round(float(rough[j]), 3), "min": float(x.min()), "max": float(x.max()),
               "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
               "n_frames": int(n), "n_passes": n_pass}
        rho_sun = C[j, ci["sunlit"]]
        ema_rhos = [C[j, ci[f"ema_{t}"]] for t in TAUS]
        k = int(np.argmax(np.abs(ema_rhos)))
        rho_lag, tau = ema_rhos[k], TAUS[k]
        options = []
        if lit.std() > 0:
            options.append(("sun", abs(rho_sun), {"rho_sun": round(float(rho_sun), 3),
                            "sun_mean": float(x[lit > 0].mean()), "shadow_mean": float(x[lit == 0].mean())}))
        if tau >= 10 and abs(rho_lag) > abs(rho_sun) + 0.05:
            options.append(("lagged", abs(rho_lag), {"rho_lag": round(float(rho_lag), 3), "tau_min": tau,
                                                      "rho_sun": round(float(rho_sun), 3)}))
        options.append(("orbit", math.sqrt(max(r2_orbit[j], 0)) * 0.9, {"r2_orbit": round(float(r2_orbit[j]), 3)}))
        if frac_up[j] > 0.97 and up[j] > 20:
            slope = _slope(unix, x)
            info = {"frac_up": round(float(frac_up[j]), 4), "per_second": slope,
                    "resets": int(((step[:, j] < 0) & (np.abs(step[:, j]) > 0.5 * (x.max() - x.min()))).sum())}
            clock = _clock(unix, x)
            # misaligned read: the true lowest byte is one up, and it, not ours, counts in-pass
            lsb_c, nxt_c = ((o + w - 1, o + w - 2) if endian == "be" else (o, o + 1)) if w > 1 else (o, None)
            misaligned = (nxt_c is not None and rolling.get(nxt_c, 0) >= 0.9 > rolling.get(lsb_c, 0) + 0.3)
            pen = 0.02 if misaligned else 0.0
            if clock:
                options.append(("clock", 0.999 - pen, {**info, **clock}))
            else:
                options.append(("counter", float(frac_up[j]) - pen, info))
        elif (frac_up_pass[j] > 0.97 and up_p[j] > 20 and 1 <= wrap_p[j] <= 0.2 * up_p[j]
              and np.unique(x).size >= 16):
            inc = sp[:, j] > 0
            rate = float(np.median(sp[inc, j] / np.maximum(dt_same[inc], 1))) if inc.any() else float("nan")
            options.append(("counter", float(frac_up_pass[j]),
                            {"frac_up_in_pass": round(float(frac_up_pass[j]), 4), "per_second": rate,
                             "rolling": True, "wraps_in_pass": int(wrap_p[j])}))
        kind, s, info = max(options, key=lambda t: t[1])
        phys_best[(o, w, endian, signed)] = max(v for k_, v, _ in options if k_ in ("sun", "lagged", "orbit"))
        n_eff = n_pass_sun if kind in ("sun", "lagged") else n_pass
        p = _p_from_rho(np.array([s]), max(n_eff, 3))[0] if kind in ("sun", "lagged", "orbit") else 0.0
        if p > alpha:
            continue
        det.update(info)
        det["p"] = float(p)
        fields.append(Field(o, w, endian, signed, kind, float(s), det, x))

    return fields, phys_best
