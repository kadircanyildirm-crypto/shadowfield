"""Structural detectors that need no sunlight: clocks, GNSS fixes, quaternions, magnetometers."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .field import Field
from .stats import _p_from_rho

EPOCHS = {"unix": 0.0, "J2000": 946727935.816, "GPS": 315964800.0}

def _slope(t: np.ndarray, x: np.ndarray, m: int = 600) -> float:
    idx = np.linspace(0, len(t) - 1, min(m, len(t))).astype(int)
    tt, xx = t[idx], x[idx]
    s = stats.theilslopes(xx, tt)[0] if len(tt) > 3 else float("nan")
    return float(s)

def _clock(t: np.ndarray, x: np.ndarray) -> dict | None:
    for scale in (1.0, 1000.0, 1 / 1000.0):
        d = t - x / scale
        med = np.median(d)
        if np.median(np.abs(d - med)) < 120:  # within two minutes of wall clock
            epoch = min(EPOCHS, key=lambda k: abs(EPOCHS[k] - med))
            return {"clock_scale": scale, "epoch_offset_s": float(med),
                    "epoch_guess": epoch if abs(EPOCHS[epoch] - med) < 86400 * 2 else "custom"}
    return None

# coordinates that swing through zero; altitude barely changes, so a constant-ish byte
# would "match" it, and it is left out
NAV = [f"{q}_{fr}_{a}" for q in ("pos", "vel") for fr in ("eci", "ecef") for a in "xyz"]

def _positions(cand, X, phys, obs=None) -> list[Field]:
    """Readings that equal one orbital coordinate times a scale: a GNSS fix in the frame.

    Real receivers often repeat their last fix for minutes, or send -1/0 when they have
    none, so a plain correlation fails. Instead the scale is estimated robustly (median of
    raw/coordinate) and a field counts when a good share of frames match the TLE orbit
    within 1% of the coordinate's range (about 70 km for positions), over several passes.
    """
    cols = [c for c in NAV if c in phys and np.isfinite(phys[c]).all() and phys[c].std() > 0]
    if not cols:
        return []
    F = phys[cols].to_numpy()
    fmax = np.abs(F).max(0)
    obs = np.zeros(len(X), dtype=int) if obs is None else obs
    out = []
    for j, c in enumerate(cand):
        if c[1] < 2:
            continue
        x = X[:, j]
        valid = (x != 0) & (x != -1)
        vals, counts = np.unique(x[valid], return_counts=True)
        if valid.sum() < 30 or vals.size < 20:
            continue
        # all coordinates at once: robust scale, then which frames agree with the orbit
        big = valid[:, None] & (np.abs(F) > 0.3 * fmax[None, :])
        enough = big.sum(0) >= 20
        if not enough.any():
            continue
        with np.errstate(all="ignore"):
            scale = np.nanmedian(np.where(big, x[:, None] / F, np.nan), axis=0)
            match = valid[:, None] & (np.abs(x[:, None] / scale[None, :] - F) < 0.01 * fmax[None, :])
        ok_k = enough & np.isfinite(scale) & (scale != 0) & (match.sum(0) >= 20)
        best = None
        for k in np.flatnonzero(ok_k):
            mk = match[:, k]
            fm = F[mk, k]
            sweep = (fm.max() - fm.min()) / (2 * fmax[k])
            if sweep < 0.5 or np.unique(x[mk]).size < 15 or abs(np.corrcoef(x[mk], fm)[0, 1]) < 0.99:
                continue
            n_pass = np.unique(obs[mk]).size
            frac = mk.sum() / valid.sum()
            if n_pass >= 5 and frac >= 0.25 and (best is None or frac > best[0]):
                best = (frac, int(k), float(scale[k]), int(mk.sum()), int(n_pass))
        if best:
            frac, k, scale, n_match, n_pass = best
            unit = "deg" if cols[k] == "lat" else ("km" if cols[k].startswith(("pos", "alt")) else "km/s")
            out.append(Field(*c, "position", 0.95 + 0.05 * frac,
                             {"component": cols[k], "matching_frames_fraction": round(frac, 3),
                              "matching_frames": n_match, "matching_passes": n_pass,
                              "raw_per_" + unit.replace("/", "_per_"): round(scale, 4),
                              "min": float(x.min()), "max": float(x.max())}, x))
    return out

def _quaternions(cand, X, tol: float = 0.02) -> list[Field]:
    """Four consecutive floats whose squares sum to one: an attitude quaternion.

    Attitude has no driver the orbit can predict, but a unit quaternion is a structural
    fingerprint no other telemetry has.
    """
    key = {c: j for j, c in enumerate(cand) if c[3] == "f"}
    out = []
    for (o, w, endian, sg), j in key.items():
        js = [key.get((o + 4 * i, 4, endian, "f")) for i in range(4)]
        if None in js:
            continue
        Q = X[:, js]
        nz = np.abs(Q).sum(1) > 0
        if nz.sum() < 30:
            continue
        norm = np.sqrt((Q[nz] ** 2).sum(1))
        frac = float((np.abs(norm - 1) < tol).mean())
        if frac >= 0.9 and all(np.unique(Q[nz, i]).size >= 10 for i in range(4)):
            out.append(Field(o, 4, endian, "f", "attitude", 0.95 + 0.05 * frac,
                             {"axes": 4, "unit_norm_fraction": round(frac, 3)}, np.sqrt((Q ** 2).sum(1))))
    return out

def _magnetometers(cand, X, b, n_pass, alpha) -> list[Field]:
    ok = np.isfinite(b)
    if ok.sum() < 30:
        return []
    key = {c: j for j, c in enumerate(cand)}
    out = []
    rb = stats.rankdata(b[ok])
    for (o, w, endian, signed), j in key.items():
        if not signed or w == 1:
            continue
        j2, j3 = key.get((o + w, w, endian, signed)), key.get((o + 2 * w, w, endian, signed))
        if j2 is None or j3 is None:
            continue
        m = np.sqrt(X[:, j] ** 2 + X[:, j2] ** 2 + X[:, j3] ** 2)
        rho = stats.spearmanr(m[ok], rb).statistic
        if rho > 0.5 and _p_from_rho(np.array([rho]), n_pass)[0] < alpha:
            out.append(Field(o, w, endian, signed, "magnet", float(rho),
                             {"axes": 3, "rho_igrf": round(float(rho), 3)}, m))
    return out
