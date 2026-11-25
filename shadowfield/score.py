"""Find telemetry fields in unknown frames by correlating raw bytes with physics.

Every (offset, width, endianness, signedness) reading of the frame bytes is a
candidate field. Each candidate is scored against the orbital physics:

  sun      steps with sunlight right now        -> solar array current/voltage, sun sensors
  lagged   follows sunlight with a time lag      -> temperatures, battery charge
  orbit    repeats once/twice per orbit          -> attitude/thermal cycles
  counter  only ever goes up (with resets)       -> uptime, packet/boot counters
  clock    tracks wall-clock time 1:1            -> on-board time stamps
  magnet   |(x,y,z)| tracks Earth's field model  -> 3-axis magnetometer
  position equals a coordinate of the orbit       -> GNSS position / velocity (ECEF or inertial)

Frames inside one pass are strongly correlated, so significance is judged on
the number of passes, not the number of frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd
from scipy import stats

from .physics import TAUS

EPOCHS = {"unix": 0.0, "J2000": 946727935.816, "GPS": 315964800.0}


@dataclass
class Field:
    offset: int
    width: int
    endian: str
    signed: bool
    kind: str
    score: float
    detail: dict = field(default_factory=dict)
    values: np.ndarray | None = field(default=None, repr=False)

    @property
    def span(self) -> range:
        return range(self.offset, self.offset + self.width * self.detail.get("axes", 1))

    @property
    def ctype(self) -> str:
        t = ("f" if self.signed == "f" else "s" if self.signed else "u") + str(self.width)
        return t if self.width == 1 else t + self.endian

    def as_dict(self) -> dict:
        return {"offset": self.offset, "type": self.ctype, "kind": self.kind,
                "score": round(self.score, 3), **{k: v for k, v in self.detail.items()}}


WIDTHS = (1, 2, 3, 4)


def _decode(B: np.ndarray, o: int, w: int, endian: str, signed) -> np.ndarray:
    if signed == "f":  # IEEE-754 single precision (GNSS fixes, attitude quaternions)
        dt = np.dtype("f4").newbyteorder(">" if endian == "be" else "<")
        x = np.ascontiguousarray(B[:, o:o + 4]).view(dt).ravel().astype(np.float64)
        return np.where(np.isfinite(x), x, 0.0)
    if w in (1, 2, 4, 8):
        dt = np.dtype(("i" if signed else "u") + str(w)).newbyteorder(">" if endian == "be" else "<")
        return np.ascontiguousarray(B[:, o:o + w]).view(dt).ravel().astype(np.float64)
    cols = B[:, o:o + w].astype(np.int64)
    if endian == "le":
        cols = cols[:, ::-1]
    v = np.zeros(len(B), dtype=np.int64)
    for c in cols.T:
        v = v * 256 + c
    if signed:
        v = np.where(v >= 2 ** (8 * w - 1), v - 2 ** (8 * w), v)
    return v.astype(np.float64)


def _split(key) -> tuple[int, int, int, int]:
    """(hi_off, hi_w, lo_off, lo_w): the reading split into its high part and least significant part."""
    o, w, endian = key[:3]
    lo_w = 1 if w == 3 else w // 2
    hi_w = w - lo_w
    return (o, hi_w, o + hi_w, lo_w) if endian == "be" else (o + lo_w, hi_w, o, lo_w)


def _ranks(M: np.ndarray) -> np.ndarray:
    R = stats.rankdata(M, axis=0)
    R -= R.mean(0)
    sd = R.std(0)
    sd[sd == 0] = np.inf
    return R / sd


def _demean(R: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = R.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(0)
    sd = out.std(0)
    sd[sd == 0] = np.inf
    return out / sd


def _p_from_rho(rho: np.ndarray, n_eff: int) -> np.ndarray:
    r = np.clip(np.abs(rho), 0, 0.999999)
    t = r * np.sqrt((n_eff - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(t, n_eff - 2)


def analyze(B: np.ndarray, df: pd.DataFrame, phys: pd.DataFrame, alpha: float = 1e-6,
            quick: bool = False) -> list[Field]:
    """All fields in one frame type. quick=True (1-2 byte integers only, no GNSS search) is for
    comparing candidate decodings of the same bytes, where only the relative evidence matters."""
    n, L = B.shape
    order = np.argsort(df["ts"].to_numpy())
    B, df, phys = B[order], df.iloc[order].reset_index(drop=True), phys.iloc[order].reset_index(drop=True)
    n_pass = int(df["obs_id"].nunique())
    unix = phys["unix"].to_numpy()

    # ---- candidate readings -------------------------------------------------
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
            if sane.mean() >= 0.98 and nz.sum() >= 30 and np.unique(x[: min(n, 2000)]).size >= 4:
                cand.append((o, 4, endian, "f"))
                X.append(x)
    if not X:
        return []
    X = np.column_stack(X)

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
    apart = dts > 5
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

    # ---- physics correlations on ranks ---------------------------------------
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
    C = RX.T @ RF / n  # (n_cand, n_feat) Spearman
    names = list(feats)
    ci = {k: i for i, k in enumerate(names)}

    # orbital periodicity: R^2 of first two harmonics of the orbital phase
    su, cu = phys["sin_u"].to_numpy(), phys["cos_u"].to_numpy()
    H = np.column_stack([np.ones(n), su, cu, 2 * su * cu, cu ** 2 - su ** 2])
    beta, *_ = np.linalg.lstsq(H, RX, rcond=None)
    r2_orbit = 1 - ((RX - H @ beta) ** 2).mean(0)

    fields: list[Field] = []
    phys_best: dict[tuple, float] = {}
    lit = feats["sunlit"]
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
            if clock:
                options.append(("clock", 0.999, {**info, **clock}))
            else:
                options.append(("counter", float(frac_up[j]), info))
        elif (frac_up_pass[j] > 0.97 and up_p[j] > 20 and 1 <= wrap_p[j] <= 0.2 * up_p[j]
              and np.unique(x).size >= 16):
            inc = sp[:, j] > 0
            rate = float(np.median(sp[inc, j] / np.maximum(dt_same[inc], 1))) if inc.any() else float("nan")
            options.append(("counter", float(frac_up_pass[j]),
                            {"frac_up_in_pass": round(float(frac_up_pass[j]), 4), "per_second": rate,
                             "rolling": True, "wraps_in_pass": int(wrap_p[j])}))
        kind, s, info = max(options, key=lambda t: t[1])
        phys_best[(o, w, endian, signed)] = max(v for k_, v, _ in options if k_ in ("sun", "lagged", "orbit"))
        p = _p_from_rho(np.array([s]), n_pass)[0] if kind in ("sun", "lagged", "orbit") else 0.0
        if p > alpha:
            continue
        det.update(info)
        det["p"] = float(p)
        fields.append(Field(o, w, endian, signed, kind, float(s), det, x))

    fields += _magnetometers(cand, X, phys["b_igrf"].to_numpy(), n_pass, alpha)
    if not quick:
        fields += _positions(cand, X, phys, df["obs_id"].to_numpy())
    ctx = {"B": B, "same": same, "phys_best": phys_best,
           "up": {c: int(up[j]) for j, c in enumerate(cand)},
           "rough": {c: float(rough[j]) for j, c in enumerate(cand)}}
    fields = _pick_encodings(fields, ctx["rough"], B)
    fields = [f for f in fields if _width_ok(f, ctx)]
    return _greedy(fields, ctx["rough"])


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
        best = None
        for k, col in enumerate(cols):
            f = F[:, k]
            big = valid & (np.abs(f) > 0.3 * fmax[k])
            if big.sum() < 20:
                continue
            scale = float(np.median(x[big] / f[big]))
            if not np.isfinite(scale) or scale == 0:
                continue
            match = valid & (np.abs(x / scale - f) < 0.01 * fmax[k])
            if match.sum() < 20:
                continue
            n_pass = np.unique(obs[match]).size
            frac = match.sum() / valid.sum()
            # stations cluster geographically, so a nearly constant byte can sit inside a
            # cluster of coordinates: the matches must sweep most of the coordinate's range,
            # with the field moving along (not just many frames near one value)
            fm = f[match]
            sweep = (fm.max() - fm.min()) / (2 * fmax[k])
            if sweep < 0.5 or np.unique(x[match]).size < 15 or abs(np.corrcoef(x[match], fm)[0, 1]) < 0.99:
                continue
            if n_pass >= 5 and frac >= 0.25 and (best is None or frac > best[0]):
                best = (frac, k, scale, int(match.sum()), int(n_pass))
        if best:
            frac, k, scale, n_match, n_pass = best
            unit = "deg" if cols[k] == "lat" else ("km" if cols[k].startswith(("pos", "alt")) else "km/s")
            out.append(Field(*c, "position", 0.95 + 0.05 * frac,
                             {"component": cols[k], "matching_frames_fraction": round(frac, 3),
                              "matching_frames": n_match, "matching_passes": n_pass,
                              "raw_per_" + unit.replace("/", "_per_"): round(scale, 4),
                              "min": float(x.min()), "max": float(x.max())}, x))
    return out


def _magnetometers(cand, X, b, n_pass, alpha) -> list[Field]:
    if np.isnan(b).all():
        return []
    key = {c: j for j, c in enumerate(cand)}
    out = []
    rb = stats.rankdata(b)
    for (o, w, endian, signed), j in key.items():
        if not signed or w == 1:
            continue
        j2, j3 = key.get((o + w, w, endian, signed)), key.get((o + 2 * w, w, endian, signed))
        if j2 is None or j3 is None:
            continue
        m = np.sqrt(X[:, j] ** 2 + X[:, j2] ** 2 + X[:, j3] ** 2)
        rho = stats.spearmanr(m, rb).statistic
        if rho > 0.5 and _p_from_rho(np.array([rho]), n_pass)[0] < alpha:
            out.append(Field(o, w, endian, signed, "magnet", float(rho),
                             {"axes": 3, "rho_igrf": round(float(rho), 3)}, m))
    return out


def _pick_encodings(fields: list[Field], rough: dict, B: np.ndarray | None = None) -> list[Field]:
    """For each byte span keep one reading: near-best physics score, then the smoothest.

    Wrong byte order or signedness puts the noisy low byte (or a sign wrap) in the
    high position, which makes the series jump inside a pass even when its rank
    correlation with the physics is almost as good.
    """
    groups: dict[tuple, list[Field]] = {}
    for f in fields:
        groups.setdefault((f.offset, f.width, f.kind == "magnet"), []).append(f)
    out = []
    for fs in groups.values():
        # two's complement: if the signed reading is far more compact than the unsigned one
        # (values near 0 and near the maximum), the number is signed. The unsigned reading
        # looks deceptively smooth because its huge, bimodal spread inflates the scale.
        def spread(f):
            lo, hi = np.percentile(f.values, [1, 99])
            return hi - lo
        drop = set()
        for u in fs:
            if u.signed:
                continue
            for sg in fs:
                if sg.signed and sg.endian == u.endian and spread(sg) < 0.25 * spread(u):
                    drop.add(id(u))
        fs = [f for f in fs if id(f) not in drop]
        floats = [f for f in fs if f.signed == "f"]
        if floats and max(f.score for f in floats) >= max(f.score for f in fs) - 0.05:
            fs = floats
        top = max(f.score for f in fs)
        near = [f for f in fs if f.score >= top - 0.03]
        # byte order and sign proven by structure beat a slightly better correlation
        if B is not None and fs[0].width == 2:
            proven = [f for f in fs if f.score >= top - 0.1 and f.kind not in ("magnet", "position")
                      and _boundary(B, f)]
            if proven:
                near = proven
        out.append(min(near, key=lambda f: (round(rough.get(_k(f), 1.0), 2), f.signed)))
    return out


def _k(f: Field) -> tuple:
    return (f.offset, f.width, f.endian, f.signed)


def _carry(B: np.ndarray, f: Field, same: np.ndarray) -> tuple[int, float]:
    """How often the low part wraps exactly when the high part steps by one."""
    hi_off, hi_w, lo_off, lo_w = _split((f.offset, f.width, f.endian))
    hi = _decode(B, hi_off, hi_w, f.endian, False)
    lo = _decode(B, lo_off, lo_w, f.endian, False)
    dhi, dlo = np.diff(hi)[same], np.diff(lo)[same]
    step = np.abs(dhi) == 1
    wrap = (np.sign(dlo) == -np.sign(dhi)) & (np.abs(dlo) > 256 ** lo_w / 2)
    n = int(step.sum())
    return n, float((wrap & step).sum() / n) if n else 0.0


def _boundary(B: np.ndarray, f: Field) -> bool:
    """A number that crosses a byte boundary leaves a fingerprint even without in-pass carries.

    Signed: a high byte that only ever holds -1/0 (0xFF/0x00, maybe 0xFE/0x01) is a sign
    extension, so the two bytes are one signed number (a battery current that reverses).
    Unsigned: wherever two neighbouring high values are both seen, the low part must meet
    at the wrap point: just under the top below it and just over zero above it (straddle),
    or, for a busy low byte, reaching both edges. An unrelated sensor byte next to a mode
    byte does neither.
    """
    hi_off, hi_w, lo_off, lo_w = _split((f.offset, f.width, f.endian))
    if hi_w != 1:
        return False
    hi = _decode(B, hi_off, 1, "be", f.signed)
    lo = _decode(B, lo_off, lo_w, f.endian, False)
    vals = np.unique(hi)
    if len(vals) < 2:
        return False
    n_lo = np.unique(lo).size
    if f.signed:
        return (len(vals) <= 8 and not np.any(np.diff(vals) != 1) and vals.min() < 0 <= vals.max()
                and n_lo >= 16)
    if len(vals) > 64 or n_lo < 8:
        return False
    top = 256 ** lo_w
    good = bad = 0
    for h in vals[:-1]:
        a, b = lo[hi == h], lo[hi == h + 1]
        if len(a) < 5 or len(b) < 5:
            continue
        if a.mean() > 0.5 * top > b.mean() and a.mean() - b.mean() > 0.35 * top:
            good += 1
        elif len(a) >= 20 and len(b) >= 20 and n_lo >= 64:
            reach = bool((a >= top - top / 16).any() and (b < top / 16).any())
            good += reach
            bad += not reach
    return good > 0 and bad == 0


def _width_ok(f: Field, ctx: dict) -> bool:
    """Accept a 2- or 4-byte reading only if its low half really belongs to it.

    Evidence for: the low half carries into the high half, or reading it adds
    resolution (better physics score / more counter increments). Without such
    evidence the wide reading would just swallow the neighbouring bytes.
    """
    if f.width == 1 or f.signed == "f" or f.kind in ("clock", "magnet", "position"):
        return True
    n, frac = _carry(ctx["B"], f, ctx["same"])
    # structural proof that these bytes form one number
    if _boundary(ctx["B"], f):
        f.detail["joined"] = "sign" if f.signed else "wrap"
        return True
    if n >= 5 and frac >= 0.5:
        f.detail["joined"] = "carry"
        return True
    top_off, h, lo_off, lo_w = _split((f.offset, f.width, f.endian))
    # the least significant byte of a real measurement takes many values; a status
    # byte with a handful of values is a neighbour, not the bottom of this number
    if np.unique(_decode(ctx["B"], lo_off, lo_w, f.endian, False)).size < 32:
        return False
    # every narrower reading that keeps the same most significant byte is a "top part";
    # some of them are not candidates (constant low byte), so compare with all that exist
    msb = f.offset if f.endian == "be" else f.offset + f.width - 1
    tops = []
    for hw in range(1, f.width):
        o2 = msb if f.endian == "be" else msb - hw + 1
        for e in (("be",) if hw == 1 else (f.endian,)):
            tops += [(o2, hw, e, sg) for sg in (False, True)]
    if f.kind == "counter":
        return ctx["up"][_k(f)] >= 1.5 * max(ctx["up"].get(t, 0) for t in tops)
    return f.score >= max(ctx["phys_best"].get(t, 0.0) for t in tops) + 0.02


def _greedy(fields: list[Field], rough: dict) -> list[Field]:
    """Non-overlapping choice: strongest evidence first, wider readings on ties.

    A reading with structural proof that its bytes form one number (carry or boundary
    test) first removes its own sub-readings, so a signed current is not split into a
    sign byte and a low byte; it still has to win against everything else on evidence.
    """
    # A sign byte is shared by the reading to its left and to its right, and both pass the
    # sign test. Frames use one byte order throughout, so let the frame's other multi-byte
    # fields decide which neighbour the sign byte belongs to.
    votes = {"be": 0.0, "le": 0.0}
    for f in fields:  # only readings whose width is proven by structure get a vote
        if f.width > 1 and (f.detail.get("joined") in ("wrap", "carry")
                            or f.kind in ("counter", "clock", "position")):
            votes[f.endian] += 1
    if votes["be"] != votes["le"]:
        dom = max(votes, key=votes.get)
        signs = [f for f in fields if f.detail.get("joined") == "sign"]
        fields = [f for f in fields if not (
            f.detail.get("joined") == "sign" and f.endian != dom and
            any(g.endian == dom and set(g.span) & set(f.span) for g in signs))]
        # a reading in the minority byte order that cuts across a structurally proven
        # reading in the frame's byte order is a misaligned read of two neighbours
        proven = [g for g in fields if g.detail.get("joined") in ("wrap", "carry", "sign") and g.endian == dom
                  and g.width > 1]
        fields = [f for f in fields if f.width == 1 or f.endian == dom
                  or f.kind in ("position", "magnet", "clock")
                  or not any(set(g.span) & set(f.span) for g in proven)]
    joined = [f for f in fields if f.detail.get("joined")]
    # carries also appear when the split point cuts through a real neighbouring field,
    # so a carry-joined reading only absorbs parts it explains about as well
    fields = [f for f in fields if f.detail.get("joined") or not any(
        set(f.span) < set(j.span) and (j.detail["joined"] != "carry" or j.score >= f.score - 0.05)
        for j in joined)]

    def key(f: Field):
        std_clock = f.kind == "clock" and f.detail.get("epoch_guess") != "custom"
        width = f.width * f.detail.get("axes", 1)
        if f.kind in ("magnet", "position"):
            # vector sensors and GNSS fixes: the full-width reading (finer than the orbit
            # can check) rather than a truncated one
            return (round(f.score, 2), std_clock, 0, width, -rough.get(_k(f), 1.0), not f.signed)
        # resolution first (a reading that drops the low byte loses distinct values), then
        # narrower (a constant top byte adds width but no information)
        distinct = np.unique(f.values).size
        return (round(f.score, 2), std_clock, distinct, -width, -rough.get(_k(f), 1.0), not f.signed)

    used: set[int] = set()
    chosen = []
    for f in sorted(fields, key=key, reverse=True):
        if used.intersection(f.span):
            continue
        chosen.append(f)
        used.update(f.span)
    return sorted(chosen, key=lambda f: f.offset)


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
        # fields that move together are often the same kind of sensor (four panel temperatures...)
        d["moves_with_offsets"] = [g.offset for g in fields if g is not f and
                                   abs(np.corrcoef(rx, ranked[id(g)])[0, 1]) > 0.85]
        f.detail.update(d)
