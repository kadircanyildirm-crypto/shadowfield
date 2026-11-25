"""Is a multi-byte reading really one number? Carry, sign extension, wrap point, low-byte continuity."""
from __future__ import annotations

import numpy as np

from .field import Field, _decode, _k, _split

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
    if f.width == 1 or f.signed == "f" or f.kind in ("clock", "magnet", "position", "attitude"):
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
    top_best = max(ctx["phys_best"].get(t, 0.0) for t in tops)
    if f.score >= top_best + 0.02:
        return True
    # The least significant byte of a measured number carries its noise and wraps, so it
    # is much rougher inside a pass than the most significant byte. Two separate smooth
    # sensors side by side are not. With that, a reading that loses nothing against its
    # top part is one number even when the sunlight correlation is saturated (a solar
    # current is near-binary, so extra resolution cannot raise it).
    if lo_w == 1 and f.score >= top_best - 0.005:
        r_lo = ctx["rough"].get((lo_off, 1, "be", False), 0.0)
        r_hi = max(ctx["rough"].get((top_off, 1, "be", sg), 9.0) for sg in (False, True)) if h == 1 else \
            min(ctx["rough"].get((top_off, h, f.endian, sg), 9.0) for sg in (False, True))
        if r_lo >= r_hi + 0.2 and _lsb_follows(ctx["B"], lo_off, top_off, h, f.endian, ctx["same"]):
            f.detail["joined"] = "lsb"
            return True
    return False

def _lsb_follows(B, lo_off, hi_off, hi_w, endian, same) -> bool:
    """Is the low byte the continuation of the high part rather than an unrelated noisy byte?

    Inside a pass, while the high part stays put, a real low byte moves by small steps
    (the signal's own noise); a random neighbour jumps like any two draws from its values.
    """
    lo = B[:, lo_off].astype(float)
    hi = _decode(B, hi_off, hi_w, endian, False)
    m = same & (np.diff(hi) == 0)
    if m.sum() < 20:
        return False
    step = np.median(np.abs(np.diff(lo))[m])
    rng = np.random.default_rng(0)
    shuffled = np.median(np.abs(lo - rng.permutation(lo)))
    return step <= 0.6 * shuffled

def _carries_into(B: np.ndarray, narrow: Field, wide: Field) -> bool:
    """Are the extra top bytes of `wide` part of the same counter?

    Yes if they change only when `narrow` wraps (or resets), or never change but are not zero.

    That is what the upper bytes of one counter do. A neighbouring field that happens
    to move would change at unrelated moments. B must be in time order.
    """
    extra = [i for i in wide.span if i not in narrow.span]
    e = B[:, extra].astype(np.int64) @ (256 ** np.arange(len(extra)))
    de = np.diff(e) != 0
    vn = _decode(B, narrow.offset, narrow.width, narrow.endian, narrow.signed)
    if de.sum() == 0:
        # Constant top byte: zero adds nothing (narrow is exact); non-zero is part of the
        # absolute value (a unix time whose top byte is 0x69 all month), unless the narrow
        # counter itself wraps at its own maximum, which a counter with more bytes cannot.
        top = 256 ** narrow.width
        wraps_own = ((np.diff(vn) < 0) & (vn[:-1] > 0.9 * top)).any()
        return bool(e[0] != 0 and not wraps_own)
    wraps = np.diff(vn) < 0
    return (de & wraps).sum() / de.sum() >= 0.8
