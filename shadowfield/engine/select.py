"""From all candidate readings to one non-overlapping layout, in a decoder author's widths."""
from __future__ import annotations

import numpy as np

from .field import Field, _decode, _k
from .structure import _boundary, _carries_into

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

def _canonical_widths(chosen: list[Field], pool: list[Field], B: np.ndarray) -> list[Field]:
    """Settle the width of each chosen field the way a decoder author would.

    - Narrower when it reads exactly the same values (a constant or sign-extension top
      byte adds nothing; decoders list the narrow type).
    - For counters and clocks, wider when the extra top byte is free and really moves:
      an uptime's top byte changes once a month, too rarely to win on evidence, but it
      is part of the number.
    """
    def same_lsb(a, b):
        if a.endian != b.endian and min(a.width, b.width) > 1:
            return False
        la = a.offset + a.width - 1 if a.endian == "be" else a.offset
        lb = b.offset + b.width - 1 if b.endian == "be" else b.offset
        return la == lb

    out = []
    for f in chosen:
        if f.kind in ("magnet", "attitude") or f.signed == "f":
            out.append(f)
            continue
        claimed = set().union(*(set(g.span) for g in chosen if g is not f)) if len(chosen) > 1 else set()
        vf = _decode(B, f.offset, f.width, f.endian, f.signed)
        best = f
        for g in pool:
            if g is f or not same_lsb(f, g) or g.signed == "f" or set(g.span) & claimed:
                continue
            vg = _decode(B, g.offset, g.width, g.endian, g.signed)
            if g.width < best.width and np.array_equal(vg, vf) and g.kind == f.kind:
                best = g
            elif (f.kind in ("counter", "clock") and g.kind in ("counter", "clock") and g.width > best.width
                  and g.width <= 4 and g.score >= f.score - 0.01 and not np.array_equal(vg, vf)
                  and _carries_into(B, f, g)):
                best = g
        out.append(best)
    return sorted(out, key=lambda f: f.offset)

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
        # With a clear frame byte order, a minority-order reading that overlaps an almost
        # equally good same-width reading in the frame's order is a misaligned read.
        if votes[dom] >= 2 and votes[dom] >= 2 * votes["le" if dom == "be" else "be"]:
            dom_reads = [g for g in fields if g.width > 1 and g.endian == dom]
            fields = [f for f in fields if not (
                f.width > 1 and f.endian != dom and not f.detail.get("joined")
                and f.kind in ("sun", "lagged", "orbit")
                and any(g.width == f.width and set(g.span) & set(f.span) and g.score >= f.score - 0.05
                        for g in dom_reads))]
    joined = [f for f in fields if f.detail.get("joined")]
    # carries also appear when the split point cuts through a real neighbouring field,
    # so a carry-joined reading only absorbs parts it explains about as well
    fields = [f for f in fields if f.detail.get("joined") or not any(
        set(f.span) < set(j.span) and (j.detail["joined"] != "carry" or j.score >= f.score - 0.05)
        for j in joined)]

    def key(f: Field):
        std_clock = f.kind == "clock" and f.detail.get("epoch_guess") != "custom"
        width = f.width * f.detail.get("axes", 1)
        if f.kind in ("magnet", "position", "attitude"):
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
