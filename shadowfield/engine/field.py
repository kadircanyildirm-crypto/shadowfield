"""A candidate field: one reading (offset, width, byte order, sign) of the frame bytes."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
        return range(self.offset, self.offset + self.width * self.detail.get("axes", 1))  # vectors: all axes

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

def _k(f: Field) -> tuple:
    return (f.offset, f.width, f.endian, f.signed)
