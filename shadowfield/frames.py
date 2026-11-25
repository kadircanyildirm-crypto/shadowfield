"""Load frames, strip AX.25 headers, dedupe, and split into frame types."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def strip_ax25(raw: bytes) -> tuple[str | None, bytes]:
    """Return (source callsign, payload) if raw looks like an AX.25 UI frame."""
    i = 0
    addrs = []
    while i + 7 <= len(raw) and len(addrs) < 10:
        a = raw[i:i + 7]
        call = bytes(b >> 1 for b in a[:6])
        if not all(48 <= c <= 90 or c == 32 for c in call):  # A-Z, 0-9, space
            return None, raw
        addrs.append(call.decode("ascii").strip())
        i += 7
        if a[6] & 1:
            break
    else:
        return None, raw
    if len(addrs) < 2 or i + 2 > len(raw):
        return None, raw
    if raw[i] != 0x03 or raw[i + 1] != 0xF0:
        return None, raw
    return addrs[1], raw[i + 2:]


@dataclass
class FrameType:
    key: str
    length: int
    df: pd.DataFrame  # columns: ts (datetime64[ns, UTC]), obs_id, tle1, tle2, payload (bytes)
    bytes_: np.ndarray = field(repr=False, default=None)  # (n, length) uint8

    def __post_init__(self):
        self.bytes_ = np.frombuffer(b"".join(self.df["payload"]), dtype=np.uint8).reshape(len(self.df), self.length)


def _split_types(g: pd.DataFrame, key: str, min_count: int, depth: int = 2, used=()):
    """Split frames on a type/sub-type byte near the start (up to two levels).

    Many satellites put a packet type in one of the first bytes, and some a sub-type a few
    bytes later (one frame length, several unrelated layouts).
    """
    if depth == 0:
        yield key, g
        return
    n = len(g["payload"].iloc[0])
    for off in range(min(8, n)):
        if off in used:
            continue
        counts = Counter(p[off] for p in g["payload"])
        if 2 <= len(counts) <= 6 and counts.most_common(1)[0][1] < 0.95 * len(g):
            big = [v for v, c in counts.items() if c >= min_count]
            if big:
                for v, gg in g.groupby(g["payload"].map(lambda p: p[off])):
                    if len(gg) >= min_count:
                        yield from _split_types(gg, f"{key}:b{off}={v:#04x}", min_count, depth - 1, used + (off,))
                return
    yield key, g


def load(path: Path, min_count: int = 150, max_types: int = 4) -> list[FrameType]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            f = json.loads(line)
        except json.JSONDecodeError:  # a download may still be appending its last line
            continue
        raw = bytes.fromhex(f["hex"])
        src, payload = strip_ax25(raw)
        rows.append({"ts": f["ts"], "obs_id": f["obs_id"], "tle1": f["tle1"], "tle2": f["tle2"],
                     "src": src or "", "payload": payload, "len": len(payload), "raw": raw})
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # the same frame is often heard by several stations: keep the earliest copy
    df = df.sort_values("ts").drop_duplicates(subset=["payload"], keep="first")
    # one pass = one physical overflight, whichever stations heard it: split on gaps > 20 min.
    # (DB frames carry no observation id, and overlapping stations would interleave anyway)
    gap = df["ts"].diff().dt.total_seconds().fillna(1e9) > 1200
    df["obs_id"] = gap.cumsum().astype(int)

    types = []
    for (src, n), g in df.groupby(["src", "len"]):
        if len(g) < min_count or n < 8:
            continue
        for key, gg in _split_types(g, f"{src or 'raw'}:len{n}", min_count):
            types.append(FrameType(key, n, gg.reset_index(drop=True)))
    types.sort(key=lambda t: -len(t.df))
    return types[:max_types]
