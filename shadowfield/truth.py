"""Blind-test scoring against satellites that already have a public decoder.

The SatNOGS decoders (Kaitai Struct) are used only AFTER Shadowfield has made its
guesses: they tell us where the real fields are and what they are called, so we
can measure how many fields physics alone recovered, and whether it understood
what they measure.
"""
from __future__ import annotations

import importlib
import re

import numpy as np
import pandas as pd

from .frames import FrameType
from .score import Field, _decode

CATEGORY_PATTERNS = [
    ("magnet", r"mag(?!ic)|mgm|magnetom"),
    ("thermal", r"temp|tmp|therm|(^|_)t(_|\d|$)"),
    ("clock", r"timestamp|epoch|rtc|unix|date|(^|_)time(_|$)|obc_?time"),
    ("nav", r"position|velocity|gnss|gps_?(x|y|z|v|lat|lon|alt)|(^|_)(pos|vel)_?[xyz]|latitude|longitude|altitude"),
    ("counter", r"count|cnt|uptime|boot|reset|seq|reboot|tick|frame_?n|packet_?n|nr_"),
    ("power", r"volt|vbat|curr|amp|pow|pwr|solar|panel|bat|pcm|charge|mppt|(^|_)v(_|\d|bat)|(^|_)i(_|\d)|bus|sun|light|photo|css"),
]

KIND_TO_CATEGORIES = {
    "sun": {"power"},
    "lagged": {"thermal", "power"},
    "orbit": {"thermal", "power", "magnet"},
    "counter": {"counter", "clock"},
    "clock": {"clock", "counter"},
    "magnet": {"magnet"},
    "position": {"nav"},
    "attitude": {"nav"},
}


FINE_POWER = [  # order matters: "solar_cells_current" is solar, "cell_voltage" is battery
    ("power_solar", r"solar|(^|_)sol_|panel.*(curr|volt)|mppt|(^|_)sun|array"),
    ("power_battery", r"bat|(^|_)cell|(^|_)soc|state_of_charge|charge"),
    ("power_bus", r"(^|_)bus|consumption|load|(^|_)power(_|$)|pwr|vcc|3v3|5v"),
]


def fine_category(name: str) -> str | None:
    """battery / solar / bus for power fields (None when the name does not say)."""
    if category(name) != "power":
        return None
    n = name.lower()
    for cat, pat in FINE_POWER:
        if re.search(pat, n):
            return cat
    return None


def category(name: str) -> str:
    n = name.lower()
    for cat, pat in CATEGORY_PATTERNS:
        if re.search(pat, n):
            return cat
    return "other"


def decode_frames(decoder: str, raws: list[bytes]) -> pd.DataFrame:
    from satnogsdecoders.decoder import get_fields
    mod = importlib.import_module(f"satnogsdecoders.decoder.{decoder}")
    cls = getattr(mod, decoder.capitalize())
    rows = []
    for raw in raws:
        try:
            rows.append(get_fields(cls.from_bytes(raw)))
        except Exception:
            rows.append({})
    return pd.DataFrame(rows)


def locate_true_fields(ft: FrameType, raws: list[bytes], decoder: str) -> list[dict]:
    """Find each decoder field's byte position by exact value matching against raw readings."""
    truth = decode_frames(decoder, raws)
    B = ft.bytes_
    out = []
    for col in truth.columns:
        y = pd.to_numeric(truth[col], errors="coerce").to_numpy(dtype=float)
        ok = ~np.isnan(y)
        if ok.sum() < 50 or np.unique(y[ok]).size < 4:
            continue
        best = None
        for w in (1, 2, 3, 4):
            for o in range(B.shape[1] - w + 1):
                for endian in (("be", "le") if w > 1 else ("be",)):
                    for signed in (False, True):
                        x = _decode(B, o, w, endian, signed)[ok]
                        if x.std() == 0:
                            continue
                        r = abs(np.corrcoef(x, y[ok])[0, 1])
                        # best match first; among exact ties the narrowest reading (constant
                        # neighbouring bytes do not change r, so wider readings tie exactly)
                        # a raw integer field equals its reading exactly; prefer that
                        rank = (bool(np.array_equal(x, y[ok])), round(r, 10), -w)
                        if r > 0.9995 and (best is None or rank > best["_rank"]):
                            best = {"name": col, "offset": o, "width": w, "endian": endian,
                                    "signed": signed, "category": category(col), "_rank": rank,
                                    "located": "exact" if rank[0] else "affine"}
        for o in range(B.shape[1] - 3):  # float32 fields
            for endian in ("be", "le"):
                x = _decode(B, o, 4, endian, "f")[ok]
                if x.std() == 0 or not np.isfinite(x).all():
                    continue
                exact = bool(np.allclose(x, y[ok], rtol=1e-6, atol=1e-9))
                r = abs(np.corrcoef(x, y[ok])[0, 1]) if np.isfinite(x).all() else 0.0
                rank = (exact, round(r, 10), -4)
                if (exact or r > 0.9995) and (best is None or rank > best["_rank"]):
                    best = {"name": col, "offset": o, "width": 4, "endian": endian,
                            "signed": "f", "category": category(col), "_rank": rank,
                            "located": "exact" if exact else "affine"}
        if best:
            best.pop("_rank")
            best["fine"] = fine_category(col)
            out.append(best)
    # drop duplicate aliases of the same bytes (decoders often expose raw + scaled)
    seen, uniq = set(), []
    for t in out:
        k = (t["offset"], t["width"])
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


VECTOR_KINDS = ("magnet", "attitude")


def evaluate(found: list[Field], truth: list[dict], B: np.ndarray | None = None) -> dict:
    """Grade found fields against located decoder fields.

    exact        same byte offset and width
    value_exact  the found reading reproduces the decoder's raw values bit for bit
                 (so byte order and signedness are right too, wherever they matter)
    """
    physical = [t for t in truth if t["category"] != "other"]
    rows = []
    for t in truth:
        t_span = set(range(t["offset"], t["offset"] + t["width"]))
        match = None
        for f in found:
            if f.kind in VECTOR_KINDS:
                continue
            if f.offset == t["offset"] and f.width == t["width"]:
                match, how = f, "exact"
                break
            if t_span & set(f.span):
                match, how = f, "overlap"
        if any(f.kind in VECTOR_KINDS and t_span <= set(f.span) for f in found):
            # one component of a vector the engine found as a whole (magnetometer, quaternion)
            match = next(f for f in found if f.kind in VECTOR_KINDS and t_span <= set(f.span))
            how = "exact" if (t["offset"] - match.offset) % match.width == 0 and t["width"] == match.width else "overlap"
        value_exact = False
        if match is not None and B is not None and match.kind not in VECTOR_KINDS and how == "exact":
            value_exact = bool(np.array_equal(
                _decode(B, match.offset, match.width, match.endian, match.signed),
                _decode(B, t["offset"], t["width"], t["endian"], t["signed"])))
        elif match is not None and match.kind in VECTOR_KINDS and how == "exact":
            value_exact = True
        rows.append({**t, "found": match is not None, "how": how if match else "",
                     "value_exact": value_exact, "found_offset": match.offset if match else None,
                     "kind": match.kind if match else "",
                     "kind_ok": bool(match) and t["category"] in KIND_TO_CATEGORIES.get(match.kind, set())})
    res = pd.DataFrame(rows)
    phys = res[res["category"] != "other"] if len(res) else res
    found_true = sum(1 for f in found if any(
        set(f.span) & set(range(t["offset"], t["offset"] + t["width"])) for t in truth))
    return {
        "true_fields": len(truth),
        "physical_true_fields": len(physical),
        "recall_physical": float(phys["found"].mean()) if len(phys) else float("nan"),
        "exact_physical": float((phys["how"] == "exact").mean()) if len(phys) else float("nan"),
        "value_exact_physical": float(phys["value_exact"].mean()) if len(phys) else float("nan"),
        "located_exact_fraction": float((res["located"] == "exact").mean()) if len(res) else float("nan"),
        "kind_accuracy": float(phys.loc[phys["found"], "kind_ok"].mean()) if phys["found"].any() else float("nan"),
        "precision": found_true / len(found) if found else float("nan"),
        "table": res,
    }
