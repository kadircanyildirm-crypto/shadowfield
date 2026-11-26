"""Build docs/data.json for the explorer page from the benchmark results. Run: python -m tools.make_explorer"""
import json
from pathlib import Path

import numpy as np

from shadowfield import bench, pipeline
from shadowfield.report import _values

ROOT = Path(__file__).resolve().parents[1]
SHOW = [(43880, "uwe4", "UWE-4", "DP0UWH:len58", "German university cubesat, 2018"),
        (43784, "snuglite", "SNUGLITE", "raw:len214", "Korean university cubesat with GNSS, 2018")]
KIND_GROUP = {"lagged": "thermal-ish", "sun": "power-ish"}


def pick(rng, n, k=320):
    return np.sort(rng.choice(n, size=min(k, n), replace=False))


out = {"satellites": []}
rng = np.random.default_rng(0)
for norad, dec, label, key, blurb in SHOW:
    res = [r for r in pipeline.run(norad, ROOT / "data", decoder=dec, use_llm=False, peel_layers=False) if r.ft.key == key][0]
    saved = next(i for i in json.loads((ROOT / "results" / f"{norad}.json").read_text(encoding="utf-8")) if i["frame_type"] == key)
    names = {g["offset"]: g for g in (saved.get("names") or {}).get("fields", [])}
    truth = {t["offset"]: t for t in saved["evaluation"]["truth"]} if saved.get("evaluation") else {}
    ph = res.phys
    order = np.argsort(ph["unix"].to_numpy())
    frame = res.ft.bytes_[order[len(order) // 2]].tolist()
    fields = []
    for f in res.fields:
        y = _values(res.ft, f)
        if f.kind in ("counter", "clock", "attitude"):
            x, xl = (ph["unix"].to_numpy() - ph["unix"].min()) / 86400, "days"
        elif f.kind == "position":
            x, xl = ph[f.detail["component"]].to_numpy(), f"orbit {f.detail['component']} (km, from the TLE)"
        else:
            x, xl = ph["since_change"].to_numpy(), "minutes since sunset (−) / sunrise (+)"
        idx = pick(rng, len(y))
        g = names.get(f.offset, {})
        span = f.width * f.detail.get("axes", 1)
        t = next((tt for o, tt in truth.items() if o < f.offset + span and f.offset < o + tt["width"]), None)
        d = f.detail
        ev = {k: d[k] for k in ("rho_sun", "tau_min", "rho_lag", "r2_orbit", "per_second", "epoch_guess", "component",
                                "matching_frames_fraction", "rho_igrf", "unit_norm_fraction", "power_class_hint",
                                "joined", "rough") if k in d and d[k] is not None}
        fields.append({"offset": f.offset, "span": span, "type": f.ctype, "kind": f.kind, "score": round(f.score, 3),
                       "name": g.get("name"), "category": g.get("category"), "unit": g.get("unit"),
                       "scale": g.get("scale"), "confidence": g.get("confidence"), "rationale": g.get("rationale"),
                       "evidence": ev, "xlabel": xl,
                       "x": np.round(x[idx], 2).tolist(), "y": np.round(y[idx].astype(float), 3).tolist(),
                       "decoder": {"name": t["name"], "exact": t["offset"] == f.offset and t["width"] == f.width}
                       if t else None})
    ksy = next((p for p in (ROOT / "results" / "ksy").glob(f"{norad}_*.ksy") if key.replace(":", "_") in p.stem), None)
    ev = saved.get("evaluation") or {}
    out["satellites"].append({
        "norad": norad, "name": label, "blurb": blurb, "frame_type": key, "length": res.ft.length,
        "frames": len(res.ft.df), "passes": int(res.ft.df["obs_id"].nunique()), "frame": frame,
        "summary": (saved.get("names") or {}).get("summary"), "health": (saved.get("names") or {}).get("health"),
        "grade": {k: ev.get(k) for k in ("recall_physical", "value_exact_physical", "precision", "llm_category_accuracy")},
        "decoder_fields": [{"offset": t["offset"], "width": t["width"], "name": t["name"], "category": t["category"]}
                           for t in truth.values() if t["category"] != "other"],
        "fields": fields, "ksy": ksy.read_text(encoding="utf-8") if ksy else ""})
rows = json.loads((ROOT / "results" / "scoreboard.json").read_text(encoding="utf-8"))
out["scores"] = {s: bench.totals([r for r in rows if r["split"] == s]) for s in ("development", "held-out")}
out["scores"]["held_out_sats"] = sorted({r["name"] for r in rows if r["split"] == "held-out"})
(ROOT / "docs" / "data.json").write_text(json.dumps(out, separators=(",", ":"), default=float), encoding="utf-8")
print("wrote", (ROOT / "docs" / "data.json").stat().st_size // 1024, "KB")
