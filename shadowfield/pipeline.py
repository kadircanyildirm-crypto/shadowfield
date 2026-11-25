"""End-to-end run: frames -> physics -> fields -> (Claude names) -> (blind score) -> report."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import frames as frames_mod
from . import physics, score, truth
from .fetch import load_env

ROOT = Path(__file__).resolve().parents[1]
MIN_PASSES = 20


@dataclass
class TypeResult:
    ft: frames_mod.FrameType
    phys: pd.DataFrame
    fields: list[score.Field]
    names: dict | None = None
    evaluation: dict | None = None
    note: str | None = None
    layer: str | None = None
    eclipse_note: str | None = None
    segment_note: str | None = None
    order: np.ndarray = field(default=None, repr=False)


def _one_season(ft, ph, gap_days: float = 14.0):
    """Keep one contiguous stretch of data when the frames come from separate seasons.

    Mixing, say, a June eclipse season with November data in permanent sunlight is a trap:
    everything that drifted between June and November (season, battery ageing, counters)
    then correlates with "had an eclipse recently". So the stretch where the sunlight
    history actually varies is analysed on its own.
    """
    u = ph["unix"].to_numpy()
    order = np.argsort(u)
    gaps = np.flatnonzero(np.diff(u[order]) > gap_days * 86400)
    if not len(gaps):
        return ft, ph, None
    bounds = np.split(order, gaps + 1)
    def value(idx):
        passes = ft.df["obs_id"].iloc[idx].nunique()
        return ph[f"ema_{physics.TAUS[2]}"].iloc[idx].std() * min(passes, 200) if passes >= MIN_PASSES else -1
    best = max(bounds, key=value)
    if value(best) <= 0:
        return ft, ph, None
    keep = np.sort(best)
    sub = ft.df.iloc[keep].reset_index(drop=True)
    ft2 = frames_mod.FrameType(ft.key, ft.length, sub)
    ph2 = ph.iloc[keep].reset_index(drop=True)
    d0, d1 = (pd.Timestamp(x, unit="s", tz="UTC").strftime("%Y-%m-%d") for x in (u[keep].min(), u[keep].max()))
    return ft2, ph2, (f"Frames come from {len(bounds)} separate seasons; analysed the one where the sunlight "
                      f"history varies ({d0} to {d1}, {len(sub)} frames) on its own, so seasonal drift "
                      f"cannot pose as physics.")


def orbit_summary(phys: pd.DataFrame) -> dict:
    return {
        "mean_altitude_km": round(float(phys["alt_km"].mean())),
        "max_abs_latitude_deg": round(float(phys["lat"].abs().max()), 1),
        "beta_angle_range_deg": [round(float(phys["beta"].min()), 1), round(float(phys["beta"].max()), 1)],
        "fraction_of_frames_in_sunlight": round(float(phys["sunlit"].mean()), 3),
        "fraction_of_frames_with_an_eclipse_in_the_previous_80_min": round(float((phys["ema_80"] < 0.999).mean()), 3),
        "days_covered": round(float((phys["unix"].max() - phys["unix"].min()) / 86400), 1),
    }


def run(norad: int, data_dir: Path, decoder: str | None = None, use_llm: bool = True,
        blind: bool = True, sat_name: str | None = None, max_types: int = 3,
        peel_layers: bool = True) -> list[TypeResult]:
    load_env(ROOT / ".env")
    path = data_dir / str(norad) / "frames.jsonl"
    types = frames_mod.load(path, max_types=max_types)
    results = []
    for ft in types:
        ph = physics.compute(ft.df, data_dir / "ephemeris")
        ft, ph, seg_note = _one_season(ft, ph)
        if seg_note:
            print(f"[{norad}] {ft.key}: {seg_note}")
        n_pass = int(ft.df["obs_id"].nunique())
        if n_pass < MIN_PASSES:
            # frames from one or two passes (often a stored-telemetry dump) cannot be tied to physics
            res = TypeResult(ft, ph, [])
            res.note = (f"Only {n_pass} pass(es) in the data: too few independent looks at the orbit "
                        f"to tell physics from chance (need {MIN_PASSES}). Skipped.")
            print(f"[{norad}] {ft.key}: {res.note}")
            results.append(res)
            continue
        found = score.analyze(ft.bytes_, ft.df, ph)
        layer = None
        if peel_layers:
            from . import peel
            got = peel.auto(ft, ph, score.analyze)
            if got and sum(f.score for f in got["fields"]) > sum(f.score for f in found):
                ft.bytes_, found, layer = got["bytes"], got["fields"], got["layer"]
                print(f"[{norad}] {ft.key}: peeled a '{layer}' whitening layer")
        score.describe(found, ft.df, ph)
        res = TypeResult(ft, ph, found)
        res.layer = layer
        res.segment_note = seg_note
        if not found:
            from .peel import looks_encrypted
            enc = looks_encrypted(ft.bytes_)
            if enc["encrypted_like"]:
                res.note = (f"Payload looks encrypted or compressed: {enc['uniform_byte_fraction']:.0%} of byte "
                            "positions are uniformly random across frames. No physics can be read from it.")
        if ph[f"ema_{physics.TAUS[-1]}"].min() > 0.999:
            # nothing to correlate against: no shadow in the hours before any frame
            res.note = (f"No eclipses in this data window (beta angle {ph['beta'].min():.0f} to "
                        f"{ph['beta'].max():.0f} deg): the spacecraft was in sunlight for every frame and the "
                        "hours before it, so sunlight-driven fields cannot be told apart. Fetch the satellite's "
                        "eclipse season (python -m shadowfield fetch NORAD --eclipse-season).")
            print(f"[{norad}] {ft.key}: {res.note}")
        elif ph["sunlit"].nunique() < 2:
            res.eclipse_note = ("Every frame was heard in sunlight, but many came minutes after an eclipse "
                                "(the shadow falls where there are no ground stations): sunlight history still "
                                "separates the fields.")
        print(f"[{norad}] {ft.key}: {len(ft.df)} frames, {ft.df['obs_id'].nunique()} passes, "
              f"{len(found)} fields found")
        if use_llm and found:
            from .llm import name_fields
            frame_info = {"key": ft.key, "payload_length_bytes": ft.length, "frames": len(ft.df),
                          "passes": int(ft.df["obs_id"].nunique())}
            if layer:
                frame_info["whitening_removed"] = layer
            guess = name_fields([f.as_dict() for f in found], orbit_summary(ph), frame_info,
                                sat_hint=None if blind else sat_name)
            res.names = guess.model_dump()
        if decoder:
            t = truth.locate_true_fields(ft, list(ft.df["raw"]), decoder)
            if t:
                res.evaluation = truth.evaluate(found, t, ft.bytes_)
                if res.names:
                    res.evaluation.update(_llm_accuracy(res.names, res.evaluation["table"]))
        results.append(res)
    return results


LLM_TO_TRUTH = {"navigation": "nav", "attitude": "nav", "thermal": "thermal", "power_solar": "power", "power_battery": "power", "power_bus": "power",
                "counter": "counter", "clock": "clock", "magnetometer": "magnet"}


def _llm_accuracy(names: dict, table: pd.DataFrame) -> dict:
    """Claude's categories against the decoder, for every physical field the engine found.

    category_accuracy   coarse (thermal / power / counter+clock / navigation / magnet); an
                        'unknown' answer counts as wrong
    answered            share of those fields Claude did not mark 'unknown'
    selective_accuracy  coarse accuracy on the answered fields only
    fine_accuracy       battery vs solar vs bus, on answered power fields whose decoder name says which
    """
    by_off = {f["offset"]: f for f in names["fields"]}
    hits = total = answered = sel_hits = fine_total = fine_hits = 0
    for _, t in table[(table["category"] != "other") & table["found"]].iterrows():
        g = by_off.get(t["found_offset"])
        if g is None:
            continue
        total += 1
        mapped = LLM_TO_TRUTH.get(g["category"])
        ok = mapped == t["category"] or {mapped, t["category"]} <= {"counter", "clock"}
        hits += ok
        if g["category"] != "unknown":
            answered += 1
            sel_hits += ok
            if t.get("fine") and g["category"].startswith("power"):
                fine_total += 1
                fine_hits += g["category"] == t["fine"]
    nan = float("nan")
    return {"llm_category_accuracy": hits / total if total else nan, "llm_judged": total,
            "llm_answered": answered / total if total else nan,
            "llm_selective_accuracy": sel_hits / answered if answered else nan,
            "llm_fine_accuracy": fine_hits / fine_total if fine_total else nan, "llm_fine_judged": fine_total}


def save(results: list[TypeResult], out_dir: Path, norad: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = []
    for r in results:
        item = {"frame_type": r.ft.key, "frames": len(r.ft.df), "passes": int(r.ft.df["obs_id"].nunique()),
                "orbit": orbit_summary(r.phys), "fields": [f.as_dict() for f in r.fields], "names": r.names}
        if r.evaluation:
            ev = {k: v for k, v in r.evaluation.items() if k != "table"}
            ev["truth"] = r.evaluation["table"].to_dict(orient="records")
            item["evaluation"] = ev
        dump.append(item)
    p = out_dir / f"{norad}.json"
    p.write_text(json.dumps(dump, indent=1, default=float), encoding="utf-8")
    return p
