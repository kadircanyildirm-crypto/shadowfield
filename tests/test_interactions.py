"""How the rules work *together*: a random synthetic fleet, and a regression lock on real satellites.

The fleet changes byte order, field order, gaps and traps from satellite to satellite, so a rule
that fixes one layout but breaks another shows up here. The real-data lock pins every frame type's
recall and value-exact rate (it is skipped if the frames are not downloaded).
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "tests" / "regression_lock.json"


def test_synthetic_fleet_no_false_fields_and_exact_layouts():
    from tests.fleet import run
    res = run(range(200, 206))
    assert res["false"] == 0, [g["false_fields"] for _, g in res["rows"]]
    assert res["recall"] >= 0.95
    assert res["exact_rate"] >= 0.80, res["exact_rate"]


def _measure():
    from shadowfield import pipeline
    from shadowfield.bench import SATS
    out = {}
    for norad, dec, _ in SATS:
        if not (ROOT / "data" / str(norad) / "frames.jsonl").exists():
            continue
        for r in pipeline.run(norad, ROOT / "data", decoder=dec, use_llm=False, peel_layers=False):
            ev = r.evaluation or {}
            if ev.get("physical_true_fields"):
                out[f"{norad}:{r.ft.key}"] = {"recall": round(ev["recall_physical"], 3),
                                              "value_exact": round(ev["value_exact_physical"], 3),
                                              "precision": round(ev["precision"], 3)}
    return out


@pytest.mark.skipif(not LOCK.exists() or not (ROOT / "data" / "43880").exists(), reason="needs local frames")
def test_real_satellites_do_not_regress():
    lock = json.loads(LOCK.read_text())
    now = _measure()
    worse = {k: (lock[k], now.get(k)) for k in lock
             if k not in now or any(now[k][m] < lock[k][m] - 0.001 for m in ("recall", "value_exact", "precision"))}
    assert not worse, worse


if __name__ == "__main__":  # python -m tests.test_interactions  -> (re)write the lock from the current engine
    LOCK.write_text(json.dumps(_measure(), indent=1))
    print(LOCK.read_text())
