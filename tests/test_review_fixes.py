"""Regression tests for the bugs found in the code review: each one fails if its bug comes back."""
import json

import numpy as np
import pandas as pd

from shadowfield import fetch
from shadowfield.engine.detectors import _magnetometers
from shadowfield.engine.field import Field
from shadowfield.engine.select import _canonical_widths
from shadowfield.pipeline import _llm_accuracy


def test_fine_accuracy_ignores_power_fields_whose_name_says_nothing():
    # pandas 3 turns None in a str column into NaN, which is truthy; it must not count as a wrong answer
    table = pd.DataFrame([
        {"category": "power", "found": True, "found_offset": 0, "fine": "power_battery"},
        {"category": "power", "found": True, "found_offset": 2, "fine": None},
    ])
    names = {"fields": [{"offset": 0, "category": "power_battery"}, {"offset": 2, "category": "power_battery"}]}
    acc = _llm_accuracy(names, table)
    assert acc["llm_fine_judged"] == 1 and acc["llm_fine_accuracy"] == 1.0


def test_fetch_resumes_after_a_half_written_line(tmp_path, monkeypatch):
    d = tmp_path / "123"
    d.mkdir()
    good = {"obs_id": 1, "ts": "2026-11-01T00:00:00+00:00", "hex": "00", "tle1": "", "tle2": "",
            "station_lat": 0, "station_lng": 0, "station_alt": 0}
    (d / "frames.jsonl").write_text(json.dumps(good) + "\n" + '{"obs_id": 2, "ts": "2026-', encoding="utf-8")
    monkeypatch.setattr(fetch, "list_observations", lambda *a, **k: iter(()))
    fetch.fetch(123, tmp_path)  # must not raise
    lines = (d / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["obs_id"] == 1


def test_widened_counters_never_share_a_byte():
    # Two slow counters that never reach their own maximum, with one constant non-zero byte
    # between them: each, on its own, would widen over that byte (little-endian left, big-endian right).
    n = 400
    B = np.zeros((n, 5), np.uint8)
    B[:, 2] = np.linspace(0, 200, n).astype(np.uint8)
    B[:, 3] = 7
    B[:, 4] = np.linspace(10, 210, n).astype(np.uint8)
    a = Field(2, 1, "be", False, "counter", 1.0, {}, B[:, 2].astype(float))
    b = Field(4, 1, "be", False, "counter", 1.0, {}, B[:, 4].astype(float))
    pool = [a, b, Field(2, 2, "le", False, "counter", 1.0), Field(3, 2, "be", False, "counter", 1.0)]
    out = _canonical_widths([a, b], pool, B)
    spans = [set(f.span) for f in out]
    assert not (spans[0] & spans[1]), [(f.offset, f.ctype) for f in out]


def test_one_missing_igrf_value_does_not_switch_magnetometers_off():
    rng = np.random.default_rng(0)
    n = 600
    b = 30000 + 15000 * np.sin(np.linspace(0, 40, n))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    xyz = (d * b[:, None] / 10).round()
    X = np.column_stack([xyz, rng.integers(0, 100, n)]).astype(float)
    cand = [(0, 2, "be", True), (2, 2, "be", True), (4, 2, "be", True), (6, 2, "be", True)]
    b_nan = b.copy()
    b_nan[17] = np.nan
    assert _magnetometers(cand, X, b_nan, n_pass=60, alpha=1e-4)
