"""Unit tests for the parts that decide byte layout. Run: python -m pytest -q"""
import struct

import numpy as np
import pandas as pd
import pytest

from shadowfield import peel
from shadowfield.llm import FieldGuess, FrameGuess, _vote
from shadowfield.score import Field, _boundary, _carries_into, _decode
from shadowfield.truth import category, fine_category


def frames_of(values, fmt):
    return np.frombuffer(b"".join(struct.pack(fmt, v) for v in values), dtype=np.uint8).reshape(len(values), -1)


# ---- decoding -------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,w,endian,signed", [("<H", 2, "le", False), (">h", 2, "be", True),
                                                  ("<I", 4, "le", False), (">i", 4, "be", True)])
def test_decode_standard_widths(fmt, w, endian, signed):
    vals = [0, 1, 255, 256, 1000, 12345] + ([-1, -300] if signed else [65535 if w == 2 else 4_000_000_000])
    B = frames_of(vals, fmt)
    assert np.array_equal(_decode(B, 0, w, endian, signed), np.array(vals, dtype=float))


def test_decode_three_bytes_both_orders_and_sign():
    vals = [0, 1, 70000, 8_000_000, -5, -4_000_000]
    le = np.array([list((v & 0xFFFFFF).to_bytes(3, "little")) for v in vals], dtype=np.uint8)
    be = np.array([list((v & 0xFFFFFF).to_bytes(3, "big")) for v in vals], dtype=np.uint8)
    assert np.array_equal(_decode(le, 0, 3, "le", True), np.array(vals, float))
    assert np.array_equal(_decode(be, 0, 3, "be", True), np.array(vals, float))
    assert _decode(le, 0, 3, "le", False)[4] == 2 ** 24 - 5


def test_decode_float32():
    vals = [0.0, 1.5, -2866739.75, 6707.125]
    B = frames_of(vals, "<f")
    assert np.allclose(_decode(B, 0, 4, "le", "f"), vals)


# ---- structure tests --------------------------------------------------------------------

def test_sign_extension_joins_signed_current():
    rng = np.random.default_rng(0)
    cur = rng.integers(-200, 300, 500)  # reverses sign: high byte is 0xFF or 0x00/0x01
    B = frames_of(cur.tolist(), "<h")
    assert _boundary(B, Field(0, 2, "le", True, "sun", 0.9))


def test_sign_test_rejects_flag_next_to_sensor():
    rng = np.random.default_rng(1)
    flag = rng.choice([0, 1], 500)
    sensor = rng.integers(20, 40, 500)  # a smooth reading that never nears the wrap point
    B = np.stack([sensor, flag], 1).astype(np.uint8)  # little-endian view: hi = flag
    assert not _boundary(B, Field(0, 2, "le", False, "sun", 0.9))


def test_counter_top_byte_carries_but_neighbour_does_not():
    t = np.arange(0, 300_000, 997)  # an uptime counter crossing 65536 several times
    rng = np.random.default_rng(2)
    other = rng.integers(0, 256, len(t))
    B = np.stack([t & 0xFF, (t >> 8) & 0xFF, (t >> 16) & 0xFF, other], 1).astype(np.uint8)
    narrow = Field(0, 2, "le", False, "counter", 1.0)
    assert _carries_into(B, narrow, Field(0, 3, "le", False, "counter", 1.0))
    assert not _carries_into(B, Field(0, 3, "le", False, "counter", 1.0), Field(0, 4, "le", False, "counter", 1.0))


# ---- peeling ----------------------------------------------------------------------------

def test_ccsds_sequence_matches_standard():
    assert peel.ccsds_pn(8).tobytes().hex() == "ff480ec09a0d70bc"


def test_g3ruh_roundtrip():
    B = np.random.default_rng(3).integers(0, 256, (20, 40), dtype=np.uint8)
    assert np.array_equal(peel.g3ruh_descramble(peel.g3ruh_scramble(B)), B)


def test_encryption_flag():
    rng = np.random.default_rng(4)
    assert peel.looks_encrypted(rng.integers(0, 256, (800, 40), dtype=np.uint8))["encrypted_like"]
    structured = np.zeros((800, 40), dtype=np.uint8)
    structured[:, 5] = np.arange(800) % 256
    assert not peel.looks_encrypted(structured)["encrypted_like"]


# ---- grading helpers ---------------------------------------------------------------------

@pytest.mark.parametrize("name,cat,fine", [
    ("beacon_payload_batt_a_voltage", "power", "power_battery"),
    ("eps_solar_cells_current", "power", "power_solar"),
    ("eps_consumption_current", "power", "power_bus"),
    ("obc_temperature_pos_x", "thermal", None),
    ("gnss_timestamp", "clock", None),
    ("position_x", "nav", None),
])
def test_categories(name, cat, fine):
    assert category(name) == cat
    assert fine_category(name) == fine


def test_vote_majority_and_unknown_never_wins_ties():
    def g(o, c, conf):
        return FieldGuess(offset=o, name=f"f{o}", category=c, unit="", scale="raw", confidence=conf, rationale="r")
    s = [FrameGuess(summary="a", health="h", fields=[g(1, "power_solar", .6), g(2, "unknown", .9)]),
         FrameGuess(summary="b", health="h", fields=[g(1, "power_battery", .5), g(2, "thermal", .4)]),
         FrameGuess(summary="c", health="h", fields=[g(1, "power_solar", .4)])]
    out = {f.offset: f for f in _vote(s).fields}
    assert out[1].category == "power_solar" and "2/3" in out[1].rationale
    assert out[2].category == "thermal"  # 1-1 tie: a real answer beats "unknown"


# ---- end to end on a synthetic satellite ----------------------------------------------------

def test_synthetic_satellite_fully_recovered():
    from tests import synth
    from shadowfield import score
    ft, ph = synth.make(n_pass=200)
    found = {f.offset: f for f in score.analyze(ft.bytes_, ft.df, ph)}
    for off, (name, ctype, kind) in synth.PLANTED.items():
        if kind is None:
            assert not any(set(f.span) & {off} for f in found.values() if f.kind != "magnet"), name
            continue
        assert off in found, name
        f = found[off]
        if kind == "magnet":
            continue
        # same values bit for bit (a signed field that never goes negative reads the same unsigned)
        w, en, sg = int(ctype[1]), ctype[2:] or "be", ctype[0] == "s"
        assert np.array_equal(_decode(ft.bytes_, off, w, en, sg),
                              _decode(ft.bytes_, f.offset, f.width, f.endian, f.signed)), (name, f.ctype)
