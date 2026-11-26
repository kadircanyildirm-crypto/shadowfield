"""Controlled experiment: whiten real UWE-4 frames three ways and let Shadowfield peel them.

Run: python -m tests.peel_experiment
"""
import copy
from pathlib import Path

import numpy as np

from shadowfield import frames, physics, score, truth
from shadowfield.peel import auto, ccsds_pn, g3ruh_scramble, looks_encrypted

ft = frames.load(Path("data/43880/frames.jsonl"), max_types=1)[0]
ph = physics.compute(ft.df)
raws = list(ft.df["raw"])
true = truth.locate_true_fields(ft, raws, "uwe4")


def grade(B, found):
    f2 = copy.copy(ft)
    f2.bytes_ = B
    ev = truth.evaluate(found, true)
    return f"recall {ev['recall_physical']:.0%}, exact {ev['exact_physical']:.0%}, precision {ev['precision']:.0%}"


base = score.analyze(ft.bytes_, ft.df, ph)
print(f"original      : {len(base):2d} fields (evidence {sum(f.score for f in base):5.1f})  {grade(ft.bytes_, base)}   {looks_encrypted(ft.bytes_)}")
rng = np.random.default_rng(7)
cases = {
    "CCSDS PN": ft.bytes_ ^ ccsds_pn(ft.length)[None, :],
    "G3RUH": g3ruh_scramble(ft.bytes_),
    "unknown key": ft.bytes_ ^ rng.integers(0, 256, ft.length, dtype=np.uint8)[None, :],
    "AES-like": rng.integers(0, 256, ft.bytes_.shape, dtype=np.uint8),  # fresh random body per frame
}
for name, B in cases.items():
    ft_s = copy.copy(ft)
    ft_s.bytes_ = B
    scr = score.analyze(B, ft.df, ph)
    res = auto(ft_s, ph, score.analyze)
    enc = looks_encrypted(B)
    ssum = sum(f.score for f in scr)
    if res:
        extra = f", key bytes recovered {res['key_bytes']}/{ft.length}" if res["key_bytes"] is not None else ""
        print(f"{name:<13} : scrambled {len(scr):2d} fields (evidence {ssum:5.1f}) -> peeled '{res['layer']}' "
              f"{len(res['fields']):2d} fields (evidence {res['score']:5.1f})  {grade(res['bytes'], res['fields'])}{extra}")
    else:
        print(f"{name:<13} : scrambled {len(scr):2d} fields -> nothing to peel; encrypted-like={enc['encrypted_like']} "
              f"(uniform bytes {enc['uniform_byte_fraction']:.0%})")
