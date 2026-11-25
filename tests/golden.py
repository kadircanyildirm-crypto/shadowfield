"""Snapshot every field the engine finds on the local satellites, to prove a refactor changes nothing.

python -m tests.golden save data/golden_before.json
python -m tests.golden check data/golden_before.json
"""
import json
import sys
from pathlib import Path

from shadowfield import frames, physics, score

NORADS = [43880, 61785, 60237, 43784, 46506, 41789, 57172, 57175, 64881]


def snapshot() -> dict:
    out = {}
    for n in NORADS:
        p = Path("data") / str(n) / "frames.jsonl"
        if not p.exists():
            continue
        for ft in frames.load(p, max_types=3):
            ph = physics.compute(ft.df)
            if ft.df["obs_id"].nunique() < 20:
                continue
            found = score.analyze(ft.bytes_, ft.df, ph)
            out[f"{n}:{ft.key}"] = [[f.offset, f.width, f.endian, str(f.signed), f.kind, round(f.score, 6)]
                                   for f in found]
            print(n, ft.key, len(found), flush=True)
    return out


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    snap = snapshot()
    if mode == "save":
        path.write_text(json.dumps(snap, indent=0))
    else:
        ref = json.loads(path.read_text())
        diff = [k for k in set(ref) | set(snap) if ref.get(k) != snap.get(k)]
        print("IDENTICAL" if not diff else f"DIFFERENT: {diff}")
