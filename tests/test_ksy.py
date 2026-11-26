"""Every generated .ksy compiles with the official Kaitai Struct compiler and reads real frames
exactly as the engine does. Needs Node.js and `npm install` in tools/ksc; skipped otherwise.
"""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from shadowfield.score import _decode

ROOT = Path(__file__).resolve().parents[1]
KSC = ROOT / "tools" / "ksc"
KSY = sorted((ROOT / "results" / "ksy").glob("*.ksy"))
pytestmark = pytest.mark.skipif(not shutil.which("node") or not (KSC / "node_modules").exists() or not KSY,
                                reason="needs node + tools/ksc/node_modules + results/ksy")

TYPE = {"u1": (1, "be", False), "s1": (1, "be", True)}


def layout(spec):
    """(id, offset, width, endian, signed) of every typed entry, offsets in the raw frame."""
    pos, out = 0, []
    for e in spec["seq"]:
        if "size" in e:
            pos += e["size"]
            continue
        t = e["type"]
        if t in TYPE:
            w, en, sg = TYPE[t]
        elif t.startswith("b24"):
            w, en, sg = 3, t[3:], False
        elif t.startswith("f4"):
            w, en, sg = 4, t[2:], "f"
        else:
            w, en, sg = int(t[1]), t[2:], t[0] == "s"
        out.append((e["id"], pos, w, en, sg))
        pos += w
    return out, pos


def _compile(path, tmp_path):
    r = subprocess.run(["node", str(KSC / "compile.js"), str(path), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return next(tmp_path.glob("*.py"))


@pytest.mark.parametrize("path", KSY, ids=[p.stem for p in KSY])
def test_ksy_compiles_with_the_official_compiler(path, tmp_path):
    py = _compile(path, tmp_path)
    compile(py.read_text(encoding="utf-8"), str(py), "exec")  # valid Python parser out


@pytest.mark.parametrize("path", KSY, ids=[p.stem for p in KSY])
def test_ksy_reads_real_frames_like_the_engine(path, tmp_path):
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    py = _compile(path, tmp_path)
    mod_spec = importlib.util.spec_from_file_location(py.stem, py)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    cls = next(v for k, v in vars(mod).items() if isinstance(v, type) and k.lower().startswith("shadowfield"))

    norad = path.stem.split("_")[0]
    frames_file = ROOT / "data" / norad / "frames.jsonl"
    if not frames_file.exists():
        pytest.skip("no local frames to parse")
    fields, total = layout(spec)
    raws = [bytes.fromhex(json.loads(l)["hex"]) for l in frames_file.read_text(encoding="utf-8").splitlines()[:3000]]
    raws = [x for x in raws if len(x) == total][:300]
    assert raws, "no frames of this length"
    B = np.frombuffer(b"".join(raws), dtype=np.uint8).reshape(len(raws), total)
    from kaitaistruct import KaitaiStream, BytesIO
    parsed = [cls(KaitaiStream(BytesIO(x))) for x in raws]
    for ident, off, w, en, sg in fields:
        ours = _decode(B, off, w, en, sg)
        theirs = np.array([getattr(p, ident) for p in parsed], dtype=float)
        if sg == "f":  # the engine reads NaN/inf floats (no-fix frames) as 0
            theirs = np.where(np.isfinite(theirs), theirs, 0.0)
        assert np.allclose(ours, theirs, equal_nan=True), ident
