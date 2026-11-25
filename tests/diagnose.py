"""List every physical decoder field the engine did not pin down exactly (development satellites only).

Run: python -m tests.diagnose [norad ...]
"""
import json
import sys
from pathlib import Path

from shadowfield import pipeline
from shadowfield.bench import SATS

DEV = {43880, 61785, 60237, 43784, 46506, 41789, 57172, 57175, 64881}  # incl. the first held-out set (now development)
want = {int(a) for a in sys.argv[1:]} or DEV
out = []
for norad, dec, name in SATS:
    if norad not in want or norad not in DEV or not (Path("data") / str(norad) / "frames.jsonl").exists():
        continue
    for r in pipeline.run(norad, Path("data"), decoder=dec, use_llm=False):
        ev = r.evaluation
        if not ev:
            continue
        found = {f.offset: f for f in r.fields}
        print(f"\n## {name} {r.ft.key}  recall {ev['recall_physical']:.2f} exact {ev['exact_physical']:.2f} "
              f"value-exact {ev['value_exact_physical']:.2f} precision {ev['precision']:.2f}")
        for _, t in ev["table"].iterrows():
            if t["category"] == "other" or t["value_exact"]:
                continue
            near = [f"{o}:{f.ctype}:{f.kind}" for o, f in sorted(found.items())
                    if set(f.span) & set(range(t["offset"], t["offset"] + t["width"]))]
            sig = "f4" if t["signed"] == "f" else ("s" if t["signed"] else "u") + str(t["width"])
            print(f"  {t['offset']:<3} {sig}{t['endian'] if t['width'] > 1 else '':<2} {t['name'][:34]:<34} "
                  f"{t['category']:<8} {t['how'] or 'missed':<8} {near}")
            out.append({"sat": name, "type": r.ft.key, "offset": int(t["offset"]), "truth": sig + t["endian"],
                        "name": t["name"], "how": t["how"], "near": near})
Path("data/diagnose.json").write_text(json.dumps(out, indent=1))
