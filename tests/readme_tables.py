"""Fill the README's DEV_TABLE / HELDOUT_TABLE from results/scoreboard.json. Run: python -m tests.readme_tables"""
import json
import re
from pathlib import Path

from shadowfield.bench import COLS, totals

ROOT = Path(__file__).resolve().parents[1]


def pct(v):
    return "–" if v is None or v != v else f"{v:.0%}"


def table(rows):
    head = "| satellite | frame type | frames / passes | decoder fields | " + " | ".join(h for _, h in COLS) + " |"
    sep = "|" + "---|" * (4 + len(COLS))
    lines = [head, sep]
    for r in rows:
        if not r["physical_truth"]:
            continue
        name = r["name"] + (" ¹" if r.get("note") else "")
        lines.append(f"| {name} | `{r['frame_type'].replace('raw:', '')}` | {r['frames']} / {r['passes']} | "
                     f"{r['physical_truth']} | " + " | ".join(pct(r.get(k)) for k, _ in COLS) + " |")
    t = totals(rows)
    lines.append(f"| **total** | | | **{t['fields']}** | " + " | ".join(f"**{pct(t[k])}**" for k, _ in COLS) + " |")
    return "\n".join(lines)


rows = json.loads((ROOT / "results" / "scoreboard.json").read_text(encoding="utf-8"))
dev = [r for r in rows if r.get("split") == "development"]
held = [r for r in rows if r.get("split") == "held-out"]
readme = ROOT / "README.md"
s = readme.read_text(encoding="utf-8")
for marker, rs in (("DEV_TABLE", dev), ("HELDOUT_TABLE", held)):
    block = f"<!-- {marker} -->\n{table(rs)}\n<!-- /{marker} -->"
    if f"<!-- {marker} -->" in s:
        s = re.sub(rf"<!-- {marker} -->.*?<!-- /{marker} -->", block, s, flags=re.S)
    else:
        s = s.replace(marker, block)
readme.write_text(s, encoding="utf-8")
print(table(dev)); print(); print(table(held))
