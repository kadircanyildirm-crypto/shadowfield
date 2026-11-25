"""Blind benchmark across satellites: run the pipeline, score against satnogs-decoders, write a scoreboard.

Each satellite's frames are decoded by its published decoder only to *grade* the result;
the engine and Claude never see the decoder or the satellite's name.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from . import pipeline, report

ROOT = Path(__file__).resolve().parents[1]
SATS = [  # norad, decoder, display name
    (43880, "uwe4", "UWE-4"),
    (61785, "geoscan", "Geoscan"),
    (60237, "grbbeta", "GRBBeta"),
    (43784, "snuglite", "SNUGLITE"),
    (46506, "netsat", "NetSat"),
    (41789, "alsat1n", "AlSat-1N"),
]
# The first held-out set (Sputnix, CubeBel-2, Geoscan #2) was inspected after its blind run and
# has joined development. The current held-out set has never been looked at while tuning.
SATS += [
    (57172, "sputnixusp", "Sputnix USP"),
    (57175, "cubebel2", "CubeBel-2"),
    (64881, "geoscan", "Geoscan (2nd unit)"),
]
HELDOUT = [
    (68458, "cp16", "CP16"),
    (67683, "knacksat2", "KNACKSAT-2"),
    (69935, "kostka", "Kostka"),
    (98243, "gbsat", "GBSAT"),
    (68635, "canvas", "CANVAS"),
    (64878, "innosat16", "InnoSat-16"),
]


def run(sats=SATS, data_dir: Path = ROOT / "data", out_dir: Path = ROOT / "results", use_llm: bool = True,
        merge: bool = False) -> list[dict]:
    rows = []
    for norad, dec, name in sats:
        if not (data_dir / str(norad) / "frames.jsonl").exists():
            print(f"[bench] {name}: no data, skipped")
            continue
        try:
            res = pipeline.run(norad, data_dir, decoder=dec, use_llm=use_llm)
        except Exception as e:  # one bad satellite should not sink the scoreboard
            print(f"[bench] {name}: failed: {type(e).__name__}: {e}")
            continue
        pipeline.save(res, out_dir, norad)
        report.render(res, norad, out_dir / f"{norad}.html", title=f"{name} (blind test)")
        from . import ksy
        ksy.write(res, norad, out_dir / "ksy")
        for r in res:
            ev = r.evaluation or {}
            rows.append({"norad": norad, "name": name, "frame_type": r.ft.key, "frames": len(r.ft.df),
                         "passes": int(r.ft.df["obs_id"].nunique()), "found": len(r.fields),
                         "physical_truth": ev.get("physical_true_fields"),
                         "recall": ev.get("recall_physical"), "exact": ev.get("exact_physical"),
                         "precision": ev.get("precision"), "kind_acc": ev.get("kind_accuracy"),
                         "value_exact": ev.get("value_exact_physical"),
                         "claude_answered": ev.get("llm_answered"), "claude_selective": ev.get("llm_selective_accuracy"),
                         "claude_fine": ev.get("llm_fine_accuracy"), "claude_fine_n": ev.get("llm_fine_judged"),
                         "split": "held-out" if norad in {h[0] for h in HELDOUT} else "development",
                         "claude_acc": ev.get("llm_category_accuracy"), "note": r.note,
                         "layer": r.layer, "health": (r.names or {}).get("health")})
    sb = out_dir / "scoreboard.json"
    if merge and sb.exists():  # keep satellites not re-run this time
        done = {r["norad"] for r in rows}
        keep = [r for r in json.loads(sb.read_text(encoding="utf-8")) if r["norad"] not in done]
        order = {n: i for i, (n, _, _) in enumerate(SATS + HELDOUT)}
        rows = sorted(keep + rows, key=lambda r: order.get(r["norad"], 99))
    sb.write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    _index(rows, out_dir)
    return rows


def _pct(v):
    return "–" if v is None or v != v else f"{v:.0%}"


COLS = [("recall", "recall"), ("value_exact", "exact"), ("precision", "precision"), ("kind_acc", "kind"),
        ("claude_acc", "Claude"), ("claude_answered", "answered"), ("claude_selective", "when answered"),
        ("claude_fine", "battery / solar / bus")]


def totals(rows: list[dict]) -> dict:
    """Weighted by the number of decoder fields; rows with a note (no eclipses, too few passes) excluded."""
    g = [r for r in rows if r["physical_truth"] and not r.get("note")]
    out = {"fields": sum(r["physical_truth"] for r in g)}
    for k, _ in COLS:
        w = "claude_fine_n" if k == "claude_fine" else "physical_truth"
        vals = [(r[k], r[w] or 0) for r in g if r.get(k) is not None and r[k] == r[k]]
        out[k] = sum(v * n for v, n in vals) / sum(n for _, n in vals) if vals and sum(n for _, n in vals) else None
    return out


def _table(rows: list[dict]) -> str:
    head = "".join(f"<th>{h}</th>" for _, h in COLS)
    body = "".join(
        f"<tr><td><a href='{r['norad']}.html'>{html.escape(r['name'])}</a></td><td>{html.escape(r['frame_type'])}</td>"
        f"<td>{r['frames']}</td><td>{r['passes']}</td><td>{r['physical_truth'] or '–'}</td>"
        + "".join(f"<td>{_pct(r.get(k))}</td>" for k, _ in COLS) + "</tr>"
        + (f"<tr><td></td><td colspan='{4 + len(COLS)}' class='meta'>{html.escape(r['note'])}</td></tr>" if r.get("note") else "")
        for r in rows)
    t = totals(rows)
    foot = (f"<tr class='tot'><td colspan='4'>total, weighted by decoder fields</td><td>{t['fields']}</td>"
            + "".join(f"<td>{_pct(t[k])}</td>" for k, _ in COLS) + "</tr>")
    return (f"<div class='wrap'><table><tr><th>satellite</th><th>frame type</th><th>frames</th><th>passes</th>"
            f"<th>decoder fields</th>{head}</tr>{body}{foot}</table></div>")


def _index(rows: list[dict], out_dir: Path) -> None:
    dev = [r for r in rows if r.get("split", "development") == "development"]
    held = [r for r in rows if r.get("split") == "held-out"]
    sections = f"<h2>Development satellites</h2><p class='sub'>The engine's rules were tuned while looking at these.</p>{_table(dev)}"
    if held:
        sections += ("<h2>Held-out satellites</h2><p class='sub'>Run once with the engine frozen; never inspected while "
                     f"tuning. This is the honest estimate for a satellite nobody has looked at.</p>{_table(held)}")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadowfield scoreboard</title><style>{report.CSS}
td:nth-child(n+3),th:nth-child(n+3){{text-align:right}} tr.tot td{{font-weight:600}} .wrap{{overflow-x:auto}}</style></head>
<body><main><h1>Shadowfield: blind decoding scoreboard</h1>
<p class="sub">Telemetry fields recovered from raw SatNOGS frames using only orbital physics (from the TLE) and Claude,
without the satellite's decoder or name. The published satnogs-decoders definitions are used afterwards, only to grade.
<b>Recall</b>: share of the decoder's physical fields (temperature, power, clock, counter, navigation, magnetometer) that were
found. <b>Exact</b>: the found reading reproduces the decoder's raw values bit for bit (offset, width, byte order, sign).
<b>Precision</b>: share of reported fields that are real decoder fields. <b>Kind</b>: the engine's physics class matches the
decoder's category. <b>Claude</b>: Claude's blind category matches (an "unknown" answer counts as wrong); <b>answered</b>:
share not marked unknown; <b>when answered</b>: accuracy on those; <b>battery / solar / bus</b>: the finer power class,
where the decoder's field name says which. Each Claude answer is the majority of three independent samples.</p>
{sections}</main></body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
