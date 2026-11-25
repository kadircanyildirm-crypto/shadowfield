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
    (62394, "crocube", "CroCube"),
    (57172, "sputnixusp", "Sputnix USP"),
    (43784, "snuglite", "SNUGLITE"),
    (46506, "netsat", "NetSat"),
    (41789, "alsat1n", "AlSat-1N"),
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
                         "claude_acc": ev.get("llm_category_accuracy"), "note": r.note,
                         "layer": r.layer, "health": (r.names or {}).get("health")})
    sb = out_dir / "scoreboard.json"
    if merge and sb.exists():  # keep satellites not re-run this time
        done = {r["norad"] for r in rows}
        keep = [r for r in json.loads(sb.read_text(encoding="utf-8")) if r["norad"] not in done]
        order = {n: i for i, (n, _, _) in enumerate(SATS)}
        rows = sorted(keep + rows, key=lambda r: order.get(r["norad"], 99))
    sb.write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    _index(rows, out_dir)
    return rows


def _pct(v):
    return "–" if v is None or v != v else f"{v:.0%}"


def _index(rows: list[dict], out_dir: Path) -> None:
    graded = [r for r in rows if r["physical_truth"] and not r.get("note")]
    tot = sum(r["physical_truth"] for r in graded) or 1

    def wavg(k):
        vals = [(r[k], r["physical_truth"]) for r in graded if r[k] is not None and r[k] == r[k]]
        return sum(v * w for v, w in vals) / sum(w for _, w in vals) if vals else None

    body = "".join(
        f"<tr><td><a href='{r['norad']}.html'>{html.escape(r['name'])}</a></td><td>{html.escape(r['frame_type'])}</td>"
        f"<td>{r['frames']}</td><td>{r['passes']}</td><td>{r['found']}</td><td>{r['physical_truth'] or '–'}</td>"
        f"<td>{_pct(r['recall'])}</td><td>{_pct(r['exact'])}</td><td>{_pct(r['precision'])}</td>"
        f"<td>{_pct(r['kind_acc'])}</td><td>{_pct(r['claude_acc'])}</td></tr>"
        + (f"<tr><td></td><td colspan='10' class='meta'>{html.escape(r['note'])}</td></tr>" if r.get("note") else "")
        for r in rows)
    foot = (f"<tr class='tot'><td colspan='5'>weighted by decoder fields ({tot})</td><td></td>"
            + "".join(f"<td>{_pct(wavg(k))}</td>" for k in ("recall", "exact", "precision", "kind_acc", "claude_acc"))
            + "</tr>")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadowfield scoreboard</title><style>{report.CSS}
td:nth-child(n+3),th:nth-child(n+3){{text-align:right}} tr.tot td{{font-weight:600}} .wrap{{overflow-x:auto}}</style></head>
<body><main><h1>Shadowfield: blind decoding scoreboard</h1>
<p class="sub">Telemetry fields recovered from raw SatNOGS frames using only orbital physics (from the TLE) and Claude,
without the satellite's decoder or name. The published satnogs-decoders definitions are used afterwards, only to grade.
<b>Recall</b>: share of the decoder's physical fields (temperatures, power, clocks, counters, magnetometers) that were found.
<b>Exact</b>: found at exactly the decoder's byte offset and width.
<b>Precision</b>: share of reported fields that are real decoder fields. <b>Kind</b>: engine's physics class matches the
decoder's category. <b>Claude</b>: Claude's blind category for the field matches the decoder's.</p>
<div class="wrap"><table><tr><th>satellite</th><th>frame type</th><th>frames</th><th>passes</th><th>fields</th><th>decoder physical</th>
<th>recall</th><th>exact</th><th>precision</th><th>kind</th><th>Claude</th></tr>{body}{foot}</table></div></main></body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
