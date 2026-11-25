"""Self-contained HTML report: one card per recovered field, plotted against the physics that found it.

Plots are inline SVG (no JS, no CDN), so the file opens anywhere and can be attached as-is.
  sun / lagged / orbit fields  value vs minutes since the last sunrise (+) or sunset (-), shadow shaded
  counter / clock              value vs time
  magnetometer                 measured |B| (raw units) vs IGRF model |B| (nT)
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .score import _decode

W, H, PAD = 360, 170, 30
CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#d9d9de;--dot:#2563eb;--shade:#e7e7ec;--ok:#15803d;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1d1d20;--ink:#ececf0;--mute:#9a9aa3;--line:#38383e;--dot:#60a5fa;--shade:#2a2a30;--ok:#4ade80;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px}
.sub{color:var(--mute);margin:0 0 16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h3{margin:0;font-size:15px}.meta{color:var(--mute);font-size:12.5px;margin:2px 0 6px}.why{font-size:13px;margin:6px 0 0}
svg{width:100%;height:auto;display:block}svg text{fill:var(--mute);font-size:10px}
.dot{fill:var(--dot);opacity:.45}.shade{fill:var(--shade)}.axis{stroke:var(--line)}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 4px}.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.stat b{display:block;font-size:20px}.ok{color:var(--ok)}.bad{color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}td,th{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}
"""


def _values(ft, f) -> np.ndarray:
    if f.kind == "magnet":
        ax = [_decode(ft.bytes_, f.offset + i * f.width, f.width, f.endian, f.signed).astype(float) for i in range(3)]
        return np.sqrt(sum(a ** 2 for a in ax))
    return _decode(ft.bytes_, f.offset, f.width, f.endian, f.signed).astype(float)


def _scatter(x, y, shade_neg=False, xlabel="", max_pts=2500) -> str:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return ""
    if len(x) > max_pts:
        idx = np.random.default_rng(0).choice(len(x), max_pts, replace=False)
        x, y = x[idx], y[idx]
    lo, hi = np.percentile(y, [0.5, 99.5]) if len(y) > 50 else (y.min(), y.max())
    hi = hi if hi > lo else lo + 1
    x0, x1 = float(x.min()), float(x.max())
    x1 = x1 if x1 > x0 else x0 + 1
    sx = lambda v: PAD + (v - x0) / (x1 - x0) * (W - PAD - 6)
    sy = lambda v: H - 18 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (H - 28)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    if shade_neg and x0 < 0:
        parts.append(f'<rect class="shade" x="{sx(x0):.1f}" y="4" width="{sx(min(0, x1)) - sx(x0):.1f}" height="{H - 22}"/>')
        parts.append(f'<text x="{sx(x0) + 3:.1f}" y="14">shadow</text>')
    parts.append(f'<line class="axis" x1="{PAD}" y1="{H - 18}" x2="{W - 6}" y2="{H - 18}"/>')
    parts += [f'<circle class="dot" cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="1.6"/>' for a, b in zip(x, y)]
    parts.append(f'<text x="{PAD}" y="{H - 5}">{x0:.4g}</text><text x="{W - 6}" y="{H - 5}" text-anchor="end">{x1:.4g}</text>')
    parts.append(f'<text x="{(W + PAD) / 2:.0f}" y="{H - 5}" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(f'<text x="2" y="12">{hi:.4g}</text><text x="2" y="{H - 20}">{lo:.4g}</text></svg>')
    return "".join(parts)


def _plot(ft, phys, f) -> str:
    y = _values(ft, f)
    if f.kind in ("counter", "clock"):
        return _scatter((phys["unix"].to_numpy() - phys["unix"].min()) / 86400, y, xlabel="days")
    if f.kind == "magnet":
        return _scatter(phys["b_igrf"].to_numpy(), y, xlabel="IGRF |B| (nT)")
    return _scatter(phys["since_change"].to_numpy(), y, shade_neg=True, xlabel="min since sunrise (+) / sunset (−)")


def _evidence(f) -> str:
    d = f.detail
    bits = [f"score {f.score:.2f}"]
    for k, lab in (("rho_sun", "ρ sun"), ("tau_min", "lag τ min"), ("r2_orbit", "R² orbit"), ("per_second", "per s"),
                   ("epoch_guess", "epoch"), ("rho_igrf", "ρ IGRF"), ("rough", "rough")):
        if k in d and d[k] is not None:
            v = d[k]
            bits.append(f"{lab} {v:.3g}" if isinstance(v, (int, float)) else f"{lab} {v}")
    return " · ".join(bits)


def render(results, norad: int, out: Path, title: str | None = None) -> Path:
    body = [f"<h1>{html.escape(title or f'NORAD {norad}')}: telemetry recovered from orbital physics</h1>",
            '<p class="sub">Every field below was found without a decoder: candidate byte readings were tested '
            'against sunlight, thermal lag, orbital phase, wall-clock time and the IGRF magnetic field computed from '
            'the TLE. Names, units and scales are Claude\'s reading of that evidence, with the satellite name withheld.</p>']
    for res in results:
        ev = res.evaluation or {}
        body.append(f"<h2>Frame type {html.escape(res.ft.key)}</h2>")
        body.append(f'<p class="sub">{len(res.ft.df)} frames, {res.ft.df["obs_id"].nunique()} passes, '
                    f'{res.ft.length}-byte payload</p>')
        if getattr(res, "segment_note", None):
            body.append(f'<p class="sub">{html.escape(res.segment_note)}</p>')
        if getattr(res, "eclipse_note", None):
            body.append(f'<p class="sub">{html.escape(res.eclipse_note)}</p>')
        if getattr(res, "layer", None):
            body.append(f'<p class="ok">Whitening layer detected and removed before analysis: {html.escape(res.layer)}.</p>')
        if getattr(res, "note", None):
            body.append(f'<p class="bad">{html.escape(res.note)}</p>')
        if res.names and res.names.get("summary"):
            body.append(f'<p>{html.escape(res.names["summary"])}</p>')
        if res.names and res.names.get("health"):
            body.append(f'<p><b>Health.</b> {html.escape(res.names["health"])}</p>')
        if ev:
            stats = [("fields found", len(res.fields), None),
                     ("physical-field recall", f"{ev['recall_physical']:.0%}", ev["recall_physical"] >= 0.5),
                     ("exact byte layout", f"{ev['exact_physical']:.0%}", ev["exact_physical"] >= 0.5),
                     ("precision", f"{ev['precision']:.0%}", ev["precision"] >= 0.7)]
            if "llm_category_accuracy" in ev:
                a = ev["llm_category_accuracy"]
                stats.append(("Claude category accuracy", f"{a:.0%}" if a == a else "n/a", a == a and a >= 0.6))
            body.append('<div class="stats">' + "".join(
                f'<div class="stat"><b class="{"" if good is None else ("ok" if good else "bad")}">{v}</b>{k}</div>'
                for k, v, good in stats) + "</div>")
        names = {g["offset"]: g for g in (res.names or {}).get("fields", [])}
        cards = []
        for f in res.fields:
            g = names.get(f.offset, {})
            head = html.escape(g.get("name") or f"field_{f.offset}")
            unit = f' [{html.escape(g["unit"])}]' if g.get("unit") else ""
            meta = f"byte {f.offset} · {f.ctype} · {f.kind}" + (f" · {html.escape(g['category'])}" if g else "")
            scale = f'<div class="meta">{html.escape(g["scale"])} · confidence {g["confidence"]:.2f}</div>' if g else ""
            why = f'<p class="why">{html.escape(g["rationale"])}</p>' if g else ""
            cards.append(f'<div class="card"><h3>{head}{unit}</h3><div class="meta">{meta}</div>'
                         f'{_plot(res.ft, res.phys, f)}<div class="meta">{_evidence(f)}</div>{scale}{why}</div>')
        body.append('<div class="grid">' + "".join(cards) + "</div>")
        if ev and "table" in ev:
            t = ev["table"]
            rows = "".join(
                f"<tr><td>{r['offset']}</td><td>{html.escape(str(r['name']))}</td><td>{html.escape(str(r['category']))}</td>"
                f"<td class=\"{'ok' if r['found'] else 'bad'}\">{'found' if r['found'] else 'missed'}</td>"
                f"<td>{html.escape(str(r.get('how', '')))}</td></tr>" for _, r in t.iterrows())
            body.append("<h2>Check against the published decoder</h2><table><tr><th>byte</th><th>decoder field</th>"
                        f"<th>category</th><th>result</th><th>how</th></tr>{rows}</table>")
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
           f'content="width=device-width,initial-scale=1"><title>Shadowfield {norad}</title><style>{CSS}</style>'
           f'</head><body><main>{"".join(body)}</main></body></html>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
