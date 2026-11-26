"""Command line: python -m shadowfield <command> ...

  fetch   NORAD [--db] [--days N]      download frames (SatNOGS Network, or DB with a token)
          [--eclipse-season]           ...plus the weeks when the orbit actually saw eclipses
  run     NORAD [--decoder NAME]       find fields, name them with Claude, score against a decoder
          [--no-llm] [--not-blind]
  bench   [--no-llm]                   blind scoreboard over the benchmark satellites -> results/index.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .fetch import load_env

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_env(ROOT / ".env")
    ap = argparse.ArgumentParser(prog="shadowfield")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("norad", type=int)
    f.add_argument("--db", action="store_true", help="use SatNOGS DB (needs SATNOGS_DB_TOKEN)")
    f.add_argument("--days", type=int, default=45)
    f.add_argument("--max-obs", type=int, default=400)
    f.add_argument("--eclipse-season", action="store_true",
                   help="also fetch the most recent weeks in which the satellite saw eclipses")
    r = sub.add_parser("run")
    r.add_argument("norad", type=int)
    r.add_argument("--decoder", help="satnogs-decoders name, for blind scoring (e.g. uwe4)")
    r.add_argument("--no-llm", action="store_true")
    r.add_argument("--not-blind", action="store_true", help="tell Claude the satellite name")
    r.add_argument("--name", help="satellite name (only sent with --not-blind)")
    b = sub.add_parser("bench")
    b.add_argument("--no-llm", action="store_true")
    b.add_argument("--dev-only", action="store_true", help="development satellites only (keeps held-out rows)")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    a = ap.parse_args()

    if a.cmd == "fetch":
        from . import fetch
        if a.db:
            path = fetch.fetch_db(a.norad, a.data, days=a.days)
            # DB frames carry no orbit: give each the historical TLE nearest its time
            import json as _json
            from datetime import datetime
            rows = [_json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
            if rows:
                ts = [datetime.fromisoformat(r["ts"]) for r in rows]
                tles = fetch.tle_history(a.norad, min(ts), max(ts))
                fetch.apply_tles(path, tles)
                print(f"[fetch] {len(tles)} historical TLEs applied")
        else:
            fetch.fetch(a.norad, a.data, max_obs=a.max_obs)
        if a.eclipse_season:
            fetch.fetch_eclipse_season(a.norad, a.data)
        return

    if a.cmd == "bench":
        from . import bench
        sats = bench.SATS if a.dev_only else bench.SATS + bench.HELDOUT
        for r in bench.run(sats, data_dir=a.data, use_llm=not a.no_llm, merge=a.dev_only):
            def p(v):
                return "-" if v is None or v != v else f"{v:.0%}"
            print(f"{r['name']:<12} {r['frame_type']:<26} recall {p(r['recall'])}, exact {p(r['value_exact'])}, "
                  f"precision {p(r['precision'])}, Claude {p(r['claude_acc'])}")
        print("scoreboard: results/index.html")
        return

    from . import pipeline
    results = pipeline.run(a.norad, a.data, decoder=a.decoder, use_llm=not a.no_llm,
                           blind=not a.not_blind, sat_name=a.name)
    # single runs go to results/run/, so they never overwrite the benchmark's results
    run_dir = ROOT / "results" / "run"
    out = pipeline.save(results, run_dir, a.norad)
    from . import ksy, report
    page = report.render(results, a.norad, run_dir / f"{a.norad}.html", title=a.name or f"NORAD {a.norad}")
    print(f"report: {page}")
    for k in ksy.write(results, a.norad, run_dir / "ksy"):
        print(f"decoder draft: {k}")
    for res in results:
        print(f"\n== {res.ft.key}: {len(res.ft.df)} frames ==")
        names = {g["offset"]: g for g in (res.names or {}).get("fields", [])}
        for fld in res.fields:
            g = names.get(fld.offset, {})
            print(f"  @{fld.offset:<3} {fld.ctype:<6} {fld.kind:<8} score {fld.score:.3f}  "
                  f"{g.get('name', ''):<28} {g.get('unit', '')}")
        if res.evaluation:
            ev = res.evaluation
            print(f"  vs decoder: recall {ev['recall_physical']:.2f}, exact {ev['value_exact_physical']:.2f}, "
                  f"precision {ev['precision']:.2f}"
                  + (f", Claude category accuracy {ev['llm_category_accuracy']:.2f}" if "llm_category_accuracy" in ev else ""))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
