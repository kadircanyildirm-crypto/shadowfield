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
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    a = ap.parse_args()

    if a.cmd == "fetch":
        from . import fetch
        if a.db:
            fetch.fetch_db(a.norad, a.data, days=a.days)
        else:
            fetch.fetch(a.norad, a.data, max_obs=a.max_obs)
        if a.eclipse_season:
            fetch.fetch_eclipse_season(a.norad, a.data)
        return

    if a.cmd == "bench":
        from . import bench
        for r in bench.run(data_dir=a.data, use_llm=not a.no_llm):
            print(f"{r['name']:<10} {r['frame_type']:<22} recall {r['recall']}, exact {r['exact']}, "
                  f"precision {r['precision']}, Claude {r['claude_acc']}")
        return

    from . import pipeline
    results = pipeline.run(a.norad, a.data, decoder=a.decoder, use_llm=not a.no_llm,
                           blind=not a.not_blind, sat_name=a.name)
    out = pipeline.save(results, ROOT / "results", a.norad)
    from . import ksy, report
    report.render(results, a.norad, ROOT / "results" / f"{a.norad}.html", title=a.name or f"NORAD {a.norad}")
    for k in ksy.write(results, a.norad, ROOT / "results" / "ksy"):
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
            print(f"  vs decoder: recall {ev['recall_physical']:.2f}, exact {ev['exact_physical']:.2f}, "
                  f"precision {ev['precision']:.2f}"
                  + (f", Claude category accuracy {ev['llm_category_accuracy']:.2f}" if "llm_category_accuracy" in ev else ""))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
