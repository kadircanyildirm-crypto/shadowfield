"""Download raw telemetry frames from the public SatNOGS Network API.

No account or token is needed: every observation lists its demodulated
frame files (one frame per file, timestamped in the file name) and the
TLE that was used for that pass, so historical orbits come for free.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://network.satnogs.org/api/observations/"
UA = {"User-Agent": "shadowfield-research/0.1"}
_TS = re.compile(r"data_\d+_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")


@dataclass
class Frame:
    obs_id: int
    ts: str  # ISO UTC
    hex: str
    tle1: str
    tle2: str
    station_lat: float
    station_lng: float
    station_alt: float


def _get(url: str, params: dict | None = None, tries: int = 5, max_wait: int = 3600) -> requests.Response:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
        except (requests.ConnectionError, requests.Timeout):  # the API is slow at times: back off, retry
            time.sleep(10 * (i + 1))
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "10")) + 5
            if wait > max_wait:
                r.raise_for_status()
            print(f"[fetch] rate limited by {url.split('/')[2]}, waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(2 * (i + 1))
            continue
        r.raise_for_status()
        return r
    raise requests.ConnectionError(f"gave up on {url} after {tries} tries")


def list_observations(norad: int, max_obs: int = 400, pause: float = 6.0,
                      start: datetime | None = None, end: datetime | None = None):
    """Yield 'good' observations that have demodulated frames, newest first (optionally in a date range)."""
    url, params, seen = API, {"norad_cat_id": norad, "status": "good", "format": "json"}, 0
    if start:
        params["start"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    if end:
        params["end"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    while url and seen < max_obs:
        r = _get(url, params)
        for o in r.json():
            if o.get("demoddata"):
                seen += 1
                yield o
                if seen >= max_obs:
                    return
        nxt = r.links.get("next", {}).get("url")
        url, params = nxt, None
        time.sleep(pause)


def _frame_ts(url: str) -> str | None:
    m = _TS.search(url)
    if not m:
        return None
    d, t = m.group(1).split("T")
    return datetime.fromisoformat(f"{d}T{t.replace('-', ':')}").replace(tzinfo=timezone.utc).isoformat()


def _download(url: str) -> bytes | None:
    try:
        r = _get(url, tries=3)
        return r.content
    except requests.RequestException:
        return None


def fetch(norad: int, out_dir: Path, max_obs: int = 400, workers: int = 16,
          start: datetime | None = None, end: datetime | None = None) -> Path:
    """Fetch frames for one satellite into out_dir/<norad>/frames.jsonl (cached)."""
    out = out_dir / str(norad)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "frames.jsonl"
    done = set()
    if path.exists():
        good = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                f = json.loads(line)
            except json.JSONDecodeError:  # a run killed mid-write leaves half a line: drop it
                continue
            good.append(line)
            done.add((f["obs_id"], f["ts"], f["hex"]))
        path.write_text("".join(g + "\n" for g in good), encoding="utf-8")
    done_obs = {k[0] for k in done}

    def work(job):
        o, url, ts = job
        raw = _download(url)
        if not raw:
            return None
        return Frame(o["id"], ts, raw.hex(), o["tle1"], o["tle2"],
                     o["station_lat"], o["station_lng"], o.get("station_alt") or 0.0)

    # frame files live on S3 (no API rate limit); write after every observation so
    # an interrupted or rate-limited run keeps its progress
    n_new = 0
    with ThreadPoolExecutor(workers) as ex, path.open("a", encoding="utf-8") as fh:
        for o in list_observations(norad, max_obs=max_obs, start=start, end=end):
            if o["id"] in done_obs:
                continue
            jobs = [(o, dd["payload_demod"], _frame_ts(dd["payload_demod"])) for dd in o["demoddata"]]
            for fr in ex.map(work, [j for j in jobs if j[2]]):
                if fr is None or (fr.obs_id, fr.ts, fr.hex) in done:
                    continue
                done.add((fr.obs_id, fr.ts, fr.hex))
                fh.write(json.dumps(asdict(fr)) + "\n")
                n_new += 1
            fh.flush()
            done_obs.add(o["id"])
    print(f"[fetch] {norad}: +{n_new} frames, {len(done)} total -> {path}", flush=True)
    return path


DB_API = "https://db.satnogs.org/api/"


def load_env(path: Path = Path(".env")) -> None:
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip() and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v.strip()


def latest_tle(norad: int) -> tuple[str, str]:
    r = _get(DB_API + "tle/", {"norad_cat_id": norad, "format": "json"})
    t = r.json()[0]
    return t["tle1"], t["tle2"]


def fetch_db(norad: int, out_dir: Path, days: int = 45, max_frames: int = 20000, pause: float = 1.5) -> Path:
    """Fetch frames from SatNOGS DB (all ground stations incl. non-SatNOGS ones). Needs SATNOGS_DB_TOKEN.

    DB frames carry no per-pass TLE, so the latest TLE is used and the window is kept
    short (default 45 days) to keep the propagated orbit accurate to well under a minute.
    """
    import os
    token = os.environ.get("SATNOGS_DB_TOKEN")
    if not token:
        raise RuntimeError("SATNOGS_DB_TOKEN is not set (put it in .env)")
    headers = {**UA, "Authorization": f"Token {token}"}
    tle1, tle2 = latest_tle(norad)
    since = datetime.now(timezone.utc).timestamp() - days * 86400
    start = datetime.fromtimestamp(since, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out_dir / str(norad)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "frames.jsonl"
    url, params, n, ts = DB_API + "telemetry/", {"satellite": norad, "start": start, "format": "json"}, 0, None
    tmp = path.with_suffix(".jsonl.part")  # the old file is only replaced once the download succeeds
    with tmp.open("w", encoding="utf-8") as fh:
        while url and n < max_frames:
            r = None
            for i in range(6):
                try:
                    r = requests.get(url, params=params, headers=headers, timeout=90)
                except (requests.ConnectionError, requests.Timeout):
                    time.sleep(10 * (i + 1))
                    r = None
                    continue
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", "30"))
                    print(f"[fetch_db] rate limited, waiting {wait}s")
                    time.sleep(min(wait, 900))
                    r = None
                    continue
                r.raise_for_status()
                break
            if r is None:
                raise requests.ConnectionError(f"SatNOGS DB did not answer for {url}")
            body = r.json()
            for t in body["results"] if isinstance(body, dict) else body:
                ts = t.get("timestamp")
                if not ts or not t.get("frame"):
                    continue
                # pass ids are rebuilt from time gaps in frames.load
                f = Frame(t.get("observation_id") or 0,
                          datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat(),
                          t["frame"].lower(), tle1, tle2,
                          float(t.get("lat") or 0), float(t.get("lng") or 0), 0.0)
                fh.write(json.dumps(asdict(f)) + "\n")
                n += 1
            url = body.get("next") if isinstance(body, dict) else r.links.get("next", {}).get("url")
            params = None
            fh.flush()
            print(f"[fetch_db] {norad}: {n} frames, back to {ts}", flush=True)
            time.sleep(pause)
    tmp.replace(path)
    return path


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def survey(norads: list[int], pages: int = 1) -> list[dict]:
    """Cheap look at how many frames each satellite has recently."""
    rows = []
    for n in norads:
        try:
            r = _get(API, {"norad_cat_id": n, "status": "good", "format": "json"})
            obs = r.json()
        except requests.RequestException as e:
            rows.append({"norad": n, "error": str(e)})
            continue
        frames = sum(len(o.get("demoddata") or []) for o in obs)
        rows.append({"norad": n, "obs": len(obs), "frames_page1": frames,
                     "newest": obs[0]["start"] if obs else None,
                     "oldest": obs[-1]["start"] if obs else None})
        time.sleep(1.0)
    return rows


def _tle_epoch(tle1: str) -> float:
    """TLE epoch as unix seconds."""
    yy, doy = int(tle1[18:20]), float(tle1[20:32])
    year = 2000 + yy if yy < 57 else 1900 + yy
    return datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() + (doy - 1) * 86400


def tle_history(norad: int, start: datetime, end: datetime, step_days: float = 4.0, pause: float = 6.0) -> list[tuple[str, str]]:
    """A handful of historical TLEs: the one used by the first SatNOGS observation in each window.

    Low satellites drift by many km per day under drag, so DB frames (which carry no TLE) are
    matched to the nearest of these instead of being propagated weeks from today's TLE.
    """
    out, t = {}, start
    while t < end:
        t2 = min(end, datetime.fromtimestamp(t.timestamp() + step_days * 86400, timezone.utc))
        try:
            r = _get(API, {"norad_cat_id": norad, "start": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "end": t2.strftime("%Y-%m-%dT%H:%M:%SZ"), "format": "json"}, max_wait=900)
            for o in r.json():
                if o.get("tle1"):
                    out[o["tle1"]] = (o["tle1"], o["tle2"])
                    break
        except requests.RequestException as e:
            print(f"[tle] {t:%Y-%m-%d}: {e}", flush=True)
        t = t2
        time.sleep(pause)
    return sorted(out.values(), key=lambda p: _tle_epoch(p[0]))


def apply_tles(path: Path, tles: list[tuple[str, str]]) -> None:
    """Give every frame in a frames.jsonl the TLE with the nearest epoch."""
    if not tles:
        return
    epochs = [_tle_epoch(t[0]) for t in tles]
    rows = _read_rows(path)
    for f in rows:
        u = datetime.fromisoformat(f["ts"]).timestamp()
        f["tle1"], f["tle2"] = tles[min(range(len(epochs)), key=lambda i: abs(epochs[i] - u))]
    path.write_text("".join(json.dumps(f) + "\n" for f in rows), encoding="utf-8")


def fetch_eclipse_season(norad: int, out_dir: Path, max_obs: int = 150, lookback_days: int = 730) -> list[tuple]:
    """Add frames from the most recent weeks in which the satellite actually saw eclipses."""
    from .eclipse import calendar, eclipse_windows
    path = out_dir / str(norad) / "frames.jsonl"
    rows = _read_rows(path) if path.exists() else []
    if rows:
        t1, t2 = max(((f["tle1"], f["tle2"]) for f in rows), key=lambda t: t[0][18:32])
    else:
        t1, t2 = latest_tle(norad)
    now = time.time()
    wins = eclipse_windows(calendar(t1, t2, now - lookback_days * 86400, now, step_days=2))
    if not wins:
        print(f"[eclipse] {norad}: no eclipse season in the last {lookback_days} days")
        return []
    a, b = wins[-1]
    start, end = (datetime.fromtimestamp(x, timezone.utc) for x in (a, b))
    print(f"[eclipse] {norad}: fetching eclipse season {start:%Y-%m-%d} .. {end:%Y-%m-%d}", flush=True)
    fetch(norad, out_dir, max_obs=max_obs, start=start, end=end)
    return wins
