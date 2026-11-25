"""Compare the fast shadow model with Skyfield's full-precision one. Run: python -m tests.check_physics"""
import datetime as dt

import numpy as np
from sgp4.api import Satrec
from skyfield.api import EarthSatellite, Loader, wgs84

from shadowfield.physics import propagate, sun_unit, in_sun, geodetic, JD_UNIX0
from tests.synth import TLE1, TLE2

ld = Loader("data/ephemeris", verbose=False)
ts, eph = ld.timescale(), ld("de421.bsp")
t0 = 1792800000.0  # 2026-10-23
unix = t0 + np.arange(0, 2 * 86400, 10.0)  # two days, 10 s steps

r, _ = propagate(Satrec.twoline2rv(TLE1, TLE2), unix)
fast = in_sun(r, sun_unit(JD_UNIX0 + unix / 86400))
# note: ts.utc(1970, 1, 1, 0, 0, unix) would add the 27 leap seconds since 1972 to unix time
sky = EarthSatellite(TLE1, TLE2, "s", ts).at(ts.from_datetimes([dt.datetime.fromtimestamp(u, dt.timezone.utc) for u in unix]))
ref = sky.is_sunlit(eph)
mism = np.flatnonzero(fast != ref)
print(f"sunlit agreement: {100 * (1 - len(mism) / len(unix)):.3f}%  ({len(mism)} of {len(unix)} 10-s samples differ)")
edges = np.flatnonzero(np.diff(ref.astype(int)))
print(f"shadow transitions: {len(edges)}; samples off per transition: {len(mism) / max(len(edges), 1):.2f}")
lat, lon, alt = geodetic(r, unix)
sp = wgs84.geographic_position_of(sky)
dlon = (lon - sp.longitude.degrees + 180) % 360 - 180
print(f"max |dlat| {np.abs(lat - sp.latitude.degrees).max():.3f} deg, max |dlon| {np.abs(dlon).max():.3f} deg, "
      f"max |dalt| {np.abs(alt - sp.elevation.km).max():.2f} km")
