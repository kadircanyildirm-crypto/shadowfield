"""Find telemetry fields in unknown frames by correlating raw bytes with physics.

Every (offset, width, endianness, signedness) reading of the frame bytes is a
candidate field. Each candidate is scored against the orbital physics:

  sun      steps with sunlight right now        -> solar array current/voltage, sun sensors
  lagged   follows sunlight with a time lag      -> temperatures, battery charge
  orbit    repeats once/twice per orbit          -> attitude/thermal cycles
  counter  only ever goes up (with resets)       -> uptime, packet/boot counters
  clock    tracks wall-clock time 1:1            -> on-board time stamps
  magnet   |(x,y,z)| tracks Earth's field model  -> 3-axis magnetometer
  position equals a coordinate of the orbit       -> GNSS position / velocity (ECEF or inertial)

Frames inside one pass are strongly correlated, so significance is judged on
the number of passes, not the number of frames.
"""
# The implementation lives in the shadowfield.engine package; this module keeps the old import path.
from .engine import ALPHA, WIDTHS, Field, analyze, describe  # noqa: F401
from .engine.detectors import _clock, _magnetometers, _positions, _quaternions  # noqa: F401
from .engine.field import _decode, _k, _split  # noqa: F401
from .engine.select import _canonical_widths, _greedy, _pick_encodings  # noqa: F401
from .engine.structure import _boundary, _carries_into, _carry, _lsb_follows, _width_ok  # noqa: F401
