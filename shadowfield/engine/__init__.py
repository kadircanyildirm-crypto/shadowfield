"""The Shadowfield engine: telemetry fields from raw frames and orbital physics."""
from .core import analyze
from .evidence import describe
from .field import WIDTHS, Field, _decode, _split
from .stats import ALPHA
from .structure import _boundary, _carries_into

__all__ = ["analyze", "describe", "Field", "WIDTHS", "ALPHA"]
