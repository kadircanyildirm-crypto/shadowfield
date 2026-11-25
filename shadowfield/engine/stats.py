"""Rank statistics. Significance is counted in passes, because frames in one pass are not independent."""
from __future__ import annotations

import os

import numpy as np
from scipy import stats

# 1e-4 per reading, counted in passes: on the development satellites this found more
# fields than 1e-6 at the same precision (1e-8 lost fields)
ALPHA = float(os.environ.get("SHADOWFIELD_ALPHA", "1e-4"))

def _ranks(M: np.ndarray) -> np.ndarray:
    R = stats.rankdata(M, axis=0)
    R -= R.mean(0)
    sd = R.std(0)
    sd[sd == 0] = np.inf
    return R / sd

def _restd(R: np.ndarray) -> np.ndarray:
    R = R - R.mean(0)
    sd = R.std(0)
    sd[sd == 0] = np.inf
    return R / sd

def _demean(R: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = R.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(0)
    sd = out.std(0)
    sd[sd == 0] = np.inf
    return out / sd

def _p_from_rho(rho: np.ndarray, n_eff: int) -> np.ndarray:
    r = np.clip(np.abs(rho), 0, 0.999999)
    t = r * np.sqrt((n_eff - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(t, n_eff - 2)
