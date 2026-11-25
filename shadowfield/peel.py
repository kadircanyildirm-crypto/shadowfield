"""Peel obfuscation layers that are not cryptography, and recognise the ones that are.

Telemetry that "looks encrypted" is usually just whitened. A scrambler or randomizer
XORs the frame with a pseudo-random sequence so the radio link has no long runs of
equal bits. Amateur-band telemetry may not be encrypted to hide its meaning (ITU Radio
Regulations, Art. 25), so a real cipher in a beacon is rare. The layers handled here:

  ccsds      XOR with the CCSDS TM pseudo-randomizer (h(x)=x^8+x^7+x^5+x^3+1)
  g3ruh      the 9k6 G3RUH self-synchronising scrambler (x^17+x^12+1)
  xor key    any *fixed* per-position whitening sequence, unknown in advance: recovered
             byte by byte from physics. Only the right key makes a byte column follow
             sunlight smoothly, or count upward inside a pass.

A real cipher (AES and the like) gives every frame a fresh, uniform-looking body. That
is reported as "encrypted: no physics left to read", never decoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---- known whitening sequences --------------------------------------------------------

def ccsds_pn(n_bytes: int) -> np.ndarray:
    """CCSDS 131.0-B pseudo-randomizer sequence (starts FF 48 0E C0 9A 0D 70 BC ...)."""
    state = [1] * 8
    bits = []
    for _ in range(n_bytes * 8):
        bits.append(state[0])
        fb = state[0] ^ state[3] ^ state[5] ^ state[7]  # x^8 + x^7 + x^5 + x^3 + 1
        state = state[1:] + [fb]
    return np.packbits(np.array(bits, dtype=np.uint8))


def g3ruh_scramble(B: np.ndarray) -> np.ndarray:
    """Scramble each frame (for tests): y[i] = x[i] ^ y[i-12] ^ y[i-17], fresh state per frame."""
    bits = np.unpackbits(B, axis=1).astype(np.uint8)
    out = np.zeros_like(bits)
    for i in range(bits.shape[1]):
        v = bits[:, i].copy()
        if i >= 12:
            v ^= out[:, i - 12]
        if i >= 17:
            v ^= out[:, i - 17]
        out[:, i] = v
    return np.packbits(out, axis=1)


def g3ruh_descramble(B: np.ndarray) -> np.ndarray:
    """x[i] = y[i] ^ y[i-12] ^ y[i-17] (self-synchronising: needs no key)."""
    y = np.unpackbits(B, axis=1).astype(np.uint8)
    x = y.copy()
    x[:, 12:] ^= y[:, :-12]
    x[:, 17:] ^= y[:, :-17]
    return np.packbits(x, axis=1)


# ---- unknown fixed key, recovered from physics ----------------------------------------

def _prep(B, df, phys):
    order = np.argsort(df["ts"].to_numpy())
    Bs, ph = B[order], phys.iloc[order].reset_index(drop=True)
    obs = df["obs_id"].to_numpy()[order]
    unix = ph["unix"].to_numpy()
    same = (obs[1:] == obs[:-1]) & (np.diff(unix) > 5) & (np.diff(unix) <= 180)
    return Bs, ph, same


def column_key_scores(B: np.ndarray, df: pd.DataFrame, phys: pd.DataFrame) -> np.ndarray:
    """S[c, k]: how physical byte column c looks after XOR with k (NaN for near-constant columns).

    The right key turns a column into a reading that tracks sunlight (rank correlation) or
    counts upward inside passes; a wrong key scrambles its value order.
    """
    Bs, ph, same = _prep(B, df, phys)
    targets = [stats.rankdata(ph[c].to_numpy()) for c in ("sunlit", "ema_10", "ema_40") if ph[c].std() > 0]
    targets = [(t - t.mean()) / t.std() for t in targets]
    keys = np.arange(256, dtype=np.uint8)
    S = np.full((B.shape[1], 256), np.nan)
    for c in range(B.shape[1]):
        if np.unique(Bs[:, c]).size < 8:
            continue
        V = (Bs[:, c][:, None] ^ keys[None, :]).astype(float)
        R = stats.rankdata(V, axis=0)
        R = (R - R.mean(0)) / np.where(R.std(0) > 0, R.std(0), np.inf)
        phys_s = np.max([np.abs(t @ R / len(t)) for t in targets], axis=0) if targets else np.zeros(256)
        d = np.diff(V, axis=0)[same]
        mono = ((d > 0).sum(0) - (d < 0).sum(0)) / np.maximum(len(d), 1)
        S[c] = np.maximum(phys_s, mono)
    return S


def _identity(S: np.ndarray) -> np.ndarray:
    # XOR 0x80 only turns a signed byte into offset binary: the same information, not a layer
    return np.fmax(S[:, 0], S[:, 0x80])


def key_gain(S: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Per-column improvement of a key over leaving the bytes alone."""
    g = S[np.arange(len(key)), key] - _identity(S)
    return np.nan_to_num(g, nan=0.0)


def recover_xor_key(B: np.ndarray, df: pd.DataFrame, phys: pd.DataFrame, min_gain: float = 0.25,
                    S: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Estimate a fixed per-position XOR key one byte column at a time.

    A column takes a key only when that key clearly beats leaving the column alone; other
    columns stay 0. A second pass gives the low byte of a number whose high byte is now
    known the key that makes the combined value smooth inside passes.
    """
    S = column_key_scores(B, df, phys) if S is None else S
    Bs, _, same = _prep(B, df, phys)
    keys = np.arange(256, dtype=np.uint8)
    key = np.zeros(B.shape[1], dtype=np.uint8)
    info = {}
    ident = _identity(S)
    for c in range(B.shape[1]):
        if np.isnan(S[c]).all():
            continue
        k = int(np.nanargmax(S[c]))
        if k not in (0, 0x80) and S[c, k] - ident[c] >= min_gain:
            key[c] = k
            info[c] = {"key": k, "score": round(float(S[c, k]), 3), "identity": round(float(ident[c]), 3)}
    for c in range(B.shape[1]):
        if c in info or np.unique(Bs[:, c]).size < 16:
            continue
        best = None
        for hi_c in (c - 1, c + 1):
            if hi_c not in info or not 0 <= hi_c < B.shape[1]:
                continue
            hi = (Bs[:, hi_c] ^ key[hi_c]).astype(float)
            V = 256 * hi[:, None] + (Bs[:, c][:, None] ^ keys[None, :]).astype(float)
            rough = np.abs(np.diff(V, axis=0))[same].mean(0)
            k = int(np.argmin(rough))
            gain = float(rough[0] / max(rough[k], 1e-9))  # against leaving the byte alone
            if k != 0 and gain > 3 and (best is None or gain > best[1]):
                best = (k, gain, hi_c)
        if best:
            key[c] = best[0]
            info[c] = {"key": best[0], "low_byte_of": best[2], "smoothness_gain": round(best[1], 1)}
    return key, info


# ---- encrypted or not? ---------------------------------------------------------------

def column_entropy(B: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of each byte position across frames."""
    out = np.zeros(B.shape[1])
    for c in range(B.shape[1]):
        _, cnt = np.unique(B[:, c], return_counts=True)
        p = cnt / cnt.sum()
        out[c] = -(p * np.log2(p)).sum()
    return out


def looks_encrypted(B: np.ndarray) -> dict:
    """Ciphertext: nearly every byte position is close to uniform (entropy near its maximum)."""
    h = column_entropy(B)
    ceiling = min(8.0, np.log2(len(B)))
    hot = h > ceiling - 0.35
    return {"mean_entropy_bits": round(float(h.mean()), 2), "ceiling_bits": round(float(ceiling), 2),
            "uniform_byte_fraction": round(float(hot.mean()), 3), "encrypted_like": bool(hot.mean() > 0.85)}


TRANSFORMS = {
    "ccsds": lambda B: B ^ ccsds_pn(B.shape[1])[None, :],
    "g3ruh": g3ruh_descramble,
}


def auto(ft, phys, analyze, min_fields: int = 3) -> dict | None:
    """Find a whitening layer, if there is one. Returns None for frames that are already clean.

    Each layer must pass a structural test before it is even tried, so clean frames are
    never "peeled":
      fixed keys (CCSDS, recovered)  the key must make several byte columns clearly more
                                     physical than leaving them alone
      G3RUH (mixes bits)             undoing a scrambler simplifies the data, so per-column
                                     entropy must drop clearly (on clean frames it rises)
    The layers that pass are compared with a quick analysis; the winner is analysed in full.
    """
    B = ft.bytes_
    S = column_key_scores(B, ft.df, phys)
    trials = {}
    pn = ccsds_pn(B.shape[1])
    g = key_gain(S, pn)
    # columns the key leaves no better are not evidence against it (a byte whose key bit
    # pattern happens to keep its order), so only the clear improvements are counted
    if (g >= 0.25).sum() >= 3 and g[g > 0].sum() >= 1.0:
        trials["ccsds"] = B ^ pn[None, :]
    key, info = recover_xor_key(B, ft.df, phys, S=S)
    g = key_gain(S, key)
    if "ccsds" not in trials and (g >= 0.25).sum() >= 3 and g[g > 0].sum() >= 1.0:
        trials["xor_key"] = B ^ key[None, :]
    h0 = column_entropy(B).mean()
    Bg = g3ruh_descramble(B)
    if column_entropy(Bg).mean() < h0 - 0.5:
        trials["g3ruh"] = Bg
    if not trials:
        return None
    base = analyze(B, ft.df, phys, quick=True)
    base_score = sum(f.score for f in base)
    best = None
    for name, Bt in trials.items():
        found = analyze(Bt, ft.df, phys, quick=True)
        sc = sum(f.score for f in found)
        if len(found) >= min_fields and sc > base_score and (best is None or sc > best["score"]):
            best = {"layer": name, "bytes": Bt, "score": sc,
                    "key_bytes": int((key != 0).sum()) if name == "xor_key" else None}
    if best:
        best["fields"] = analyze(best["bytes"], ft.df, phys)
        best["baseline_fields"] = len(base)
        best["baseline_score"] = base_score
    return best
