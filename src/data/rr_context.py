"""RR feature sets computed from the stored per-beat RR arrays.

The model interface is fixed at 6 RR inputs. Alternative feature sets keep the
6-column shape but change what each column means, so a set other than "base"
requires the same transform in the deployed preprocessing.

"base" (E12): [RR_actual, RR_prev, RR_mean_local(5), SDNN(5), RMSSD(5), RR_actual/RR_prev]

Rows of one patient are stored in time order; consecutive beats are
contiguous when X_rr[i+1, RR_prev] == X_rr[i, RR_actual] (99.6 % of rows in
mitbih_aami3). RR_next is taken from the next row only when contiguous.
"""

from __future__ import annotations

import numpy as np


def _next_rr(X_rr: np.ndarray, pid: np.ndarray) -> np.ndarray:
    nxt = X_rr[:, 0].copy()  # fallback: current RR (ratio = 1, neutral)
    same = pid[1:] == pid[:-1]
    contig = same & np.isclose(X_rr[1:, 1], X_rr[:-1, 0], atol=1e-6)
    nxt[:-1][contig] = X_rr[1:, 0][contig]
    return nxt


def _rolling_mean_per_patient(x: np.ndarray, pid: np.ndarray, window: int) -> np.ndarray:
    """Causal mean of the last `window` values (current included), per patient."""

    out = np.empty_like(x, dtype=np.float64)
    for p in np.unique(pid):
        idx = np.nonzero(pid == p)[0]
        v = x[idx].astype(np.float64)
        c = np.cumsum(np.r_[0.0, v])
        n = np.minimum(np.arange(1, len(v) + 1), window)
        out[idx] = (c[1:] - c[np.arange(1, len(v) + 1) - n]) / n
    return out


def build_rr_features(X_rr: np.ndarray, patient_id: np.ndarray, rr_set: str = "base") -> np.ndarray:
    X_rr = np.asarray(X_rr, dtype=np.float32)
    if rr_set == "base":
        return X_rr
    pid = np.asarray(patient_id).astype(str)
    rr = X_rr[:, 0].astype(np.float64)
    prev = X_rr[:, 1].astype(np.float64)
    nxt = _next_rr(X_rr, pid).astype(np.float64)
    eps = 1e-6
    if rr_set == "context":
        # [RR_actual, RR_next, RR_mean_local(5), RR_actual/RR_local_long, RR_next/RR_actual, RR_actual/RR_prev]
        # RR_local_long: mean of the last 50 RR (~40 s), a patient-adaptive reference
        # that turns absolute RR into "premature relative to this patient's rhythm".
        # Requires a one-beat delay at inference (RR_next).
        valid = rr > 0
        rr_f = np.where(valid, rr, np.nan)
        long_mean = _rolling_mean_per_patient(np.nan_to_num(rr_f, nan=np.nanmedian(rr_f)), pid, 50)
        out = np.stack([
            rr,
            nxt,
            X_rr[:, 2],
            np.where(valid, rr / (long_mean + eps), 1.0),
            np.where(valid & (nxt > 0), nxt / (rr + eps), 1.0),
            X_rr[:, 5],
        ], axis=1)
        return out.astype(np.float32)
    raise ValueError(f"unknown rr_set={rr_set}")
