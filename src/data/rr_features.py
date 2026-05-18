"""RR-interval handcrafted feature extraction.

This module must NOT perform R-peak detection.
R-peaks are detected and validated in `src.data.rpeaks`.

Input/Output contract
---------------------
- Input: `r_peaks` sample indices (1D, sorted, unique), and `fs`.
- Output: NumPy array of shape (N, 6) aligned with `r_peaks`.

Feature set (per thesis spec)
-----------------------------
[RR_actual, RR_prev, RR_mean_local, SDNN, RMSSD, RR_ratio]

Definitions
-----------
RR intervals are computed in seconds:
- rr_i = (r_peaks[i] - r_peaks[i-1]) / fs, for i>=1

For beat i:
- RR_actual: rr_i (0 for i==0)
- RR_prev: rr_{i-1} (0 for i<2)
- RR_mean_local: mean of last `window` RR intervals up to rr_i (requires i>=1)
- SDNN: std of last `window` RR intervals up to rr_i
- RMSSD: sqrt(mean(diff(rr_window)^2)) over last `window` RR intervals (>=2 intervals)
- RR_ratio: RR_actual / RR_prev (0 if RR_prev==0)
"""

from __future__ import annotations

import numpy as np


def extract_rr_features(r_peaks: np.ndarray, fs: int = 360, window: int = 5) -> np.ndarray:
    """Extract 6 RR-interval features per beat from pre-detected R-peaks.

    IMPORTANT
    ---------
    `r_peaks` must already be detected and validated (Se/PPV criteria).
    This function does NOT detect R-peaks.

    Parameters
    ----------
    r_peaks:
        1D array of R-peak sample indices.
    fs:
        Sampling rate (Hz).
    window:
        Number of most recent RR intervals to use for local statistics.

    Returns
    -------
    X_rr: np.ndarray
        Array of shape (N, 6) aligned with `r_peaks`.
        Columns: [RR_actual, RR_prev, RR_mean_local, SDNN, RMSSD, RR_ratio]
    """

    rp = np.asarray(r_peaks, dtype=int).reshape(-1)
    n = int(rp.size)

    if n == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if n == 1:
        # No RR intervals can be computed
        return np.zeros((1, 6), dtype=np.float32)

    if window < 1:
        raise ValueError(f"window must be >= 1; got window={window}")

    # Ensure strictly increasing to avoid negative/zero RR intervals
    if np.any(np.diff(rp) <= 0):
        raise ValueError("r_peaks must be strictly increasing (sorted, unique).")

    # RR intervals in seconds for i>=1
    rr = np.zeros(n, dtype=np.float64)
    rr[1:] = np.diff(rp).astype(np.float64) / float(fs)

    X = np.zeros((n, 6), dtype=np.float32)

    for i in range(n):
        rr_actual = float(rr[i]) if i >= 1 else 0.0
        rr_prev = float(rr[i - 1]) if i >= 2 else 0.0

        if i >= 1:
            # RR window is defined over intervals rr[1..i]
            start = max(1, i - window + 1)
            rr_w = rr[start : i + 1]
            rr_mean_local = float(np.mean(rr_w)) if rr_w.size > 0 else 0.0
            sdnn = float(np.std(rr_w, ddof=0)) if rr_w.size > 0 else 0.0

            if rr_w.size >= 2:
                d = np.diff(rr_w)
                rmssd = float(np.sqrt(np.mean(d * d)))
            else:
                rmssd = 0.0
        else:
            rr_mean_local = 0.0
            sdnn = 0.0
            rmssd = 0.0

        rr_ratio = float(rr_actual / rr_prev) if rr_prev > 0 else 0.0

        X[i, 0] = rr_actual
        X[i, 1] = rr_prev
        X[i, 2] = rr_mean_local
        X[i, 3] = sdnn
        X[i, 4] = rmssd
        X[i, 5] = rr_ratio

    # Safety checks (requested)
    assert X.shape == (len(rp), 6)
    assert not np.any(np.isnan(X))

    return X