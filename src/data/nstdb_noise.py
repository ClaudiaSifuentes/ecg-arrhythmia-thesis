"""Real ECG noise from the MIT-BIH Noise Stress Test Database (NSTDB).

NSTDB provides three 30-min noise recordings made on active volunteers with
electrodes placed so the ECG is not visible:
  bw  baseline wander
  em  electrode motion artifact (the hardest for beat detectors)
  ma  muscle (EMG) artifact
These are real wearable-type artifacts, unlike the synthetic generators in
src/data/domain_shift_sim.py. Each record is split in time: the first half
may be used for training-time augmentation (E19), the second half only for
robustness evaluation, so evaluation noise is never seen in training.

Noise is preprocessed like the ECG (bandpass 0.5–40 Hz) and added to each
250-sample beat window with a random offset. SNR is defined per beat:
SNR_dB = 10·log10(var(beat) / var(noise segment)). NSTDB's own records define
SNR from QRS amplitude over the whole record, so values are not identical to
the NSTDB "_e12"/"_e06" records, but the ordering (12 dB mild, 6 dB severe) is
comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NOISE_TYPES = ("bw", "em", "ma")


def load_nstdb_noise(raw_dir: str | Path = "data/raw/nstdb", part: str = "eval") -> dict[str, np.ndarray]:
    """Download (once) and return {type: bandpassed channel-0 signal at 360 Hz}.

    part: "train" = first half of each 30-min noise record (training-time
    augmentation only), "eval" = second half (robustness evaluation only),
    "all" = whole record. Train and eval never share noise samples.
    """

    import wfdb

    from src.data.preprocessing import apply_bandpass

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not all((raw_dir / f"{r}.hea").exists() for r in NOISE_TYPES):
        wfdb.dl_database("nstdb", dl_dir=str(raw_dir), records=list(NOISE_TYPES))
    out = {}
    for r in NOISE_TYPES:
        rec = wfdb.rdrecord(str(raw_dir / r))
        assert rec.fs == 360, rec.fs
        raw = rec.p_signal[:, 0]
        half = len(raw) // 2
        # split BEFORE filtering so the zero-phase filter cannot mix the halves
        seg = {"train": raw[:half], "eval": raw[half:], "all": raw}[part]
        out[r] = apply_bandpass(seg, fs=360).astype(np.float32)
    return out


def add_nstdb_noise(X: np.ndarray, noise: dict[str, np.ndarray], types: tuple[str, ...], snr_db: float,
                    seed: int) -> np.ndarray:
    """Add a random real-noise segment to every beat at the given per-beat SNR.

    `types`: noise type(s) to draw from; with several, each beat gets one type
    chosen uniformly at random.
    """

    rng = np.random.default_rng(seed)
    n, L = X.shape
    kinds = rng.integers(0, len(types), size=n)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(n):
        src = noise[types[kinds[i]]]
        start = int(rng.integers(0, len(src) - L))
        seg = src[start:start + L]
        seg = seg - seg.mean()
        p_sig = float(np.var(X[i])) + 1e-8
        p_noise = float(np.var(seg)) + 1e-12
        scale = np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
        out[i] = X[i] + scale * seg
    return out
