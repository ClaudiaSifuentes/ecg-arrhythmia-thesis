"""Synthetic Polar H10-like degradation for domain-shift robustness testing.

Approximates three effects of moving from MIT-BIH's 360 Hz clinical,
two-lead acquisition to a Polar H10's 130 Hz, single-lead, ambulatory
chest-strap signal, applied directly to already-extracted 250-sample
(694 ms) beat windows:

1. Resample round-trip: 360 Hz -> ~130 Hz-equivalent -> 360 Hz via polyphase
   filtering, matching the deployed backend's resampling step (Section
   III.B) -- this is the only piece that is an exact match to the real
   pipeline; it captures the information loss from the wearable's lower
   native sampling rate.
2. Baseline wander: low-frequency drift from breathing/movement against a
   chest strap, largely absent with adhesive Holter electrodes.
3. Motion artifact / contact noise: bursty high-amplitude noise and an
   elevated broadband noise floor from a chest strap's less stable skin
   contact.

This is a SIMULATION, not real Polar H10 data. It cannot substitute for
validation on the physical device (roadmap Section V, stage 1) -- there is
no ground truth that this noise model matches real chest-strap artifacts in
kind or magnitude. What it gives is a reproducible, quantitative lower
bound: if performance degrades even under a plausible synthetic version of
the domain shift, that is informative; if it does not, that is a necessary
(not sufficient) condition for real-device robustness.

Caveat: R-peak locations and RR-interval features are kept from the clean
signal (not re-estimated under the simulated noise), since RR features are
derived upstream of the beat-window extraction this module operates on.
This likely UNDERESTIMATES real degradation, since noisy R-peak detection
would also add RR-timing jitter.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

FS_CLINICAL = 360
FS_WEARABLE = 130


def resample_round_trip(x: np.ndarray, fs_in: int = FS_CLINICAL, fs_wearable: int = FS_WEARABLE) -> np.ndarray:
    """Downsample each window to the wearable rate and back up again, matching
    the deployed backend's polyphase resampling of incoming 130 Hz segments.
    """
    n = x.shape[-1]
    n_down = max(int(round(n * fs_wearable / fs_in)), 8)
    down = resample_poly(x, n_down, n, axis=-1)
    up = resample_poly(down, n, n_down, axis=-1)
    if up.shape[-1] != n:
        if up.shape[-1] > n:
            up = up[..., :n]
        else:
            pad_width = [(0, 0)] * (up.ndim - 1) + [(0, n - up.shape[-1])]
            up = np.pad(up, pad_width, mode="edge")
    return up.astype(np.float32)


def _add_baseline_wander(x: np.ndarray, fs: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[-1]
    t = np.arange(n) / fs
    freq = rng.uniform(0.15, 0.5)  # Hz, respiration-band drift
    phase = rng.uniform(0, 2 * np.pi)
    wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return x + wander


def _add_motion_artifact(x: np.ndarray, prob: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    n = x.shape[-1]
    if rng.random() < prob:
        burst_len = int(rng.integers(10, 40))
        start = int(rng.integers(0, max(1, n - burst_len)))
        x[start:start + burst_len] += amplitude * rng.standard_normal(burst_len)
    return x


def _add_contact_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = float(np.mean(x ** 2)) + 1e-8
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def simulate_polar_h10_degradation(
    X_beats: np.ndarray,
    *,
    seed: int = 42,
    baseline_amplitude: float = 0.15,
    motion_prob: float = 0.15,
    motion_amplitude: float = 0.8,
    contact_snr_db: float = 15.0,
) -> np.ndarray:
    """Apply the full degradation chain to a batch of (N, 250) beat windows,
    then re-normalize per-window (z-score) to match the original
    preprocessing (Section IV.A.2).
    """

    rng = np.random.default_rng(seed)
    resampled = resample_round_trip(X_beats)

    degraded = np.empty_like(resampled)
    for i in range(resampled.shape[0]):
        w = _add_baseline_wander(resampled[i], FS_CLINICAL, baseline_amplitude, rng)
        w = _add_motion_artifact(w, motion_prob, motion_amplitude, rng)
        w = _add_contact_noise(w, contact_snr_db, rng)
        degraded[i] = w

    mu = degraded.mean(axis=-1, keepdims=True)
    sd = degraded.std(axis=-1, keepdims=True) + 1e-8
    return ((degraded - mu) / sd).astype(np.float32)
