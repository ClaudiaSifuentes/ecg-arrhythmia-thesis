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

Contact noise is modeled as TIERED, not uniform: most beats get mild,
typical chest-strap noise, and a small fraction get severe noise, echoing
Skala et al. [4] -- the same Polar H10 validation study already cited in
this paper -- who report only 2.16% of their real Polar H10 recordings as
artifact-affected. An earlier uniform-severe-noise version of this model
(every beat at 15 dB SNR) collapsed VEB recall to 0.00; a component
ablation (scripts/eval_domain_shift_ablation.py) showed this was driven
entirely by that uniform noise assumption, not by the resample round-trip
(which matches the real deployed pipeline step and had almost no effect on
its own) or baseline wander (fully absorbed by the existing per-beat
z-score normalization). The tiered model below is the literature-grounded
replacement.

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


def _add_tiered_contact_noise(
    x: np.ndarray,
    *,
    artifact_prob: float,
    typical_snr_db: float,
    severe_snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Most beats get mild, typical contact noise; a small fraction
    (artifact_prob) get severe noise -- see module docstring for why this
    replaced a uniform-severe-noise assumption.
    """
    snr_db = severe_snr_db if rng.random() < artifact_prob else typical_snr_db
    return _add_contact_noise(x, snr_db, rng)


def simulate_polar_h10_degradation(
    X_beats: np.ndarray,
    *,
    seed: int = 42,
    apply_resample: bool = True,
    baseline_amplitude: float = 0.15,
    motion_prob: float = 0.15,
    motion_amplitude: float = 0.8,
    contact_artifact_prob: float = 0.0216,  # Skala et al. [4]: 2.16% of Polar H10 recordings artifact-affected
    contact_typical_snr_db: float = 28.0,
    contact_severe_snr_db: float = 15.0,
) -> np.ndarray:
    """Apply the degradation chain to a batch of (N, 250) beat windows.

    Does NOT re-normalize per window afterward. The real preprocessing
    (src/data/dataset_builder.py) z-scores each full record ONCE, before
    slicing out 250-sample beat windows -- so a window's local amplitude
    (std) is not 1.0 by construction, it reflects the beat's actual
    deflection size relative to the record. VEB beats have a mean std of
    1.27 vs. 0.96 for N in this corpus (measured on X_beats.npy) -- this is
    real morphological signal, not noise. An earlier version of this
    function forcibly re-z-scored each output window to std=1, which erased
    that amplitude cue and caused a large, spurious VEB recall drop even
    with all degradation parameters set to their off-values. Do not add
    that step back without re-deriving it from the real per-record
    normalization.

    `apply_resample=False` and/or zeroing individual noise parameters
    (baseline_amplitude=0, motion_prob=0, contact_artifact_prob=0 with
    contact_typical_snr_db>=60) isolates each component -- see
    scripts/eval_domain_shift_ablation.py.
    """

    rng = np.random.default_rng(seed)
    resampled = resample_round_trip(X_beats) if apply_resample else X_beats.copy()

    degraded = np.empty_like(resampled)
    for i in range(resampled.shape[0]):
        w = _add_baseline_wander(resampled[i], FS_CLINICAL, baseline_amplitude, rng)
        w = _add_motion_artifact(w, motion_prob, motion_amplitude, rng)
        w = _add_tiered_contact_noise(
            w,
            artifact_prob=contact_artifact_prob,
            typical_snr_db=contact_typical_snr_db,
            severe_snr_db=contact_severe_snr_db,
            rng=rng,
        )
        degraded[i] = w

    return degraded.astype(np.float32)
