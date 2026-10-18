"""Candidate E21: E20 CNN ensemble (3 seeds) + RR-only gradient boosting.

Interface (unchanged from E12)
------------------------------
    input : beats (N, 250) float32 — same preprocessing as E16 (heartly_e16.py)
            rr    (N, 6)   float32 — the 6 RR inputs, "context2" set:
                  [RR, RR_next, RR_mean5, RR/RR_median300, RR_next/RR, RR/RR_prev]
                  (use rr_features_from_peaks(peaks, rr_set="context2"))
    output: classes (N,) in {0: N, 1: SVEB, 2: VEB} and probabilities (N, 3)

    p = (1 - w) · mean_seeds(softmax(CNN)) + w · p_RR
then the decision rule `decide(p, t_sveb, t_veb)` (src/training/decision.py).
w comes from the CV (median of the per-fold choices on inner validation);
t_sveb / t_veb are None (argmax) unless set from the CV as well.
The RR model needs the 300-beat RR median, so the first beats of a stream use
a shorter window (as in training, the rolling median starts with min_periods=1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.inference.heartly_e16 import HeartlyE16, rr_features_from_peaks  # noqa: F401  (re-exported)
from src.training.decision import decide


class HeartlyE21:
    def __init__(self, cnn_checkpoints: Sequence[str | Path], rr_model_path: str | Path, w: float,
                 t_sveb: float | None = None, t_veb: float | None = None, device: str = "cpu"):
        import joblib

        self.cnn = HeartlyE16(cnn_checkpoints, device=device)
        self.rr = joblib.load(rr_model_path)
        self.w, self.t_sveb, self.t_veb = float(w), t_sveb, t_veb

    def predict_proba(self, beats: np.ndarray, rr: np.ndarray) -> np.ndarray:
        p_cnn = self.cnn.predict_proba(beats, rr)
        p_rr = self.rr.predict_proba(np.asarray(rr, dtype=np.float64))
        return (1 - self.w) * p_cnn + self.w * p_rr

    def predict(self, beats: np.ndarray, rr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prob = self.predict_proba(beats, rr)
        return decide(prob, self.t_sveb, self.t_veb), prob
