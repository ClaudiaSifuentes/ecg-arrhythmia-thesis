"""Deployable candidate: E16 ensemble (3 seeds), argmax decision.

Interface (unchanged from E12)
------------------------------
    input : beats (N, 250) float32 — record-z-scored, bandpassed 0.5–40 Hz,
            window [R-100, R+150) at 360 Hz
            rr    (N, 6)   float32 — the 6 RR inputs
    output: classes (N,) in {0: N, 1: SVEB, 2: VEB} and probabilities (N, 3)

What changed vs E12 for the deployment pipeline
-----------------------------------------------
1. The 6 RR inputs are the "context" set (src/data/rr_context.py):
   [RR, RR_next, RR_mean5, RR/RR_mean50, RR_next/RR, RR/RR_prev].
   RR_next needs the following R peak, so a beat is classified with a
   one-beat delay. Use `rr_features_from_peaks` below: it runs exactly the
   training code path (extract_rr_features -> build_rr_features).
2. Decision rule is plain argmax of the averaged probabilities (no SVEB
   threshold; the legacy 0.24 must be removed from the deployed service).
3. Three models are averaged (ensemble of the 3 training seeds).

Training-time difference to keep in mind: in training the 50-beat rolling RR
mean ran over annotated (AAMI-mapped) beats only; at inference it runs over
every detected beat. With few unmapped beats the difference is negligible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.data.rr_context import build_rr_features
from src.data.rr_features import extract_rr_features
from src.models.fusion import FusionClassifier

CLASSES = ("N", "SVEB", "VEB")


def rr_features_from_peaks(r_peaks: np.ndarray, fs: int = 360, rr_set: str = "context") -> np.ndarray:
    """6 RR features per detected R peak, identical to training ("context" for E16)."""

    base = extract_rr_features(np.asarray(r_peaks), fs=fs)
    return build_rr_features(base, np.full(len(base), "stream"), rr_set)


class HeartlyE16:
    def __init__(self, checkpoints: Sequence[str | Path], device: str = "cpu"):
        self.device = torch.device(device)
        self.models = []
        for c in checkpoints:
            m = FusionClassifier(dropout=0.5, use_rr=True)
            m.load_state_dict(torch.load(c, map_location="cpu"))
            m.eval().to(self.device)
            self.models.append(m)
        if not self.models:
            raise ValueError("no checkpoints")

    @torch.no_grad()
    def predict_proba(self, beats: np.ndarray, rr: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        beats = np.asarray(beats, dtype=np.float32)
        rr = np.asarray(rr, dtype=np.float32)
        assert beats.ndim == 2 and beats.shape[1] == 250, beats.shape
        assert rr.shape == (len(beats), 6), rr.shape
        out = np.zeros((len(beats), 3), dtype=np.float64)
        for i in range(0, len(beats), batch_size):
            e = torch.from_numpy(beats[i:i + batch_size]).unsqueeze(1).to(self.device)
            r = torch.from_numpy(rr[i:i + batch_size]).to(self.device)
            p = sum(torch.softmax(m(e, r), dim=1) for m in self.models) / len(self.models)
            out[i:i + batch_size] = p.cpu().numpy()
        return out

    def predict(self, beats: np.ndarray, rr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prob = self.predict_proba(beats, rr)
        return prob.argmax(axis=1), prob
