"""Single decision rule shared by CV, test evaluation and inference.

The model outputs softmax probabilities p = [p_N, p_SVEB, p_VEB]. The class
decision is NOT plain argmax in the deployed E12 setup: an SVEB threshold
(0.24) was tuned on validation. Before this module existed, CV used argmax
while the test report used the threshold, so CV and test numbers were not
comparable. Every evaluation path must now call `decide()` with thresholds
read from the experiment config.

Rule (applied in this order)
----------------------------
1. pred = argmax(p)
2. if t_sveb is set:  pred[p_SVEB > t_sveb] = SVEB          (E12 legacy rule)
                      beats predicted SVEB with p_SVEB <= t_sveb fall back to
                      the best of {N, VEB} (only possible for t_sveb >= 1/3)
3. if t_veb is set:   pred[p_VEB >= t_veb]  = VEB
                      beats predicted VEB with p_VEB < t_veb fall back to the
                      argmax of {N, SVEB}
With t_veb=None the rule is exactly the E12 rule; with both None it is argmax.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def decide(prob: np.ndarray, t_sveb: Optional[float] = None, t_veb: Optional[float] = None) -> np.ndarray:
    prob = np.asarray(prob)
    pred = prob.argmax(axis=1)
    if t_sveb is not None:
        t = float(t_sveb)
        # gate: argmax-SVEB beats with p_SVEB <= t fall back to the best of {N, VEB}
        # (never triggers for t < 1/3, so the legacy 0.24 rule is unchanged)
        demote = (pred == 1) & (prob[:, 1] <= t)
        pred[demote] = np.where(prob[demote, 0] >= prob[demote, 2], 0, 2)
        pred[prob[:, 1] > t] = 1
    if t_veb is not None:
        t = float(t_veb)
        demote = (pred == 2) & (prob[:, 2] < t)
        pred[demote] = prob[demote, :2].argmax(axis=1)
        pred[prob[:, 2] >= t] = 2
    return pred


def decide_from_config(prob: np.ndarray, decision_cfg: Optional[dict]) -> np.ndarray:
    cfg = decision_cfg or {}
    return decide(prob, cfg.get("t_sveb"), cfg.get("t_veb"))
