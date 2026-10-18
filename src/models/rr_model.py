"""RR-only classifier used by the E21 hybrid (CNN probabilities mixed with it).

Same 6 RR inputs the CNN gets (no new signal); features are [x, log x],
standardised. Class weights are the focal-loss alpha of the config.
    p_hybrid = (1 - w) · p_CNN + w · p_RR
w is chosen on validation patients (max macro-F1 of argmax), never on the
patients that are reported.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

W_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


def rr_design(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.concatenate([X, np.log(np.clip(X, 1e-3, None))], axis=1)


def make_rr_model(kind: str, alpha) -> object:
    """Unfitted sklearn pipeline: [x, log x] -> scaler -> classifier."""

    cw = {c: float(a) for c, a in enumerate(alpha or [1.0, 1.0, 1.0])}
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, class_weight=cw, max_iter=2000)
    elif kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             class_weight=cw, random_state=0)
    else:
        raise ValueError(kind)
    return make_pipeline(FunctionTransformer(rr_design), StandardScaler(), clf)


def choose_w(y_val: np.ndarray, p_cnn_val: np.ndarray, p_rr_val: np.ndarray) -> float:
    return float(max(W_GRID, key=lambda w: f1_score(y_val, ((1 - w) * p_cnn_val + w * p_rr_val).argmax(1),
                                                     average="macro")))
