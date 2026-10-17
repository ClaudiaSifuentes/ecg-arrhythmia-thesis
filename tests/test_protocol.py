import numpy as np

from src.training.cv_protocol import DEFAULTS, deep_update, inner_split, predict_logits, train_model
from src.training.decision import decide


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    Xb = (rng.standard_normal((n, 250)) + y[:, None] * 0.3).astype(np.float32)
    Xr = rng.random((n, 6)).astype(np.float32)
    return Xb, Xr, y


import pytest


@pytest.mark.parametrize("compile_,augment", [(False, None), (True, None),
                                               (False, {"p": 0.5, "gauss_std": 0.04, "baseline_amp": 0.15,
                                                        "shift_max": 8, "scale": 0.1})])
def test_training_is_bitwise_deterministic(compile_, augment):
    cfg = deep_update(DEFAULTS, {"optim": {"max_epochs": 2, "batch_size": 64}, "compile": compile_,
                                 "augment": augment})
    Xb, Xr, y = _toy()
    outs = []
    for _ in range(2):
        m, _ = train_model(cfg, Xb[:400], Xr[:400], y[:400], Xb[400:], Xr[400:], y[400:], seed=3, log=lambda s: None)
        outs.append(predict_logits(m, Xb, Xr))
    assert np.array_equal(outs[0], outs[1])


def test_decision_rule_matches_e12_and_argmax():
    p = np.array([[0.70, 0.25, 0.05], [0.2, 0.2, 0.6], [0.5, 0.1, 0.4], [0.3, 0.3, 0.4]])
    assert decide(p).tolist() == [0, 2, 0, 2]
    assert decide(p, t_sveb=0.24).tolist() == [1, 2, 0, 1]
    # VEB gate: needs p_VEB >= 0.5, otherwise best of {N, SVEB}
    assert decide(p, t_veb=0.5).tolist() == [0, 2, 0, 0]
    # SVEB gate: a high t_sveb removes low-confidence SVEB calls
    q = np.array([[0.2, 0.45, 0.35], [0.1, 0.8, 0.1], [0.40, 0.42, 0.18]])
    assert decide(q).tolist() == [1, 1, 1]
    assert decide(q, t_sveb=0.6).tolist() == [2, 1, 0]


def test_inner_split_is_patient_disjoint():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(20).astype(str), 50)
    y = rng.integers(0, 3, len(pid))
    tr, va = inner_split(y, pid, 5, 1)
    assert set(pid[tr]).isdisjoint(set(pid[va]))
    assert len(np.unique(y[va])) == 3


def test_rr_features_from_peaks_match_training_path():
    from src.data.rr_context import build_rr_features
    from src.data.rr_features import extract_rr_features
    from src.inference.heartly_e16 import rr_features_from_peaks

    rng = np.random.default_rng(0)
    peaks = np.cumsum(rng.integers(200, 400, size=120))
    a = rr_features_from_peaks(peaks)
    b = build_rr_features(extract_rr_features(peaks), np.full(120, "p"), "context")
    assert a.shape == (120, 6) and np.array_equal(a, b)


def test_heartly_e16_ensemble_interface(tmp_path):
    import torch

    from src.inference.heartly_e16 import HeartlyE16
    from src.models.fusion import FusionClassifier

    paths = []
    for s in range(3):
        torch.manual_seed(s)
        p = tmp_path / f"seed{s}.pt"
        torch.save(FusionClassifier().state_dict(), p)
        paths.append(p)
    clf = HeartlyE16(paths)
    Xb, Xr, _ = _toy(50)
    cls, prob = clf.predict(Xb, Xr)
    assert cls.shape == (50,) and prob.shape == (50, 3)
    assert np.allclose(prob.sum(1), 1, atol=1e-5) and set(cls) <= {0, 1, 2}
