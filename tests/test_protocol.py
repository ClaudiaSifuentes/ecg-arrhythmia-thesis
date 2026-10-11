import numpy as np

from src.training.cv_protocol import DEFAULTS, deep_update, inner_split, predict_logits, train_model
from src.training.decision import decide


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    Xb = (rng.standard_normal((n, 250)) + y[:, None] * 0.3).astype(np.float32)
    Xr = rng.random((n, 6)).astype(np.float32)
    return Xb, Xr, y


def test_training_is_bitwise_deterministic():
    cfg = deep_update(DEFAULTS, {"optim": {"max_epochs": 2, "batch_size": 64}})
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


def test_inner_split_is_patient_disjoint():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(20).astype(str), 50)
    y = rng.integers(0, 3, len(pid))
    tr, va = inner_split(y, pid, 5, 1)
    assert set(pid[tr]).isdisjoint(set(pid[va]))
    assert len(np.unique(y[va])) == 3
