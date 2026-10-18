"""E21 — hybrid ensemble: CNN probabilities (from an existing CV run) averaged
with a linear model trained ONLY on the same 6 RR inputs.

    python scripts/hybrid_rr.py --exp E20_rrmed300 --data_dir data/processed/mitbih_incartII_svdb

Why: classical inter-patient methods (de Chazal 2004, Llamedo 2011) use RR
features with a linear classifier and generalise to unseen patients better
than a CNN trained on few SVEB patients. Same model interface (beat 250 + 6 RR
-> 3 classes): the RR model sees only the 6 RR inputs the CNN already gets.

Protocol (no new information is used):
- Same outer folds and the same inner-validation patients as the CNN run
  (read from oof_{exp}.npz). For each fold the RR model is trained on the
  fold's training patients minus its inner-validation patients — exactly the
  data the CNN was trained on.
- RR model: multinomial logistic regression on [x, log x] of the 6 RR inputs
  (standardised), class weights as the focal-loss alpha of the config.
- Mixing weight w in p = (1-w)·p_CNN + w·p_RR is chosen per (seed, fold) on the
  inner-validation patients (max macro-F1 of argmax); deployment uses the
  median w. OOF patients are never used to choose anything.
Output: reports/oof/oof_E21_hybrid.npz (same format as a CV run: drop-in for
log_experiment.py) and reports/cv/E21_hybrid/hybrid.json.
Needs only X_rr.npy, y_beats.npy, patient_id.npy of the data dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.rr_context import build_rr_features  # noqa: E402
from src.data.splits import TEST_PATIENTS  # noqa: E402
from src.training.cv_protocol import _norm_pid, is_mitbih, softmax  # noqa: E402

W_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


def rr_design(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.concatenate([X, np.log(np.clip(X, 1e-3, None))], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="CNN CV experiment whose OOF/inner-val logits are reused")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_exp", default="E21_hybrid")
    ap.add_argument("--rr_model", choices=["logreg", "gbm"], default="logreg")
    args = ap.parse_args()

    d = dict(np.load(Path("reports") / "oof" / f"oof_{args.exp}.npz"))
    cfg = json.loads(str(d["config"]))
    dd = Path(args.data_dir)
    X_rr = np.load(dd / "X_rr.npy"); y_all = np.load(dd / "y_beats.npy"); pid_all = np.load(dd / "patient_id.npy").astype(str)
    # same row selection as cv_protocol.load_dev_data
    keep = ~(np.isin(_norm_pid(pid_all), list(set(TEST_PATIENTS) | set(map(str, cfg.get("exclude_patients") or []))))
             & is_mitbih(pid_all))
    R = build_rr_features(X_rr, pid_all, cfg["features"]["rr_set"])[keep]
    y, pid = y_all[keep], pid_all[keep]
    assert np.array_equal(y, d["y"]) and np.array_equal(pid, d["patient_id"].astype(str)), "data rows != OOF rows"
    Z = rr_design(R)
    fold, seeds = d["fold"], d["seeds"].tolist()
    alpha = cfg["loss"].get("alpha") or [1.0, 1.0, 1.0]
    cw = {c: float(alpha[c]) for c in range(3)}

    p_rr_oof = np.zeros((len(y), 3))
    p_rr_val = {}  # fold -> (val_idx, probs)
    for k in np.unique(fold):
        m0 = (d["val_seed"] == seeds[0]) & (d["val_fold"] == k)
        va = d["val_idx"][m0]
        tr = np.setdiff1d(np.nonzero(fold != k)[0], va)
        sc = StandardScaler().fit(Z[tr])
        if args.rr_model == "logreg":
            lr = LogisticRegression(C=1.0, class_weight=cw, max_iter=2000)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            lr = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                class_weight=cw, random_state=0)
        lr.fit(sc.transform(Z[tr]), y[tr])
        te = fold == k
        p_rr_oof[te] = lr.predict_proba(sc.transform(Z[te]))
        p_rr_val[int(k)] = (va, lr.predict_proba(sc.transform(Z[va])))
        print(f"fold {k}: RR-LR trained on {len(tr)} beats", flush=True)

    probs, chosen = [], []
    for si, s in enumerate(seeds):
        p_cnn = d["prob"][si]
        mix = np.zeros_like(p_cnn)
        for k in np.unique(fold):
            m = (d["val_seed"] == s) & (d["val_fold"] == k)
            va, pv_rr = p_rr_val[int(k)]
            assert np.array_equal(d["val_idx"][m], va)
            pv_cnn = softmax(d["val_logits"][m])
            yv = d["val_y"][m]
            w = max(W_GRID, key=lambda w: f1_score(yv, ((1 - w) * pv_cnn + w * pv_rr).argmax(1), average="macro"))
            chosen.append(float(w))
            te = fold == k
            mix[te] = (1 - w) * p_cnn[te] + w * p_rr_oof[te]
        probs.append(mix)
    prob = np.stack(probs)
    out = Path("reports") / "oof" / f"oof_{args.out_exp}.npz"
    cfg_out = dict(cfg, exp_id=args.out_exp, hybrid={"cnn_exp": args.exp, "rr_model": args.rr_model + " [x, log x]",
                                                     "w_chosen": chosen, "w_deploy": float(np.median(chosen))})
    np.savez_compressed(out, y=y, prob=prob, logits=np.log(np.clip(prob, 1e-12, None)), fold=fold,
                        patient_id=d["patient_id"], subject=d["subject"], seeds=d["seeds"],
                        val_idx=d["val_idx"], val_fold=d["val_fold"], val_seed=d["val_seed"],
                        val_logits=d["val_logits"], val_y=d["val_y"], best_epochs=d["best_epochs"],
                        train_sec=d["train_sec"], config=json.dumps(cfg_out), p_rr_oof=p_rr_oof)
    rep = Path("reports") / "cv" / args.out_exp
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "hybrid.json").write_text(json.dumps({"cnn_exp": args.exp, "w_chosen": chosen,
                                                 "w_deploy": float(np.median(chosen))}, indent=1))
    print(f"w chosen per (seed, fold): {chosen} -> deploy median {np.median(chosen):.2f}; wrote {out}")


if __name__ == "__main__":
    main()
