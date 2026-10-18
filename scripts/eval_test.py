"""Train on the canonical train patients and report on the fixed test patients.

    python scripts/eval_test.py --config configs/experiments/E12.yaml          # train 3 seeds + test
    python scripts/eval_test.py --config configs/experiments/E12.yaml \
        --checkpoint models/checkpoints/E12_best.pt                           # evaluate an existing model
    python scripts/eval_test.py --config configs/experiments/E20_rrmed300.yaml \
        --hybrid E21_hybrid_gbm                                               # E21: CNN + RR model

Training: TRAIN_PATIENTS, early stopping on VAL_PATIENTS (as E12 was trained).
Decision: the config's rule (same as CV) and argmax for reference.
Test is reported but NEVER used to choose anything: it is small (6 patients)
and ~96 % of its SVEB beats come from record 232, so the "excl. 232" view
is always printed next to it.
--hybrid <cv exp>: also report the E21 hybrid. The RR model (kind and mixing
weight w and, if present, the decision thresholds from
reports/cv/<cv exp>/hybrid.json, i.e. chosen in CV) is trained on
the same training beats as the CNN and saved next to the CNN weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.rr_context import build_rr_features  # noqa: E402
from src.data.splits import TEST_PATIENTS, TRAIN_PATIENTS, VAL_PATIENTS  # noqa: E402
from src.data.torch_datasets import load_processed_arrays  # noqa: E402
from src.models.fusion import FusionClassifier  # noqa: E402
from src.training.cv_protocol import _norm_pid, is_mitbih, load_config, predict_logits, softmax, train_model  # noqa: E402
from src.training.decision import decide_from_config  # noqa: E402
from src.training.metrics import class_metrics  # noqa: E402


def _arrays(cfg):
    arr = load_processed_arrays(cfg["data_dir"])
    pid = _norm_pid(arr.patient_id)
    Xr = build_rr_features(arr.X_rr, arr.patient_id, cfg["features"]["rr_set"])
    sel = {n: np.isin(pid, lst) for n, lst in (("train", TRAIN_PATIENTS), ("val", VAL_PATIENTS), ("test", TEST_PATIENTS))}
    # every non-MIT-BIH source (INCART, SVDB, ...) is training-only
    sel["train"] |= ~is_mitbih(arr.patient_id)
    return arr, Xr, pid, sel


def _train(args):
    cfg, seed, out = args
    if Path(out).exists():
        return out
    arr, Xr, _, sel = _arrays(cfg)
    tr, va = sel["train"], sel["val"]
    model, info = train_model(cfg, arr.X_beats[tr], Xr[tr], arr.y[tr], arr.X_beats[va], Xr[va], arr.y[va],
                              seed=seed, log=lambda s: None)
    torch.save(model.state_dict(), out)
    Path(out).with_suffix(".json").write_text(json.dumps({k: v for k, v in info.items() if k != "history"}))
    return out


def report(cfg, prob_list, names, extra_rule=None):
    arr, _, pid, sel = _arrays(cfg)
    te = sel["test"]
    y = arr.y[te]
    keep = pid[te] != "232"
    rows = []
    for name, prob in zip(names, prob_list):
        rules = [("argmax", {}), ("config", cfg["decision"])] + ([extra_rule] if extra_rule else [])
        for rule_name, rule in rules:
            pred = decide_from_config(prob, rule)
            rows.append((name, rule_name, "full", class_metrics(y, pred, prob)))
            rows.append((name, rule_name, "excl232", class_metrics(y[keep], pred[keep], prob[keep])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--hybrid", default=None, help="CV hybrid experiment (reports/cv/<name>/hybrid.json)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    arr, Xr, _, sel = _arrays(cfg)
    te = sel["test"]

    if args.checkpoint:
        ckpts = [args.checkpoint]
        names = [Path(args.checkpoint).stem]
    else:
        out_dir = Path("models") / "final" / cfg["exp_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpts = [str(out_dir / f"seed{s}.pt") for s in cfg["seeds"]]
        todo = [(cfg, int(s), c) for s, c in zip(cfg["seeds"], ckpts)]
        if args.jobs <= 1 or torch.cuda.is_available():
            for t in todo:
                _train(t)
        else:
            import multiprocessing
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
                list(ex.map(_train, todo))
        names = [f"{cfg['exp_id']}_seed{s}" for s in cfg["seeds"]]

    probs = []
    for c in ckpts:
        m = FusionClassifier(dropout=float(cfg["model"]["dropout"]), use_rr=bool(cfg["model"]["use_rr"]))
        m.load_state_dict(torch.load(c, map_location="cpu"))
        probs.append(softmax(predict_logits(m, arr.X_beats[te], Xr[te]), T=float(cfg.get("temperature", 1.0))))
    if len(probs) > 1:
        probs.append(np.mean(probs, axis=0))
        names.append(f"{cfg['exp_id']}_ensemble")
    if args.hybrid:
        import joblib

        from src.models.rr_model import make_rr_model

        h = json.loads((Path("reports") / "cv" / args.hybrid / "hybrid.json").read_text())
        # decision (optional): thresholds = median of the per-(seed, fold) inner-val choices of the CV
        kind, w = h.get("rr_model", "gbm"), float(h["w_deploy"])
        rr_path = Path(ckpts[0]).parent / f"rr_{kind}.joblib"
        if rr_path.exists():
            rr = joblib.load(rr_path)
        else:
            tr = sel["train"]
            rr = make_rr_model(kind, cfg["loss"].get("alpha")).fit(Xr[tr], arr.y[tr])
            joblib.dump(rr, rr_path)
            rr_path.with_suffix(".json").write_text(json.dumps({"rr_model": kind, "w": w, "cv_exp": args.hybrid,
                                                                     "decision": h.get("decision")}))
        p_rr = rr.predict_proba(Xr[te])
        probs += [(1 - w) * p + w * p_rr for p in list(probs)]
        names += [f"{n}+RR{kind}(w={w:.2f})" for n in list(names)]

    extra = None
    if args.hybrid and h.get("decision"):
        extra = (f"hybrid CV {h['decision']}", h["decision"])
    rows = report(cfg, probs, names, extra)
    keys = ("f1_macro", "auc_macro", "sveb_n", "sveb_se", "sveb_ppv", "sveb_f1", "veb_n", "veb_se", "veb_ppv", "veb_f1")
    lines = ["| model | rule | subset | " + " | ".join(keys) + " |", "|" + "---|" * (3 + len(keys))]
    for name, rule, subset, m in rows:
        lines.append(f"| {name} | {rule} | {subset} | " + " | ".join(
            f"{m.get(k, float('nan')):.3f}" if isinstance(m.get(k), float) else str(m.get(k)) for k in keys) + " |")
    stem = names[0] if args.checkpoint else cfg["exp_id"]
    out = Path("reports") / "test" / f"{stem}{'_' + args.hybrid if args.hybrid else ''}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"Decision rule (config): {cfg['decision']}\n\n" + "\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
