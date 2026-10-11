"""Train on the canonical train patients and report on the fixed test patients.

    python scripts/eval_test.py --config configs/experiments/E12.yaml          # train 3 seeds + test
    python scripts/eval_test.py --config configs/experiments/E12.yaml \
        --checkpoint models/checkpoints/E12_best.pt                           # evaluate an existing model

Training: TRAIN_PATIENTS, early stopping on VAL_PATIENTS (as E12 was trained).
Decision: the config's rule (same as CV) and argmax for reference.
Test is reported but NEVER used to choose anything: it is small (6 patients)
and ~96 % of its SVEB beats come from record 232, so the "excl. 232" view
is always printed next to it.
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
from src.training.cv_protocol import _norm_pid, load_config, predict_logits, softmax, train_model  # noqa: E402
from src.training.decision import decide_from_config  # noqa: E402
from src.training.metrics import class_metrics  # noqa: E402


def _arrays(cfg):
    arr = load_processed_arrays(cfg["data_dir"])
    pid = _norm_pid(arr.patient_id)
    Xr = build_rr_features(arr.X_rr, arr.patient_id, cfg["features"]["rr_set"])
    sel = {n: np.isin(pid, lst) for n, lst in (("train", TRAIN_PATIENTS), ("val", VAL_PATIENTS), ("test", TEST_PATIENTS))}
    if arr.patient_id[0].startswith(("mit_", "inc_")):
        sel["train"] |= np.char.startswith(arr.patient_id.astype(str), "inc_")
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


def report(cfg, prob_list, names):
    arr, _, pid, sel = _arrays(cfg)
    te = sel["test"]
    y = arr.y[te]
    keep = pid[te] != "232"
    rows = []
    for name, prob in zip(names, prob_list):
        for rule_name, rule in (("argmax", {}), ("config", cfg["decision"])):
            pred = decide_from_config(prob, rule)
            rows.append((name, rule_name, "full", class_metrics(y, pred, prob)))
            rows.append((name, rule_name, "excl232", class_metrics(y[keep], pred[keep], prob[keep])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--jobs", type=int, default=3)
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
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(_train, [(cfg, int(s), c) for s, c in zip(cfg["seeds"], ckpts)]))
        names = [f"{cfg['exp_id']}_seed{s}" for s in cfg["seeds"]]

    probs = []
    for c in ckpts:
        m = FusionClassifier(dropout=float(cfg["model"]["dropout"]), use_rr=bool(cfg["model"]["use_rr"]))
        m.load_state_dict(torch.load(c, map_location="cpu"))
        probs.append(softmax(predict_logits(m, arr.X_beats[te], Xr[te]), T=float(cfg.get("temperature", 1.0))))
    if len(probs) > 1:
        probs.append(np.mean(probs, axis=0))
        names.append(f"{cfg['exp_id']}_ensemble")

    rows = report(cfg, probs, names)
    keys = ("f1_macro", "auc_macro", "sveb_n", "sveb_se", "sveb_ppv", "sveb_f1", "veb_n", "veb_se", "veb_ppv", "veb_f1")
    lines = ["| model | rule | subset | " + " | ".join(keys) + " |", "|" + "---|" * (3 + len(keys))]
    for name, rule, subset, m in rows:
        lines.append(f"| {name} | {rule} | {subset} | " + " | ".join(
            f"{m.get(k, float('nan')):.3f}" if isinstance(m.get(k), float) else str(m.get(k)) for k in keys) + " |")
    out = Path("reports") / "test" / f"{names[0] if args.checkpoint else cfg['exp_id']}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"Decision rule (config): {cfg['decision']}\n\n" + "\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
