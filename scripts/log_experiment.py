"""Compare an experiment against a baseline from their OOF files and log it.

    python scripts/log_experiment.py --exp E13_rrctx --base E12 --rule argmax \
        --change "RR context features" [--ens 2] [--subset mit] [--no_log]

--rule   argmax | config (the experiment's config decision) | t_sveb=<x>[,t_veb=<y>]
         | nested_sveb | nested_both: thresholds picked per (seed, fold) on the
         inner-validation patients (max F1 of the class; both = SVEB then VEB),
         applied to the fold. Not available with --ens.
--ens    0: one model per seed (default). 2: mean of probabilities of every
         pair of seeds (3 replicates -> a range). 3: mean of all seeds.
--subset all | mit (MIT-BIH records only) | nosvdb (subjects shared with E14) | svdb
The decision (adoptar/descartar) follows src/training/experiment_report.compare.
Appends a row to reports/experiments_log.csv and prints the comparison table.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.cv_protocol import is_mitbih, metrics_for, softmax  # noqa: E402
from src.training.decision import decide  # noqa: E402
from src.training.metrics import class_metrics  # noqa: E402
from src.training.experiment_report import append_log, compare, format_summary  # noqa: E402


GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def nested_preds(d, both: bool) -> list[np.ndarray]:
    """Per seed: predictions with t_SVEB (and t_VEB) chosen on each fold's inner-val."""

    y, fold, out = d["y"], d["fold"], []
    for si, s in enumerate(d["seeds"]):
        pred = np.empty(len(y), dtype=int)
        for k in np.unique(fold):
            m = (d["val_seed"] == s) & (d["val_fold"] == k)
            pv, yv = softmax(d["val_logits"][m]), d["val_y"][m]
            ts = max(GRID, key=lambda t: np.nan_to_num(class_metrics(yv, decide(pv, float(t)))["sveb_f1"], nan=-1))
            tv = None
            if both:
                tv = max(GRID, key=lambda t: np.nan_to_num(
                    class_metrics(yv, decide(pv, float(ts), float(t)))["veb_f1"], nan=-1))
            te = fold == k
            pred[te] = decide(d["prob"][si][te], float(ts), None if tv is None else float(tv))
        out.append(pred)
    return out


def build(exp: str, rule: str, ens: int, subset: str) -> dict:
    d = np.load(Path("reports") / "oof" / f"oof_{exp}.npz")
    cfg = json.loads(str(d["config"]))
    if rule in ("nested_sveb", "nested_both"):
        assert ens == 0, "nested rules are per seed"
        m = _subset_mask(d, subset)
        y, fold, P = d["y"][m], d["fold"][m], d["prob"]
        per_seed = []
        for si, pred in enumerate(nested_preds(d, rule == "nested_both")):
            pr, p = pred[m], P[si][m]
            per_seed.append({"pooled": class_metrics(y, pr, p),
                             "per_fold": [class_metrics(y[fold == k], pr[fold == k], p[fold == k])
                                          for k in np.unique(fold)]})
        return {"seeds": d["seeds"].tolist(), "best_epochs": d["best_epochs"].tolist(),
                "n_subjects": len(set(d["subject"][m])), "rules": {"r": {"decision": rule, "per_seed": per_seed}}}
    if rule == "argmax":
        dec = {"t_sveb": None, "t_veb": None}
    elif rule == "config":
        dec = cfg["decision"]
    elif rule.startswith("t_sveb="):  # t_sveb=<x>[,t_veb=<y>]
        kv = dict(part.split("=") for part in rule.split(","))
        dec = {"t_sveb": float(kv["t_sveb"]), "t_veb": float(kv["t_veb"]) if "t_veb" in kv else None}
    else:
        raise ValueError(rule)
    P = d["prob"]
    S = len(P)
    if ens == 0:
        reps = [P[i] for i in range(S)]
    elif ens == S:
        reps = [P.mean(axis=0)]
    else:
        reps = [P[list(c)].mean(axis=0) for c in itertools.combinations(range(S), ens)]
    m = _subset_mask(d, subset)
    return {"seeds": d["seeds"].tolist(), "best_epochs": d["best_epochs"].tolist(), "n_subjects": len(set(d["subject"][m])),
            "rules": {"r": {"decision": dec, "per_seed": [metrics_for(d["y"][m], p[m], d["fold"][m], dec) for p in reps]}}}


def _subset_mask(d, subset: str) -> np.ndarray:
    m = np.ones(len(d["y"]), dtype=bool)
    if subset == "mit":
        m = is_mitbih(d["patient_id"])
    elif subset == "nosvdb":  # the 66 subjects shared with E12–E14 (identical beats)
        m = ~np.char.startswith(d["patient_id"].astype(str), "svdb_")
    elif subset == "svdb":
        m = np.char.startswith(d["patient_id"].astype(str), "svdb_")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--base")
    ap.add_argument("--base_rule", default=None, help="defaults to --rule")
    ap.add_argument("--base_ens", type=int, default=0)
    ap.add_argument("--rule", default="argmax")
    ap.add_argument("--ens", type=int, default=0)
    ap.add_argument("--subset", default="all")
    ap.add_argument("--change", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--decision", default=None, help="override the automatic decision text")
    ap.add_argument("--no_log", action="store_true")
    args = ap.parse_args()

    exp = build(args.exp, args.rule, args.ens, args.subset)
    tag = args.exp + (f"+ens{args.ens}" if args.ens else "")
    decision = args.decision or "referencia"
    if args.base:
        base = build(args.base, args.base_rule or args.rule, args.base_ens, args.subset)
        btag = args.base + (f"+ens{args.base_ens}" if args.base_ens else "")
        table, checks = compare(base, exp, btag, tag, rule="r")
        print(f"rule={args.rule} subset={args.subset}\n{table}\n{checks}")
        if args.decision is None:
            decision = "adoptar" if checks["adopt"] else "descartar"
            decision += f" (vs {btag}; PPV↑ {checks['improved_classes']})"
    else:
        print(format_summary(exp))
    if not args.no_log:
        rule_txt = args.rule + ("" if args.subset == "all" else f", subset={args.subset}") + (f", ens{args.ens}" if args.ens else "")
        append_log(Path("reports") / "experiments_log.csv", tag, args.change, exp, "r", decision,
                   notes=f"[{rule_txt}] {args.notes}".strip(), n_patients=exp["n_subjects"])


if __name__ == "__main__":
    main()
