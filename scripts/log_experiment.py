"""Compare an experiment against a baseline from their OOF files and log it.

    python scripts/log_experiment.py --exp E13_rrctx --base E12 --rule argmax \
        --change "RR context features" [--ens 2] [--subset mit] [--no_log]

--rule   argmax | config (the experiment's config decision) | t_sveb=<x>
--ens    0: one model per seed (default). 2: mean of probabilities of every
         pair of seeds (3 replicates -> a range). 3: mean of all seeds.
--subset all | mit (MIT-BIH records only)
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

from src.training.cv_protocol import is_mitbih, metrics_for  # noqa: E402
from src.training.experiment_report import append_log, compare, format_summary  # noqa: E402


def build(exp: str, rule: str, ens: int, subset: str) -> dict:
    d = np.load(Path("reports") / "oof" / f"oof_{exp}.npz")
    cfg = json.loads(str(d["config"]))
    if rule == "argmax":
        dec = {"t_sveb": None, "t_veb": None}
    elif rule == "config":
        dec = cfg["decision"]
    elif rule.startswith("t_sveb="):
        dec = {"t_sveb": float(rule.split("=")[1]), "t_veb": None}
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
    m = np.ones(len(d["y"]), dtype=bool)
    if subset == "mit":
        m = is_mitbih(d["patient_id"])
    return {"seeds": d["seeds"].tolist(), "best_epochs": d["best_epochs"].tolist(), "n_subjects": len(set(d["subject"][m])),
            "rules": {"r": {"decision": dec, "per_seed": [metrics_for(d["y"][m], p[m], d["fold"][m], dec) for p in reps]}}}


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
