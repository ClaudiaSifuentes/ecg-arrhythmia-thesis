"""Post-hoc variants of a trained CV experiment (no retraining).

    python scripts/posthoc_variants.py --exp E12

Variants, each evaluated with (a) the config decision rule and (b) a nested
threshold rule (t_sveb picked per fold on the INNER validation patients,
maximising SVEB F1; t_veb kept as in config):

- base        : the experiment itself (one model per seed)
- temp        : temperature scaling, T fitted per (seed, fold) by NLL on the
                inner validation logits, applied to that fold's OOF logits
- ens2        : average of probabilities of 2 seeds (3 leave-one-out pairs
                -> 3 replicates, so a range can be reported)
- ens3        : average of the 3 seeds (single replicate, no range)

Results: reports/cv/{exp}/posthoc.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.cv_protocol import metrics_for, softmax  # noqa: E402
from src.training.decision import decide  # noqa: E402
from src.training.experiment_report import KEYS, aggregate, compare  # noqa: E402
from src.training.metrics import class_metrics  # noqa: E402

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
T_GRID = np.round(np.exp(np.linspace(np.log(0.25), np.log(4.0), 81)), 4)


def nll(logits, y, T):
    p = softmax(logits, T)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def fit_temperature(logits, y):
    return float(min(T_GRID, key=lambda T: nll(logits, y, T)))


def pick_t_sveb(pv, yv, t_veb):
    best_t, best = 0.5, -1.0
    for t in GRID:
        v = class_metrics(yv, decide(pv, float(t), t_veb))["sveb_f1"]
        if not np.isnan(v) and v > best:
            best, best_t = v, float(t)
    return best_t


def summary_from(y, fold, oof_probs, val_probs, val_y, val_fold, cfg_rule):
    """oof_probs: list over replicates of (N,3); val_probs: list over replicates of dict fold->(n,3)."""

    out = {"seeds": list(range(len(oof_probs))), "best_epochs": [], "rules": {}}
    out["rules"]["config"] = {"decision": cfg_rule,
                              "per_seed": [metrics_for(y, p, fold, cfg_rule) for p in oof_probs]}
    nested_runs, chosen = [], []
    for p, vp in zip(oof_probs, val_probs):
        pred = np.empty(len(y), dtype=int)
        for k in np.unique(fold):
            t = pick_t_sveb(vp[k], val_y[k], cfg_rule.get("t_veb"))
            chosen.append(t)
            te = fold == k
            pred[te] = decide(p[te], t, cfg_rule.get("t_veb"))
        pooled = class_metrics(y, pred, p)
        per_fold = [class_metrics(y[fold == k], pred[fold == k], p[fold == k]) for k in np.unique(fold)]
        nested_runs.append({"pooled": pooled, "per_fold": per_fold})
    out["rules"]["nested"] = {"decision": {"t_sveb": "inner-val max SVEB F1", "chosen": chosen,
                                           "t_veb": cfg_rule.get("t_veb")}, "per_seed": nested_runs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    d = dict(np.load(Path("reports") / "oof" / f"oof_{args.exp}.npz"))
    cfg = json.loads(str(d["config"]))
    rule = cfg["decision"]
    y, fold, seeds = d["y"], d["fold"], d["seeds"].tolist()
    folds = np.unique(fold)
    S = len(seeds)

    # inner-val logits per (seed, fold); inner-val patients do not depend on the seed
    val = {(s, k): d["val_logits"][(d["val_seed"] == s) & (d["val_fold"] == k)] for s in seeds for k in folds}
    val_y = {k: d["val_y"][(d["val_seed"] == seeds[0]) & (d["val_fold"] == k)] for k in folds}
    val_fold = None

    variants = {}
    # base
    variants["base"] = summary_from(y, fold, [d["prob"][i] for i in range(S)],
                                    [{k: softmax(val[(s, k)]) for k in folds} for s in seeds], val_y, val_fold, rule)
    # temperature
    temps, oof_T, val_T = [], [], []
    for i, s in enumerate(seeds):
        p = np.empty_like(d["prob"][i])
        vp = {}
        for k in folds:
            T = fit_temperature(val[(s, k)], val_y[k])
            temps.append(T)
            p[fold == k] = softmax(d["logits"][i][fold == k], T)
            vp[k] = softmax(val[(s, k)], T)
        oof_T.append(p)
        val_T.append(vp)
    variants["temp"] = summary_from(y, fold, oof_T, val_T, val_y, val_fold, rule)
    # ensembles (on temperature-scaled probabilities: averaging calibrated members)
    pairs = list(itertools.combinations(range(S), 2))
    variants["ens2"] = summary_from(y, fold, [np.mean([oof_T[a], oof_T[b]], axis=0) for a, b in pairs],
                                    [{k: (val_T[a][k] + val_T[b][k]) / 2 for k in folds} for a, b in pairs],
                                    val_y, val_fold, rule)
    variants["ens3"] = summary_from(y, fold, [np.mean(oof_T, axis=0)],
                                    [{k: np.mean([v[k] for v in val_T], axis=0) for k in folds}],
                                    val_y, val_fold, rule)

    lines = [f"# Post-hoc variants — {args.exp}", "",
             f"Temperatures fitted on inner-val (per seed×fold): {temps}", ""]
    for r in ("config", "nested"):
        lines += [f"## Rule: {r}", ""]
        for name in ("temp", "ens2", "ens3"):
            table, checks = compare(variants["base"], variants[name], f"{args.exp} ({r})", f"{name} ({r})", rule=r)
            lines += [f"### {name} vs base", "", table, "", f"checks: {checks}", ""]
    lines += ["## Base under nested threshold vs base under config rule", ""]
    a_cfg, a_nest = aggregate(variants["base"], "config"), aggregate(variants["base"], "nested")
    lines += ["| metric | config rule | nested t_sveb |", "|---|---|---|"]
    for k in KEYS:
        lines.append(f"| {k} | {a_cfg[k]['mean']:.3f} [{a_cfg[k]['min']:.3f}–{a_cfg[k]['max']:.3f}] | "
                     f"{a_nest[k]['mean']:.3f} [{a_nest[k]['min']:.3f}–{a_nest[k]['max']:.3f}] |")
    out = Path("reports") / "cv" / args.exp / "posthoc.md"
    out.write_text("\n".join(lines))
    (out.with_suffix(".json")).write_text(json.dumps(variants, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
