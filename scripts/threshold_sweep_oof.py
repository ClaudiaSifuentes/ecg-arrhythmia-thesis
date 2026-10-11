"""Threshold sweep and operating point selection on OOF outputs (never test).

    python scripts/threshold_sweep_oof.py --exp E12 [--ensemble]

Outputs
-------
reports/thresholds/{exp}.md        Se/PPV/F1 per threshold (mean [min–max] over seeds)
reports/thresholds/{exp}_pr.png    PR curves (one-vs-rest) for SVEB and VEB, per seed
Nested estimate: per fold, the threshold is picked on that fold's INNER
validation patients and applied to the fold's held-out patients, so the
reported number is not optimistically biased by picking on the same data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.cv_protocol import softmax  # noqa: E402
from src.training.decision import decide  # noqa: E402
from src.training.metrics import class_metrics  # noqa: E402

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def sweep(y, probs, cls: str, base_rule: dict):
    """probs: (S, N, 3). Returns rows of (t, {metric: (mean,min,max)})."""

    rows = []
    for t in GRID:
        rule = dict(base_rule)
        rule["t_sveb" if cls == "sveb" else "t_veb"] = float(t)
        ms = [class_metrics(y, decide(p, rule["t_sveb"], rule["t_veb"])) for p in probs]
        rows.append((float(t), {k: (np.nanmean([m[k] for m in ms]), np.nanmin([m[k] for m in ms]),
                                    np.nanmax([m[k] for m in ms]))
                                for k in ("f1_macro", "sveb_se", "sveb_ppv", "sveb_f1", "veb_se", "veb_ppv", "veb_f1")}))
    return rows


def nested(d, cls: str, base_rule: dict, objective: str = "f1"):
    """Pick t per (seed, fold) on inner-val, apply to the fold. Returns pooled metrics per seed."""

    y, fold = d["y"], d["fold"]
    out = []
    chosen = []
    for si, s in enumerate(d["seeds"]):
        pred = np.empty(len(y), dtype=int)
        for k in np.unique(fold):
            m = (d["val_seed"] == s) & (d["val_fold"] == k)
            pv, yv = softmax(d["val_logits"][m]), d["val_y"][m]
            best_t, best = None, -1
            for t in GRID:
                rule = dict(base_rule)
                rule["t_sveb" if cls == "sveb" else "t_veb"] = float(t)
                v = class_metrics(yv, decide(pv, rule["t_sveb"], rule["t_veb"]))[f"{cls}_{objective}"]
                if not np.isnan(v) and v > best:
                    best, best_t = v, float(t)
            rule = dict(base_rule)
            rule["t_sveb" if cls == "sveb" else "t_veb"] = best_t
            te = fold == k
            pred[te] = decide(d["prob"][si][te], rule["t_sveb"], rule["t_veb"])
            chosen.append(best_t)
        out.append(class_metrics(y, pred))
    return out, chosen


def fmt(v):
    return f"{v[0]:.3f} [{v[1]:.3f}–{v[2]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ensemble", action="store_true", help="average probabilities over seeds first")
    args = ap.parse_args()

    d = dict(np.load(Path("reports") / "oof" / f"oof_{args.exp}.npz"))
    cfg = json.loads(str(d["config"]))
    y = d["y"]
    probs = d["prob"]
    tag = args.exp
    if args.ensemble:
        probs = probs.mean(axis=0, keepdims=True)
        tag += "_ens"
    out_dir = Path("reports") / "thresholds"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_rule = cfg["decision"]
    lines = [f"# Threshold sweep on OOF — {tag}", "",
             f"Seeds: {d['seeds'].tolist()}  |  dev beats: {len(y)}  |  SVEB n={int((y == 1).sum())}  VEB n={int((y == 2).sum())}",
             f"Reference rule (config): {ref_rule}", ""]

    # PR curves + AP
    from sklearn.metrics import average_precision_score, precision_recall_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (c, name) in zip(axes, ((1, "SVEB"), (2, "VEB"))):
        aps = []
        for si, p in enumerate(probs):
            pr, rc, _ = precision_recall_curve(y == c, p[:, c])
            aps.append(average_precision_score(y == c, p[:, c]))
            ax.plot(rc, pr, lw=1.2, label=f"seed {si} AP={aps[-1]:.3f}")
        ax.set_xlabel("Se (recall)"); ax.set_ylabel("PPV (precision)"); ax.set_title(f"{name} — OOF {tag}")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        lines.append(f"- AP {name}: {np.mean(aps):.3f} [{np.min(aps):.3f}–{np.max(aps):.3f}]")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_pr.png", dpi=120); plt.close(fig)
    lines.append("")

    for cls, other in (("sveb", {"t_veb": ref_rule.get("t_veb")}), ("veb", {"t_sveb": ref_rule.get("t_sveb")})):
        base_rule = {"t_sveb": ref_rule.get("t_sveb"), "t_veb": ref_rule.get("t_veb")}
        rows = sweep(y, probs, cls, base_rule)
        lines += [f"## {cls.upper()} threshold sweep (other threshold fixed at {other})", "",
                  "| t | F1-macro | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | VEB F1 |",
                  "|---|---|---|---|---|---|---|---|"]
        for t, m in rows:
            if (round(t * 100) % 5 == 0) or t in (0.24,):
                lines.append(f"| {t:.2f} | " + " | ".join(fmt(m[k]) for k in
                             ("f1_macro", "sveb_se", "sveb_ppv", "sveb_f1", "veb_se", "veb_ppv", "veb_f1")) + " |")
        best_t, best_m = max(rows, key=lambda r: r[1][f"{cls}_f1"][0])
        lines += ["", f"**Max mean {cls.upper()} F1 on OOF:** t={best_t:.2f} → Se {fmt(best_m[f'{cls}_se'])}, "
                  f"PPV {fmt(best_m[f'{cls}_ppv'])}, F1 {fmt(best_m[f'{cls}_f1'])}"]
        if probs.shape[0] > 0 and "val_logits" in d and not args.ensemble:
            ms, chosen = nested(d, cls, base_rule)
            agg = {k: (np.nanmean([m[k] for m in ms]), np.nanmin([m[k] for m in ms]), np.nanmax([m[k] for m in ms]))
                   for k in (f"{cls}_se", f"{cls}_ppv", f"{cls}_f1", "f1_macro")}
            lines += [f"**Nested (t picked on inner-val per fold, max {cls.upper()} F1):** chosen t per (seed,fold) "
                      f"{chosen} → Se {fmt(agg[f'{cls}_se'])}, PPV {fmt(agg[f'{cls}_ppv'])}, "
                      f"F1 {fmt(agg[f'{cls}_f1'])}, F1-macro {fmt(agg['f1_macro'])}"]
        lines.append("")
    (out_dir / f"{tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
