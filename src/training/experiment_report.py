"""Aggregation, baseline comparison, adoption rule and experiments_log.csv.

Conventions
-----------
- "pooled": metrics on the concatenated OOF predictions of the 5 folds for
  one seed (every dev patient predicted exactly once). Primary number.
- "fold-mean": mean over the 5 folds of per-fold metrics (what the old CV
  printed). Reported as secondary; per-fold SVEB PPV is very noisy because
  some folds hold only a few dozen SVEB beats.
- Each number is the mean over seeds with [min, max] across seeds.

Adoption rule (from the project brief), operationalised:
  For the class whose PPV is targeted (SVEB and/or VEB):
    1. mean pooled PPV improves (Δ > 0),
    2. mean pooled Se does not drop by more than 0.03 (both classes),
    3. per-fold PPV (seed-averaged) improves in the majority of folds (>= 3/5).
  A drop > 0.01 in the other class's PPV is reported as a warning, not a veto.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

KEYS = ["f1_macro", "auc_macro", "sveb_se", "sveb_ppv", "sveb_f1", "veb_se", "veb_ppv", "veb_f1"]


def aggregate(summary: dict, rule: str = "config") -> dict:
    per_seed = summary["rules"][rule]["per_seed"]
    out = {}
    for k in KEYS:
        pooled = np.array([s["pooled"].get(k, np.nan) for s in per_seed], dtype=float)
        foldmean = np.array([np.nanmean([f.get(k, np.nan) for f in s["per_fold"]]) for s in per_seed])
        per_fold = np.nanmean(np.array([[f.get(k, np.nan) for f in s["per_fold"]] for s in per_seed]), axis=0)
        out[k] = {"mean": float(np.nanmean(pooled)), "min": float(np.nanmin(pooled)), "max": float(np.nanmax(pooled)),
                  "foldmean": float(np.nanmean(foldmean)), "per_fold": per_fold.tolist()}
    out["sveb_n_per_fold"] = [f["sveb_n"] for f in per_seed[0]["per_fold"]]
    out["veb_n_per_fold"] = [f["veb_n"] for f in per_seed[0]["per_fold"]]
    return out


def fmt(a: dict, k: str) -> str:
    return f"{a[k]['mean']:.3f} [{a[k]['min']:.3f}–{a[k]['max']:.3f}]"


def format_summary(summary: dict) -> str:
    lines = []
    for rule in summary["rules"]:
        a = aggregate(summary, rule)
        lines.append(f"\n### rule={rule} {summary['rules'][rule]['decision']}  (pooled OOF, mean [min–max] over seeds)")
        lines.append("| metric | pooled | fold-mean | per fold (seed-avg) |")
        lines.append("|---|---|---|---|")
        for k in KEYS:
            pf = " / ".join(f"{v:.2f}" for v in a[k]["per_fold"])
            lines.append(f"| {k} | {fmt(a, k)} | {a[k]['foldmean']:.3f} | {pf} |")
        lines.append(f"SVEB n per fold: {a['sveb_n_per_fold']}  VEB n per fold: {a['veb_n_per_fold']}")
    lines.append(f"best epochs: {summary['best_epochs']}")
    return "\n".join(lines)


def compare(base: dict, exp: dict, base_name: str, exp_name: str, rule: str = "config",
            target: tuple[str, ...] = ("sveb", "veb")) -> tuple[str, dict]:
    a, b = aggregate(base, rule), aggregate(exp, rule)
    lines = [f"| metric | {base_name} | {exp_name} | Δ mean | folds better |", "|---|---|---|---|---|"]
    for k in KEYS:
        better = sum(y > x for x, y in zip(a[k]["per_fold"], b[k]["per_fold"]))
        lines.append(f"| {k} | {fmt(a, k)} | {fmt(b, k)} | {b[k]['mean'] - a[k]['mean']:+.3f} | "
                     f"{better}/{len(a[k]['per_fold'])} |")
    checks = {}
    for c in ("sveb", "veb"):
        checks[f"{c}_se_ok"] = b[f"{c}_se"]["mean"] - a[f"{c}_se"]["mean"] >= -0.03
        checks[f"{c}_ppv_delta"] = b[f"{c}_ppv"]["mean"] - a[f"{c}_ppv"]["mean"]
        checks[f"{c}_ppv_folds_better"] = int(sum(y > x for x, y in zip(a[f"{c}_ppv"]["per_fold"],
                                                                         b[f"{c}_ppv"]["per_fold"])))
    improved = [c for c in target if checks[f"{c}_ppv_delta"] > 0 and checks[f"{c}_ppv_folds_better"] >= 3]
    adopt = bool(improved) and checks["sveb_se_ok"] and checks["veb_se_ok"]
    warnings = [f"{c} PPV {checks[f'{c}_ppv_delta']:+.3f}" for c in ("sveb", "veb")
                if c not in improved and checks[f"{c}_ppv_delta"] < -0.01]
    checks.update({"improved_classes": improved, "adopt": adopt, "warnings": warnings})
    return "\n".join(lines), checks


LOG_FIELDS = ["timestamp", "exp_id", "change", "seeds", "rule", "n_patients", "f1_macro", "auc_macro",
              "sveb_se", "sveb_ppv", "sveb_f1", "veb_se", "veb_ppv", "veb_f1", "decision", "notes"]


def append_log(path: Path, exp_id: str, change: str, summary: dict, rule: str, decision: str,
               notes: str = "", n_patients: int | None = None) -> None:
    a = aggregate(summary, rule)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "exp_id": exp_id, "change": change,
           "seeds": " ".join(map(str, summary["seeds"])),
           "rule": str(summary["rules"][rule]["decision"]), "n_patients": n_patients or "",
           "decision": decision, "notes": notes}
    for k in KEYS:
        row[k] = fmt(a, k)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
