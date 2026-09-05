"""Multi-patient validation of patient-adaptive fine-tuning.

The record-232 result (Table VII: SVEB sensitivity 0.01 -> 0.97-0.99) was a
single-patient, single-session proof of concept -- exactly the kind of
result a reviewer would ask to see repeated elsewhere before trusting it as
a general method. This script repeats the same protocol (freeze CNN, fine-
tune only the fusion head on a calibration split of one patient's own
beats, evaluate on the disjoint remainder) across every validation and test
patient, and reports both a per-patient breakdown and a pooled aggregate.

Patients with very few SVEB/VEB beats will have noisy per-patient recall
(small support n) -- that is reported explicitly, not hidden, so read the
support column before trusting any single row.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from src.data.splits import TEST_PATIENTS, VAL_PATIENTS
from src.training.patient_adaptive import run_finetune
from src.utils.reproducibility import set_global_seeds

# Record 232 is excluded here -- it's already covered separately in Table VII
# with a 3-seed robustness check; this script targets the *other* val/test
# patients to test whether the method generalizes beyond that one case.
DEFAULT_RECORDS = [p for p in VAL_PATIENTS + TEST_PATIENTS if p != "232"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default="data/processed/mitbih_incart")
    ap.add_argument("--records", nargs="*", default=DEFAULT_RECORDS, help="Record ids to test (default: all val+test patients)")
    ap.add_argument("--calib_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="reports/finetune_multi_patient.csv")
    args = ap.parse_args()

    set_global_seeds(args.seed)

    rows = []
    all_y_held_before, all_pred_before = [], []
    all_y_held_after, all_pred_after = [], []

    for record in args.records:
        try:
            result = run_finetune(
                checkpoint=args.checkpoint,
                data_dir=args.data_dir,
                record=record,
                calib_frac=args.calib_frac,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
            )
        except RuntimeError as e:
            print(f"[skip] {record}: {e}")
            continue

        rb, ra = result["report_before"], result["report_after"]
        n_sveb = int(rb["SVEB"]["support"])
        n_veb = int(rb["VEB"]["support"])

        print(f"\n=== Record {record} | held-out n={result['n_held']} (SVEB={n_sveb}, VEB={n_veb}) ===")
        print(f"  SVEB sens BEFORE={rb['SVEB']['recall']:.3f} -> AFTER={ra['SVEB']['recall']:.3f} "
              f"(PPV {rb['SVEB']['precision']:.3f} -> {ra['SVEB']['precision']:.3f})")
        print(f"  N recall BEFORE={rb['N']['recall']:.3f} -> AFTER={ra['N']['recall']:.3f}")

        rows.append({
            "record": record,
            "n_held": result["n_held"],
            "n_sveb": n_sveb,
            "n_veb": n_veb,
            "sveb_sens_before": round(rb["SVEB"]["recall"], 4),
            "sveb_sens_after": round(ra["SVEB"]["recall"], 4),
            "sveb_ppv_before": round(rb["SVEB"]["precision"], 4),
            "sveb_ppv_after": round(ra["SVEB"]["precision"], 4),
            "n_recall_before": round(rb["N"]["recall"], 4),
            "n_recall_after": round(ra["N"]["recall"], 4),
            "veb_sens_before": round(rb["VEB"]["recall"], 4),
            "veb_sens_after": round(ra["VEB"]["recall"], 4),
        })

        all_y_held_before.append(result["y_held"])
        all_pred_before.append(result["y_pred_before"])
        all_y_held_after.append(result["y_held"])
        all_pred_after.append(result["y_pred_after"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-record breakdown to {out_path}")

    if all_y_held_before:
        y_before = np.concatenate(all_y_held_before)
        p_before = np.concatenate(all_pred_before)
        y_after = np.concatenate(all_y_held_after)
        p_after = np.concatenate(all_pred_after)

        print(f"\n{'=' * 70}\nPOOLED ACROSS {len(rows)} RECORDS (n={len(y_after)} held-out beats)\n{'=' * 70}")
        print("--- BEFORE fine-tuning ---")
        print(classification_report(y_before, p_before, labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], zero_division=0))
        print("--- AFTER fine-tuning (each record fine-tuned independently) ---")
        print(classification_report(y_after, p_after, labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], zero_division=0))


if __name__ == "__main__":
    main()
