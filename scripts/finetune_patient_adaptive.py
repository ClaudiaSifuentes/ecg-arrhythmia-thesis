"""Patient-adaptive fine-tuning proof of concept for a single MIT-BIH record.

Default target is record 232, the test-set outlier: ~1,380 SVEB beats
(95.7% of all test SVEB) with an atypical LBBB + supraventricular ectopy
morphology absent from both training databases. Population-trained models
(E10, and focal-loss / no-SMOTE variants) detect essentially none of them
(SVEB sensitivity ~0.01-0.02 even when record 232 is excluded from the
denominator -- see reports/test_comparison.csv).

This script tests whether a small amount of the *same patient's* labeled
beats can recover SVEB sensitivity that the population model misses
entirely: freeze the CNN backbone (keep the learned morphology features),
fine-tune only the fusion head on a small calibration split of the record's
own beats, and evaluate on the remaining held-out beats from that patient.

This is a proof of concept, not a deployment recipe: it assumes labeled
beats for the target patient are available, which is not true at real
deployment time. Use --record to point it at any other patient (e.g. to
check whether the record-232 result generalizes as a method); see
scripts/finetune_multi_patient.py to run this across several records at once.
"""

from __future__ import annotations

import argparse

from src.training.patient_adaptive import run_finetune
from src.utils.reproducibility import set_global_seeds

from sklearn.metrics import classification_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint (e.g. E10, E11, E12)")
    ap.add_argument("--data_dir", type=str, default="data/processed/mitbih_incart")
    ap.add_argument("--record", type=str, default="232", help="Target patient record id (e.g. 232, 228, 219)")
    ap.add_argument("--calib_frac", type=float, default=0.2, help="Fraction of the record's beats used to fine-tune")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seeds(args.seed)

    result = run_finetune(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        record=args.record,
        calib_frac=args.calib_frac,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    print(f"Record {args.record}: {result['n_total']} beats total | calib={result['n_calib']} | held-out={result['n_held']}")
    print(f"Held-out SVEB count: {(result['y_held'] == 1).sum()} | VEB count: {(result['y_held'] == 2).sum()}")

    print("\n=== BEFORE fine-tuning (held-out split) ===")
    print(classification_report(result["y_held"], result["y_pred_before"], labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], zero_division=0))

    print("\n=== AFTER fine-tuning (held-out split) ===")
    print(classification_report(result["y_held"], result["y_pred_after"], labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], zero_division=0))


if __name__ == "__main__":
    main()
