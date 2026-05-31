"""Patient-wise stratified 5-fold cross-validation (Week 3).

Design decisions
----------------
- StratifiedGroupKFold: preserves class distribution per fold while guaranteeing
  each patient (group) appears in exactly one fold — never shared between train/val.
- SMOTE applied INSIDE each fold, train split only — never touches val data.
- Fold isolation verified at runtime with an explicit assert.
- Model config mirrors E06 (current best): lr=5e-4, dropout=0.5, batch=256, wd=1e-4.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.data.torch_datasets import (
    ECGBeatDataset,
    SplitArrays,
    apply_smote_train_only,
    load_processed_arrays,
)
from src.models.fusion import FusionClassifier
from src.training.metrics import compute_metrics
from src.training.trainer import evaluate, fit
from src.utils.reproducibility import set_global_seeds


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    best_epoch: int
    best_val_loss: float
    f1_macro: float
    recall_veb: float
    specificity_veb: float
    auc_macro: float


def cross_validate_patient_wise(
    data_dir: str | Path,
    *,
    n_splits: int = 5,
    lr: float = 5e-4,
    dropout: float = 0.5,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    max_epochs: int = 80,
    patience: int = 15,
    seed: int = 42,
    use_rr: bool = True,
    device: Optional[torch.device] = None,
    log_path: str | Path = "reports/cv_results.csv",
    exclude_patients: list[str] = None,
    notes: str = "",
) -> list[FoldResult]:
    """Run patient-wise stratified k-fold cross-validation.

    Parameters
    ----------
    data_dir:
        Path to processed .npy arrays (X_beats, X_rr, y_beats, patient_id).
    n_splits:
        Number of CV folds (default 5).
    lr, dropout, batch_size, weight_decay:
        Optimizer / model config — mirrors E06 defaults.
    max_epochs, patience:
        Training budget per fold.
    seed:
        Master seed for reproducibility (SMOTE, DataLoader shuffle, model init).
    use_rr:
        Whether to use RR features in FusionClassifier (False for CNN-only ablation).
    device:
        Torch device. Auto-detected (CUDA > CPU) if None.
    log_path:
        CSV path for per-fold results.
    exclude_patients:
        List of patient IDs to exclude from training/evaluation (for ablation studies).
    notes:
        Optional notes for this CV run (saved to log file).

    Returns
    -------
    list[FoldResult], one entry per fold.
    Summary statistics (mean ± std) printed to stdout.
    """
    set_global_seeds(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load all data ---
    arr = load_processed_arrays(data_dir)
    X_beats = arr.X_beats    # (N, 250)
    X_rr = arr.X_rr          # (N, 6)
    y = arr.y                # (N,)
    patient_ids = arr.patient_id  # (N,) string patient IDs

    # --- Exclude patients if requested ---
    if exclude_patients:
        mask = ~np.isin(patient_ids, exclude_patients)
        X_beats = X_beats[mask]
        X_rr = X_rr[mask]
        y = y[mask]
        patient_ids = patient_ids[mask]

    # --- CV splitter ---
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # --- Prepare log CSV ---
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    _fields = list(asdict(FoldResult(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)).keys())
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_fields).writeheader()

    criterion = nn.CrossEntropyLoss()
    fold_results: list[FoldResult] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(X_beats, y, groups=patient_ids), start=1
    ):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{n_splits}  |  device={device}")
        print(f"{'='*60}")

        # --- Patient isolation assert ---
        train_pids = set(patient_ids[train_idx].tolist())
        val_pids = set(patient_ids[val_idx].tolist())
        assert train_pids.isdisjoint(val_pids), (
            f"Fold {fold_idx}: patient overlap detected — "
            f"shared patients: {train_pids & val_pids}"
        )

        print(f"Train: {len(train_idx):>6} beats | {len(train_pids):>2} patients")
        print(f"Val:   {len(val_idx):>6} beats | {len(val_pids):>2} patients")

        # --- Build SplitArrays for this fold ---
        train_split = SplitArrays(
            X_beats=X_beats[train_idx],
            X_rr=X_rr[train_idx],
            y=y[train_idx],
            patient_id=patient_ids[train_idx],
        )
        val_split = SplitArrays(
            X_beats=X_beats[val_idx],
            X_rr=X_rr[val_idx],
            y=y[val_idx],
            patient_id=patient_ids[val_idx],
        )

        # --- SMOTE only on train ---
        train_sm = apply_smote_train_only(train_split, seed=seed)

        # --- DataLoaders ---
        g = torch.Generator()
        g.manual_seed(seed + fold_idx)

        train_loader = DataLoader(
            ECGBeatDataset(train_sm.X_beats, train_sm.X_rr, train_sm.y),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            ECGBeatDataset(val_split.X_beats, val_split.X_rr, val_split.y),
            batch_size=max(batch_size, 256),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        # --- Model (fresh init per fold) ---
        set_global_seeds(seed)
        model = FusionClassifier(dropout=dropout, use_rr=use_rr)

        ckpt_path = f"models/checkpoints/cv_fold{fold_idx}_best.pt"
        train_log = f"reports/cv_fold{fold_idx}_training_log.csv"

        model = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            checkpoint_path=ckpt_path,
            log_path=train_log,
            device=device,
        )

        # --- Evaluate best checkpoint on val ---
        val_m = evaluate(model, val_loader, criterion, device)

        # --- Read best epoch/loss from per-fold training log ---
        import pandas as pd

        df_log = pd.read_csv(train_log)
        best_row = df_log.loc[df_log["val_loss"].idxmin()]
        best_epoch = int(best_row["epoch"])
        best_val_loss = float(best_row["val_loss"])

        result = FoldResult(
            fold=fold_idx,
            n_train=len(train_idx),
            n_val=len(val_idx),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            f1_macro=float(val_m["f1_macro"]),
            recall_veb=float(val_m["recall_veb"]),
            specificity_veb=float(val_m["specificity_veb"]),
            auc_macro=float(val_m["auc_macro"])
            if not np.isnan(val_m["auc_macro"])
            else float("nan"),
        )
        fold_results.append(result)

        print(
            f"\nFold {fold_idx} — F1_macro={result.f1_macro:.4f} | "
            f"Recall_VEB={result.recall_veb:.4f} | epoch={best_epoch}"
        )
        print(val_m["report_str"])

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_fields).writerow(asdict(result))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY (mean ± std)")
    print("=" * 60)

    f1s = [r.f1_macro for r in fold_results]
    recalls = [r.recall_veb for r in fold_results]
    specs = [r.specificity_veb for r in fold_results]
    aucs = [r.auc_macro for r in fold_results if not np.isnan(r.auc_macro)]

    print(f"F1_macro:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Recall_VEB:      {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"Specificity_VEB: {np.mean(specs):.4f} ± {np.std(specs):.4f}")
    if aucs:
        print(f"AUC_macro:       {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    return fold_results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/mitbih_aami3")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_rr", dest="use_rr", action="store_true", default=True)
    ap.add_argument("--no_rr", dest="use_rr", action="store_false")
    ap.add_argument("--log_path", type=str, default="reports/cv_results.csv")
    ap.add_argument("--exclude_patients", type=str, default=None, help="Comma-separated patient IDs to exclude (e.g., '232,104')")
    ap.add_argument("--notes", type=str, default="", help="Optional notes for this run")
    args = ap.parse_args()

    exclude_patients = []
    if args.exclude_patients:
        exclude_patients = [p.strip() for p in args.exclude_patients.split(",")]

    cross_validate_patient_wise(
        data_dir=args.data_dir,
        n_splits=args.n_splits,
        lr=args.lr,
        dropout=args.dropout,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        use_rr=args.use_rr,
        log_path=args.log_path,
        exclude_patients=exclude_patients,
        notes=args.notes,
    )
