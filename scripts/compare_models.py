"""Model comparison: E10 (CNN Fusion) vs SVM, RF, XGBoost.

Reproducible baseline comparison with identical data splits and preprocessing.
Metrics: Accuracy, F1_macro, Recall_VEB, Precision_VEB, Specificity_VEB.
Results saved to reports/model_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, roc_auc_score
)

from src.data.torch_datasets import build_dataloaders, load_processed_arrays, make_patient_wise_splits
from src.data.splits import get_mitbih_splits
from src.models.fusion import FusionClassifier
from src.training.metrics import compute_metrics
from src.utils.reproducibility import set_global_seeds


def _ensure_comparison_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="") as f:
        fieldnames = [
            "timestamp", "model", "accuracy", "f1_macro", "recall_veb",
            "precision_veb", "specificity_veb", "roc_auc", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _load_e10_model(checkpoint_path: Path | str, device: str = "cpu") -> nn.Module:
    """Load E10 checkpoint (CNN Fusion)."""
    model = FusionClassifier(dropout=0.5)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both direct state_dict and checkpoint wrapper
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def _get_predictions_cnn(model: nn.Module, X_beats: np.ndarray, X_rr: np.ndarray,
                         batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """Get predictions from CNN model."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_beats), batch_size):
            b_end = min(i + batch_size, len(X_beats))
            beats = torch.from_numpy(X_beats[i:b_end]).unsqueeze(1).to(device)  # (B, 1, 250)
            rr = torch.from_numpy(X_rr[i:b_end]).to(device)  # (B, 6)
            
            logits = model(beats, rr)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    
    return np.concatenate(all_preds)


def _compute_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute clinical metrics (VEB-centric)."""
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    # VEB-specific metrics (class 2)
    recall_veb = float(recall_score(y_true, y_pred, labels=[2], average="micro", zero_division=0))
    precision_veb = float(precision_score(y_true, y_pred, labels=[2], average="micro", zero_division=0))
    
    # Specificity for VEB (true negative rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred != 2, labels=[True, False]).ravel()
    specificity_veb = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    
    # ROC-AUC (one-vs-rest, macro average)
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro", zero_division=0))
    except:
        roc_auc = 0.0
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "recall_veb": recall_veb,
        "precision_veb": precision_veb,
        "specificity_veb": specificity_veb,
        "roc_auc": roc_auc,
    }


def compare_models(
    data_dir: str = "data/processed/mitbih_incart",
    checkpoint_e10: str = "models/E10_best.pt",
    output_csv: str = "reports/model_comparison.csv",
    seed: int = 42,
    device: str = "cpu",
) -> None:
    """Compare E10 (CNN) vs SVM, RF, XGBoost on test set."""
    
    set_global_seeds(seed)
    _ensure_comparison_log(Path(output_csv))
    
    print("=" * 80)
    print("MODEL COMPARISON: E10 (CNN) vs Baselines")
    print("=" * 80)
    
    # Load data with reproducible splits
    print("\n[1/5] Loading data with patient-wise splits...")
    arr = load_processed_arrays(data_dir)
    splits = make_patient_wise_splits(arr, get_mitbih_splits())
    
    X_train_beats = splits["train"].X_beats
    X_train_rr = splits["train"].X_rr
    y_train = splits["train"].y
    
    X_test_beats = splits["test"].X_beats
    X_test_rr = splits["test"].X_rr
    y_test = splits["test"].y
    
    print(f"  Train: {len(y_train)} beats")
    print(f"  Test:  {len(y_test)} beats")
    print(f"  Classes: {np.unique(y_test)}")
    
    # Concatenate beats + RR features for sklearn models
    X_train_combined = np.concatenate([X_train_beats, X_train_rr], axis=1)
    X_test_combined = np.concatenate([X_test_beats, X_test_rr], axis=1)
    
    # Standardize features for sklearn
    scaler = StandardScaler()
    X_train_combined = scaler.fit_transform(X_train_combined)
    X_test_combined = scaler.transform(X_test_combined)
    
    # === E10 (CNN Fusion) ===
    print("\n[2/5] Evaluating E10 (CNN Fusion)...")
    try:
        model_e10 = _load_e10_model(checkpoint_e10, device=device)
        y_pred_e10 = _get_predictions_cnn(model_e10, X_test_beats, X_test_rr, device=device)
        metrics_e10 = _compute_clinical_metrics(y_test, y_pred_e10)
        print(f"  ✓ E10 loaded and evaluated")
        print(f"    F1_macro: {metrics_e10['f1_macro']:.4f}, Recall_VEB: {metrics_e10['recall_veb']:.4f}")
    except Exception as e:
        print(f"  ✗ E10 failed: {e}")
        metrics_e10 = {k: 0.0 for k in ["accuracy", "f1_macro", "recall_veb", "precision_veb", "specificity_veb", "roc_auc"]}
    
    # === SVM ===
    print("\n[3/5] Training SVM...")
    try:
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed, verbose=0)
        svm.fit(X_train_combined, y_train)
        y_pred_svm = svm.predict(X_test_combined)
        metrics_svm = _compute_clinical_metrics(y_test, y_pred_svm)
        print(f"  ✓ SVM trained")
        print(f"    F1_macro: {metrics_svm['f1_macro']:.4f}, Recall_VEB: {metrics_svm['recall_veb']:.4f}")
    except Exception as e:
        print(f"  ✗ SVM failed: {e}")
        metrics_svm = {k: 0.0 for k in ["accuracy", "f1_macro", "recall_veb", "precision_veb", "specificity_veb", "roc_auc"]}
    
    # === Random Forest ===
    print("\n[4/5] Training Random Forest...")
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1, verbose=0)
        rf.fit(X_train_combined, y_train)
        y_pred_rf = rf.predict(X_test_combined)
        metrics_rf = _compute_clinical_metrics(y_test, y_pred_rf)
        print(f"  ✓ Random Forest trained")
        print(f"    F1_macro: {metrics_rf['f1_macro']:.4f}, Recall_VEB: {metrics_rf['recall_veb']:.4f}")
    except Exception as e:
        print(f"  ✗ Random Forest failed: {e}")
        metrics_rf = {k: 0.0 for k in ["accuracy", "f1_macro", "recall_veb", "precision_veb", "specificity_veb", "roc_auc"]}
    
    # === XGBoost ===
    print("\n[5/5] Training XGBoost...")
    try:
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            verbosity=0,
            eval_metric="mlogloss"
        )
        xgb.fit(X_train_combined, y_train, verbose=False)
        y_pred_xgb = xgb.predict(X_test_combined)
        metrics_xgb = _compute_clinical_metrics(y_test, y_pred_xgb)
        print(f"  ✓ XGBoost trained")
        print(f"    F1_macro: {metrics_xgb['f1_macro']:.4f}, Recall_VEB: {metrics_xgb['recall_veb']:.4f}")
    except Exception as e:
        print(f"  ✗ XGBoost failed: {e}")
        metrics_xgb = {k: 0.0 for k in ["accuracy", "f1_macro", "recall_veb", "precision_veb", "specificity_veb", "roc_auc"]}
    
    # === Log results ===
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    results = [
        ("E10 (CNN Fusion)", metrics_e10),
        ("SVM", metrics_svm),
        ("Random Forest", metrics_rf),
        ("XGBoost", metrics_xgb),
    ]
    
    timestamp = datetime.now().isoformat()
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "model", "accuracy", "f1_macro", "recall_veb",
            "precision_veb", "specificity_veb", "roc_auc", "notes"
        ])
        
        for model_name, metrics in results:
            row = {
                "timestamp": timestamp,
                "model": model_name,
                **{k: f"{v:.6f}" for k, v in metrics.items()},
                "notes": "mitbih_incart_test_split",
            }
            writer.writerow(row)
            
            # Print table row
            print(f"\n{model_name:20} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f} | "
                  f"Recall_VEB: {metrics['recall_veb']:.4f} | Precision_VEB: {metrics['precision_veb']:.4f}")
    
    print("\n✅ Results saved to:", output_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare E10 (CNN) vs baseline models")
    ap.add_argument("--data_dir", type=str, default="data/processed/mitbih_incart",
                    help="Path to processed dataset")
    ap.add_argument("--checkpoint", type=str, default="models/E10_best.pt",
                    help="Path to E10 checkpoint")
    ap.add_argument("--output_csv", type=str, default="reports/model_comparison.csv",
                    help="Output CSV file")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = ap.parse_args()
    
    compare_models(
        data_dir=args.data_dir,
        checkpoint_e10=args.checkpoint,
        output_csv=args.output_csv,
        seed=args.seed,
        device=args.device,
    )
