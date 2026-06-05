"""Ablation study: E10 vs baselines using ONLY X_rr (6 RR features).

Compares with compare_models.py which uses combined X_beats + X_rr (256 features).
Results saved to reports/model_comparison_ablation_rr_only.csv
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
    roc_auc_score
)

from src.data.torch_datasets import load_processed_arrays, make_patient_wise_splits
from src.data.splits import get_mitbih_splits
from src.models.fusion import FusionClassifier
from src.utils.reproducibility import set_global_seeds


def _ensure_comparison_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="") as f:
        fieldnames = [
            "timestamp", "model", "feature_set", "accuracy", "f1_macro", "recall_veb",
            "precision_veb", "specificity_veb", "roc_auc", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _load_e10_model(checkpoint_path: Path | str, device: str = "cpu") -> nn.Module:
    """Load E10 checkpoint (CNN Fusion)."""
    model = FusionClassifier(dropout=0.5)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def _get_predictions_cnn(model: nn.Module, X_beats: np.ndarray, X_rr: np.ndarray,
                         batch_size: int = 256, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Get hard predictions and probability scores from CNN model."""
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X_beats), batch_size):
            b_end = min(i + batch_size, len(X_beats))
            beats = torch.from_numpy(X_beats[i:b_end]).unsqueeze(1).to(device)
            rr = torch.from_numpy(X_rr[i:b_end]).to(device)
            
            logits = model(beats, rr)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_probs)


def _compute_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> Dict[str, float]:
    """Compute clinical metrics (VEB-centric)."""
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    recall_veb = float(recall_score(y_true, y_pred, labels=[2], average="micro", zero_division=0))
    precision_veb = float(precision_score(y_true, y_pred, labels=[2], average="micro", zero_division=0))
    
    y_true_binary = (y_true == 2).astype(int)
    y_pred_binary = (y_pred == 2).astype(int)
    tn = int(((y_true_binary == 0) & (y_pred_binary == 0)).sum())
    fp = int(((y_true_binary == 0) & (y_pred_binary == 1)).sum())
    specificity_veb = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    roc_auc = 0.0
    if y_score is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except Exception as e:
            print(f"    [Warning] ROC-AUC computation failed: {e}")
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "recall_veb": recall_veb,
        "precision_veb": precision_veb,
        "specificity_veb": specificity_veb,
        "roc_auc": roc_auc,
    }


def compare_models_ablation(
    data_dir: str = "data/processed/mitbih_incart",
    checkpoint_e10: str = "models/checkpoints/E10_best.pt",
    output_csv: str = "reports/model_comparison_ablation_rr_only.csv",
    seed: int = 42,
    device: str = "cpu",
) -> None:
    """Compare E10 (CNN, uses X_beats+X_rr) vs baselines using ONLY X_rr (6 features)."""
    
    set_global_seeds(seed)
    _ensure_comparison_log(Path(output_csv))
    
    print("=" * 80)
    print("ABLATION STUDY: E10 (full) vs Baselines (X_rr only)")
    print("=" * 80)
    
    # Load data
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
    
    # For ablation: use ONLY RR features (6 features)
    X_train_rr_only = X_train_rr
    X_test_rr_only = X_test_rr
    
    scaler = StandardScaler()
    X_train_rr_scaled = scaler.fit_transform(X_train_rr_only)
    X_test_rr_scaled = scaler.transform(X_test_rr_only)
    
    # === E10 (CNN Fusion) — uses X_beats + X_rr ===
    print("\n[2/5] Evaluating E10 (CNN Fusion, uses X_beats + X_rr)...")
    try:
        model_e10 = _load_e10_model(checkpoint_e10, device=device)
        y_pred_e10, y_score_e10 = _get_predictions_cnn(model_e10, X_test_beats, X_test_rr, device=device)
        metrics_e10 = _compute_clinical_metrics(y_test, y_pred_e10, y_score_e10)
        print(f"  ✓ E10 loaded and evaluated (using 250 beats + 6 RR features)")
        print(f"    F1_macro: {metrics_e10['f1_macro']:.4f}, Recall_VEB: {metrics_e10['recall_veb']:.4f}")
    except Exception as e:
        print(f"  ✗ E10 failed: {e}")
        metrics_e10 = {k: 0.0 for k in ["accuracy", "f1_macro", "recall_veb", "precision_veb", "specificity_veb", "roc_auc"]}
    
    # === SVM (X_rr only) ===
    print("\n[3/5] Training SVM (X_rr only, 6 features)...")
    try:
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed, verbose=0, probability=True)
        svm.fit(X_train_rr_scaled, y_train)
        y_pred_svm = svm.predict(X_test_rr_scaled)
        y_score_svm = svm.predict_proba(X_test_rr_scaled)
        metrics_svm = _compute_clinical_metrics(y_test, y_pred_svm, y_score_svm)
        print(f"  ✓ SVM trained")
        print(f"    F1_macro: {metrics_svm['f1_macro']:.4f}, Recall_VEB: {metrics_svm['recall_veb']:.4f}")
    except Exception as e:
        print(f"  ✗ SVM failed: {e}")
        metrics_svm = {k: 0.0 for k in ["accuracy", "f1_macro", "recall_veb", "precision_veb", "specificity_veb", "roc_auc"]}
    
    # === Random Forest (X_rr only) ===
    print("\n[4/5] Training Random Forest (X_rr only, 6 features)...")
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1, verbose=0)
        rf.fit(X_train_rr_scaled, y_train)
        y_pred_rf = rf.predict(X_test_rr_scaled)
        y_score_rf = rf.predict_proba(X_test_rr_scaled)
        metrics_rf = _compute_clinical_metrics(y_test, y_pred_rf, y_score_rf)
        print(f"  ✓ Random Forest trained")
        print(f"    F1_macro: {metrics_rf['f1_macro']:.4f}, Recall_VEB: {metrics_rf['recall_veb']:.4f}")
    except Exception as e:
        print(f"  ✗ Random Forest failed: {e}")
        metrics_rf = {k: 0.0 for k in ["accuracy", "f1_macro", "recall_veb", "precision_veb", "specificity_veb", "roc_auc"]}
    
    # === XGBoost (X_rr only) ===
    print("\n[5/5] Training XGBoost (X_rr only, 6 features)...")
    try:
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            verbosity=0,
            eval_metric="mlogloss"
        )
        xgb.fit(X_train_rr_scaled, y_train, verbose=False)
        y_pred_xgb = xgb.predict(X_test_rr_scaled)
        y_score_xgb = xgb.predict_proba(X_test_rr_scaled)
        metrics_xgb = _compute_clinical_metrics(y_test, y_pred_xgb, y_score_xgb)
        print(f"  ✓ XGBoost trained")
        print(f"    F1_macro: {metrics_xgb['f1_macro']:.4f}, Recall_VEB: {metrics_xgb['recall_veb']:.4f}")
    except Exception as e:
        print(f"  ✗ XGBoost failed: {e}")
        metrics_xgb = {k: 0.0 for k in ["accuracy", "f1_macro", "recall_veb", "precision_veb", "specificity_veb", "roc_auc"]}
    
    # === Log results ===
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)
    
    results = [
        ("E10 (CNN Fusion)", "X_beats (250) + X_rr (6)", metrics_e10),
        ("SVM", "X_rr only (6)", metrics_svm),
        ("Random Forest", "X_rr only (6)", metrics_rf),
        ("XGBoost", "X_rr only (6)", metrics_xgb),
    ]
    
    timestamp = datetime.now().isoformat()
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "model", "feature_set", "accuracy", "f1_macro", "recall_veb",
            "precision_veb", "specificity_veb", "roc_auc", "notes"
        ])
        
        for model_name, feature_set, metrics in results:
            row = {
                "timestamp": timestamp,
                "model": model_name,
                "feature_set": feature_set,
                **{k: f"{v:.6f}" for k, v in metrics.items()},
                "notes": "ablation_rr_only_vs_e10_full",
            }
            writer.writerow(row)
            
            print(f"\n{model_name:20} | Features: {feature_set:25}")
            print(f"  Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f} | "
                  f"Recall_VEB: {metrics['recall_veb']:.4f} | Prec_VEB: {metrics['precision_veb']:.4f}")
    
    print("\n✅ Ablation results saved to:", output_csv)
    print("\nKEY INSIGHT: Compare E10 (256 features) vs baselines (6 features)")
    print("             Difference shows value of learning temporal morphology from X_beats")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ablation study: E10 (full) vs baselines (X_rr only)")
    ap.add_argument("--data_dir", type=str, default="data/processed/mitbih_incart",
                    help="Path to processed dataset")
    ap.add_argument("--checkpoint", type=str, default="models/checkpoints/E10_best.pt",
                    help="Path to E10 checkpoint")
    ap.add_argument("--output_csv", type=str, default="reports/model_comparison_ablation_rr_only.csv",
                    help="Output CSV file")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = ap.parse_args()
    
    compare_models_ablation(
        data_dir=args.data_dir,
        checkpoint_e10=args.checkpoint,
        output_csv=args.output_csv,
        seed=args.seed,
        device=args.device,
    )
