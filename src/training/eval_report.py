"""Test-set evaluation with an explicit record-232 breakdown.

Record 232 concentrates ~95.7% of the SVEB beats in the MIT-BIH test
partition (see src/data/splits.py). Reporting test metrics with and without
it separately is required to tell whether a training change (focal loss,
SMOTE ratio, fine-tuning) actually improves SVEB generalization to unseen
patients broadly, or only shifts performance on this single outlier.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report

from src.data.splits import get_mitbih_splits
from src.data.torch_datasets import load_processed_arrays, make_patient_wise_splits

OUTLIER_RECORD = "232"


def _normalize_pid(patient_id: np.ndarray) -> np.ndarray:
    pid = patient_id.astype(str)
    pid = np.char.replace(pid, "mit_", "")
    pid = np.char.replace(pid, "inc_", "")
    return pid


def _predict(model, X_beats: np.ndarray, X_rr: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_beats), batch_size):
            j = i + batch_size
            ecg = torch.from_numpy(X_beats[i:j]).unsqueeze(1).float().to(device)
            rr = torch.from_numpy(X_rr[i:j]).float().to(device)
            logits = model(ecg, rr)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def evaluate_test_with_outlier_breakdown(model, data_dir, device) -> dict:
    """Return classification reports for the full test set and for the test
    set with record 232 excluded.
    """

    arr = load_processed_arrays(data_dir)
    split = make_patient_wise_splits(arr, get_mitbih_splits())
    test = split["test"]

    y_pred = _predict(model, test.X_beats, test.X_rr, device)

    pid_norm = _normalize_pid(test.patient_id)
    keep = pid_norm != OUTLIER_RECORD

    report_full = classification_report(
        test.y, y_pred, target_names=["N", "SVEB", "VEB"], output_dict=True, zero_division=0
    )
    report_excl = classification_report(
        test.y[keep], y_pred[keep], target_names=["N", "SVEB", "VEB"], output_dict=True, zero_division=0
    )

    return {
        "full": report_full,
        "excl_232": report_excl,
        "n_test_full": int(len(test.y)),
        "n_test_excl_232": int(keep.sum()),
        "n_sveb_excl_232": int(report_excl["SVEB"]["support"]),
    }


def print_outlier_breakdown(test_report: dict) -> None:
    full, excl = test_report["full"], test_report["excl_232"]
    print(f"\n=== TEST full (n={test_report['n_test_full']}, includes record 232) ===")
    print(
        f"F1-macro: {full['macro avg']['f1-score']:.4f} | "
        f"SVEB sens: {full['SVEB']['recall']:.4f} | SVEB PPV: {full['SVEB']['precision']:.4f} | "
        f"VEB recall: {full['VEB']['recall']:.4f}"
    )
    print(f"\n=== TEST excluding record 232 (n={test_report['n_test_excl_232']}, SVEB n={test_report['n_sveb_excl_232']}) ===")
    print(
        f"F1-macro: {excl['macro avg']['f1-score']:.4f} | "
        f"SVEB sens: {excl['SVEB']['recall']:.4f} | SVEB PPV: {excl['SVEB']['precision']:.4f} | "
        f"VEB recall: {excl['VEB']['recall']:.4f}"
    )


def append_test_comparison_row(csv_path, exp_id: str, timestamp: str, notes: str, test_report: dict) -> None:
    import csv
    from pathlib import Path

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exp_id", "timestamp", "notes",
        "test_f1_macro", "test_sveb_sens", "test_sveb_ppv", "test_recall_veb",
        "test_f1_macro_excl232", "test_sveb_sens_excl232", "test_sveb_ppv_excl232",
        "n_sveb_excl232",
    ]
    write_header = not path.exists()
    full, excl = test_report["full"], test_report["excl_232"]
    row = {
        "exp_id": exp_id,
        "timestamp": timestamp,
        "notes": notes,
        "test_f1_macro": round(full["macro avg"]["f1-score"], 4),
        "test_sveb_sens": round(full["SVEB"]["recall"], 4),
        "test_sveb_ppv": round(full["SVEB"]["precision"], 4),
        "test_recall_veb": round(full["VEB"]["recall"], 4),
        "test_f1_macro_excl232": round(excl["macro avg"]["f1-score"], 4),
        "test_sveb_sens_excl232": round(excl["SVEB"]["recall"], 4),
        "test_sveb_ppv_excl232": round(excl["SVEB"]["precision"], 4),
        "n_sveb_excl232": test_report["n_sveb_excl_232"],
    }
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
