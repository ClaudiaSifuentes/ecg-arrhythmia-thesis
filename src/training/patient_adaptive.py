"""Shared logic for patient-adaptive fine-tuning (head-only, CNN frozen).

Used by scripts/finetune_patient_adaptive.py (single record, CLI) and
scripts/finetune_multi_patient.py (loops over several records to test
whether the record-232 result generalizes as a *method*, not a one-off).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.data.torch_datasets import ECGBeatDataset, load_processed_arrays
from src.models.fusion import FusionClassifier


def load_record_beats(data_dir: str, record: str):
    arr = load_processed_arrays(data_dir)
    pid = arr.patient_id.astype(str)
    pid_norm = np.char.replace(np.char.replace(pid, "mit_", ""), "inc_", "")
    mask = pid_norm == record
    if not mask.any():
        raise RuntimeError(f"Record {record} not found in {data_dir}")
    return arr.X_beats[mask], arr.X_rr[mask], arr.y[mask]


def predict(model, X_beats: np.ndarray, X_rr: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_beats), batch_size):
            j = i + batch_size
            ecg = torch.from_numpy(X_beats[i:j]).unsqueeze(1).float().to(device)
            rr = torch.from_numpy(X_rr[i:j]).float().to(device)
            preds.append(model(ecg, rr).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def run_finetune(
    checkpoint: str,
    data_dir: str,
    record: str,
    *,
    calib_frac: float = 0.2,
    epochs: int = 10,
    lr: float = 1e-4,
    seed: int = 42,
    device=None,
) -> dict:
    """Fine-tune the fusion head on a calibration split of `record`'s own
    beats; evaluate on the disjoint held-out remainder.

    Returns a dict with before/after classification reports, held-out
    predictions/labels (for pooled aggregation across records), and split
    sizes.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_beats, X_rr, y = load_record_beats(data_dir, record)
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_calib = int(n * calib_frac)
    calib_idx, held_idx = idx[:n_calib], idx[n_calib:]

    model = FusionClassifier()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    y_pred_before = predict(model, X_beats[held_idx], X_rr[held_idx], device)
    report_before = classification_report(
        y[held_idx], y_pred_before, labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], output_dict=True, zero_division=0
    )

    model_ft = deepcopy(model)
    for p in model_ft.cnn.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model_ft.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    calib_ds = ECGBeatDataset(X_beats[calib_idx], X_rr[calib_idx], y[calib_idx])
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=True)

    model_ft.train()
    for _ in range(epochs):
        for ecg, rr, yb in calib_loader:
            ecg, rr, yb = ecg.to(device), rr.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model_ft(ecg, rr), yb)
            loss.backward()
            optimizer.step()

    y_pred_after = predict(model_ft, X_beats[held_idx], X_rr[held_idx], device)
    report_after = classification_report(
        y[held_idx], y_pred_after, labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], output_dict=True, zero_division=0
    )

    return {
        "record": record,
        "n_total": n,
        "n_calib": len(calib_idx),
        "n_held": len(held_idx),
        "y_held": y[held_idx],
        "y_pred_before": y_pred_before,
        "y_pred_after": y_pred_after,
        "report_before": report_before,
        "report_after": report_after,
    }
