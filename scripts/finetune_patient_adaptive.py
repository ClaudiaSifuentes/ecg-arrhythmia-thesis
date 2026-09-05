"""Patient-adaptive fine-tuning proof of concept for MIT-BIH record 232.

Record 232 is the test-set outlier: ~1,380 SVEB beats (95.7% of all test
SVEB) with an atypical LBBB + supraventricular ectopy morphology absent from
both training databases. Population-trained models (E10, and focal-loss /
no-SMOTE variants) detect essentially none of them (SVEB sensitivity ~0.01-0.02
even when record 232 is excluded from the denominator — see
reports/test_comparison.csv).

This script tests whether a small amount of the *same patient's* labeled
beats can recover SVEB sensitivity that the population model misses
entirely: freeze the CNN backbone (keep the learned morphology features),
fine-tune only the fusion head on a small calibration split of record 232's
own beats, and evaluate on the remaining held-out beats from that patient.

This is a proof of concept, not a deployment recipe: it assumes labeled
beats for the target patient are available, which is not true at real
deployment time. It answers a narrower question raised in review: is the
SVEB failure on record 232 fixable with patient-specific data, or is the
morphology simply unlearnable from any amount of data this model sees.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.data.torch_datasets import ECGBeatDataset, load_processed_arrays
from src.models.fusion import FusionClassifier
from src.utils.reproducibility import set_global_seeds

TARGET_RECORD = "232"


def _load_record_beats(data_dir: str, record: str):
    arr = load_processed_arrays(data_dir)
    pid = arr.patient_id.astype(str)
    pid_norm = np.char.replace(np.char.replace(pid, "mit_", ""), "inc_", "")
    mask = pid_norm == record
    if not mask.any():
        raise RuntimeError(f"Record {record} not found in {data_dir}")
    return arr.X_beats[mask], arr.X_rr[mask], arr.y[mask]


def _predict(model, X_beats: np.ndarray, X_rr: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_beats), batch_size):
            j = i + batch_size
            ecg = torch.from_numpy(X_beats[i:j]).unsqueeze(1).float().to(device)
            rr = torch.from_numpy(X_rr[i:j]).float().to(device)
            preds.append(model(ecg, rr).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint (e.g. E10, E11, E12)")
    ap.add_argument("--data_dir", type=str, default="data/processed/mitbih_incart")
    ap.add_argument("--calib_frac", type=float, default=0.2, help="Fraction of record 232's beats used to fine-tune")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_beats, X_rr, y = _load_record_beats(args.data_dir, TARGET_RECORD)
    n = len(y)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(n)
    n_calib = int(n * args.calib_frac)
    calib_idx, held_idx = idx[:n_calib], idx[n_calib:]

    print(f"Record {TARGET_RECORD}: {n} beats total | calib={len(calib_idx)} | held-out={len(held_idx)}")
    print(f"Held-out SVEB count: {(y[held_idx] == 1).sum()} | VEB count: {(y[held_idx] == 2).sum()}")

    model = FusionClassifier()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    print("\n=== BEFORE fine-tuning (held-out split) ===")
    y_pred_before = _predict(model, X_beats[held_idx], X_rr[held_idx], device)
    print(classification_report(y[held_idx], y_pred_before, target_names=["N", "SVEB", "VEB"], zero_division=0))

    # Freeze the CNN backbone (keep learned morphology features) and fine-tune
    # only the fusion head on the patient's own calibration beats.
    for p in model.cnn.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    calib_ds = ECGBeatDataset(X_beats[calib_idx], X_rr[calib_idx], y[calib_idx])
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss, n_seen = 0.0, 0
        for ecg, rr, yb in calib_loader:
            ecg, rr, yb = ecg.to(device), rr.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ecg, rr), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            n_seen += len(yb)
        print(f"epoch {epoch:>2}: calib_loss={total_loss / n_seen:.4f}")

    print("\n=== AFTER fine-tuning (held-out split) ===")
    y_pred_after = _predict(model, X_beats[held_idx], X_rr[held_idx], device)
    print(classification_report(y[held_idx], y_pred_after, target_names=["N", "SVEB", "VEB"], zero_division=0))


if __name__ == "__main__":
    main()
