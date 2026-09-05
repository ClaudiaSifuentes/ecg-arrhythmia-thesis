"""Evaluate robustness to a simulated Polar H10 acquisition domain shift.

Applies simulate_polar_h10_degradation (src/data/domain_shift_sim.py) to the
MIT-BIH test partition and compares classification performance against the
clean 360 Hz clinical signal, for a given checkpoint. This is a synthetic
proxy for the real-device validation that is not possible in this cycle
(no physical Polar H10 available) -- see the module docstring for what the
simulation does and does not capture, and Section V of the paper for why
this cannot replace real-device validation.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report

from src.data.domain_shift_sim import simulate_polar_h10_degradation
from src.data.splits import get_mitbih_splits
from src.data.torch_datasets import load_processed_arrays, make_patient_wise_splits
from src.models.fusion import FusionClassifier

OUTLIER_RECORD = "232"


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
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default="data/processed/mitbih_incart")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr = load_processed_arrays(args.data_dir)
    split = make_patient_wise_splits(arr, get_mitbih_splits())
    test = split["test"]

    pid_norm = np.char.replace(np.char.replace(test.patient_id.astype(str), "mit_", ""), "inc_", "")
    keep = pid_norm != OUTLIER_RECORD

    model = FusionClassifier()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    X_degraded = simulate_polar_h10_degradation(test.X_beats, seed=args.seed)

    for label, X in [
        ("CLEAN (360 Hz clinical)", test.X_beats),
        ("SIMULATED POLAR H10 (130 Hz round-trip + wander + motion + noise)", X_degraded),
    ]:
        y_pred = _predict(model, X, test.X_rr, device)

        print(f"\n{'=' * 70}\n{label} -- TEST full (n={len(test.y)})\n{'=' * 70}")
        print(classification_report(test.y, y_pred, labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], zero_division=0))

        print(f"--- {label} -- TEST excluding record 232 (n={int(keep.sum())}) ---")
        print(classification_report(test.y[keep], y_pred[keep], labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], zero_division=0))


if __name__ == "__main__":
    main()
