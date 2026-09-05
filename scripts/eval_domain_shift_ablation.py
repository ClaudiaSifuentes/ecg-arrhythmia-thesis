"""Isolate which component of the Polar H10 degradation simulation drives
the VEB collapse reported in Table IX (recall 0.99 -> 0.00-0.20).

Table IX applies all four effects at once (resample round-trip + baseline
wander + motion artifact + contact noise), so it cannot tell us whether the
model is fragile to the resampling bandwidth loss specifically (which
exactly matches the deployed backend's real preprocessing step, Section
III.B), to the additive noise terms (which are plausible but unmeasured
guesses at chest-strap artifact severity), or to their combination. This
matters: if resampling alone explains most of the drop, that is a strong,
well-grounded finding (it uses the *real* pipeline step, no guessed noise
parameters); if only the noise terms do, the finding is contingent on noise
parameters nobody has validated against a real device.

Each configuration is evaluated independently against the same clean
baseline on the MIT-BIH test partition, for a given checkpoint.
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
NOISE_OFF = dict(contact_artifact_prob=0.0, contact_typical_snr_db=60.0)

CONFIGS = [
    ("clean (no degradation)", dict(apply_resample=False, baseline_amplitude=0.0, motion_prob=0.0, **NOISE_OFF)),
    ("resample round-trip only (360->130->360 Hz)", dict(apply_resample=True, baseline_amplitude=0.0, motion_prob=0.0, **NOISE_OFF)),
    ("baseline wander only (no resample)", dict(apply_resample=False, baseline_amplitude=0.15, motion_prob=0.0, **NOISE_OFF)),
    ("motion artifact only (no resample)", dict(apply_resample=False, baseline_amplitude=0.0, motion_prob=0.15, motion_amplitude=0.8, **NOISE_OFF)),
    ("uniform severe noise only (no resample, SNR=15dB, 100% of beats)", dict(apply_resample=False, baseline_amplitude=0.0, motion_prob=0.0, contact_artifact_prob=1.0, contact_typical_snr_db=15.0, contact_severe_snr_db=15.0)),
    ("tiered contact noise only (no resample, 2.16% severe per Skala et al.)", dict(apply_resample=False, baseline_amplitude=0.0, motion_prob=0.0, contact_artifact_prob=0.0216, contact_typical_snr_db=28.0, contact_severe_snr_db=15.0)),
    ("full, tiered noise (resample + wander + motion + tiered noise)", dict(apply_resample=True, baseline_amplitude=0.15, motion_prob=0.15, motion_amplitude=0.8, contact_artifact_prob=0.0216, contact_typical_snr_db=28.0, contact_severe_snr_db=15.0)),
]


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

    print(f"{'Config':45} | {'F1-macro':>8} | {'SVEB sens':>9} | {'VEB sens':>8} | (excl.232: F1/SVEB/VEB)")
    print("-" * 110)
    for label, kwargs in CONFIGS:
        X = simulate_polar_h10_degradation(test.X_beats, seed=args.seed, **kwargs)
        y_pred = _predict(model, X, test.X_rr, device)

        rep_full = classification_report(
            test.y, y_pred, labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], output_dict=True, zero_division=0
        )
        rep_excl = classification_report(
            test.y[keep], y_pred[keep], labels=[0, 1, 2], target_names=["N", "SVEB", "VEB"], output_dict=True, zero_division=0
        )

        print(
            f"{label:45} | {rep_full['macro avg']['f1-score']:>8.3f} | "
            f"{rep_full['SVEB']['recall']:>9.3f} | {rep_full['VEB']['recall']:>8.3f} | "
            f"({rep_excl['macro avg']['f1-score']:.3f}/{rep_excl['SVEB']['recall']:.3f}/{rep_excl['VEB']['recall']:.3f})"
        )


if __name__ == "__main__":
    main()
