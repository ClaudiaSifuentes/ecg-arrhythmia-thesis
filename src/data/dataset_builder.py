"""Dataset builder: MIT-BIH -> .npy files ready for training.

This module connects the existing pipeline pieces in a strict order:
1) preprocessing.apply_bandpass(signal)
2) rpeaks.detect_r_peaks(signal)  [neurokit2]
3) mitbih_dataset.MITBIHDataset.get_ecg_segments_and_labels()  [PRE=100, POST=150]
4) rr_features.extract_rr_features(r_peaks)  [6 RR features]
5) patient-wise split (splits.get_mitbih_splits)  [handled by choosing `records`]
6) SMOTE on train only (optional; only if imblearn exists)
7) export: X_beats.npy, X_rr.npy, y_beats.npy, patient_id.npy

Critical alignment constraint
----------------------------
We build beat segments and labels by iterating over *annotated beats* and
filtering by AAMI mapping. Therefore, we must align RR features to the same
beats. We do this by:
- detecting R peaks on the filtered ECG signal
- for each annotated beat used in L1, matching its annotation sample to the
  nearest detected R peak within ±150ms
- extracting RR features on the full detected peak sequence, then selecting the
  rows corresponding to the matched peaks in the same order as the segments.

Design decisions (project spec)
-------------------------------
- F -> VEB
- PRE=100 / POST=150
- No LabelEncoder (explicit AAMI ids)
- '/' and 'f' excluded (map to None)
- neurokit2 as R-peak detector
- tolerance ±150ms (ANSI/AAMI EC57)
- excluded records list lives in `splits.py`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import wfdb

from .aami_mapping import map_symbol_to_aami_class
from .mitbih_dataset import MITBIHDataset, PRE_SAMPLES, POST_SAMPLES
from .preprocessing import apply_bandpass
from .rpeaks import detect_r_peaks
from .rr_features import extract_rr_features
from .splits import EXCLUDED_RECORDS, filter_excluded


def _match_annotations_to_detected(
    ann_samples: np.ndarray,
    det_peaks: np.ndarray,
    fs: int = 360,
    tolerance_ms: int = 150,
    *,
    strict: bool = False,
) -> np.ndarray:
    """Return indices into `det_peaks` for each `ann_sample`.

    If `strict=True`, raise on the first unmatched annotation.
    If `strict=False` (default), return -1 for unmatched samples so the caller
    can drop those beats while keeping the rest aligned.
    """

    ann_samples = np.asarray(ann_samples, dtype=int).reshape(-1)
    det_peaks = np.asarray(det_peaks, dtype=int).reshape(-1)

    tol = int(round((tolerance_ms / 1000.0) * fs))
    if ann_samples.size == 0:
        return np.zeros((0,), dtype=int)

    if det_peaks.size == 0:
        if strict:
            raise RuntimeError("No detected R-peaks available for matching.")
        return -np.ones((ann_samples.size,), dtype=int)

    used = np.zeros(det_peaks.size, dtype=bool)
    matched_idx: List[int] = []

    for a in ann_samples:
        j = int(np.searchsorted(det_peaks, a))
        cand = []
        if 0 <= j < det_peaks.size:
            cand.append(j)
        if j - 1 >= 0:
            cand.append(j - 1)

        cand = [c for c in cand if (not used[c]) and abs(int(det_peaks[c]) - int(a)) <= tol]
        if not cand:
            # Optional: try a tiny local scan near j (rare if duplicates/close peaks)
            lo = max(0, j - 3)
            hi = min(det_peaks.size, j + 4)
            cand = [c for c in range(lo, hi) if (not used[c]) and abs(int(det_peaks[c]) - int(a)) <= tol]

        if not cand:
            if strict:
                raise RuntimeError(
                    f"Cannot match annotation sample {a} within ±{tol} samples to any detected peak. "
                    f"Detected peaks around: {det_peaks[max(0, j-2):min(det_peaks.size, j+3)]}"
                )
            matched_idx.append(-1)
            continue

        # choose closest
        c_best = min(cand, key=lambda c: abs(int(det_peaks[c]) - int(a)))
        used[c_best] = True
        matched_idx.append(int(c_best))

    return np.asarray(matched_idx, dtype=int)


def _collect_record(
    record_id: str,
    *,
    fs: int = 360,
    tolerance_ms: int = 150,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned (X_beats, X_rr, y, patient_ids) for a single record."""

    ds = MITBIHDataset(record_id, sampling_rate=fs)
    ds.load_data()  # loads signal + annotations

    # 1) bandpass filter on raw signal (prior to peak detection)
    x_f = apply_bandpass(ds.ecg_data, fs=fs)

    # 2) R-peak detection (neurokit2 default is set in rpeaks.py)
    det_peaks = detect_r_peaks(x_f, fs=fs, method="neurokit")

    # 3) Build segments/labels using annotated peaks and AAMI mapping
    #    We reproduce the logic explicitly here so we also collect the annotation
    #    samples that were actually kept.
    ann = wfdb.rdann(record_id, "atr")
    ann_samples = np.asarray(ann.sample, dtype=int)
    ann_symbols = np.asarray(ann.symbol)

    segments: List[np.ndarray] = []
    y: List[int] = []
    kept_ann_samples: List[int] = []

    win_len = PRE_SAMPLES + POST_SAMPLES

    # z-score normalization (as used in MITBIHDataset)
    x_norm = (x_f - np.mean(x_f)) / (np.std(x_f) + 1e-12)

    for a_samp, sym in zip(ann_samples, ann_symbols):
        cls = map_symbol_to_aami_class(str(sym))
        if cls is None:
            continue

        start = int(a_samp) - PRE_SAMPLES
        end = int(a_samp) + POST_SAMPLES
        if start < 0 or end > len(x_norm) or (end - start) != win_len:
            continue

        segments.append(np.asarray(x_norm[start:end], dtype=np.float32))
        y.append(int(cls))
        kept_ann_samples.append(int(a_samp))

    X_beats = np.asarray(segments, dtype=np.float32)
    y_beats = np.asarray(y, dtype=np.int64)
    kept_ann_samples_arr = np.asarray(kept_ann_samples, dtype=int)

    # 4) RR features from detected peaks, then pick rows aligned to beats
    X_rr_all = extract_rr_features(det_peaks, fs=fs)

    matched_det_idx = _match_annotations_to_detected(
        kept_ann_samples_arr,
        det_peaks,
        fs=fs,
        tolerance_ms=tolerance_ms,
        strict=False,
    )

    valid = matched_det_idx >= 0
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        # Keep dataset build robust: drop only unmatched beats, preserve alignment
        X_beats = X_beats[valid]
        y_beats = y_beats[valid]
        matched_det_idx = matched_det_idx[valid]
        print(f"[dataset_builder:{record_id}] dropped_unmatched_beats={n_dropped} / kept={int(valid.sum())}")

    X_rr = X_rr_all[matched_det_idx]

    patient_ids = np.asarray([str(record_id)] * len(y_beats), dtype="<U8")

    # Alignment asserts
    assert X_beats.shape[0] == X_rr.shape[0] == len(y_beats) == len(patient_ids)
    assert X_beats.shape[1] == (PRE_SAMPLES + POST_SAMPLES)
    assert X_rr.shape[1] == 6
    assert not np.any(np.isnan(X_rr))

    return X_beats, X_rr, y_beats, patient_ids


def build_dataset(data_dir: str, records: list, output_dir: str) -> Dict[str, str]:
    """Pipeline completo: MIT-BIH -> 4 archivos .npy listos para entrenamiento.

    Parameters
    ----------
    data_dir:
        Directory containing MIT-BIH records (e.g. data/raw/mitdb).
    records:
        List of record ids to process.
        NOTE: patient-wise split is applied by choosing which records to pass.
    output_dir:
        Where to store the `.npy` exports.

    Returns
    -------
    dict with paths for: X_beats, X_rr, y_beats, patient_id
    """

    data_dir = str(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply excluded record policy
    records_in = [str(r) for r in records]
    records_ok = filter_excluded(records_in)

    if len(records_ok) < len(records_in):
        dropped = sorted(set(records_in) - set(records_ok))
        print(f"[dataset_builder] Dropping excluded records: {dropped} (policy in splits.py)")

    X_beats_all: List[np.ndarray] = []
    X_rr_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    pid_all: List[np.ndarray] = []

    old_cwd = os.getcwd()
    try:
        os.chdir(data_dir)  # WFDB local read compatibility

        for rec in records_ok:
            xb, xr, y, pid = _collect_record(rec)
            X_beats_all.append(xb)
            X_rr_all.append(xr)
            y_all.append(y)
            pid_all.append(pid)

    finally:
        os.chdir(old_cwd)

    X_beats = np.concatenate(X_beats_all, axis=0) if X_beats_all else np.zeros((0, 250), dtype=np.float32)
    X_rr = np.concatenate(X_rr_all, axis=0) if X_rr_all else np.zeros((0, 6), dtype=np.float32)
    y_beats = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    patient_id = np.concatenate(pid_all, axis=0) if pid_all else np.zeros((0,), dtype="<U8")

    # 6) SMOTE train-only is intentionally NOT applied here.
    # Reason: this builder exports raw arrays. SMOTE should be applied in the
    # training dataloader to avoid accidental leakage into val/test.

    # 7) Export
    p_xb = output_dir / "X_beats.npy"
    p_xr = output_dir / "X_rr.npy"
    p_y = output_dir / "y_beats.npy"
    p_pid = output_dir / "patient_id.npy"

    np.save(p_xb, X_beats)
    np.save(p_xr, X_rr)
    np.save(p_y, y_beats)
    np.save(p_pid, patient_id)

    # Final verification requested
    assert X_beats.shape[1] == 250
    assert X_rr.shape[1] == 6
    assert X_beats.shape[0] == X_rr.shape[0] == len(y_beats) == len(patient_id)
    assert not np.any(np.isnan(X_rr))
    print("✅ Pipeline OK — listo para entrenamiento")

    return {
        "X_beats": str(p_xb),
        "X_rr": str(p_xr),
        "y_beats": str(p_y),
        "patient_id": str(p_pid),
    }
