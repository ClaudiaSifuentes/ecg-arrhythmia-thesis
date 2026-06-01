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
from .preprocessing import apply_bandpass, resample_signal
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

    # 1) resample if needed
    x = ds.ecg_data
    if fs != 360:
        x = resample_signal(x, fs, 360)
        fs_used = 360
    else:
        fs_used = fs

    # 2) bandpass filter on raw signal (prior to peak detection)
    x_f = apply_bandpass(x, fs=fs_used)

    # 3) R-peak detection (neurokit2 default is set in rpeaks.py)
    det_peaks = detect_r_peaks(x_f, fs=fs_used, method="neurokit")

    # 4) Build segments/labels using annotated peaks and AAMI mapping
    #    We reproduce the logic explicitly here so we also collect the annotation
    #    samples that were actually kept.
    ann = wfdb.rdann(record_id, "atr")
    ann_samples = np.asarray(ann.sample, dtype=int)
    ann_symbols = np.asarray(ann.symbol)

    # IMPORTANT: if the source fs != 360, annotation sample indices are in the
    # original sampling grid, but we detect peaks after resampling to 360.
    # Convert annotation indices to 360Hz before any windowing/matching.
    if fs != 360 and ann_samples.size > 0:
        ann_samples = np.asarray(np.round(ann_samples * (360.0 / float(fs))), dtype=int)

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

    # 5) RR features from detected peaks, then pick rows aligned to beats
    X_rr_all = extract_rr_features(det_peaks, fs=fs_used)

    matched_det_idx = _match_annotations_to_detected(
        kept_ann_samples_arr,
        det_peaks,
        fs=fs_used,
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


def build_dataset(sources: list[dict], output_dir: str) -> Dict[str, str]:
    """Build a combined dataset from multiple WFDB sources.

    sources = [
        {'name': 'mit', 'data_dir': 'data/raw/mitdb', 'records': [...], 'fs': 360},
        {'name': 'inc', 'data_dir': 'data/raw/incartdb', 'records': [...], 'fs': 257},
    ]

    Notes
    -----
    - Each source is processed independently; arrays are concatenated at the end.
    - If fs != 360, resample happens inside _collect_record (already implemented).
    - patient_id is prefixed with '{name}_' to keep sources separable.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_beats_all: List[np.ndarray] = []
    X_rr_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    pid_all: List[np.ndarray] = []

    for src in sources:
        name = str(src.get('name', 'src'))
        data_dir = str(src['data_dir'])
        records = [str(r) for r in src['records']]
        fs = int(src.get('fs', 360))

        # Apply excluded-record policy only for MIT-BIH sources
        if name.startswith('mit'):
            records_ok = filter_excluded(records)
        else:
            records_ok = records

        old_cwd = os.getcwd()
        try:
            os.chdir(data_dir)
            for rec in records_ok:
                xb, xr, yb, pid = _collect_record(rec, fs=fs)
                pid = np.asarray([f"{name}_{p}" for p in pid], dtype=pid.dtype)
                X_beats_all.append(xb)
                X_rr_all.append(xr)
                y_all.append(yb)
                pid_all.append(pid)
        finally:
            os.chdir(old_cwd)

    X_beats = np.concatenate(X_beats_all, axis=0) if X_beats_all else np.zeros((0, 250), dtype=np.float32)
    X_rr = np.concatenate(X_rr_all, axis=0) if X_rr_all else np.zeros((0, 6), dtype=np.float32)
    y_beats = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    patient_id = np.concatenate(pid_all, axis=0) if pid_all else np.zeros((0,), dtype='<U16')

    p_xb = output_dir / 'X_beats.npy'
    p_xr = output_dir / 'X_rr.npy'
    p_y = output_dir / 'y_beats.npy'
    p_pid = output_dir / 'patient_id.npy'

    np.save(p_xb, X_beats)
    np.save(p_xr, X_rr)
    np.save(p_y, y_beats)
    np.save(p_pid, patient_id)

    assert X_beats.shape[1] == 250
    assert X_rr.shape[1] == 6
    assert X_beats.shape[0] == X_rr.shape[0] == len(y_beats) == len(patient_id)
    assert not np.any(np.isnan(X_rr))

    print('✅ Combined dataset built successfully')

    return {
        'X_beats': str(p_xb),
        'X_rr': str(p_xr),
        'y_beats': str(p_y),
        'patient_id': str(p_pid),
    }


def build_dataset_multi_source(*, output_dir: str = "data/processed/mitbih_incart") -> Dict[str, str]:
    """Build combined processed dataset: MIT-BIH (train/val/test as before) + INCART (train-only).

    Spec (user)
    -----------
    - INCART: 64 clean records go to train (val/test remain MIT-BIH only).
    - Patient IDs are prefixed with `mit_` and `inc_`.
    - Writes arrays to `data/processed/mitbih_incart/`.
    - Prints summary:
      * Total beats + per-class counts/%
      * SVEB from INCART vs MIT-BIH
      * Total patients in train/val/test

    Notes
    -----
    - This function does not modify nor depend on any existing `build_dataset()` behavior.
    - Output dataset is the *concatenation* (train+val+test) for downstream code that
      already uses explicit patient-wise lists from `splits.py`.
    """

    from collections import Counter

    # Local imports to avoid changing module-level imports/behavior
    from .splits import TRAIN_PATIENTS, VAL_PATIENTS, TEST_PATIENTS

    INCART_EXCLUDE = {"I08", "I09", "I12", "I29", "I30", "I32", "I38", "I39", "I57", "I58", "I62"}
    INCART_ALL = [f"I{idx:02d}" for idx in range(1, 76)]  # INCART has I01..I75
    incart_records = [r for r in INCART_ALL if r not in INCART_EXCLUDE]

    mit_records = [str(r) for r in (list(TRAIN_PATIENTS) + list(VAL_PATIENTS) + list(TEST_PATIENTS))]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xb_parts: List[np.ndarray] = []
    Xr_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    pid_parts: List[np.ndarray] = []

    def _append_source(*, name: str, data_dir: str, records: Sequence[str], fs: int) -> None:
        old_cwd = os.getcwd()
        try:
            os.chdir(data_dir)
            for rec in records:
                xb, xr, yb, pid = _collect_record(str(rec), fs=fs)
                pid = np.asarray([f"{name}_{p}" for p in pid], dtype="<U16")
                Xb_parts.append(xb)
                Xr_parts.append(xr)
                y_parts.append(yb)
                pid_parts.append(pid)
        finally:
            os.chdir(old_cwd)

    # MIT-BIH (fs=360) + INCART (fs=257)
    _append_source(name="mit", data_dir="data/raw/mitdb", records=mit_records, fs=360)
    _append_source(name="inc", data_dir="data/raw/incartdb", records=incart_records, fs=257)

    X_beats = np.concatenate(Xb_parts, axis=0)
    X_rr = np.concatenate(Xr_parts, axis=0)
    y_beats = np.concatenate(y_parts, axis=0)
    patient_id = np.concatenate(pid_parts, axis=0)

    p_xb = out_dir / "X_beats.npy"
    p_xr = out_dir / "X_rr.npy"
    p_y = out_dir / "y_beats.npy"
    p_pid = out_dir / "patient_id.npy"

    np.save(p_xb, X_beats)
    np.save(p_xr, X_rr)
    np.save(p_y, y_beats)
    np.save(p_pid, patient_id)

    # Sanity checks
    assert X_beats.shape[1] == 250
    assert X_rr.shape[1] == 6
    assert X_beats.shape[0] == X_rr.shape[0] == len(y_beats) == len(patient_id)

    # --- Required prints ---
    total_beats = int(len(y_beats))
    c = Counter(y_beats.tolist())

    def _fmt(cls_id: int, name: str) -> str:
        n = int(c.get(cls_id, 0))
        pct = 100.0 * n / total_beats if total_beats else 0.0
        return f"{name}={n} ({pct:.2f}%)"

    print("Total beats:", total_beats)
    print(_fmt(0, "N"), _fmt(1, "SVEB"), _fmt(2, "VEB"))

    is_inc = np.char.startswith(patient_id.astype(str), "inc_")
    is_mit = np.char.startswith(patient_id.astype(str), "mit_")
    s_from_inc = int(((y_beats == 1) & is_inc).sum())
    s_from_mit = int(((y_beats == 1) & is_mit).sum())
    print(f"SVEB desde INCART: {s_from_inc}")
    print(f"SVEB desde MIT-BIH: {s_from_mit}")

    # Patient counts by split policy (INCART all train)
    train_pats = {f"mit_{p}" for p in TRAIN_PATIENTS} | {f"inc_{r}" for r in incart_records}
    val_pats = {f"mit_{p}" for p in VAL_PATIENTS}
    test_pats = {f"mit_{p}" for p in TEST_PATIENTS}

    print(f"Total pacientes en train: {len(train_pats)}")
    print(f"Total pacientes en val: {len(val_pats)}")
    print(f"Total pacientes en test: {len(test_pats)}")

    print("✅ Combined dataset (MIT-BIH+INCART) built successfully")

    return {"X_beats": str(p_xb), "X_rr": str(p_xr), "y_beats": str(p_y), "patient_id": str(p_pid)}
