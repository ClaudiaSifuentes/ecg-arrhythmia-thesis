"""PyTorch datasets and split utilities for Week 2 training.

This module is intentionally small and explicit:
- Loads the prepared `.npy` arrays from `data/processed/mitbih_aami3/`
- Splits patient-wise using `splits.get_mitbih_splits()` and `patient_id.npy`
- Applies SMOTE ONLY on the train split to target ratio N:SVEB:VEB ≈ 3:1:1
- Builds a `torch.utils.data.Dataset` that returns:
    ecg: (1, 250) float32
    rr:  (6,)     float32
    y:   ()       int64

No overengineering: keep it reproducible and shape-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitArrays:
    X_beats: np.ndarray  # (N, 250)
    X_rr: np.ndarray  # (N, 6)
    y: np.ndarray  # (N,)
    patient_id: np.ndarray  # (N,)


def load_processed_arrays(processed_dir: str | Path) -> SplitArrays:
    processed_dir = Path(processed_dir)

    X_beats = np.load(processed_dir / "X_beats.npy")
    X_rr = np.load(processed_dir / "X_rr.npy")
    y = np.load(processed_dir / "y_beats.npy")
    patient_id = np.load(processed_dir / "patient_id.npy")

    # Basic checks (Week 2 gate)
    assert X_beats.ndim == 2 and X_beats.shape[1] == 250
    assert X_rr.ndim == 2 and X_rr.shape[1] == 6
    assert y.ndim == 1
    assert patient_id.ndim == 1
    assert X_beats.shape[0] == X_rr.shape[0] == y.shape[0] == patient_id.shape[0]
    assert not np.any(np.isnan(X_rr))

    return SplitArrays(
        X_beats=X_beats.astype(np.float32, copy=False),
        X_rr=X_rr.astype(np.float32, copy=False),
        y=y.astype(np.int64, copy=False),
        patient_id=patient_id.astype(str, copy=False),
    )


def make_patient_wise_splits(arr: SplitArrays, splits) -> Dict[str, SplitArrays]:
    """Create patient-wise splits using the provided record id lists.

    Supports datasets where `patient_id` is prefixed (e.g. `mit_100`, `inc_I01`).
    - MIT-BIH samples are matched after stripping `mit_`.
    - INCART samples (`inc_`) are forced into train.
    """

    pid = np.asarray(arr.patient_id).astype(str)

    # Normalize ids for matching against MIT-BIH split lists
    pid_norm = pid.copy()
    pid_norm = np.char.replace(pid_norm, "mit_", "")
    pid_norm = np.char.replace(pid_norm, "inc_", "")

    is_inc = np.char.startswith(pid, "inc_")

    train_mask = np.isin(pid_norm, np.asarray(splits.train)) | is_inc
    val_mask = np.isin(pid_norm, np.asarray(splits.val))
    test_mask = np.isin(pid_norm, np.asarray(splits.test))

    def _sub(mask: np.ndarray) -> SplitArrays:
        return SplitArrays(
            X_beats=arr.X_beats[mask],
            X_rr=arr.X_rr[mask],
            y=arr.y[mask],
            patient_id=arr.patient_id[mask],
        )

    out = {
        "train": _sub(train_mask),
        "val": _sub(val_mask),
        "test": _sub(test_mask),
    }

    # Guards: disjoint and non-empty
    assert out["train"].y.size > 0 and out["val"].y.size > 0 and out["test"].y.size > 0
    assert set(out["train"].patient_id.tolist()).isdisjoint(set(out["val"].patient_id.tolist()))
    assert set(out["train"].patient_id.tolist()).isdisjoint(set(out["test"].patient_id.tolist()))
    assert set(out["val"].patient_id.tolist()).isdisjoint(set(out["test"].patient_id.tolist()))

    return out


def apply_smote_train_only(
    train: SplitArrays,
    *,
    seed: int = 42,
    target_ratio: Tuple[int, int, int] = (3, 1, 1),
) -> SplitArrays:
    """Apply SMOTE ONLY on train to approximate N:SVEB:VEB ~= 3:1:1.

    Important constraint
    --------------------
    SMOTE is an *oversampling* method: it can only increase class counts.
    Therefore we do NOT downsample class 0 here.

    Targeting strategy used
    -----------------------
    Let N0 = count(class 0). To achieve ~3:1:1 without downsampling, we set:
      target_1 = target_2 = floor(N0 / 3)
    and oversample classes 1 and 2 up to those targets (when they are smaller).

    This yields N:SVEB:VEB ≈ 3:1:1, while leaving N unchanged.
    """

    from imblearn.over_sampling import SMOTE

    Xb = train.X_beats
    Xr = train.X_rr
    y = train.y

    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)
    n2 = counts.get(2, 0)

    if n0 == 0 or n1 == 0 or n2 == 0:
        raise RuntimeError(f"Cannot SMOTE train split: missing class. counts={counts}")

    # Target ~3:1:1 without downsampling class 0
    target_minority = int(n0 // target_ratio[0])
    target_1 = max(n1, target_minority)
    target_2 = max(n2, target_minority)

    sampling_strategy = {1: target_1, 2: target_2}

    X = np.concatenate([Xb, Xr], axis=1)

    # k_neighbors must be < minority count (pre-SMOTE)
    min_pre = min(n1, n2)
    k = int(min(5, min_pre - 1))
    if k < 1:
        raise RuntimeError(f"Not enough minority samples for SMOTE: min_pre={min_pre}")

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=seed, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)

    Xb_res = X_res[:, :250].astype(np.float32, copy=False)
    Xr_res = X_res[:, 250:].astype(np.float32, copy=False)

    pid_res = np.empty((len(y_res),), dtype="<U8")
    pid_res[: len(train.patient_id)] = train.patient_id.astype("<U8", copy=False)
    pid_res[len(train.patient_id) :] = "SMOTE"

    assert Xb_res.shape[0] == Xr_res.shape[0] == y_res.shape[0] == pid_res.shape[0]
    assert Xb_res.shape[1] == 250
    assert Xr_res.shape[1] == 6
    assert set(np.unique(y_res)) == {0, 1, 2}
    assert not np.any(np.isnan(Xr_res))

    return SplitArrays(X_beats=Xb_res, X_rr=Xr_res, y=y_res.astype(np.int64), patient_id=pid_res)


class ECGBeatDataset(Dataset):
    """Torch Dataset returning (ecg, rr, y) with strict shapes."""

    def __init__(self, X_beats: np.ndarray, X_rr: np.ndarray, y: np.ndarray):
        self.X_beats = np.asarray(X_beats, dtype=np.float32)
        self.X_rr = np.asarray(X_rr, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

        assert self.X_beats.ndim == 2 and self.X_beats.shape[1] == 250
        assert self.X_rr.ndim == 2 and self.X_rr.shape[1] == 6
        assert self.y.ndim == 1
        assert self.X_beats.shape[0] == self.X_rr.shape[0] == self.y.shape[0]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        ecg = torch.from_numpy(self.X_beats[idx]).unsqueeze(0)  # (1, 250)
        rr = torch.from_numpy(self.X_rr[idx])  # (6,)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return ecg, rr, y


def build_dataloaders(
    data_dir: str | Path,
    *,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Build train/val/test DataLoaders from prepared `.npy` arrays.

    Strict requirements (Week 2)
    ----------------------------
    - Patient-wise split using `splits.get_mitbih_splits()` + patient_id.npy
    - SMOTE ONLY on train
    - Reproducible order and shuffling (seed)

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """

    from torch.utils.data import DataLoader

    from src.data.splits import get_mitbih_splits

    arr = load_processed_arrays(data_dir)
    split = make_patient_wise_splits(arr, get_mitbih_splits())

    train_sm = apply_smote_train_only(split["train"], seed=seed, target_ratio=(3, 1, 1))

    train_ds = ECGBeatDataset(train_sm.X_beats, train_sm.X_rr, train_sm.y)
    val_ds = ECGBeatDataset(split["val"].X_beats, split["val"].X_rr, split["val"].y)
    test_ds = ECGBeatDataset(split["test"].X_beats, split["test"].X_rr, split["test"].y)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(batch_size, 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(batch_size, 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Smoke checks
    b = next(iter(train_loader))
    assert b[0].shape[1:] == (1, 250)
    assert b[1].shape[1:] == (6,)
    assert b[2].ndim == 1

    return train_loader, val_loader, test_loader
