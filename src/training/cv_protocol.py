"""Reproducible patient-wise CV protocol with out-of-fold (OOF) outputs.

Fixes the protocol issues found in the audit (reports/AUDIT.md):

- Outer folds are FIXED by `cv.split_seed` and do not move with the training
  seed. Different training seeds therefore see the same patients in the same
  fold, which makes fold-by-fold comparisons between experiments valid.
- Early stopping / checkpoint selection uses an INNER validation split made
  of whole patients drawn from the outer-train patients only. The evaluation
  fold is never looked at during training (previously it was used for early
  stopping, which is optimistic).
- Test patients (splits.TEST_PATIENTS) are excluded from the CV pool so the
  CV never sees the hold-out test set.
- The same decision rule (src/training/decision.py, thresholds in the config)
  is applied to CV and test. Argmax metrics are reported alongside.
- Every (seed, fold) job is deterministic: seeded RNGs, deterministic
  algorithms, a fixed thread count and an explicit generator for shuffling.
- OOF logits/probabilities are saved to `reports/oof/oof_{exp}.npz` with
  y, prob, fold, patient_id (and the inner-validation outputs, used later
  for threshold selection and temperature scaling without touching OOF).
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

from src.data.splits import TEST_PATIENTS
from src.data.subjects import subject_ids
from src.data.torch_datasets import load_processed_arrays
from src.models.fusion import FusionClassifier
from src.training.decision import decide_from_config
from src.training.losses import FocalLoss
from src.training.metrics import class_metrics
from src.utils.reproducibility import enable_determinism


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULTS: dict = {
    "exp_id": "E12",
    "data_dir": "data/processed/mitbih_aami3",
    "exclude_patients": [],
    "model": {"dropout": 0.5, "use_rr": True},
    "loss": {"type": "focal", "gamma": 2.0, "alpha": [1.0, 8.0, 3.0], "label_smoothing": 0.0},
    "optim": {"lr": 5e-4, "weight_decay": 1e-4, "batch_size": 256, "max_epochs": 50, "patience": 8,
              "lr_patience": 7, "lr_factor": 0.5},
    "cv": {"n_splits": 5, "split_seed": 42, "inner_val_frac_splits": 5},
    "features": {"rr_set": "base"},
    "augment": None,
    "decision": {"t_sveb": 0.24, "t_veb": None},
    "seeds": [0, 1, 2],
    "num_threads": 1,
    "compile": False,
    "device": "auto",
}


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    base = load_config(Path(path).parent / parent) if parent else DEFAULTS
    return deep_update(base, cfg)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _norm_pid(pid: np.ndarray) -> np.ndarray:
    pid = np.asarray(pid).astype(str)
    return np.char.replace(np.char.replace(pid, "mit_", ""), "inc_", "")


@dataclass
class DevData:
    X_beats: np.ndarray
    X_rr: np.ndarray
    y: np.ndarray
    pid: np.ndarray  # record id (kept in OOF for per-record analysis)
    subject: np.ndarray  # grouping key for every split


def load_dev_data(cfg: dict) -> DevData:
    """All non-test patients (the CV pool)."""

    from src.data.rr_context import build_rr_features

    arr = load_processed_arrays(cfg["data_dir"])
    pid_norm = _norm_pid(arr.patient_id)
    X_rr = build_rr_features(arr.X_rr, arr.patient_id, cfg["features"]["rr_set"])
    drop = set(TEST_PATIENTS) | {str(p) for p in cfg.get("exclude_patients") or []}
    keep = ~(np.isin(pid_norm, list(drop)) & is_mitbih(arr.patient_id))
    pid = arr.patient_id[keep].astype(str)
    return DevData(arr.X_beats[keep], X_rr[keep], arr.y[keep], pid, subject_ids(pid))


def is_mitbih(patient_id: np.ndarray) -> np.ndarray:
    """MIT-BIH rows: `mit_` prefix, or every row of an unprefixed (MIT-only) dataset."""

    pid = np.asarray(patient_id).astype(str)
    if not (np.char.find(pid, "_") >= 0).any():
        return np.ones(len(pid), dtype=bool)
    return np.char.startswith(pid, "mit_")


def outer_folds(data: DevData, n_splits: int, split_seed: int, fold_map: str | None = None) -> np.ndarray:
    """Fold index per beat.

    Without `fold_map`: StratifiedGroupKFold over subjects (depends on the
    labels, so changing the data — e.g. rebuilding INCART — reshuffles subjects).
    With `fold_map` (JSON {subject: fold}): mapped subjects keep their fold, so
    experiments on different data versions stay comparable fold by fold;
    unmapped subjects (a new database) are split on their own with
    StratifiedGroupKFold and appended fold k -> fold k.
    """

    fold = np.full(len(data.y), -1, dtype=int)
    if fold_map:
        fmap = json.loads(Path(fold_map).read_text())
        for subj, k in fmap.items():
            fold[data.subject == subj] = int(k)
    rest = np.nonzero(fold < 0)[0]
    if len(rest):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for k, (_, te) in enumerate(skf.split(np.zeros(len(rest)), data.y[rest], groups=data.subject[rest])):
            fold[rest[te]] = k
    assert (fold >= 0).all()
    return fold


def inner_split(y: np.ndarray, pid: np.ndarray, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Patient-grouped, class-stratified inner train/val split (≈1/n_splits val).

    Picks, among the n_splits candidate partitions, the first one whose inner
    validation contains all three classes (SVEB is concentrated in few patients).
    """

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    first = None
    for tr, va in skf.split(np.zeros(len(y)), y, groups=pid):
        first = first or (tr, va)
        if len(np.unique(y[va])) == 3:
            return tr, va
    return first


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def build_criterion(loss_cfg: dict) -> nn.Module:
    if loss_cfg["type"] == "focal":
        return FocalLoss(gamma=float(loss_cfg["gamma"]), alpha=loss_cfg.get("alpha"),
                         label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    if loss_cfg["type"] == "ce":
        w = loss_cfg.get("alpha")
        return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32) if w else None,
                                   label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    raise ValueError(loss_cfg["type"])


class MeanPool1d(nn.Module):
    """Global average pooling over time as an explicit mean.

    Same output as nn.AdaptiveAvgPool1d(1) and no parameters (state_dict keys
    are unchanged, so checkpoints load into the original FusionClassifier),
    but its CUDA backward is deterministic, unlike adaptive_avg_pool's.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1, keepdim=True)


def get_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


@torch.no_grad()
def predict_logits(model: nn.Module, Xb: np.ndarray, Xr: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(Xb), batch_size):
        ecg = torch.from_numpy(np.ascontiguousarray(Xb[i:i + batch_size])).unsqueeze(1).to(dev)
        rr = torch.from_numpy(np.ascontiguousarray(Xr[i:i + batch_size])).to(dev)
        out.append(model(ecg, rr).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_model(cfg: dict, Xb_tr, Xr_tr, y_tr, Xb_va, Xr_va, y_va, *, seed: int, log=print):
    """E12 training recipe on in-memory arrays. Returns (model, info)."""

    from src.data.augment import make_augmenter

    enable_determinism(seed, num_threads=cfg.get("num_threads"))
    o = cfg["optim"]
    device = get_device(cfg)
    model = FusionClassifier(dropout=float(cfg["model"]["dropout"]), use_rr=bool(cfg["model"]["use_rr"]))
    model.cnn.block3[3] = MeanPool1d()
    model = model.to(device)
    # torch.compile only speeds up the CPU training step (~1.5x); weights are
    # shared with `model`, which is what gets evaluated and saved.
    fwd = torch.compile(model) if (cfg.get("compile") and device.type == "cpu") else model
    criterion = build_criterion(cfg["loss"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(o["lr"]), weight_decay=float(o["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=float(o["lr_factor"]),
                                                       patience=int(o["lr_patience"]))
    augment = make_augmenter(cfg.get("augment"), seed=seed)

    Xb_t = torch.from_numpy(np.ascontiguousarray(Xb_tr, dtype=np.float32))
    Xr_t = torch.from_numpy(np.ascontiguousarray(Xr_tr, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64))
    y_va_t = torch.from_numpy(np.asarray(y_va, dtype=np.int64))
    g = torch.Generator().manual_seed(seed)
    bs = int(o["batch_size"])

    best = (float("inf"), 0, None)
    history = []
    no_improve = 0
    for epoch in range(1, int(o["max_epochs"]) + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(y_t), generator=g)
        tot, n = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            ecg, rr, yb = Xb_t[idx], Xr_t[idx], y_t[idx]
            if augment is not None:
                ecg = augment(ecg, yb)  # on CPU, with its own seeded generator
            ecg, rr, yb = ecg.to(device), rr.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(fwd(ecg.unsqueeze(1), rr), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        va_logits = torch.from_numpy(predict_logits(model, Xb_va, Xr_va))
        with torch.no_grad():
            va_loss = float(criterion(va_logits, y_va_t))
        sched.step(va_loss)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": va_loss,
                        "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)})
        if va_loss < best[0]:
            best = (va_loss, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1
        log(f"  ep{epoch:02d} train={tot / n:.5f} val={va_loss:.5f} lr={opt.param_groups[0]['lr']:.1e} "
            f"({history[-1]['sec']}s)")
        if no_improve >= int(o["patience"]):
            break
    model.load_state_dict(best[2])
    model = model.cpu()
    return model, {"device": str(device), "best_epoch": best[1], "best_val_loss": best[0], "epochs_run": len(history), "history": history}


# --------------------------------------------------------------------------
# One CV job (seed, fold)
# --------------------------------------------------------------------------


def run_job(cfg: dict, seed: int, fold_k: int, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    part = out_dir / f"part_s{seed}_f{fold_k}.npz"
    if part.exists():
        return part
    logf = open(out_dir / f"log_s{seed}_f{fold_k}.txt", "w")

    def log(msg):
        logf.write(msg + "\n")
        logf.flush()

    data = load_dev_data(cfg)
    fold = outer_folds(data, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["split_seed"]), cfg["cv"].get("fold_map"))
    te = np.nonzero(fold == fold_k)[0]
    tr_all = np.nonzero(fold != fold_k)[0]
    assert set(data.subject[te]).isdisjoint(set(data.subject[tr_all]))

    # Inner validation split depends only on split_seed and fold (not on seed).
    itr, iva = inner_split(data.y[tr_all], data.subject[tr_all], int(cfg["cv"]["inner_val_frac_splits"]),
                           int(cfg["cv"]["split_seed"]) + 1000 + fold_k)
    tr, va = tr_all[itr], tr_all[iva]
    assert set(data.subject[tr]).isdisjoint(set(data.subject[va]))
    log(f"fold {fold_k} seed {seed}: train {len(tr)} beats/{len(set(data.pid[tr]))} pts, "
        f"inner-val {len(va)}/{len(set(data.pid[va]))} pts {sorted(set(data.pid[va]))}, "
        f"test {len(te)}/{len(set(data.pid[te]))} pts {sorted(set(data.pid[te]))}")

    t0 = time.time()
    model, info = train_model(cfg, data.X_beats[tr], data.X_rr[tr], data.y[tr],
                              data.X_beats[va], data.X_rr[va], data.y[va],
                              seed=seed * 1000 + fold_k, log=log)
    log(f"done in {time.time() - t0:.0f}s best_epoch={info['best_epoch']}")
    ckpt_dir = Path("models") / "cv" / cfg["exp_id"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"seed{seed}_fold{fold_k}.pt")

    np.savez_compressed(
        part,
        te_idx=te, te_logits=predict_logits(model, data.X_beats[te], data.X_rr[te]),
        va_idx=va, va_logits=predict_logits(model, data.X_beats[va], data.X_rr[va]),
        best_epoch=info["best_epoch"], epochs_run=info["epochs_run"],
        history=json.dumps(info["history"]), train_sec=time.time() - t0,
    )
    logf.close()
    return part


# --------------------------------------------------------------------------
# Merge parts -> oof_{exp}.npz + metrics
# --------------------------------------------------------------------------


def merge(cfg: dict, out_dir: Path, oof_path: Path) -> dict:
    data = load_dev_data(cfg)
    fold = outer_folds(data, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["split_seed"]), cfg["cv"].get("fold_map"))
    seeds = [int(s) for s in cfg["seeds"]]
    K = int(cfg["cv"]["n_splits"])
    N = len(data.y)
    logits = np.full((len(seeds), N, 3), np.nan, dtype=np.float32)
    va_idx, va_fold, va_seed, va_logits = [], [], [], []
    best_epochs = np.zeros((len(seeds), K), dtype=int)
    train_sec = np.zeros((len(seeds), K))
    for si, s in enumerate(seeds):
        for k in range(K):
            p = np.load(out_dir / f"part_s{s}_f{k}.npz")
            logits[si, p["te_idx"]] = p["te_logits"]
            va_idx.append(p["va_idx"]); va_logits.append(p["va_logits"])
            va_fold.append(np.full(len(p["va_idx"]), k)); va_seed.append(np.full(len(p["va_idx"]), s))
            best_epochs[si, k] = int(p["best_epoch"])
            train_sec[si, k] = float(p["train_sec"])
    assert not np.isnan(logits).any()
    prob = np.stack([softmax(l) for l in logits])
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        oof_path, y=data.y, prob=prob, logits=logits, fold=fold, patient_id=data.pid, subject=data.subject, seeds=np.array(seeds),
        val_idx=np.concatenate(va_idx), val_fold=np.concatenate(va_fold), val_seed=np.concatenate(va_seed),
        val_logits=np.concatenate(va_logits), val_y=data.y[np.concatenate(va_idx)],
        best_epochs=best_epochs, train_sec=train_sec, config=json.dumps(cfg),
    )
    return summarize_oof(oof_path, cfg["decision"])


def metrics_for(y, prob, fold, decision_cfg) -> dict:
    """Pooled OOF metrics + per-fold metrics for one seed."""

    pred = decide_from_config(prob, decision_cfg)
    pooled = class_metrics(y, pred, prob)
    per_fold = [class_metrics(y[fold == k], pred[fold == k], prob[fold == k]) for k in np.unique(fold)]
    return {"pooled": pooled, "per_fold": per_fold}


def summarize_oof(oof_path: Path, decision_cfg: dict) -> dict:
    d = np.load(oof_path)
    out = {"oof": str(oof_path), "seeds": d["seeds"].tolist(), "best_epochs": d["best_epochs"].tolist(),
           "rules": {}}
    rules = {"argmax": {"t_sveb": None, "t_veb": None}, "config": decision_cfg}
    is_mit = is_mitbih(d["patient_id"])
    for name, rule in rules.items():
        out["rules"][name] = {"decision": rule,
                              "per_seed": [metrics_for(d["y"], d["prob"][i], d["fold"], rule)
                                           for i in range(len(d["seeds"]))]}
        if (~is_mit).any():
            # MIT-BIH-only view: comparable with the MIT-BIH test set and the literature
            out["rules"][f"{name}_mit"] = {"decision": rule, "per_seed": [
                metrics_for(d["y"][is_mit], d["prob"][i][is_mit], d["fold"][is_mit], rule)
                for i in range(len(d["seeds"]))]}
    return out
