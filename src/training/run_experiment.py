"""Run a single baseline experiment (Week 2) and log results.

This script is intentionally simple and reproducible:
- seed=42
- patient-wise split
- SMOTE ONLY on train
- CNN 1D + RR late fusion (params < 500k)
- Metrics: F1_macro (primary) + Recall_VEB (clinical)
- Append every run to `reports/experiments_log.csv`
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data.torch_datasets import build_dataloaders
from src.models.fusion_model import FusionClassifier
from src.training.metrics import compute_metrics
from src.training.trainer import fit
from src.utils.seeds import set_seeds


@dataclass
class ExperimentResult:
    exp_id: str
    timestamp: str
    lr: float
    dropout: float
    batch_size: int
    max_epochs: int
    best_epoch: int
    best_val_loss: float
    val_f1_macro: float
    val_recall_veb: float
    val_specificity_veb: float
    notes: str
    checkpoint_path: str


def _ensure_experiments_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in asdict(ExperimentResult("", "", 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, "", "")).keys()])
        writer.writeheader()


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", type=str, required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--notes", type=str, default="")
    ap.add_argument(
        "--data_dir",
        type=str,
        default=str(Path("data") / "processed" / "mitbih_aami3"),
    )
    args = ap.parse_args()

    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(
        Path(args.data_dir),
        batch_size=int(args.batch),
        seed=42,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = FusionClassifier(dropout=float(args.dropout)).to(device)
    total_params = _count_params(model)
    assert total_params < 500_000, f"Model too large: {total_params:,} params"

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    ckpt_dir = Path("models") / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{args.exp_id}_best.pt"

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        checkpoint_path=str(checkpoint_path),
    )

    # Load best checkpoint (trainer saves best by val_loss)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Final VAL metrics (for log + criteria gate)
    model.eval()
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_prob_all: list[np.ndarray] = []
    with torch.no_grad():
        for ecg, rr, y in val_loader:
            ecg = ecg.to(device)
            rr = rr.to(device)
            logits = model(ecg, rr)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            y_true_all.extend(y.numpy().tolist())
            y_pred_all.extend(pred.tolist())
            y_prob_all.append(prob)

    y_true = np.asarray(y_true_all, dtype=int)
    y_pred = np.asarray(y_pred_all, dtype=int)
    y_prob = np.concatenate(y_prob_all, axis=0)

    m = compute_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    best_epoch = int(history.get("best_epoch", -1))
    best_val_loss = float(history.get("best_val_loss", float("inf")))

    result = ExperimentResult(
        exp_id=str(args.exp_id),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        lr=float(args.lr),
        dropout=float(args.dropout),
        batch_size=int(args.batch),
        max_epochs=int(args.max_epochs),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        val_f1_macro=float(m["f1_macro"]),
        val_recall_veb=float(m["recall_veb"]),
        val_specificity_veb=float(m["specificity_veb"]),
        notes=str(args.notes),
        checkpoint_path=str(checkpoint_path),
    )

    print("\n=== E01 RESULTS ===")
    print(f"exp_id: {result.exp_id}")
    print(f"device: {device}")
    print(f"params: {total_params:,}")
    print(f"best_epoch: {result.best_epoch}")
    print(f"best_val_loss: {result.best_val_loss:.6f}")
    print(f"val_f1_macro: {result.val_f1_macro:.4f}")
    print(f"val_recall_veb: {result.val_recall_veb:.4f}")
    print(f"val_specificity_veb: {result.val_specificity_veb:.4f}")
    print("\nClassification report (VAL):")
    print(m["report_str"])

    log_path = Path("reports") / "experiments_log.csv"
    _ensure_experiments_log(log_path)
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        writer.writerow(asdict(result))


if __name__ == "__main__":
    main()
