import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.training.metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for ecg, rr, y in loader:
        ecg, rr, y = ecg.to(device), rr.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(ecg, rr)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred, all_prob = [], [], []

    for ecg, rr, y in loader:
        ecg, rr, y = ecg.to(device), rr.to(device), y.to(device)
        logits = model(ecg, rr)
        loss = criterion(logits, y)

        prob = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * len(y)
        n += len(y)
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    y_prob = np.concatenate(all_prob) if all_prob else np.zeros((0, 3))

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = (total_loss / n) if n > 0 else float("nan")
    return metrics


def fit(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    max_epochs=100,
    patience=15,
    lr_patience=7,
    lr_factor=0.5,
    checkpoint_path="models/checkpoints/best_model.pt",
    log_path="reports/training_log.csv",
    device=None,
):
    """Full training loop with EarlyStopping + ReduceLROnPlateau + Checkpoint.

    Metric monitored: val_loss (lower is better).
    Saves best model weights to `checkpoint_path`.
    Logs per-epoch metrics to `log_path` (CSV).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improv = 0
    best_state = None

    log_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_f1_macro",
        "val_recall_veb",
        "val_auc_macro",
        "lr",
    ]

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    print(f"Training on {device} | max_epochs={max_epochs} | patience={patience}")
    print(
        f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>9} "
        f"{'F1mac':>7} {'RecVEB':>7} {'LR':>8}"
    )
    print("-" * 55)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, criterion, device)
        val_loss = float(val_m["loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improv = 0
            torch.save(best_state, checkpoint_path)
        else:
            epochs_no_improv += 1

        # CSV log
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(float(train_loss), 5),
                    "val_loss": round(val_loss, 5),
                    "val_acc": round(float(val_m["accuracy"]), 4),
                    "val_f1_macro": round(float(val_m["f1_macro"]), 4),
                    "val_recall_veb": round(float(val_m["recall_veb"]), 4),
                    "val_auc_macro": round(float(val_m["auc_macro"]), 4)
                    if not np.isnan(val_m["auc_macro"])
                    else float("nan"),
                    "lr": lr_now,
                }
            )

        print(
            f"{epoch:>6} {train_loss:>10.5f} {val_loss:>9.5f} "
            f"{val_m['f1_macro']:>7.4f} {val_m['recall_veb']:>7.4f} "
            f"{lr_now:>8.2e}  ({time.time() - t0:.1f}s)"
        )

        # Early stopping
        if epochs_no_improv >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest val_loss: {best_val_loss:.5f} — weights restored.")

    return model