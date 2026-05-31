import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from src.models.fusion import FusionClassifier
from src.data.torch_datasets import build_dataloaders
from src.data.splits import VAL_PATIENTS
from src.utils.reproducibility import set_global_seeds

set_global_seeds(42)

# Cargar modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = FusionClassifier()
model.load_state_dict(torch.load('models/checkpoints/E06_best.pt',
                                  map_location=device))
model.to(device)
model.eval()

# Cargar val loader
_, val_loader, _ = build_dataloaders(
    data_dir='data/processed/mitbih_aami3',
    batch_size=256,
)

# Recoger probabilidades del val set completo
all_probs, all_true = [], []
with torch.no_grad():
    for ecg, rr, y in val_loader:
        ecg, rr = ecg.to(device), rr.to(device)
        logits  = model(ecg, rr)
        probs   = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_true.append(y.numpy())

probs_val = np.concatenate(all_probs)   # (N, 3)
y_val     = np.concatenate(all_true)    # (N,)

print(f"Val set: {len(y_val)} beats")
print(f"P(SVEB) stats — mean:{probs_val[:,1].mean():.4f} "
      f"max:{probs_val[:,1].max():.4f} "
      f"p95:{np.percentile(probs_val[:,1], 95):.4f}")

# Evaluar thresholds
print(f"\n{'Threshold':>10} | {'SVEB_P':>7} | {'SVEB_R':>7} | "
      f"{'SVEB_F1':>8} | {'F1_macro':>9} | {'VEB_R':>6}")
print("-" * 62)

for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    # Predicción con threshold
    y_pred = probs_val.argmax(axis=1).copy()
    y_pred[probs_val[:, 1] > thr] = 1

    sveb_p  = precision_score(y_val, y_pred, labels=[1],
                               average='macro', zero_division=0)
    sveb_r  = recall_score(y_val, y_pred, labels=[1],
                            average='macro', zero_division=0)
    sveb_f1 = f1_score(y_val, y_pred, labels=[1],
                        average='macro', zero_division=0)
    f1_mac  = f1_score(y_val, y_pred, average='macro', zero_division=0)
    veb_r   = recall_score(y_val, y_pred, labels=[2],
                            average='macro', zero_division=0)

    print(f"{thr:>10.2f} | {sveb_p:>7.4f} | {sveb_r:>7.4f} | "
          f"{sveb_f1:>8.4f} | {f1_mac:>9.4f} | {veb_r:>6.4f}")

# Mostrar el mejor threshold por SVEB F1
best_thr = None
best_sveb_f1 = 0
for thr in np.arange(0.05, 0.55, 0.01):
    y_pred = probs_val.argmax(axis=1).copy()
    y_pred[probs_val[:, 1] > thr] = 1
    sf1 = f1_score(y_val, y_pred, labels=[1],
                   average='macro', zero_division=0)
    if sf1 > best_sveb_f1:
        best_sveb_f1 = sf1
        best_thr = thr

print(f"\nMejor threshold para SVEB F1: {best_thr:.2f} → SVEB F1={best_sveb_f1:.4f}")

# Report completo con el mejor threshold
y_pred_best = probs_val.argmax(axis=1).copy()
y_pred_best[probs_val[:, 1] > best_thr] = 1
print(f"\nClassification report con threshold={best_thr:.2f}:")
print(classification_report(y_val, y_pred_best,
                             target_names=['N','SVEB','VEB'],
                             zero_division=0))