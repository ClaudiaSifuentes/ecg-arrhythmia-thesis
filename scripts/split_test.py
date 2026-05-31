import torch
import numpy as np
from src.data.splits import TEST_PATIENTS
from src.models.fusion import FusionClassifier
from src.data.torch_datasets import ECGBeatDataset
from src.training.trainer import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Cargar datos completos
X_beats    = np.load('data/processed/mitbih_aami3/X_beats.npy')
X_rr       = np.load('data/processed/mitbih_aami3/X_rr.npy')
y          = np.load('data/processed/mitbih_aami3/y_beats.npy')
patient_id = np.load('data/processed/mitbih_aami3/patient_id.npy')

# Filtrar SOLO pacientes de test
test_mask = np.isin(patient_id, TEST_PATIENTS)
print(f"Test beats: {test_mask.sum()}")  # debe ser ~13,400
print(f"Test patients: {np.unique(patient_id[test_mask])}")

X_beats_test = X_beats[test_mask]
X_rr_test    = X_rr[test_mask]
y_test       = y[test_mask]

# DataLoader
test_loader = torch.utils.data.DataLoader(
    ECGBeatDataset(X_beats_test, X_rr_test, y_test),
    batch_size=512,
    shuffle=False,
    num_workers=0
)

# Cargar modelo E06
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionClassifier(dropout=0.5, use_rr=True)
model.load_state_dict(torch.load("models/checkpoints/E06_best.pt", map_location=device))
model.to(device)
model.eval()

# Evaluar
criterion = torch.nn.CrossEntropyLoss()
metrics = evaluate(model, test_loader, criterion, device)
print(metrics["report_str"])  # Classification report con el support correcto

# Confusion matrix
y_true = []
y_pred = []
y_proba = []

with torch.no_grad():
    for xb, xrr, yb in test_loader:
        xb, xrr = xb.to(device), xrr.to(device)
        logits = model(xb, xrr)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true.append(yb.numpy())
        y_pred.append(preds)
        y_proba.append(probs)

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
y_proba = np.concatenate(y_proba)

cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["N","SVEB","VEB"], yticklabels=["N","SVEB","VEB"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix E06")
plt.tight_layout()
plt.savefig("reports/confusion_matrix_E06.png")
plt.close()

# ROC curve OvR
n_classes = y_proba.shape[1]
y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
plt.figure(figsize=(7,6))
for i, label in enumerate(["N","SVEB","VEB"]):
    fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_proba[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve E06 (OvR)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/roc_curve_E06.png")
plt.close()