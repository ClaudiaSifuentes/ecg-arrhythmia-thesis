import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute AAMI-relevant metrics for 3-class ECG classification.

    Primary metric: F1_macro (not accuracy — imbalanced dataset).
    Clinical metric: Recall_VEB — false negatives are clinically dangerous.

    Args:
        y_true: ground truth labels (N,)
        y_pred: predicted class indices (N,)
        y_prob: softmax probabilities (N, 3)

    Returns
    -------
    dict with:
        accuracy, f1_macro, recall_veb, specificity_veb, auc_macro, report_str
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = float(np.mean(y_true == y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        target_names=["N", "SVEB", "VEB"],
        output_dict=True,
        zero_division=0,
    )
    recall_veb = float(report["VEB"]["recall"])

    # Specificity VEB = TN / (TN + FP) in one-vs-rest sense for class 2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    tn = int(cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    fp = int(cm[0, 2] + cm[1, 2])
    spec_veb = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    far_veb = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # AUC One-vs-Rest (macro)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    try:
        auc = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        auc = float("nan")

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=["N", "SVEB", "VEB"],
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "recall_veb": recall_veb,
        "specificity_veb": spec_veb,
        "far_veb": far_veb,
        "auc_macro": auc,
        "report_str": report_str,
    }


def heuristic_rr_baseline(X_rr: np.ndarray, y_true: np.ndarray, *, low: float = 0.7, high: float = 1.5) -> dict:
    """Heuristic baseline using only RR_ratio.

    Rule (user-specified):
    - if RR_ratio < 0.7 or > 1.5 => predict VEB (class 2)
    - else => predict N (class 0)

    Notes
    -----
    - This baseline intentionally never predicts SVEB.
    - Assumes RR_ratio is the last column of X_rr (shape: (N, 6)).
    """

    X_rr = np.asarray(X_rr)
    y_true = np.asarray(y_true)
    assert X_rr.ndim == 2 and X_rr.shape[1] == 6, f"Expected X_rr (N,6), got {X_rr.shape}"

    rr_ratio = X_rr[:, -1]
    pred_is_veb = (rr_ratio < low) | (rr_ratio > high)
    y_pred = np.where(pred_is_veb, 2, 0).astype(int)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=["N", "SVEB", "VEB"],
        zero_division=0,
    )

    return {
        "y_pred": y_pred,
        "report_str": report_str,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

CLASS_NAMES = ("N", "SVEB", "VEB")


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    """Flat dict with Se / PPV / F1 per ectopic class, F1-macro and AUC.

    Se (sensitivity) = recall, PPV (positive predictive value) = precision,
    both one-vs-rest, as in AAMI EC57 reporting. PPV is undefined when the
    class is never predicted; it is reported as NaN rather than 0 so means
    across folds are not silently dragged down by an undefined value.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "n": int(len(y_true)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
    }
    for c, name in ((1, "sveb"), (2, "veb")):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        se = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        out.update({f"{name}_n": tp + fn, f"{name}_se": se, f"{name}_ppv": ppv, f"{name}_f1": f1})
    if y_prob is not None:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        try:
            out["auc_macro"] = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
        except ValueError:
            out["auc_macro"] = float("nan")
    return out
