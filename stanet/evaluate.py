"""Değerlendirme metrikleri ve istatistiksel karşılaştırma."""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config as C


# --------------------------------------------------------------------- metrikler
def compute_metrics(y_true, y_prob, threshold=C.THRESHOLD) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob > threshold).astype(int)

    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()
    out["tn"], out["fp"], out["fn"], out["tp"] = (int(cm[0, 0]), int(cm[0, 1]),
                                                  int(cm[1, 0]), int(cm[1, 1]))
    return out


def pick_threshold(y_true, y_prob, grid=None) -> float:
    """Validation üzerinde F1'i maksimize eden eşik (test'e asla bakılmaz)."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    scores = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics_by_type(y_true, y_prob, atype, threshold=C.THRESHOLD) -> dict:
    """
    Alt-tip bazında performans (R2-4).
    Negatifler her alt-küme için ortak tutulur; yalnızca pozitif alt-tipi değişir.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    atype = np.asarray(atype).ravel()
    neg = y_true == 0
    out = {}
    for t in np.unique(atype[y_true == 1]):
        mask = neg | ((y_true == 1) & (atype == t))
        name = C.ANOMALY_NAMES.get(int(t), str(t))
        out[name] = compute_metrics(y_true[mask], y_prob[mask], threshold)
    return out


# --------------------------------------------------------------------- çıkarım
@torch.no_grad()
def predict(model, loader, node_x, edge_index, device):
    """Döner: y_true, y_prob, gate_weight"""
    model.eval()
    yt, yp, gw = [], [], []
    for x_b, y_b, idx_b in loader:
        x_b, idx_b = x_b.to(device), idx_b.to(device)
        pred, w = model(x_b, node_x, edge_index, idx_b)
        yt.append(y_b.numpy().ravel())
        yp.append(pred.cpu().numpy().ravel())
        gw.append(w.cpu().numpy().ravel() if w is not None
                  else np.full(len(y_b), np.nan))
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(gw)
