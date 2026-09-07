"""
Ortak eğitim döngüsü — tüm varyantlar için AYNI protokol.

Kritik ilke: checkpoint seçim kriteri (`select_by`) tüm varyantlar için aynıdır.
Orijinal notebook'ta hibrit model val-loss'a göre early-stop edip val-F1'e göre
checkpoint kaydediyordu; bu karışım varyantlar arası karşılaştırmayı bozar.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .evaluate import compute_metrics, predict
from .utils import count_parameters, get_device, set_seed


class SeqDataset(Dataset):
    def __init__(self, X, y, node_idx):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
        self.idx = torch.tensor(node_idx, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.idx[i]


def weighted_bce(pred, target, pos_weight, eps=1e-7):
    """Makale Eş. 18. Model sigmoid uygulanmış olasılık döndürdüğü için clamp gerekli."""
    pred = torch.clamp(pred, eps, 1.0 - eps)
    loss = -(pos_weight * target * torch.log(pred)
             + (1 - target) * torch.log(1 - pred))
    return loss.mean()


def make_loaders(ds, node_idx, batch_size=C.BATCH_SIZE):
    tr = DataLoader(SeqDataset(ds.X_train, ds.y_train, node_idx["train"]),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(SeqDataset(ds.X_val, ds.y_val, node_idx["val"]),
                    batch_size=batch_size, shuffle=False)
    te = DataLoader(SeqDataset(ds.X_test, ds.y_test, node_idx["test"]),
                    batch_size=batch_size, shuffle=False)
    return tr, va, te


def train_model(model, ds, node_idx, graph, seed=42, device=None,
                max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE,
                lr=C.LEARNING_RATE, select_by="val_f1", verbose=True, tag=""):
    """
    select_by: 'val_f1' | 'val_loss'  -> hem checkpoint hem early-stop bu kritere bağlı
    Döner: dict(history, best_epoch, best_score, state_dict, params, seconds)
    """
    device = device or get_device()
    set_seed(seed)
    model = model.to(device)
    node_x, edge_index = graph.to_tensors(device)

    tr_loader, va_loader, te_loader = make_loaders(ds, node_idx)
    pw = torch.tensor(ds.pos_weight, dtype=torch.float, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=C.SCHEDULER_FACTOR, patience=C.SCHEDULER_PATIENCE)

    best_score = -np.inf
    best_state, best_epoch = None, -1
    bad = 0
    history = []
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        tl = 0.0
        for x_b, y_b, i_b in tr_loader:
            x_b, y_b, i_b = x_b.to(device), y_b.to(device), i_b.to(device)
            opt.zero_grad()
            pred, _ = model(x_b, node_x, edge_index, i_b)
            loss = weighted_bce(pred, y_b, pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            tl += loss.item()
        train_loss = tl / len(tr_loader)

        # doğrulama
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x_b, y_b, i_b in va_loader:
                x_b, y_b, i_b = x_b.to(device), y_b.to(device), i_b.to(device)
                pred, _ = model(x_b, node_x, edge_index, i_b)
                vl += weighted_bce(pred, y_b, pw).item()
        val_loss = vl / len(va_loader)

        yv, pv, _ = predict(model, va_loader, node_x, edge_index, device)
        vm = compute_metrics(yv, pv)
        sched.step(val_loss)

        score = vm["f1"] if select_by == "val_f1" else -val_loss
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val_loss, "val_f1": vm["f1"],
                        "val_auc": vm.get("roc_auc"), "lr": opt.param_groups[0]["lr"]})

        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(f"  [{tag}] ep{epoch+1:3d}/{max_epochs} "
                  f"tr={train_loss:.4f} vl={val_loss:.4f} vF1={vm['f1']:.4f}"
                  f"{'  *' if bad == 0 else ''}")

        if bad >= patience:
            if verbose:
                print(f"  [{tag}] early stop @ep{epoch+1} (best ep{best_epoch})")
            break

    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch,
            "best_score": float(best_score), "state_dict": best_state,
            "params": count_parameters(model), "seconds": time.time() - t0,
            "select_by": select_by, "seed": seed, "model": model,
            "loaders": {"train": tr_loader, "val": va_loader, "test": te_loader}}
