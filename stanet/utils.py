"""Tohum sabitleme, cihaz seçimi, JSON kayıt yardımcıları."""
import json
import os
import random
from pathlib import Path

import sys

import numpy as np
import torch

from . import config as C


def init_console() -> None:
    """Windows cp1252 konsolunda Türkçe karakterlerin patlamasını önler."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def set_seed(seed: int, deterministic: bool = None) -> None:
    """
    numpy + random + torch (CPU & CUDA) tohumlarını sabitler.

    `deterministic` VARSAYILAN OLARAK KAPALI. Gerekçe:

    1. Bit-düzeyinde tekrar zaten mümkün değil — `GATConv` mesaj toplama için
       `scatter_add` kullanır; bu işlem CUDA'da atomik toplama yaptığı için
       kayan nokta toplama sırası deterministik değildir ve
       `cudnn.deterministic` bunu düzeltmez.
    2. `cudnn.deterministic=True`, dilated Conv1d geri yayılımında patolojik
       bir algoritma seçimine yol açıyor: ölçülen 627 ms/batch (deterministik)
       vs 14 ms/batch (normal) — **43 kat yavaşlama**, Graph WaveNet
       kodlayıcısında 50 epoch'u 3,6 dakikadan 157 dakikaya çıkarıyor.

    Yani ayar, sağlamadığı bir garanti için çok yüksek maliyet getiriyordu.
    Tekrarlanabilirlik bunun yerine **çok seed üzerinden ortalama ± std**
    raporlanarak sağlanır.
    """
    if deterministic is None:
        deterministic = C.DETERMINISTIC
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _NpEncoder(json.JSONEncoder):
    """numpy tiplerini JSON'a çevirir."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model) -> dict:
    """Toplam ve modül bazında parametre sayısı (R2-5 için)."""
    out = {"total": sum(p.numel() for p in model.parameters()),
           "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    by_module = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        by_module[top] = by_module.get(top, 0) + p.numel()
    out["by_module"] = by_module
    return out


# --------------------------------------------------------------- protokol damgasi
def protocol_stamp(**extra) -> dict:
    """
    Bir deneyin altinda calistigi TAM protokolu ve 12 haneli ozetini dondurur.

    Amac: farkli protokollerde uretilmis sonuclarin ayni tabloya girmesini
    engellemek. Makalenin ilk surumunde F2/F3 `cudnn.deterministic=True` ile,
    F1b/F4/F5/F6 `False` ile kosulmus; ayni konfigurasyon iki tabloda
    0,8484 ve 0,8836 vermisti. Damga bu hatanin sessizce tekrarlanmasini
    imkansiz kilar (bkz. makale/_verify_numbers.py).
    """
    import hashlib

    stamp = {
        "seeds": list(C.SEEDS),
        "max_epochs": C.MAX_EPOCHS,
        "patience": C.PATIENCE,
        "batch_size": C.BATCH_SIZE,
        "learning_rate": C.LEARNING_RATE,
        "grad_clip": C.GRAD_CLIP,
        "scheduler": [C.SCHEDULER_FACTOR, C.SCHEDULER_PATIENCE],
        "dropout": C.DROPOUT,
        "select_by": "val_f1",
        "threshold_rule": "argmax_val_f1",
        "cudnn_deterministic": bool(C.DETERMINISTIC),
        "cudnn_benchmark": not bool(C.DETERMINISTIC),
        "sequence_length": C.SEQUENCE_LENGTH,
        "step_size": C.STEP_SIZE,
        "split_seed": C.SPLIT_SEED,
        "legacy_pipeline": False,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        **extra,
    }
    # Ozet SEED LISTESINI ICERMEZ: tek bir kosu (konfig+seed) seed sayisindan
    # bagimsiz olarak yeniden kullanilabilsin diye. Seed listesi ayri alanda
    # tutulur ve tablo duzeyinde ayrica karsilastirilir.
    core = {k: v for k, v in stamp.items() if k != "seeds"}
    payload = json.dumps(core, sort_keys=True, default=str)
    stamp["hash"] = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return stamp
