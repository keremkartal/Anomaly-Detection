"""
Paylasilan kosu deposu — ayni (konfigurasyon, seed, protokol) BIR KEZ egitilir.

GEREKCE
-------
Makalenin ilk surumunde `lstm + gat + weighted_sum` konfigurasyonu dort ayri
tabloda gorunuyordu (Tablo 9/10 hibrit satiri, Tablo 11 `T:lstm`, Tablo 12
`weighted_sum`, Tablo 15 trip-instance) ve her biri AYRI bir kosudan
geliyordu. F2/F3 `cudnn.deterministic=True`, F4/F1b `False` ile kosuldugu
icin ayni konfigurasyon 0,8484 ve 0,8836 verdi. Hakem bunu "ayni model
farkli tablolarda farkli sonuc" diye isaretledi.

COZUM
-----
Iki ayri kosunun anlasmasini ummak yerine TEK kosuyu dort yerde okumak.
Bu modul her (spec, seed) ciftini `results/_runs/<protocol_hash>/` altinda
onbellege alir; F2, F3, F4 ve F1b ayni anahtari istediginde ayni diskteki
kaydi doner. Sonuclar bit-ayni olur, GPU nondeterminizmi devreye girmez.

Onbellek anahtari protokol ozetini icerir: protokol degisirse (epoch sayisi,
cuDNN ayari, esik kurali, veri hatti...) eski kayitlar otomatik gecersizlesir
ve yeniden egitilir. Sessiz karisim mumkun degildir.
"""
import json
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .evaluate import compute_metrics, metrics_by_type, pick_threshold, predict
from .hybrid import GenericHybrid
from .models import build_model
from .utils import protocol_stamp, save_json, set_seed

# ------------------------------------------------------------------ spec

# Grafi tanimlayan etiketler — kosu anahtarina girer.
GRAPH_TAGS = {False: "trip_instance", True: "osm_merged"}


def make_spec(temporal="lstm", spatial="gat", fusion="weighted_sum",
              merge_by_osm=False, arch="generic_hybrid", **kw) -> dict:
    """Bir kosuyu benzersiz sekilde tanimlayan sozluk."""
    spec = {"arch": arch, "graph": GRAPH_TAGS[bool(merge_by_osm)]}
    if arch == "generic_hybrid":
        spec.update({"temporal": temporal, "spatial": spatial, "fusion": fusion})
    else:                       # gat_only / lstm_only gibi tek-dal temeller
        spec["model"] = arch
    spec.update(kw)
    return spec


def spec_key(spec: dict, seed: int) -> str:
    """Dosya adina donusen okunabilir anahtar."""
    if spec["arch"] == "generic_hybrid":
        base = f"{spec['temporal']}-{spec['spatial']}-{spec['fusion']}"
    else:
        base = spec["model"]
    extra = "".join(f"-{k}{v}" for k, v in sorted(spec.items())
                    if k not in ("arch", "graph", "temporal", "spatial",
                                 "fusion", "model"))
    return f"{base}{extra}__{spec['graph']}__s{seed}"


# ------------------------------------------------------------------ depo

def _dirs():
    h = protocol_stamp()["hash"]
    d = C.RUNS_DIR / h
    d.mkdir(parents=True, exist_ok=True)
    return d, h


def _build(spec, graph):
    if spec["arch"] == "generic_hybrid":
        return GenericHybrid(spec["temporal"], spec["spatial"], spec["fusion"],
                             num_node_features=graph.num_features,
                             num_nodes=graph.num_nodes)
    kw = {"hidden_dim": 64} if spec["arch"] == "gat_only" else {}
    return build_model(spec["arch"], num_node_features=graph.num_features, **kw)


def get_or_train(spec, seed, ds, node_idx, graph, device,
                 force=False, verbose=True):
    """
    Onbellekte varsa diskten okur, yoksa egitir ve kaydeder.

    Doner: dict
      metrics   : compute_metrics ciktisi + esik, epoch, parametre bilgisi
      prob_test : test olasiliklari (np.ndarray)
      prob_val  : dogrulama olasiliklari
      gate      : kapi agirliklari (yoksa None)
      cached    : diskten mi geldi
    """
    from .train import train_model     # dairesel import'u onlemek icin burada

    d, phash = _dirs()
    key = spec_key(spec, seed)
    jf, nf = d / f"{key}.json", d / f"{key}.npz"

    if jf.exists() and nf.exists() and not force:
        meta = json.loads(jf.read_text(encoding="utf-8"))
        arr = np.load(nf)
        if verbose:
            print(f"  [onbellek] {key}: F1={meta['metrics']['f1']:.4f}", flush=True)
        return {"metrics": meta["metrics"], "spec": meta["spec"],
                "protocol_hash": phash, "cached": True,
                "prob_test": arr["prob_test"], "prob_val": arr["prob_val"],
                "gate": arr["gate"] if "gate" in arr.files else None}

    set_seed(seed)
    model = _build(spec, graph)
    out = train_model(model, ds, node_idx, graph, seed=seed, device=device,
                      select_by="val_f1", verbose=False, tag=key)
    nx_, ei_ = graph.to_tensors(device)
    yv, pv, _ = predict(out["model"], out["loaders"]["val"], nx_, ei_, device)
    yt, pt, gw = predict(out["model"], out["loaders"]["test"], nx_, ei_, device)
    th = pick_threshold(yv, pv)

    m = compute_metrics(yt, pt, th)
    m.update({
        "f1_at_0.5": compute_metrics(yt, pt, 0.5)["f1"],
        "threshold_val": float(th),
        "best_epoch": out["best_epoch"],
        "epochs_run": len(out["history"]),
        "seconds": out["seconds"],
        "sec_per_epoch": out["seconds"] / max(1, len(out["history"])),
        "total_params": out["params"]["total"],
        "seed": seed,
    })
    if hasattr(model, "fusion_parameters"):
        m["fusion_params"] = model.fusion_parameters()
        m["branch_params"] = model.branch_parameters()
    if gw is not None and not np.all(np.isnan(gw)):
        m["gate"] = {
            "min": float(np.nanmin(gw)), "max": float(np.nanmax(gw)),
            "mean": float(np.nanmean(gw)), "std": float(np.nanstd(gw)),
            "mean_normal": float(np.nanmean(gw[yt == 0])),
            "mean_anomaly": float(np.nanmean(gw[yt == 1])),
            "range": float(np.nanmax(gw) - np.nanmin(gw)),
        }
    m["by_type"] = metrics_by_type(yt, pt, ds.type_test, th)

    save_json({"spec": spec, "seed": seed, "protocol": protocol_stamp(),
               "metrics": m}, jf)
    payload = {"prob_test": pt, "prob_val": pv, "y_test": yt, "y_val": yv}
    if gw is not None:
        payload["gate"] = np.asarray(gw)
    np.savez_compressed(nf, **payload)

    if verbose:
        print(f"  {key}: F1={m['f1']:.4f} AUC={m.get('roc_auc', float('nan')):.4f} "
              f"ep{m['best_epoch']}/{m['epochs_run']} {m['seconds']/60:.1f}dk",
              flush=True)

    del model, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": m, "spec": spec, "protocol_hash": phash, "cached": False,
            "prob_test": pt, "prob_val": pv, "gate": gw}


def run_seeds(spec, seeds, ds, node_idx, graph, device, **kw):
    """Bir konfigurasyonu tum seed'lerde kosar; (metrik listesi, olasilik matrisi)."""
    runs, probs = [], []
    for s in seeds:
        r = get_or_train(spec, s, ds, node_idx, graph, device, **kw)
        runs.append(r["metrics"])
        probs.append(r["prob_test"])
    return runs, np.stack(probs)


def inventory() -> dict:
    """Mevcut protokol altinda onbellekteki kosularin dokumu."""
    d, h = _dirs()
    items = {}
    for jf in sorted(d.glob("*.json")):
        meta = json.loads(jf.read_text(encoding="utf-8"))
        items[jf.stem] = {"f1": meta["metrics"]["f1"],
                          "spec": meta["spec"], "seed": meta["seed"]}
    return {"protocol_hash": h, "n_runs": len(items), "runs": items}
