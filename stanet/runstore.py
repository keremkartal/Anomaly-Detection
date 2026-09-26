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
                # eski kayitlarda history yok; None doner, cagiran taraf bunu
                # "bu kosu egitim egrisi olmadan onbelleklenmis" diye okur
                "history": meta.get("history"),
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
        # SECIM metrigi. Hiperparametre aramasi bunu kullanmali; test
        # metrigiyle secim yapmak sizintidir.
        "best_val_score": float(out["best_score"]),
        "select_by": out["select_by"],
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

    # Egitim seyri de saklanir. Hakem 1 madde 9 ve Hakem 3 madde 7, gate
    # cokmesinin mekanizmasi icin egitim egrileri ve epoch bazinda gate
    # dagilimi istiyor; bunlar yalnizca son metrikten cikarilamaz.
    save_json({"spec": spec, "seed": seed, "protocol": protocol_stamp(),
               "metrics": m, "history": out["history"]}, jf)
    payload = {"prob_test": pt, "prob_val": pv, "y_test": yt, "y_val": yv}
    if gw is not None:
        payload["gate"] = np.asarray(gw)
    np.savez_compressed(nf, **payload)
    # Agirliklar da saklanir. Cikarim anindaki duyarlilik deneyleri (gurultu,
    # kenar silme, indukfif kurulum) egitilmis modeli tekrar CALISTIRMAYI
    # gerektirir; agirlik yoksa yeniden egitmek gerekir ve GAT'in
    # nondeterminizmi yuzunden onbellekteki sayiyi birebir tutturamayiz.
    torch.save(out["model"].state_dict(), d / f"{key}.pt")

    if verbose:
        print(f"  {key}: F1={m['f1']:.4f} AUC={m.get('roc_auc', float('nan')):.4f} "
              f"ep{m['best_epoch']}/{m['epochs_run']} {m['seconds']/60:.1f}dk",
              flush=True)

    hist = out["history"]
    del model, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": m, "spec": spec, "protocol_hash": phash, "cached": False,
            "history": hist, "prob_test": pt, "prob_val": pv, "gate": gw}


def weights_path(spec, seed) -> Path:
    """Bu kosunun agirlik dosyasi (var olmak zorunda degil)."""
    d, _ = _dirs()
    return d / f"{spec_key(spec, seed)}.pt"


def load_or_train_weights(spec, seed, ds, node_idx, graph, device, verbose=True):
    """
    Cikarim deneyleri icin EGITILMIS MODELI dondurur.

    Uc durum var:

      1. Agirlik dosyasi varsa: model kurulur, agirliklar yuklenir. Hicbir
         sey yeniden egitilmez, metrik onbellegi de degismez.
      2. Agirlik yok ama metrik onbellegi VAR (eski kosular): model yeniden
         egitilir ve agirliklar kaydedilir, ama **metrik onbellegine
         dokunulmaz**. GAT'in `scatter_add` toplamasi GPU'da deterministik
         olmadigi icin yeniden egitim ayni sayiyi birebir vermez; makaledeki
         degeri degistirmek yerine ikisini birden raporluyoruz
         (`f1_cached`, `f1_retrained`) ki okur farki gorsun.
      3. Hicbiri yoksa: `get_or_train` normal yolu isletir, agirlik da yazilir.

    Doner: dict(model, f1_cached, f1_retrained, weights_reused, threshold_val)
    """
    from .train import train_model

    d, _ = _dirs()
    key = spec_key(spec, seed)
    jf, wf = d / f"{key}.json", d / f"{key}.pt"

    cached_metrics = None
    if jf.exists():
        cached_metrics = json.loads(jf.read_text(encoding="utf-8"))["metrics"]

    if wf.exists():
        model = _build(spec, graph).to(device)
        model.load_state_dict(torch.load(wf, map_location=device))
        model.eval()
        if verbose:
            print(f"  [agirlik] {key}", flush=True)
        return {"model": model, "weights_reused": True,
                "f1_cached": cached_metrics["f1"] if cached_metrics else None,
                "f1_retrained": None,
                "threshold_val": (cached_metrics or {}).get("threshold_val")}

    if cached_metrics is None:                    # 3. durum — normal yol
        get_or_train(spec, seed, ds, node_idx, graph, device, verbose=verbose)
        return load_or_train_weights(spec, seed, ds, node_idx, graph, device,
                                     verbose=False)

    # 2. durum — yalnizca agirlik icin yeniden egit, metrigi BOZMA
    set_seed(seed)
    model = _build(spec, graph)
    out = train_model(model, ds, node_idx, graph, seed=seed, device=device,
                      select_by="val_f1", verbose=False, tag=key + "-weights")
    nx_, ei_ = graph.to_tensors(device)
    yv, pv, _ = predict(out["model"], out["loaders"]["val"], nx_, ei_, device)
    yt, pt, _ = predict(out["model"], out["loaders"]["test"], nx_, ei_, device)
    th = pick_threshold(yv, pv)
    f1_new = compute_metrics(yt, pt, th)["f1"]
    torch.save(out["model"].state_dict(), wf)
    if verbose:
        print(f"  [yeniden egitildi, yalnizca agirlik] {key}: "
              f"onbellek F1={cached_metrics['f1']:.4f} yeni F1={f1_new:.4f}",
              flush=True)
    m = out["model"].to(device)
    m.eval()
    return {"model": m, "weights_reused": False,
            "f1_cached": cached_metrics["f1"], "f1_retrained": float(f1_new),
            "threshold_val": float(th)}


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
