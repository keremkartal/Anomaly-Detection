"""
F21 — Karsilastirma modelleri yeterince arandi mi (degerlendirme: baseline'lar
kendi en iyi ayarlarinda mi calisiyor).

Isaretlenen sorun: STGCN / DCRNN / Graph WaveNet operatorleri tek bir ayarla
kosuldu; arama uzayi, erken durdurma, baslatma ve yakinsama belgelenmedi.
Bu durumda "GAT digerlerini geciyor" bulgusu, GAT'in daha iyi olmasindan
degil, digerlerinin daha kotu ayarlanmasindan kaynaklaniyor olabilir.

TASARIM — simetrik arama
------------------------
Arama YALNIZCA baseline'lara degil, ONERILEN yapilandirmaya da uygulanir.
Aksi halde baseline'lara avantaj tanimis olurduk ve bu da baska yonde
tarafli olurdu.

  arama uzayi   ogrenme orani in {5e-4, 1e-3, 2e-3}
  secim         YALNIZCA dogrulama F1'i (test hicbir asamada gorulmez)
  tohum         arama 2 tohumla, secilen ayar 5 tohumla dogrulanir
  erken durdurma  tum hucrelerde ayni (sabir = C.PATIENCE, olcut val F1)
  esik          her kosu kendi dogrulama esigini kullanir

Ogrenme orani PROTOKOL DAMGASININ icindedir, dolayisiyla her arama noktasi
kendi kosu deposuna yazilir ve farkli ayarlarin sonuclari kazara ayni
tabloya giremez.

YAKINSAMA
---------
Her hucre icin secilen epoch, kosulan epoch ve son epoch'taki gradyan normu
raporlanir; boylece "model daha egitilebilirdi" itirazi sayiyla karsilanir.

KULLANIM
  python experiments/f21_baseline_search.py --pilot
  python experiments/f21_baseline_search.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.data import build_dataset
from stanet.evaluate import compute_metrics
from stanet.graph import build_graph, map_segments
from stanet.runstore import get_or_train, make_spec
from stanet.utils import get_device, init_console, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F21_baseline_search.json"

LRS = [5e-4, 1e-3, 2e-3]
SEARCH_SEEDS = [42, 43]
CONFIRM_SEEDS = C.SEEDS            # 5 tohum
MERGE = False

# F4'teki karsilastirma eksenleri. Referans dahil, arama simetrik.
CANDIDATES = {
    "reference_lstm_gat": dict(temporal="lstm",        spatial="gat"),
    "stgcn_cheb":         dict(temporal="lstm",        spatial="cheb"),
    "dcrnn_diffusion":    dict(temporal="lstm",        spatial="diffusion"),
    "gwn_adaptive":       dict(temporal="lstm",        spatial="adaptive"),
    # kayit adlari `stanet/encoders.py` ile birebir ayni olmali; F4 de
    # bunlari kullaniyor, boylece arama sonuclari Tablo 11 ile karsilastirilabilir
    "stgcn_temporal":     dict(temporal="stgcn_temporal",       spatial="gat"),
    "gwn_temporal":       dict(temporal="wavenet_temporal",     spatial="gat"),
    "transformer":        dict(temporal="transformer_temporal", spatial="gat"),
    "gru":                dict(temporal="gru",         spatial="gat"),
}
FUSION = "weighted_sum"


def run_at_lr(lr, names, seeds, ds, node_idx, graph, device, candidates):
    """Bir ogrenme oraninda tum adaylari kosar. C.LEARNING_RATE gecici degisir."""
    old = C.LEARNING_RATE
    C.LEARNING_RATE = lr
    try:
        stamp = protocol_stamp()
        out = {"lr": lr, "protocol_hash": stamp["hash"], "models": {}}
        for name in names:
            cfg = candidates[name]
            runs = []
            for seed in seeds:
                spec = make_spec(merge_by_osm=MERGE, fusion=FUSION, **cfg)
                r = get_or_train(spec, seed, ds, node_idx, graph, device,
                                 verbose=False)
                m = dict(r["metrics"])
                m["seed"] = seed
                # SECIM METRIGI dogrulama kumesinden hesaplanir. Onbellekteki
                # eski kayitlarda `best_val_score` yok; o durumda kosunun
                # saklanmis dogrulama olasiliklarindan yeniden hesapliyoruz.
                # Test metrigiyle secim yapmak sizinti olurdu.
                if "best_val_score" in m:
                    m["val_f1"] = float(m["best_val_score"])
                else:
                    m["val_f1"] = float(compute_metrics(
                        ds.y_val, r["prob_val"], m["threshold_val"])["f1"])
                runs.append(m)
                print("    %-20s lr=%.0e s%d: val-secimli test F1=%.4f "
                      "(ep %d/%d)%s"
                      % (name, lr, seed, m["f1"], m["best_epoch"], m["epochs_run"],
                         "  [onbellek]" if r["cached"] else ""), flush=True)
            out["models"][name] = {
                "runs": runs,
                # SECIM OLCUTU: dogrulama F1'i. Test yalnizca raporlanir.
                "val_f1_mean": float(np.mean([m["val_f1"] for m in runs])),
                "test_f1_mean": float(np.mean([m["f1"] for m in runs])),
                "test_f1_sd": float(np.std([m["f1"] for m in runs], ddof=1))
                if len(runs) > 1 else 0.0,
                "best_epoch_mean": float(np.mean([m["best_epoch"] for m in runs])),
                "epochs_run_mean": float(np.mean([m["epochs_run"] for m in runs])),
                "hit_epoch_budget": bool(np.mean([m["epochs_run"] for m in runs])
                                         >= C.MAX_EPOCHS - 0.5),
            }
        return out
    finally:
        C.LEARNING_RATE = old


def main():
    pilot = "--pilot" in sys.argv
    names = (["reference_lstm_gat", "stgcn_cheb"] if pilot else list(CANDIDATES))
    lrs = (LRS[:2] if pilot else LRS)
    search_seeds = (SEARCH_SEEDS[:1] if pilot else SEARCH_SEEDS)
    confirm_seeds = (SEARCH_SEEDS[:1] if pilot else CONFIRM_SEEDS)

    device = get_device()
    graph = build_graph(merge_by_osm=MERGE)
    ds = build_dataset(legacy=False)
    node_idx = {sp: map_segments(getattr(ds, "meta_" + sp), graph)[0]
                for sp in ("train", "val", "test")}
    print("cihaz %s | %d aday x %d ogrenme orani x %d tohum = %d arama kosusu\n"
          % (device, len(names), len(lrs), len(search_seeds),
             len(names) * len(lrs) * len(search_seeds)), flush=True)

    results = {"baseline_protocol": protocol_stamp(), "pilot": pilot,
               "search_space": {"learning_rate": lrs},
               "selection": "validation F1 only; test never used for selection",
               "seeds": {"search": search_seeds, "confirm": confirm_seeds},
               "early_stopping": {"criterion": "val_f1", "patience": C.PATIENCE,
                                  "max_epochs": C.MAX_EPOCHS},
               "search": {}, "selected": {}, "confirmed": {}}

    for lr in lrs:
        print("OGRENME ORANI %.0e" % lr, flush=True)
        results["search"]["%.0e" % lr] = run_at_lr(
            lr, names, search_seeds, ds, node_idx, graph, device, CANDIDATES)
        print(flush=True)

    # --------------------------------------------------- secim (dogrulama ile)
    print("=" * 96)
    print("ARAMA SONUCU — secim olcutu dogrulama F1'i")
    print("=" * 96)
    print("  %-20s" % "aday" + "".join("%18s" % ("lr=%.0e" % lr) for lr in lrs)
          + "   secilen")
    for name in names:
        vals = {lr: results["search"]["%.0e" % lr]["models"][name]["val_f1_mean"]
                for lr in lrs}
        best = max(vals, key=vals.get)
        results["selected"][name] = {"lr": best, "val_f1": vals[best],
                                     "val_f1_by_lr": {"%.0e" % k: v
                                                      for k, v in vals.items()}}
        print("  %-20s" % name + "".join("%18.4f" % vals[lr] for lr in lrs)
              + "   %.0e" % best)

    # --------------------------------------------------- dogrulama (5 tohum)
    print("\n" + "=" * 96)
    print("SECILEN AYARDA DOGRULAMA — %d tohum" % len(confirm_seeds))
    print("=" * 96)
    for name in names:
        lr = results["selected"][name]["lr"]
        c = run_at_lr(lr, [name], confirm_seeds, ds, node_idx, graph, device,
                      CANDIDATES)
        results["confirmed"][name] = {**c["models"][name], "lr": lr,
                                      "protocol_hash": c["protocol_hash"]}
    print("\n  %-20s %8s %18s %10s %11s %9s"
          % ("aday", "lr", "test F1", "secilen ep", "kosulan ep", "butce doldu"))
    for name in names:
        r = results["confirmed"][name]
        print("  %-20s %8.0e %8.4f±%.4f %10.1f %11.1f %9s"
              % (name, r["lr"], r["test_f1_mean"], r["test_f1_sd"],
                 r["best_epoch_mean"], r["epochs_run_mean"],
                 "EVET" if r["hit_epoch_budget"] else "-"))

    # varsayilan ayarla fark
    ref_lr = C.LEARNING_RATE
    print("\n  Makaledeki varsayilan ogrenme orani: %.0e" % ref_lr)
    moved = [n for n in names if abs(results["selected"][n]["lr"] - ref_lr) > 1e-12]
    print("  Arama sonunda varsayilandan FARKLI bir orana giden aday: %s"
          % (", ".join(moved) if moved else "yok"))
    results["moved_from_default"] = moved

    path = (C.RESULTS_DIR / "F21_pilot.json") if pilot else OUT
    save_json(results, path)
    print("\nkaydedildi: %s" % path)


if __name__ == "__main__":
    main()
