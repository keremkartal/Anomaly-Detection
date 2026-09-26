"""
F18 — Transduktif mi induktif mi (degerlendirme: test dugumleri egitimde goruluyor mu).

SORUN
-----
Model egitilirken graf TAMAMEN verilir: mesaj gecisi, test trip'lerine ait
dugumler uzerinden de akar. Bu TRANSDUKTIF bir kurulumdur. Test trip'lerinin
ETIKETLERI hicbir zaman kullanilmaz, ama yol kesitlerinin oznitelikleri ve
komsuluklari egitim sirasinda temsile katkida bulunur. Yeni bir yola
tasindiginda model bu avantaji bulamaz.

Makale bunu "veri kokeni" tablosunda acikca belgeliyordu; burada BEDELI
olculuyor.

TASARIM
-------
Her fold icin iki kosu, her sey ayni, tek fark grafin egitim sirasindaki hali:

  transductive  egitimde tam graf (mevcut kurulum, F5'ten onbellek)
  inductive     egitimde test trip'lerinin dugumleri VE onlara degen
                kenarlar cikarilir; dugum oznitelikleri sifirlanir.
                Cikarimda tam graf geri verilir.

Dugumler silinmek yerine YALITILIR ve oznitelikleri sifirlanir: boylece dugum
indeksleri (dolayisiyla `seg_idx` esleme) degismez ve iki kosu birebir ayni
mimariyi, ayni parametre sayisini kullanir. Etkisi silmekle aynidir — yalitik
ve sifir oznitelikli bir dugum ne bilgi tasir ne de baskasina bilgi gonderir.

Dogrulama kumesi egitimde GORULUR (dogrulama zaten egitim sirasinda
kullaniliyor: erken durdurma ve esik secimi oradan geliyor). Yalnizca TEST
trip'leri gizlenir.

Okuma: fark kucukse transduktif kurulumun avantaji ihmal edilebilir ve
makaledeki sayilar induktif bir dagitimda da gecerlidir. Fark buyukse
makalenin genelleme iddiasi bu kadar daraltilmalidir.

KULLANIM
  python experiments/f18_inductive.py --pilot
  python experiments/f18_inductive.py
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from stanet import config as C
from stanet.cluster_stats import (holm_correction, nadeau_bengio, tost_equivalence)
from stanet.data import build_cv_folds
from stanet.graph import build_graph, map_segments
from stanet.runstore import get_or_train, make_spec
from stanet.utils import get_device, init_console, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F18_inductive.json"
SEEDS = [42, 43]
MERGE = False
DELTA = 0.02
CONFIGS = {
    "hybrid_gat":   dict(temporal="lstm", spatial="gat", fusion="weighted_sum"),
    "stanet_gated": dict(temporal="lstm", spatial="gat", fusion="gated"),
}


def hide_nodes(graph, hidden_idx):
    """Verilen dugumleri yalitir ve ozniteliklerini sifirlar.

    Silmek yerine yalitmak, dugum indekslerini ve dolayisiyla `seg_idx`
    eslemesini korur; iki kosu ayni mimariyi ve ayni parametre sayisini
    kullanir. Yalitik ve sifir oznitelikli bir dugum mesaj gecisine katkida
    bulunmaz, dolayisiyla etki silmekle aynidir.
    """
    hidden = np.zeros(graph.num_nodes, dtype=bool)
    hidden[np.asarray(sorted(set(int(i) for i in hidden_idx)))] = True
    ei = graph.edge_index.numpy()
    keep = ~(hidden[ei[0]] | hidden[ei[1]])
    feats = graph.node_features.copy()
    feats[hidden] = 0.0
    return replace(graph,
                   node_features=feats,
                   edge_index=torch.tensor(ei[:, keep], dtype=torch.long),
                   info={**graph.info, "setting": "inductive",
                         "n_hidden_nodes": int(hidden.sum()),
                         "n_edges_removed": int((~keep).sum()),
                         "n_edges_kept": int(keep.sum())})


def train_hidden_eval_full(spec, seed, ds, node_idx, g_train, g_full, device):
    """Egitim GIZLENMIS grafla, degerlendirme TAM grafla.

    Induktif kurulumun tanimi budur: model test kesitlerini egitim sirasinda
    hic gormez, ama cikarim aninda onlarin gercek oznitelikleri ve komsuluklari
    verilir ve modelin bunlari genellemesi beklenir. GAT'in parametreleri
    dugume bagli olmadigi icin bu mumkundur.

    `get_or_train` tek graf aldigi icin burada kendi kosu fonksiyonumuz var;
    sonuc `ind=1` tasiyan ayri bir onbellek anahtarina yazilir.
    """
    from stanet.evaluate import compute_metrics, pick_threshold, predict
    from stanet.runstore import _build, _dirs, spec_key
    from stanet.train import train_model
    from stanet.utils import load_json, set_seed

    d, _ = _dirs()
    key = spec_key(spec, seed)
    jf, nf = d / (key + ".json"), d / (key + ".npz")
    if jf.exists() and nf.exists():
        return dict(load_json(jf)["metrics"]), True

    set_seed(seed)
    model = _build(spec, g_train)
    out = train_model(model, ds, node_idx, g_train, seed=seed, device=device,
                      select_by="val_f1", verbose=False, tag=key)
    # DEGERLENDIRME tam grafla
    nx_, ei_ = g_full.to_tensors(device)
    yv, pv, _ = predict(out["model"], out["loaders"]["val"], nx_, ei_, device)
    yt, pt, _ = predict(out["model"], out["loaders"]["test"], nx_, ei_, device)
    th = pick_threshold(yv, pv)
    m = compute_metrics(yt, pt, th)
    m.update({"threshold_val": float(th), "best_epoch": out["best_epoch"],
              "epochs_run": len(out["history"]), "seed": seed,
              "total_params": out["params"]["total"],
              "setting": "inductive",
              "graph_train": g_train.info.get("setting"),
              "graph_eval": "full"})
    save_json({"spec": spec, "seed": seed, "protocol": protocol_stamp(),
               "metrics": m, "history": out["history"]}, jf)
    np.savez_compressed(nf, prob_test=pt, prob_val=pv, y_test=yt, y_val=yv)
    torch.save(out["model"].state_dict(), d / (key + ".pt"))
    del model, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return m, False


def main():
    pilot = "--pilot" in sys.argv
    device = get_device()
    stamp = protocol_stamp()
    full = build_graph(merge_by_osm=MERGE)
    folds = build_cv_folds(n_folds=C.N_FOLDS, legacy=False)
    if pilot:
        folds, seeds, configs = folds[:1], SEEDS[:1], {"hybrid_gat": CONFIGS["hybrid_gat"]}
    else:
        seeds, configs = SEEDS, CONFIGS

    print("protokol %s | cihaz %s | %d fold x %d tohum x %d konfig x 2 kurulum = %d kosu\n"
          % (stamp["hash"], device, len(folds), len(seeds), len(configs),
             len(folds) * len(seeds) * len(configs) * 2), flush=True)

    fold_idx, fold_graph = {}, {}
    for f in folds:
        k = f.info["fold"]
        idx = {sp: map_segments(getattr(f, "meta_" + sp), full)[0]
               for sp in ("train", "val", "test")}
        fold_idx[k] = idx
        # test trip'lerinin dugumleri; egitim/dogrulamada da gorunen bir dugum
        # varsa o GIZLENMEZ (ayni fiziksel kesit zaten egitimde var demektir)
        test_only = set(int(i) for i in idx["test"]) - set(int(i) for i in idx["train"]) \
            - set(int(i) for i in idx["val"])
        fold_graph[k] = hide_nodes(full, test_only)
        info = fold_graph[k].info
        print("  fold %d: test-ozel dugum %d/%d gizlendi, %d kenar silindi (%d kaldi)"
              % (k, info["n_hidden_nodes"], full.num_nodes,
                 info["n_edges_removed"], info["n_edges_kept"]), flush=True)
    print()

    results = {"protocol": stamp,
               "design": {"n_folds": len(folds), "seeds": seeds, "pilot": pilot,
                          "merge_by_osm": MERGE,
                          "what_is_hidden": ("yalnizca TEST trip'lerine ozel dugumler; "
                                             "egitim ya da dogrulamada da gorunen bir "
                                             "kesit gizlenmez"),
                          "fold_graphs": {k: g.info for k, g in fold_graph.items()}},
               "models": {}}
    runmap = {}

    for name, cfg in configs.items():
        for setting in ("transductive", "inductive"):
            runs = []
            for f in folds:
                k = f.info["fold"]
                for seed in seeds:
                    if setting == "transductive":
                        spec = make_spec(merge_by_osm=MERGE, fold=k, cv=C.N_FOLDS,
                                         **cfg)
                        r = get_or_train(spec, seed, f, fold_idx[k], full, device,
                                         verbose=False)
                        m = dict(r["metrics"])
                        cached = r["cached"]
                    else:
                        spec = make_spec(merge_by_osm=MERGE, fold=k, cv=C.N_FOLDS,
                                         ind=1, **cfg)
                        m, cached = train_hidden_eval_full(
                            spec, seed, f, fold_idx[k], fold_graph[k], full, device)
                    m.update({"fold": k, "seed": seed})
                    runs.append(m)
                    runmap.setdefault((name, setting), {})[(k, seed)] = float(m["f1"])
                    print("  %-13s %-12s fold%d s%d: F1=%.4f%s"
                          % (name, setting, k, seed, m["f1"],
                             "  [onbellek]" if cached else ""), flush=True)
            f1s = np.array([m["f1"] for m in runs])
            results["models"].setdefault(name, {})[setting] = {
                "runs": runs,
                "f1_mean": float(f1s.mean()),
                "f1_sd": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
                "f1_min": float(f1s.min()), "n": int(len(f1s))}
            print("  -> %s / %s: F1 %.4f+-%.4f\n"
                  % (name, setting, f1s.mean(),
                     results["models"][name][setting]["f1_sd"]), flush=True)

    # ----------------------------------------------------- induktif - transduktif
    fi = [f.info for f in folds]
    n_tr = int(np.mean([x["trips"]["train"] for x in fi]))
    n_te = int(np.mean([x["trips"]["test"] for x in fi]))
    print("=" * 96)
    print("INDUKTIF - TRANSDUKTIF (eslestirilmis, Nadeau-Bengio)")
    print("=" * 96)
    print("  %-16s %9s %22s %8s %8s %11s"
          % ("konfig", "ort fark", "NB %95 GA", "NB p", "Holm p", "TOST"))
    rows, pv = {}, {}
    for name in configs:
        a, b = (name, "inductive"), (name, "transductive")
        keys = sorted(set(runmap.get(a, {})) & set(runmap.get(b, {})))
        if len(keys) < 2:
            continue
        diffs = [runmap[a][k] - runmap[b][k] for k in keys]
        nb = nadeau_bengio(diffs, n_train=n_tr, n_test=n_te)
        ts = tost_equivalence(diffs, DELTA, n_train=n_tr, n_test=n_te)
        rows[name] = {"n_paired": len(keys), "diffs": diffs,
                      "nadeau_bengio": nb, "tost": ts}
        pv[name] = nb["p_corrected"]
    for k, v in (holm_correction(pv).items() if pv else []):
        rows[k]["holm"] = v
    for k, v in rows.items():
        nb, ts = v["nadeau_bengio"], v["tost"]
        verdict = ("esdeger" if ts["equivalent"]
                   else ("FARK VAR" if v.get("holm", {}).get("significant")
                         else "belirsiz"))
        v["verdict"] = verdict
        print("  %-16s %+9.4f [%+.4f, %+.4f] %8.4f %8.4f %11s"
              % (k, nb["mean_diff"], nb["ci_lo"], nb["ci_hi"], nb["p_corrected"],
                 v.get("holm", {}).get("p_holm", float("nan")), verdict))

    results["comparison"] = {"n_train_trips": n_tr, "n_test_trips": n_te,
                             "delta": DELTA, "rows": rows}
    path = (C.RESULTS_DIR / "F18_pilot.json") if pilot else OUT
    save_json(results, path)
    print("\nkaydedildi: %s" % path)


if __name__ == "__main__":
    main()
