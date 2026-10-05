"""
F14 — Uzamsal dalin katkisi OZNITELIK mi MESAJ GECISI mi (degerlendirme: ablasyonun cozunurlugu).

Isaretlenen sorun: ablasyon uzamsal dali TAMAMEN kaldiriyor, bu yuzden dugum
ozniteliklerinin katkisini yol topolojisi uzerindeki mesaj gecisinin
katkisindan ayiramiyor.

Dogru. `lstm_only` grafi tamamen kaldiriyor; GAT kazancinin dugum
ozniteliklerinden mi yoksa komsuluk toplamasindan mi geldigi belli olmuyor.
Burada dort kontrol AYRI AYRI EGITILIR — cikarim aninda mudahale degil,
kosulun kendisiyle egitim, cunku sorulan sey dayaniklilik degil katkidir:

  full          referans: gercek oznitelikler, gercek topoloji
  no_edges      oznitelikler ayni, kenarlar KALDIRILDI (yalnizca oz-dongu)
                -> mesaj gecisi yok; GAT dugum bazli bir MLP'ye dusuyor
  shuffled_feat topoloji ayni, oznitelik satirlari KARISTIRILDI
                -> mesaj gecisi var ama tasidigi bilgi yok
  random_edges  oznitelikler ayni, kenar SAYISI ayni, uclar rastgele
                -> "gercek topoloji" ile "herhangi bir topoloji" ayrimi
  lstm_only     uzamsal dal hic yok (F5'ten, karsilastirma icin)

Okuma kilavuzu:
  full ~ no_edges           -> kazanc OZNITELIKTEN, mesaj gecisi gereksiz
  full >> no_edges          -> mesaj gecisi gercekten is yapiyor
  full ~ shuffled_feat      -> kazanc topolojiden, ozniteliklerden degil
  full ~ random_edges       -> yol agi yapisi degil, herhangi bir komsuluk yeter

Butun kontroller F5 ile AYNI fold'lar, ayni tohumlar, ayni esik politikasi
altinda kosar; boylece Tablo 13 ile dogrudan karsilastirilabilir.

KULLANIM
  python experiments/f14_graph_controls.py --pilot   # 1 fold x 1 tohum
  python experiments/f14_graph_controls.py
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from stanet import config as C
from stanet.cluster_stats import (cluster_bootstrap_diff, cluster_permutation_test,
                                  cluster_summary, holm_correction, nadeau_bengio,
                                  tost_equivalence)
from stanet.data import build_cv_folds
from stanet.graph import build_graph, map_segments
from stanet.runstore import get_or_train, make_spec
from stanet.utils import get_device, init_console, protocol_stamp, save_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f5_cv import pooled_by_type, pooled_metrics          # noqa: E402

init_console()
SEEDS = [int(x) for x in os.environ.get("F14_SEEDS", "42,43").split(",")]
OUT_NAME = os.environ.get("F14_OUT", "F14_graph_controls") + ".json"
OUT = C.RESULTS_DIR / OUT_NAME
MERGE = False
BASE = dict(temporal="lstm", spatial="gat", fusion="weighted_sum")
CONTROL_SEED = 20260926        # oznitelik karistirma ve rastgele kenar icin
DELTA = 0.02                   # TOST esdegerlik siniri (F12 ile ayni gerekce)


def self_loops_only(g):
    """Kenarlari kaldirir; yalnizca oz-donguler kalir.

    GATConv oz-donguyu kendisi ekler ama kutuphane varsayilanina guvenmek
    yerine acik yaziyoruz: boylece 'mesaj gecisi yok' kosulu kodda gorunur.
    """
    n = g.num_nodes
    idx = torch.arange(n, dtype=torch.long)
    return replace(g, edge_index=torch.stack([idx, idx]),
                   info={**g.info, "control": "no_edges",
                         "num_edges_original": int(g.edge_index.shape[1])})


def shuffled_features(g, seed=CONTROL_SEED):
    """Dugum oznitelik satirlarini karistirir; topoloji degismez."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(g.num_nodes)
    return replace(g, node_features=g.node_features[perm].copy(),
                   info={**g.info, "control": "shuffled_feat",
                         "control_seed": seed})


def random_edges(g, seed=CONTROL_SEED):
    """Kenar sayisini korur, uclari rastgele secer (oz-dongu ve tekrar yok)."""
    e = int(g.edge_index.shape[1])
    n = g.num_nodes
    rng = np.random.default_rng(seed)
    seen, src, dst = set(), [], []
    while len(src) < e:
        a = int(rng.integers(n))
        b = int(rng.integers(n))
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        src.append(a)
        dst.append(b)
    return replace(g, edge_index=torch.tensor([src, dst], dtype=torch.long),
                   info={**g.info, "control": "random_edges",
                         "control_seed": seed, "num_edges": e})


CONTROLS = {
    "full":          lambda g: g,
    "no_edges":      self_loops_only,
    "shuffled_feat": shuffled_features,
    "random_edges":  random_edges,
}


def main():
    pilot = "--pilot" in sys.argv
    device = get_device()
    stamp = protocol_stamp()
    base_graph = build_graph(merge_by_osm=MERGE)
    folds = build_cv_folds(n_folds=C.N_FOLDS, legacy=False)
    if pilot:
        folds, seeds = folds[:1], SEEDS[:1]
    else:
        seeds = SEEDS

    graphs = {name: fn(base_graph) for name, fn in CONTROLS.items()}
    print("protokol %s | cihaz %s" % (stamp["hash"], device))
    for name, g in graphs.items():
        print("  %-14s dugum %d | kenar %d | oznitelik %d"
              % (name, g.num_nodes, int(g.edge_index.shape[1]), g.num_features))
    print("  %d kontrol x %d fold x %d tohum = %d kosu\n"
          % (len(CONTROLS), len(folds), len(seeds),
             len(CONTROLS) * len(folds) * len(seeds)), flush=True)

    # dugum indeksleri graf yapisindan bagimsiz (segment_to_idx ayni kalir)
    fold_idx, fold_trips = {}, {}
    for f in folds:
        k = f.info["fold"]
        fold_idx[k] = {sp: map_segments(getattr(f, "meta_" + sp), base_graph)[0]
                       for sp in ("train", "val", "test")}
        fold_trips[k] = np.asarray(f.meta_test)[:, 0]

    results = {"protocol": stamp,
               "config": {"n_folds": len(folds), "seeds": seeds, "pilot": pilot,
                          "merge_by_osm": MERGE, "base": BASE,
                          "control_seed": CONTROL_SEED,
                          "graph_info": {n: g.info.get("control", "reference")
                                         for n, g in graphs.items()},
                          "n_edges": {n: int(g.edge_index.shape[1])
                                      for n, g in graphs.items()}},
               "models": {}}
    pooled, runmap = {}, {}

    for ctrl, g in graphs.items():
        runs, y_all, p_all, b_all, t_all, g_all, th_all = [], [], [], [], [], [], []
        for f in folds:
            k = f.info["fold"]
            for seed in seeds:
                spec = make_spec(merge_by_osm=MERGE, fold=k, cv=C.N_FOLDS,
                                 **({} if ctrl == "full" else {"ctrl": ctrl}),
                                 **BASE)
                r = get_or_train(spec, seed, f, fold_idx[k], g, device, verbose=False)
                m = dict(r["metrics"])
                m.update({"fold": k, "seed": seed})
                runs.append(m)
                runmap.setdefault(ctrl, {})[(k, seed)] = float(m["f1"])
                print("  %-14s fold%d s%d: F1=%.4f AUC=%.4f%s"
                      % (ctrl, k, seed, m["f1"], m.get("roc_auc", float("nan")),
                         "  [onbellek]" if r["cached"] else ""), flush=True)
                if seed == seeds[0]:
                    y_all.append(f.y_test)
                    p_all.append(r["prob_test"])
                    b_all.append((r["prob_test"] > m["threshold_val"]).astype(int))
                    t_all.append(f.type_test)
                    g_all.append(fold_trips[k])
                    th_all.append(float(m["threshold_val"]))
        y = np.concatenate(y_all)
        p = np.concatenate(p_all)
        b = np.concatenate(b_all)
        t = np.concatenate(t_all)
        gg = np.concatenate(g_all)
        f1s = np.array([m["f1"] for m in runs])
        note = "per_fold_validation"
        results["models"][ctrl] = {
            "runs": runs,
            "aggregate": {"f1": {"mean": float(f1s.mean()),
                                 "std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
                                 "n": int(len(f1s)), "min": float(f1s.min())}},
            "pooled": pooled_metrics(y, p, b, note),
            "by_type_pooled": pooled_by_type(y, p, b, t, note),
            "pooled_fold_thresholds": th_all,
            "pooled_threshold_policy": note,
        }
        a = results["models"][ctrl]["aggregate"]["f1"]
        print("  -> %s: kosu-ort F1 %.4f+-%.4f | havuz F1 %.4f\n"
              % (ctrl, a["mean"], a["std"], results["models"][ctrl]["pooled"]["f1"]),
              flush=True)
        pooled[ctrl] = (y, p, t, gg, b)

    # --------------------------------------------------------- karsilastirmalar
    fi = [f.info for f in folds]
    n_tr = int(np.mean([x["trips"]["train"] for x in fi]))
    n_te = int(np.mean([x["trips"]["test"] for x in fi]))

    print("=" * 96)
    print("KOSU DUZEYI — full'e gore eslestirilmis fark (Nadeau-Bengio, egitim %d / test %d trip)"
          % (n_tr, n_te))
    print("=" * 96)
    print("  %-18s %9s %22s %8s %8s %11s"
          % ("kontrol", "ort fark", "NB %95 GA", "NB p", "Holm p", "TOST"))
    rl, pv = {}, {}
    for ctrl in CONTROLS:
        if ctrl == "full":
            continue
        keys = sorted(set(runmap["full"]) & set(runmap.get(ctrl, {})))
        if len(keys) < 2:
            continue
        diffs = [runmap[ctrl][k] - runmap["full"][k] for k in keys]
        nb = nadeau_bengio(diffs, n_train=n_tr, n_test=n_te)
        ts = tost_equivalence(diffs, DELTA, n_train=n_tr, n_test=n_te)
        rl[ctrl + "_minus_full"] = {"n_paired": len(keys), "diffs": diffs,
                                    "nadeau_bengio": nb, "tost": ts}
        pv[ctrl + "_minus_full"] = nb["p_corrected"]
    for k, v in (holm_correction(pv).items() if pv else []):
        rl[k]["holm"] = v
    for k, v in rl.items():
        nb, ts = v["nadeau_bengio"], v["tost"]
        verdict = ("esdeger" if ts["equivalent"]
                   else ("FARK VAR" if v.get("holm", {}).get("significant") else "belirsiz"))
        print("  %-18s %+9.4f [%+.4f, %+.4f] %8.4f %8.4f %11s"
              % (k, nb["mean_diff"], nb["ci_lo"], nb["ci_hi"], nb["p_corrected"],
                 v.get("holm", {}).get("p_holm", float("nan")), verdict))

    print("\n" + "=" * 96)
    print("TRIP DUZEYI — havuzlanmis, kume bootstrap + permutasyon (referans full)")
    print("=" * 96)
    print("  %-18s %9s %22s %8s %8s"
          % ("kontrol", "dF1", "%95 GA (kume)", "perm p", "Holm p"))
    ry, rb, rg = pooled["full"][0], pooled["full"][4], pooled["full"][3]
    tl, pv2 = {}, {}
    for ctrl in CONTROLS:
        if ctrl == "full":
            continue
        y2, _, _, g2, b2 = pooled[ctrl]
        if not np.array_equal(y2, ry):
            tl[ctrl + "_minus_full"] = {"error": "diziler hizalanmiyor"}
            continue
        bo = cluster_bootstrap_diff(ry, b2, rb, rg)
        pe = cluster_permutation_test(ry, b2, rb, rg)
        tl[ctrl + "_minus_full"] = {"bootstrap": bo, "permutation": pe}
        pv2[ctrl + "_minus_full"] = pe["p_value"]
    for k, v in (holm_correction(pv2).items() if pv2 else []):
        tl[k]["holm"] = v
    for k, v in tl.items():
        if "error" in v:
            print("  %-18s  %s" % (k, v["error"]))
            continue
        bo, pe = v["bootstrap"], v["permutation"]
        print("  %-18s %+9.4f [%+.4f, %+.4f] %8.4f %8.4f"
              % (k, bo["observed_diff"], bo["lo"], bo["hi"], pe["p_value"],
                 v.get("holm", {}).get("p_holm", float("nan"))))

    results["run_level"] = {"n_train_trips": n_tr, "n_test_trips": n_te,
                            "delta": DELTA, "comparisons": rl}
    results["trip_level"] = {"reference": "full",
                             "cluster_info": cluster_summary(rg),
                             "comparisons": tl}
    path = (C.RESULTS_DIR / "F14_pilot.json") if pilot else OUT
    save_json(results, path)
    print("\nkaydedildi: %s" % path)


if __name__ == "__main__":
    main()
