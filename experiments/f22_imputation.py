"""
F22 — Dugum ozniteliklerindeki IMPUTASYONA duyarlilik (degerlendirme notu:
eksik oznitelikler nasil dolduruldu ve sonuc buna bagli mi).

DURUM
-----
Graf dugum ozniteliklerinin ikisi eksik veriden dolduruluyor ve dolan pay
kucuk degil:

    lanes_source   osm %53,8   varsayilan %46,2
    speed_source   osm %35,3   varsayilan %49,9   yumusatilmis %14,8

Yani `max_speed` dugumlerin yarisinda gercek bir OSM degeri degil, bir
varsayilan. Uzamsal dalin katkisiyla ilgili her bulgu bu doldurmaya bagimli
olabilir: model "gercek hiz limiti" ogreniyor da olabilir, "bu kesit
varsayilan aldi" ayrimini ogreniyor da.

DORT VARYANT — hepsi ayni fold'lar, ayni tohumlar, ayni esik politikasi
----------------------------------------------------------------------
  as_is            mevcut kurulum (F5'ten onbellek)
  mask_indicator   deger kalir, YANINA iki ikili gosterge eklenir
                   (lanes_imputed, speed_imputed) -> model doldurmayi
                   ayirt edebilir
  type_median      varsayilanlar, AYNI yol tipindeki OSM kaynakli
                   kesitlerin medyani ile degistirilir -> doldurma daha
                   bilgili ama hala doldurma
  drop_feature     `max_speed` ve `lanes` ozniteligi tamamen cikarilir
                   -> doldurmanin hicbir izi kalmaz

Okuma kilavuzu:
  hepsi ~ as_is            -> bulgu imputasyona duyarsiz, rahat yaziyoruz
  drop_feature << as_is    -> kazancin bir kismi bu iki oznitelikten geliyor
  mask_indicator >> as_is  -> model doldurma GOSTERGESINI kullaniyor,
                              yani sizintiya benzer bir ipucu var
  type_median != as_is     -> sonuc doldurma KURALINA bagli, bu bir sinirdir

Medyan YALNIZCA OSM kaynakli kesitlerden hesaplanir ve tum bolumlemede
sabittir; bu bir veri hazirligi karari olduğu icin fold'a gore degismez ve
etiket kullanmaz, dolayisiyla sizinti yaratmaz.

KULLANIM
  python experiments/f22_imputation.py --pilot
  python experiments/f22_imputation.py
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from stanet import config as C
from stanet.cluster_stats import (holm_correction, nadeau_bengio, tost_equivalence)
from stanet.data import build_cv_folds
from stanet.graph import RoadGraph, _build_edges, build_graph, map_segments
from stanet.runstore import get_or_train, make_spec
from stanet.utils import get_device, init_console, protocol_stamp, save_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f5_cv import pooled_metrics                         # noqa: E402

init_console()
OUT = C.RESULTS_DIR / "F22_imputation.json"
SEEDS = [42, 43]
MERGE = False
DELTA = 0.02
DEFAULT_TAG = "varsayilan"
CONFIGS = {"hybrid_gat": dict(temporal="lstm", spatial="gat", fusion="weighted_sum")}
VARIANTS = ["as_is", "mask_indicator", "type_median", "drop_feature"]


def imputation_report(df):
    """Hangi oznitelik ne kadar dolduruldu — makaleye giren sayi."""
    out = {}
    for col, src in (("lanes", "lanes_source"), ("max_speed", "speed_source")):
        vc = df[src].value_counts()
        out[col] = {"source_column": src, "n_nodes": int(len(df)),
                    "counts": {str(k): int(v) for k, v in vc.items()},
                    "imputed_fraction": float((df[src] == DEFAULT_TAG).mean())}
    return out


def build_variant(variant):
    """Grafi verilen imputasyon varyantiyla kurar.

    `build_graph` ile ayni adimlari izler; tek fark oznitelik matrisinin
    kurulusu. Kenarlar ve dugum siralamasi degismez, dolayisiyla
    `segment_to_idx` esleme ve fold indeksleri aynen gecerlidir.
    """
    df = pd.read_csv(C.GNN_DATA)
    nodes = df.drop_duplicates(subset=["segment_id"]).copy()
    nodes = nodes.sort_values("segment_id").reset_index(drop=True)
    nodes["length_log"] = np.log1p(nodes["length_meters"])
    segment_to_idx = {int(s): i for i, s in enumerate(nodes["segment_id"])}

    num_cols = list(C.GNN_NUM_COLS)
    extra_names, extra_cols = [], []

    if variant == "type_median":
        # varsayilanlari, AYNI yol tipindeki OSM kaynakli kesitlerin
        # medyaniyle degistir; hicbir yerde etiket kullanilmaz
        for col, src in (("lanes", "lanes_source"), ("max_speed", "speed_source")):
            real = nodes[nodes[src] != DEFAULT_TAG]
            med = real.groupby("road_type")[col].median()
            glob = float(real[col].median())
            mask = nodes[src] == DEFAULT_TAG
            # sutunu once float'a cevir: pandas tamsayi bir sutuna float
            # atamayi artik uyariyla karsiliyor ve ileride hata verecek
            nodes[col] = nodes[col].astype(float)
            nodes.loc[mask, col] = [
                float(med.get(rt, glob)) for rt in nodes.loc[mask, "road_type"]]
    elif variant == "drop_feature":
        num_cols = [c for c in num_cols if c not in ("max_speed", "lanes")]
    elif variant == "mask_indicator":
        nodes["lanes_imputed"] = (nodes["lanes_source"] == DEFAULT_TAG).astype(float)
        nodes["speed_imputed"] = (nodes["speed_source"] == DEFAULT_TAG).astype(float)
        extra_names = ["lanes_imputed", "speed_imputed"]
        extra_cols = nodes[extra_names].values.astype(float)

    num = MinMaxScaler().fit_transform(nodes[num_cols].values)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    road_type = ohe.fit_transform(nodes[["road_type"]])
    binary = nodes[C.GNN_BIN_COLS].values.astype(float)

    parts = [num, binary, road_type]
    names = num_cols + C.GNN_BIN_COLS + list(ohe.get_feature_names_out(["road_type"]))
    if len(extra_names):
        parts.append(np.asarray(extra_cols))
        names = names + extra_names
    feats = np.hstack(parts).astype(np.float32)

    edge_index = _build_edges(nodes)
    info = {"merge_by_osm": False, "variant": variant,
            "num_nodes": len(nodes), "num_edges": int(edge_index.shape[1]),
            "num_features": feats.shape[1], "feature_names": names,
            "imputation": imputation_report(nodes)}
    return RoadGraph(feats, edge_index, segment_to_idx, names, info)


def main():
    pilot = "--pilot" in sys.argv
    device = get_device()
    stamp = protocol_stamp()
    folds = build_cv_folds(n_folds=C.N_FOLDS, legacy=False)
    variants = VARIANTS[:2] if pilot else VARIANTS
    if pilot:
        folds, seeds = folds[:1], SEEDS[:1]
    else:
        seeds = SEEDS

    base = build_graph(merge_by_osm=MERGE)
    graphs = {v: (base if v == "as_is" else build_variant(v)) for v in variants}
    print("protokol %s | cihaz %s" % (stamp["hash"], device))
    rep = build_variant("as_is").info["imputation"] if "as_is" in graphs else None
    if rep:
        for col, d in rep.items():
            print("  %-10s doldurulan pay %.1f%%  (%s)"
                  % (col, 100 * d["imputed_fraction"], d["counts"]))
    for v, g in graphs.items():
        print("  %-15s dugum %d | kenar %d | oznitelik %d"
              % (v, g.num_nodes, int(g.edge_index.shape[1]), g.num_features))
    print("  %d varyant x %d fold x %d tohum = %d kosu\n"
          % (len(variants), len(folds), len(seeds),
             len(variants) * len(folds) * len(seeds)), flush=True)

    fold_idx, fold_trips = {}, {}
    for f in folds:
        k = f.info["fold"]
        fold_idx[k] = {sp: map_segments(getattr(f, "meta_" + sp), base)[0]
                       for sp in ("train", "val", "test")}
        fold_trips[k] = np.asarray(f.meta_test)[:, 0]

    results = {"protocol": stamp,
               "config": {"n_folds": len(folds), "seeds": seeds, "pilot": pilot,
                          "variants": variants, "delta": DELTA},
               "imputation_report": rep,
               "graph_info": {v: g.info for v, g in graphs.items()},
               "models": {}}
    runmap = {}

    for name, cfg in CONFIGS.items():
        for v in variants:
            g = graphs[v]
            runs, y_all, p_all, b_all = [], [], [], []
            for f in folds:
                k = f.info["fold"]
                for seed in seeds:
                    kw = {} if v == "as_is" else {"imp": v}
                    spec = make_spec(merge_by_osm=MERGE, fold=k, cv=C.N_FOLDS,
                                     **kw, **cfg)
                    r = get_or_train(spec, seed, f, fold_idx[k], g, device,
                                     verbose=False)
                    m = dict(r["metrics"])
                    m.update({"fold": k, "seed": seed})
                    runs.append(m)
                    runmap.setdefault(v, {})[(k, seed)] = float(m["f1"])
                    print("  %-15s fold%d s%d: F1=%.4f%s"
                          % (v, k, seed, m["f1"],
                             "  [onbellek]" if r["cached"] else ""), flush=True)
                    if seed == seeds[0]:
                        y_all.append(f.y_test)
                        p_all.append(r["prob_test"])
                        b_all.append((r["prob_test"] > m["threshold_val"]).astype(int))
            f1 = np.array([m["f1"] for m in runs])
            y = np.concatenate(y_all)
            p = np.concatenate(p_all)
            b = np.concatenate(b_all)
            results["models"].setdefault(name, {})[v] = {
                "runs": runs, "f1_mean": float(f1.mean()),
                "f1_sd": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                "f1_min": float(f1.min()), "n": int(len(f1)),
                "n_features": graphs[v].num_features,
                "pooled": pooled_metrics(y, p, b, "per_fold_validation")}
            print("  -> %s: F1 %.4f+-%.4f | havuz %.4f\n"
                  % (v, f1.mean(), results["models"][name][v]["f1_sd"],
                     results["models"][name][v]["pooled"]["f1"]), flush=True)

    # ------------------------------------------------- as_is'e gore farklar
    fi = [f.info for f in folds]
    n_tr = int(np.mean([x["trips"]["train"] for x in fi]))
    n_te = int(np.mean([x["trips"]["test"] for x in fi]))
    print("=" * 96)
    print("IMPUTASYON VARYANTI - as_is (eslestirilmis, Nadeau-Bengio)")
    print("=" * 96)
    print("  %-16s %9s %22s %8s %8s %11s"
          % ("varyant", "ort fark", "NB %95 GA", "NB p", "Holm p", "TOST"))
    rows, pv = {}, {}
    for v in variants:
        if v == "as_is" or v not in runmap:
            continue
        keys = sorted(set(runmap["as_is"]) & set(runmap[v]))
        if len(keys) < 2:
            continue
        diffs = [runmap[v][k] - runmap["as_is"][k] for k in keys]
        nb = nadeau_bengio(diffs, n_train=n_tr, n_test=n_te)
        ts = tost_equivalence(diffs, DELTA, n_train=n_tr, n_test=n_te)
        rows[v] = {"n_paired": len(keys), "diffs": diffs,
                   "nadeau_bengio": nb, "tost": ts}
        pv[v] = nb["p_corrected"]
    for k, val in (holm_correction(pv).items() if pv else []):
        rows[k]["holm"] = val
    for k, val in rows.items():
        nb, ts = val["nadeau_bengio"], val["tost"]
        verdict = ("esdeger" if ts["equivalent"]
                   else ("FARK VAR" if val.get("holm", {}).get("significant")
                         else "belirsiz"))
        val["verdict"] = verdict
        print("  %-16s %+9.4f [%+.4f, %+.4f] %8.4f %8.4f %11s"
              % (k, nb["mean_diff"], nb["ci_lo"], nb["ci_hi"], nb["p_corrected"],
                 val.get("holm", {}).get("p_holm", float("nan")), verdict))

    results["comparison"] = {"reference": "as_is", "n_train_trips": n_tr,
                             "n_test_trips": n_te, "delta": DELTA, "rows": rows}
    path = (C.RESULTS_DIR / "F22_pilot.json") if pilot else OUT
    save_json(results, path)
    print("\nkaydedildi: %s" % path)


if __name__ == "__main__":
    main()
