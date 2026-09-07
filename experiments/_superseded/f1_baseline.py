"""
F1 — Zemin ölçümü. KARAR A'yı belirler.

Soru: STANet'in performansı, kural tabanlı veya klasik ML yaklaşımlarıyla
      elde edilebilenin üzerinde mi?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from stanet import config as C
from stanet.baselines import feature_names, rule_predictions, window_features
from stanet.data import build_dataset, inverse_kinematics
from stanet.evaluate import compute_metrics, metrics_by_type, pick_threshold, predict
from stanet.graph import build_graph, map_segments
from stanet.models import FusionHybrid
from stanet.stats import bootstrap_ci, bootstrap_diff, mcnemar
from stanet.train import SeqDataset
from stanet.utils import get_device, init_console, save_json, set_seed

init_console()
SEED = 42


def raw_windows(ds, split):
    X = getattr(ds, f"X_{split}")
    speed, accel = inverse_kinematics(X, ds.scalers)
    return np.stack([speed, accel, X[:, :, 2], X[:, :, 3]], axis=2)


def main():
    set_seed(SEED)
    device = get_device()
    print(f"cihaz: {device}\n")

    # legacy=True -> STANet checkpoint'i ile aynı split/normalizasyon şartları
    ds = build_dataset(legacy=True)
    Xtr, Xva, Xte = (raw_windows(ds, s) for s in ("train", "val", "test"))
    ytr, yva, yte = ds.y_train, ds.y_val, ds.y_test

    results = {"seed": SEED, "dataset_info": ds.info, "models": {}}
    store = {}   # model -> (y_prob veya y_pred)

    # ------------------------------------------------------------- 1) KURAL
    print("=" * 62)
    print("1) KURAL TABANLI (sıfır parametre, öğrenme yok)")
    print("=" * 62)
    _, variants = rule_predictions(Xte[:, :, 0], Xte[:, :, 1], Xte[:, :, 2], Xte[:, :, 3])
    best_rule_name, best_rule_f1 = None, -1
    for name, mask in variants.items():
        pred = mask.astype(int)
        m = compute_metrics(yte, pred, threshold=0.5)
        m["type"] = "rule"
        results["models"][f"rule_{name}"] = m
        store[f"rule_{name}"] = pred.astype(float)
        print(f"  {name:14s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        if m["f1"] > best_rule_f1:
            best_rule_f1, best_rule_name = m["f1"], f"rule_{name}"
    print(f"  -> en iyi kural: {best_rule_name} (F1={best_rule_f1:.4f})")

    # ------------------------------------------------------------- 2) AĞAÇ / LİNEER
    print("\n" + "=" * 62)
    print("2) KLASİK ML — pencere agregatları")
    print("=" * 62)
    Ftr, Fva, Fte = window_features(Xtr), window_features(Xva), window_features(Xte)
    print(f"  öznitelik boyutu: {Ftr.shape[1]}")
    spw = (len(ytr) - ytr.sum()) / ytr.sum()

    sc = StandardScaler().fit(Ftr)
    clfs = {
        "xgboost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=SEED, n_jobs=4, tree_method="hist"),
        "lightgbm": LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=spw, random_state=SEED,
                                   n_jobs=4, verbose=-1),
        "random_forest": RandomForestClassifier(n_estimators=400, max_depth=None,
                                                class_weight="balanced",
                                                random_state=SEED, n_jobs=4),
        "logistic_reg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=SEED),
    }
    for name, clf in clfs.items():
        A, B, D = (sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)) \
            if name == "logistic_reg" else (Ftr, Fva, Fte)
        clf.fit(A, ytr)
        pv = clf.predict_proba(B)[:, 1]
        pt = clf.predict_proba(D)[:, 1]
        th = pick_threshold(yva, pv)
        m50 = compute_metrics(yte, pt, 0.5)
        mth = compute_metrics(yte, pt, th)
        best = mth if mth["f1"] >= m50["f1"] else m50
        best["type"] = "classic_ml"
        best["threshold_source"] = "val" if best is mth else "fixed_0.5"
        best["f1_at_0.5"] = m50["f1"]
        best["by_type"] = metrics_by_type(yte, pt, ds.type_test, best["threshold"])
        results["models"][name] = best
        store[name] = pt
        print(f"  {name:14s} F1@0.5={m50['f1']:.4f}  F1@val({th:.2f})={mth['f1']:.4f}  "
              f"AUC={m50.get('roc_auc', float('nan')):.4f}")

    # ------------------------------------------------------------- 3) DENETİMSİZ
    print("\n" + "=" * 62)
    print("3) DENETİMSİZ (etiket kullanmadan)")
    print("=" * 62)
    iso = IsolationForest(n_estimators=300, contamination=float(ytr.mean()),
                          random_state=SEED, n_jobs=4).fit(Ftr)
    sc_te = -iso.score_samples(Fte)          # yüksek = daha anormal
    sc_va = -iso.score_samples(Fva)
    th = pick_threshold(yva, sc_va, grid=np.quantile(sc_va, np.linspace(0.5, 0.999, 80)))
    m = compute_metrics(yte, sc_te, th)
    m["type"] = "unsupervised"
    results["models"]["isolation_forest"] = m
    store["isolation_forest"] = sc_te
    print(f"  isolation_forest F1={m['f1']:.4f} AUC={m.get('roc_auc', float('nan')):.4f}")

    # ------------------------------------------------------------- 4) STANet
    print("\n" + "=" * 62)
    print("4) STANet (mevcut checkpoint)")
    print("=" * 62)
    graph = build_graph(merge_by_osm=False)
    node_x, edge_index = graph.to_tensors(device)
    iv, _ = map_segments(ds.meta_val, graph)
    it, _ = map_segments(ds.meta_test, graph)
    model = FusionHybrid("gated", num_node_features=graph.num_features, legacy_head=True)
    model.load_state_dict(torch.load(C.LEGACY_HYBRID_CKPT, map_location=device,
                                     weights_only=False))
    model = model.to(device).eval()
    dv = DataLoader(SeqDataset(ds.X_val, ds.y_val, iv), batch_size=C.BATCH_SIZE)
    dt = DataLoader(SeqDataset(ds.X_test, ds.y_test, it), batch_size=C.BATCH_SIZE)
    _, pv, _ = predict(model, dv, node_x, edge_index, device)
    _, pt, _ = predict(model, dt, node_x, edge_index, device)
    th_st = pick_threshold(yva, pv)
    m50 = compute_metrics(yte, pt, 0.5)
    mth = compute_metrics(yte, pt, th_st)
    mth["type"] = "deep"
    mth["f1_at_0.5"] = m50["f1"]
    mth["by_type"] = metrics_by_type(yte, pt, ds.type_test, th_st)
    results["models"]["stanet"] = mth
    store["stanet"] = pt
    print(f"  stanet         F1@0.5={m50['f1']:.4f}  F1@val({th_st:.2f})={mth['f1']:.4f}  "
          f"AUC={m50['roc_auc']:.4f}")

    # ------------------------------------------------------------- 5) KARŞILAŞTIRMA
    print("\n" + "=" * 62)
    print("5) STANet vs rakipler — istatistiksel karşılaştırma")
    print("=" * 62)
    st_pred = (store["stanet"] > th_st).astype(int)
    comp = {}
    for name in ["xgboost", "lightgbm", "random_forest", best_rule_name]:
        p = store[name]
        thr = results["models"][name]["threshold"]
        other = (p > thr).astype(int) if name != best_rule_name else p.astype(int)
        mc = mcnemar(yte, st_pred, other)
        bd = bootstrap_diff(yte, store["stanet"], p, threshold=th_st,
                            threshold_b=th, seed=SEED)
        comp[name] = {"mcnemar": mc, "bootstrap_diff": bd}
        print(f"  STANet vs {name:16s} ΔF1={bd['mean_diff']:+.4f} "
              f"GA[{bd['lo']:+.4f},{bd['hi']:+.4f}] McNemar p={mc['p_value']:.4f} "
              f"{'ANLAMLI' if mc['significant'] else 'anlamsız'}")
    results["comparisons_vs_stanet"] = comp
    results["stanet_ci"] = bootstrap_ci(yte, store["stanet"], threshold=th_st, seed=SEED)

    # ------------------------------------------------------------- KARAR A
    ml_best = max(("xgboost", "lightgbm", "random_forest"),
                  key=lambda k: results["models"][k]["f1"])
    ml_f1 = results["models"][ml_best]["f1"]
    st_f1 = results["models"]["stanet"]["f1"]
    verdict = "STANET_USTUN" if st_f1 > ml_f1 else "KLASIK_ML_USTUN"
    results["karar_A"] = {"stanet_f1": st_f1, "best_classic": ml_best,
                          "best_classic_f1": ml_f1, "best_rule_f1": best_rule_f1,
                          "verdict": verdict,
                          "significant": comp[ml_best]["mcnemar"]["significant"]}
    print("\n" + "=" * 62)
    print(f"KARAR A: STANet F1={st_f1:.4f} | en iyi klasik ({ml_best}) F1={ml_f1:.4f} "
          f"| en iyi kural F1={best_rule_f1:.4f}")
    print(f"         -> {verdict}"
          f" ({'anlamlı' if comp[ml_best]['mcnemar']['significant'] else 'anlamlı değil'})")
    print("=" * 62)

    save_json(results, C.RESULTS_DIR / "F1_zemin.json")
    np.savez_compressed(C.CACHE_DIR / "F1_predictions.npz",
                        y_test=yte, atype=ds.type_test,
                        **{k: np.asarray(v) for k, v in store.items()})
    print(f"\nkaydedildi: {C.RESULTS_DIR / 'F1_zemin.json'}")


if __name__ == "__main__":
    main()
