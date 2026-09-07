"""
F1b — Klasik ML baseline'ları ADİL koşulda (KARAR A'nın kesinleştirilmesi).

F1'deki karşılaştırma iki yönden adil değildi:
  - STANet tarafı: eski checkpoint (gated füzyon, sızıntılı scaler, TEK koşu)
  - Klasik ML tarafı: tek koşu

Burada her iki taraf da:
  - düzeltilmiş veri hattı (scaler yalnızca train'e fit)
  - çok seed (klasik ML 5 seed, derin model F4'ten 3 seed)
  - eşik validation'da seçilir
  - aynı test seti, eşleştirilmiş McNemar + bootstrap
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from stanet import config as C
from stanet.baselines import rule_predictions, window_features
from stanet.data import build_dataset, inverse_kinematics
from stanet.evaluate import compute_metrics, metrics_by_type, pick_threshold
from stanet.stats import aggregate_seeds, bootstrap_diff, mcnemar
from stanet.utils import (init_console, load_json, protocol_stamp, save_json,
                          set_seed)

init_console()
SEEDS = C.SEEDS          # v2: F2/F3/F4 ile ayni 5 seed


def raw_windows(ds, split):
    X = getattr(ds, f"X_{split}")
    s, a = inverse_kinematics(X, ds.scalers)
    return np.stack([s, a, X[:, :, 2], X[:, :, 3]], axis=2)


def main():
    ds = build_dataset(legacy=False)          # DUZELTILMIS hat
    Xtr, Xva, Xte = (raw_windows(ds, s) for s in ("train", "val", "test"))
    ytr, yva, yte = ds.y_train, ds.y_val, ds.y_test
    Ftr, Fva, Fte = window_features(Xtr), window_features(Xva), window_features(Xte)
    spw = (len(ytr) - ytr.sum()) / ytr.sum()
    print(f"duzeltilmis hat | oznitelik {Ftr.shape[1]} | pos_weight {spw:.2f} | "
          f"{len(SEEDS)} seed\n", flush=True)

    results = {"protocol": protocol_stamp(),
               "config": {"seeds": SEEDS, "legacy": False}, "models": {}}
    store = {}

    # ---------------------------------------------------------- kural (deterministik)
    _, variants = rule_predictions(Xte[:, :, 0], Xte[:, :, 1], Xte[:, :, 2], Xte[:, :, 3])
    best_name, best_f1 = None, -1
    for name, mask in variants.items():
        m = compute_metrics(yte, mask.astype(int), 0.5)
        results["models"][f"rule_{name}"] = {"aggregate": {"f1": {"mean": m["f1"], "std": 0.0}},
                                             "single": m, "median_f1": m["f1"],
                                             "type": "rule"}
        if m["f1"] > best_f1:
            best_f1, best_name = m["f1"], f"rule_{name}"
            store[best_name] = mask.astype(float)
    print(f"en iyi kural: {best_name} F1={best_f1:.4f}  (deterministik, std=0)\n", flush=True)

    # ---------------------------------------------------------- agac modelleri
    def make(name, seed):
        if name == "xgboost":
            return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=seed, n_jobs=4, tree_method="hist")
        if name == "lightgbm":
            return LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=spw, random_state=seed,
                                  n_jobs=4, verbose=-1)
        return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                      random_state=seed, n_jobs=4)

    for name in ("xgboost", "lightgbm", "random_forest"):
        runs, probs = [], []
        for seed in SEEDS:
            set_seed(seed)
            clf = make(name, seed).fit(Ftr, ytr)
            pv, pt = clf.predict_proba(Fva)[:, 1], clf.predict_proba(Fte)[:, 1]
            th = pick_threshold(yva, pv)
            m = compute_metrics(yte, pt, th)
            m["threshold_val"] = th
            m["by_type"] = metrics_by_type(yte, pt, ds.type_test, th)
            runs.append(m)
            probs.append(pt)
        agg = aggregate_seeds(runs)
        results["models"][name] = {"runs": runs, "aggregate": agg, "type": "classic_ml"}
        store[name] = np.mean(probs, axis=0)          # seed ortalamasi (ensemble degil, referans)
        results["models"][name]["median_prob_idx"] = len(SEEDS) // 2
        store[f"{name}_median"] = probs[len(SEEDS) // 2]
        results["models"][name]["median_threshold"] = runs[len(SEEDS) // 2]["threshold_val"]
        results["models"][name]["median_f1"] = runs[len(SEEDS) // 2]["f1"]
        print(f"  {name:14s} F1 {agg['f1']['mean']:.4f}+-{agg['f1']['std']:.4f} | "
              f"AUC {agg['roc_auc']['mean']:.4f}+-{agg['roc_auc']['std']:.4f}", flush=True)

    # ---------------------------------------------------------- derin model (F4'ten)
    f4 = load_json(C.RESULTS_DIR / "F4_benchmark.json")
    pred4 = {k: v for k, v in np.load(C.CACHE_DIR / "F4_predictions.npz").items()}
    assert np.array_equal(pred4["y_test"], yte), "F4 ve F1b test setleri uyusmuyor"
    # PROTOKOL KILIDI: F4 baska bir protokolde kosulduysa satirlari ithal etme.
    # Ilk surumde F4 3 seed + cudnn=False, F3 5 seed + cudnn=True idi ve ayni
    # konfigurasyon iki tabloda 0,8836 ve 0,8484 gorundu.
    assert f4.get("protocol", {}).get("hash") == results["protocol"]["hash"], (
        f"F4 protokolu ({f4.get('protocol', {}).get('hash')}) F1b protokolu "
        f"({results['protocol']['hash']}) ile uyusmuyor — F4'u yeniden kosun")
    assert list(f4["config"]["seeds"]) == list(SEEDS), (
        f"F4 seed listesi {f4['config']['seeds']} != F1b {SEEDS}")

    for key, label in (("T:lstm", "hybrid_lstm_gat"), ("S:none", "lstm_only")):
        cfg = f4["configs"][key]
        agg = cfg["aggregate"]
        med = len(f4["config"]["seeds"]) // 2
        results["models"][label] = {"aggregate": agg, "type": "deep",
                                    "source": f"F4:{key}",
                                    "spec": cfg.get("spec"),
                                    "median_threshold": cfg["runs"][med]["threshold_val"],
                                    "median_f1": cfg["runs"][med]["f1"]}
        store[f"{label}_median"] = pred4[key][med]
        print(f"  {label:14s} F1 {agg['f1']['mean']:.4f}+-{agg['f1']['std']:.4f} | "
              f"AUC {agg['roc_auc']['mean']:.4f}+-{agg['roc_auc']['std']:.4f}  (F4:{key})",
              flush=True)

    # ---------------------------------------------------------- eslestirilmis karsilastirma
    print("\n" + "=" * 78)
    print("HIBRIT (LSTM+GAT) vs RAKIPLER — eslestirilmis, medyan seed")
    print("=" * 78)
    ref = store["hybrid_lstm_gat_median"]
    ref_th = results["models"]["hybrid_lstm_gat"]["median_threshold"]
    comp = {}
    targets = [("lightgbm_median", "lightgbm"), ("xgboost_median", "xgboost"),
               ("random_forest_median", "random_forest"),
               (best_name, best_name), ("lstm_only_median", "lstm_only")]
    for skey, mkey in targets:
        p = store[skey]
        th = (0.5 if mkey.startswith("rule")
              else results["models"][mkey].get("median_threshold",
                                               results["models"][mkey].get("median_threshold", 0.5)))
        other = (p > th).astype(int) if not mkey.startswith("rule") else p.astype(int)
        mc = mcnemar(yte, (ref > ref_th).astype(int), other)
        bd = bootstrap_diff(yte, ref, p, threshold=ref_th, threshold_b=th, seed=0)
        dm = (results["models"]["hybrid_lstm_gat"]["aggregate"]["f1"]["mean"]
              - results["models"][mkey]["aggregate"]["f1"]["mean"])
        # KRITIK AYRIM (ilk surumdeki Tablo 10 hatasi):
        #   mean_delta        : 5 seed ORTALAMALARININ farki
        #   median_seed_delta : bootstrap GA'nin uzerinde hesaplandigi tek
        #                       kosu farki — GA bu degeri icermelidir, ortalamayi degil.
        # Ilk surumde RF satirinda GA [+0,038, +0,116] iken mean_delta +0,0137
        # yaziyordu; iki farkli nicelik ayni satira konmustu.
        dmed = (results["models"]["hybrid_lstm_gat"]["median_f1"]
                - results["models"][mkey]["median_f1"])
        ga_ok = bd["lo"] <= dmed <= bd["hi"]
        comp[mkey] = {"mcnemar": mc, "bootstrap_diff": bd, "mean_delta": dm,
                      "median_seed_delta": dmed, "median_seed_index": len(SEEDS) // 2,
                      "ci_contains_median_delta": bool(ga_ok)}
        print(f"  vs {mkey:18s} dF1(ort)={dm:+.4f}  dF1(medyan)={dmed:+.4f}  "
              f"GA[{bd['lo']:+.4f},{bd['hi']:+.4f}]{'' if ga_ok else ' <-GA UYUMSUZ'}  "
              f"McNemar p={mc['p_value']:.4f} "
              f"{'ANLAMLI' if mc['significant'] else 'anlamsiz'}")
    results["comparisons"] = comp

    # ---------------------------------------------------------- KARAR A (kesin)
    ml = {k: results["models"][k]["aggregate"]["f1"]["mean"]
          for k in ("xgboost", "lightgbm", "random_forest")}
    best_ml = max(ml, key=ml.get)
    hyb = results["models"]["hybrid_lstm_gat"]["aggregate"]["f1"]["mean"]
    verdict = "DERIN_MODEL_USTUN" if hyb > ml[best_ml] else "KLASIK_ML_USTUN"
    results["karar_A_final"] = {
        "hybrid_f1": hyb, "best_classic": best_ml, "best_classic_f1": ml[best_ml],
        "best_rule_f1": best_f1, "delta_vs_classic": hyb - ml[best_ml],
        "significant_vs_classic": comp[best_ml]["mcnemar"]["significant"],
        "significant_vs_rule": comp[best_name]["mcnemar"]["significant"],
        "verdict": verdict}
    print("\n" + "=" * 78)
    print(f"KARAR A (kesin): hibrit {hyb:.4f} | en iyi klasik ({best_ml}) {ml[best_ml]:.4f} "
          f"| en iyi kural {best_f1:.4f}")
    print(f"  -> {verdict} | klasik ML'e karsi anlamli: "
          f"{comp[best_ml]['mcnemar']['significant']} | kurala karsi anlamli: "
          f"{comp[best_name]['mcnemar']['significant']}")
    print("=" * 78)

    save_json(results, C.RESULTS_DIR / "F1b_fair.json")
    np.savez_compressed(C.CACHE_DIR / "F1b_predictions.npz", y_test=yte,
                        atype=ds.type_test, **{k: np.asarray(v) for k, v in store.items()})
    print(f"\nkaydedildi: {C.RESULTS_DIR / 'F1b_fair.json'}")


if __name__ == "__main__":
    main()
