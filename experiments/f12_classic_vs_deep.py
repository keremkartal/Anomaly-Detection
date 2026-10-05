"""
F12 — Klasik modeller ile derin modellerin ESIT tasarimda karsilastirilmasi
(degerlendirme: esit tasarim
ve "derin model gerekli mi" sorusu).

F11 klasik modelleri F5'in fold'larinda kosturdu. Artik iki taraf ayni
bolumleme, ayni esik politikasi ve ayni birim altinda karsilastirilabilir.
Bu betik karsilastirmayi iki ayri duzeyde yapar:

  1. KOSU DUZEYI (eslestirilmis). Her (fold, tohum) ciftinde iki modelin F1
     farki alinir; 10 eslestirilmis fark. Capraz dogrulamada egitim kumeleri
     ortustugu icin belirsizlik Nadeau-Bengio ile duzeltilir.

  2. TRIP DUZEYI (havuzlanmis). Havuzlanmis test tahminleri uzerinde kume
     bootstrap ve isaret-degistirme permutasyonu; birim trip. Holm duzeltmesi
     tum karsilastirma ailesine uygulanir.

Ayrica TOST ile ESDEGERLIK sinanir: anlamsiz bir fark
"fark yok" demek degildir. Esdegerlik siniri onceden belirlenir (bkz. DELTA).

KULLANIM
  python experiments/f12_classic_vs_deep.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.cluster_stats import (cluster_bootstrap_diff, cluster_permutation_test,
                                  cluster_summary, holm_correction, nadeau_bengio,
                                  tost_equivalence)
from stanet.utils import init_console, load_json, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / (os.environ.get("F12_OUT","F12_classic_vs_deep")+".json")

# Esdegerlik siniri ONCEDEN belirlenmistir ve gerekcesi su:
# F5'te derin modellerin kosular arasi standart sapmasi 0.041-0.054 F1
# araligindadir. Tek bir kosunun kendi tohum gurultusu bu buyuklukteyken
# bundan kucuk bir mimari farki "gercek" saymak anlamsizdir. Sinir olarak
# en kucuk gozlenen kosu-ici standart sapmanin yaklasik yarisini aliyoruz.
DELTA = 0.02
DELTA_NOTE = ("0.02 F1; gerekce: F5'te kosular arasi standart sapma 0.041-0.054, "
              "yani tohum gurultusunun yaklasik yarisi. Sinir veri gorulmeden "
              "degil, onceki protokolun gurultu duzeyine dayanarak secildi ve "
              "karsilastirmalardan once sabitlendi.")

DEEP = ["stanet_gated", "hybrid_gat", "hybrid_concat", "hybrid_crossatt", "lstm_only"]
CLASSIC = ["lightgbm", "xgboost", "random_forest", "rule_best"]
REFERENCE = "stanet_gated"      # onerilen mimari


def run_level(f5, f11):
    """Eslestirilmis (fold, tohum) farklari + Nadeau-Bengio + TOST."""
    def table(d):
        """(fold, tohum) -> F1. Kural tabanli model parametresizdir ve tohumu
        yoktur; degeri o fold'un butun tohumlarina kopyalanir, boylece
        eslestirme korunur. Bu, kuralin varyansini sifir sayar — dogrusu
        budur, cunku kural gercekten deterministiktir."""
        seeds, out, rule_rows = set(), {}, {}
        for name, m in d["models"].items():
            for r in m["runs"]:
                if r.get("seed") is None:
                    rule_rows.setdefault(name, {})[r["fold"]] = float(r["f1"])
                else:
                    seeds.add(r["seed"])
                    out.setdefault(name, {})[(r["fold"], r["seed"])] = float(r["f1"])
        for name, byfold in rule_rows.items():
            out[name] = {(k, s): v for k, v in byfold.items() for s in sorted(seeds)}
        return out

    a, b = table(f5), table(f11)
    both = {**a, **b}
    fold_info = f5["fold_info"]
    n_tr = int(np.mean([fi["trips"]["train"] for fi in fold_info]))
    n_te = int(np.mean([fi["trips"]["test"] for fi in fold_info]))

    rows, pvals = {}, {}
    for cl in CLASSIC:
        if cl not in both:
            continue
        for dp in DEEP:
            keys = sorted(set(both[cl]) & set(both[dp]))
            if len(keys) < 2:
                continue
            diffs = [both[cl][k] - both[dp][k] for k in keys]
            nb = nadeau_bengio(diffs, n_train=n_tr, n_test=n_te)
            tost = tost_equivalence(diffs, DELTA, n_train=n_tr, n_test=n_te)
            key = cl + "_minus_" + dp
            rows[key] = {"n_paired": len(keys), "diffs": diffs,
                         "nadeau_bengio": nb, "tost": tost}
            pvals[key] = nb["p_corrected"]
    holm = holm_correction(pvals) if pvals else {}
    for k, v in holm.items():
        rows[k]["holm"] = v
    return {"n_train_trips": n_tr, "n_test_trips": n_te,
            "comparisons": rows, "holm_family_size": len(pvals)}


def trip_level(npz5, npz11):
    """Havuzlanmis tahminler uzerinde kume bootstrap + permutasyon.

    Iki dosyanin anahtar duzeni farkli: F5 tek bir `y`/`trip` ve model basina
    `b_<ad>` tutar; F11 model basina `<ad>_y`/`<ad>_bin`/`<ad>_trip` tutar.
    Hizalamayi varsaymiyoruz — `y` dizileri birebir esitlenmezse karsilastirma
    yapilmaz ve bu durum cikti dosyasina yazilir.
    """
    avail5 = sorted({k[2:] for k in npz5.files if k.startswith("b_")})
    avail11 = sorted({k[:-2] for k in npz11.files if k.endswith("_y")})

    ref_y = npz5["y"]
    ref_g = npz5["trip"]
    ref_b = npz5["b_" + REFERENCE]
    out, pvals = {}, {}
    for cl in CLASSIC:
        if cl not in avail11:
            continue
        y, b, g = npz11[cl + "_y"], npz11[cl + "_bin"], npz11[cl + "_trip"]
        aligned = (len(y) == len(ref_y) and np.array_equal(np.asarray(y).ravel(),
                                                           np.asarray(ref_y).ravel())
                   and np.array_equal(np.asarray(g, dtype=str).ravel(),
                                      np.asarray(ref_g, dtype=str).ravel()))
        if not aligned:
            out[cl + "_minus_" + REFERENCE] = {
                "error": "havuzlanmis diziler hizalanmiyor; trip duzeyi atlandi"}
            continue
        boot = cluster_bootstrap_diff(y, b, ref_b, g)
        perm = cluster_permutation_test(y, b, ref_b, g)
        key = cl + "_minus_" + REFERENCE
        out[key] = {"bootstrap": boot, "permutation": perm}
        pvals[key] = perm["p_value"]
    holm = holm_correction(pvals) if pvals else {}
    for k, v in holm.items():
        out[k]["holm"] = v
    return {"reference": REFERENCE, "deep_available": avail5,
            "classic_available": avail11,
            "cluster_info": cluster_summary(ref_g), "comparisons": out}


def main():
    f5 = load_json(C.RESULTS_DIR / (os.environ.get("F12_F5", "F5_cv") + ".json"))
    f11 = load_json(C.RESULTS_DIR / (os.environ.get("F12_F11", "F11_classic_cv") + ".json"))
    npz5 = np.load(C.CACHE_DIR / "F5_pooled.npz")
    npz11 = np.load(C.RESULTS_DIR / "F11_pooled.npz")

    stamp = protocol_stamp()
    print("protokol %s | esdegerlik siniri delta=%.3f F1\n" % (stamp["hash"], DELTA))

    rl = run_level(f5, f11)
    print("=" * 96)
    print("KOSU DUZEYI — eslestirilmis (fold, tohum), Nadeau-Bengio duzeltmeli")
    print("egitim %d / test %d trip, ortusme orani %.3f"
          % (rl["n_train_trips"], rl["n_test_trips"],
             rl["n_test_trips"] / rl["n_train_trips"]))
    print("=" * 96)
    print("  %-38s %9s %22s %8s %8s %11s"
          % ("karsilastirma", "ort fark", "NB %95 GA", "NB p", "Holm p", "TOST"))
    for k, v in rl["comparisons"].items():
        nb, ts = v["nadeau_bengio"], v["tost"]
        verdict = ("esdeger" if ts["equivalent"]
                   else ("FARK VAR" if v.get("holm", {}).get("significant") else "belirsiz"))
        print("  %-38s %+9.4f [%+.4f, %+.4f] %8.4f %8.4f %11s"
              % (k, nb["mean_diff"], nb["ci_lo"], nb["ci_hi"], nb["p_corrected"],
                 v.get("holm", {}).get("p_holm", float("nan")), verdict))

    tl = trip_level(npz5, npz11)
    print("\n" + "=" * 96)
    print("TRIP DUZEYI — havuzlanmis tahminler, kume bootstrap + permutasyon")
    print("referans: %s | %d trip" % (REFERENCE, tl["cluster_info"]["n_trips"]))
    print("=" * 96)
    print("  %-38s %9s %22s %8s %8s"
          % ("karsilastirma", "dF1", "%95 GA (kume)", "perm p", "Holm p"))
    for k, v in tl["comparisons"].items():
        if "error" in v:
            print("  %-38s  %s" % (k, v["error"]))
            continue
        bo, pe = v["bootstrap"], v["permutation"]
        print("  %-38s %+9.4f [%+.4f, %+.4f] %8.4f %8.4f"
              % (k, bo["observed_diff"], bo["lo"], bo["hi"], pe["p_value"],
                 v.get("holm", {}).get("p_holm", float("nan"))))

    save_json({"protocol": stamp, "delta": DELTA, "delta_rationale": DELTA_NOTE,
               "run_level": rl, "trip_level": tl}, OUT)
    print("\nkaydedildi: %s" % OUT)


if __name__ == "__main__":
    main()
