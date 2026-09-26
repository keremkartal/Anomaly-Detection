"""
F15 — Ayni kosunun tekrari ne kadar oynuyor (degerlendirme: tekrarlanabilirlik ve cokme mekanizmasi).

Makale bastan beri "GAT'in `scatter_add` toplamasi GPU'da deterministik degil,
bu yuzden ayni konfigurasyon ayni tohumda bile birebir ayni sonucu vermez"
diyordu. Bu bir ONERME idi, olculmus bir sayi degildi. Burada olculuyor.

TASARIM
-------
B0 pilotu sirasinda `lstm-gat-weighted_sum` ve `lstm-none-weighted_sum`
konfigurasyonlari, AYNI protokol ozetinde, AYNI tohumla (42), AYNI bes
fold'da ikinci kez egitildi. Iki kosu arasindaki tek fark GPU'nun kayan
nokta toplama sirasidir. Ikisi arasinda tek bir kontrollu degisken var:
uzamsal dalda GATConv olup olmamasi.

  lstm_only   -> GATConv yok   -> tekrar birebir ayni mi?
  hybrid_gat  -> GATConv var   -> tekrar ne kadar oynuyor?

NEDEN ONEMLI
------------
Makaledeki mimari farklarin buyuklugu ile TEKRAR HATASININ buyuklugu
karsilastirilabilir olursa, tek kosuluk farklardan mimari sonuc cikarmanin
neden gecersiz oldugu bir ilke olarak degil bir OLCU olarak gosterilmis olur.

Bu betik yalnizca onbellegi okur; egitim yapmaz.

KULLANIM
  python experiments/f15_reproducibility.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.utils import init_console, load_json, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F15_reproducibility.json"

# (rapor adi, dosya tabani, uzamsal dalda GATConv var mi)
PAIRS = [("hybrid_gat", "lstm-gat-weighted_sum", True),
         ("lstm_only",  "lstm-none-weighted_sum", False)]
SEED = 42
GRAPH = "trip_instance"


def main():
    stamp = protocol_stamp()
    d = C.RUNS_DIR / stamp["hash"]
    print("protokol %s | ayni tohum (%d), ayni bolumleme, ayni protokol\n"
          % (stamp["hash"], SEED))

    out = {"protocol": stamp,
           "design": {"seed": SEED, "graph": GRAPH, "n_folds": C.N_FOLDS,
                      "what_differs": ("yalnizca GPU kayan nokta toplama sirasi; "
                                       "konfigurasyon, tohum, bolumleme ve "
                                       "protokol ozeti ayni"),
                      "replicate_source": "B0 pilotunun rep0 dilimi"},
           "configs": {}}

    for name, base, has_gat in PAIRS:
        rows = []
        for k in range(C.N_FOLDS):
            a = d / ("%s-cv%d-fold%d__%s__s%d.json" % (base, C.N_FOLDS, k, GRAPH, SEED))
            b = d / ("%s-cv%d-fold%d-rep0__%s__s%d.json"
                     % (base, C.N_FOLDS, k, GRAPH, SEED))
            if not (a.exists() and b.exists()):
                print("  eksik cift, fold %d atlandi" % k)
                continue
            ma = json.loads(a.read_text(encoding="utf-8"))["metrics"]
            mb = json.loads(b.read_text(encoding="utf-8"))["metrics"]
            rows.append({
                "fold": k,
                "f1_run1": float(ma["f1"]), "f1_run2": float(mb["f1"]),
                "f1_delta": float(mb["f1"] - ma["f1"]),
                "threshold_run1": float(ma["threshold_val"]),
                "threshold_run2": float(mb["threshold_val"]),
                "best_epoch_run1": int(ma["best_epoch"]),
                "best_epoch_run2": int(mb["best_epoch"]),
                "identical": bool(ma["f1"] == mb["f1"]),
            })
        if not rows:
            continue
        dlt = np.array([r["f1_delta"] for r in rows])
        out["configs"][name] = {
            "has_gatconv": has_gat, "n_pairs": len(rows), "pairs": rows,
            "abs_delta_mean": float(np.abs(dlt).mean()),
            "abs_delta_max": float(np.abs(dlt).max()),
            "delta_sd": float(dlt.std(ddof=1)) if len(dlt) > 1 else 0.0,
            "delta_mean_signed": float(dlt.mean()),
            "n_bit_identical": int(sum(r["identical"] for r in rows)),
        }

    print("=" * 88)
    print("AYNI KOSUNUN IKI KEZ EGITILMESI")
    print("=" * 88)
    print("  %-12s %5s %9s %9s %10s %9s %9s"
          % ("konfig", "fold", "kosu 1", "kosu 2", "fark", "ep 1", "ep 2"))
    for name, d2 in out["configs"].items():
        for r in d2["pairs"]:
            print("  %-12s %5d %9.4f %9.4f %+10.4f %9d %9d"
                  % (name, r["fold"], r["f1_run1"], r["f1_run2"], r["f1_delta"],
                     r["best_epoch_run1"], r["best_epoch_run2"]))
    print("\n  %-12s %10s %12s %12s %10s"
          % ("konfig", "GATConv", "birebir ayni", "ort |fark|", "en buyuk"))
    for name, d2 in out["configs"].items():
        print("  %-12s %10s %8d/%-3d %12.4f %10.4f"
              % (name, "var" if d2["has_gatconv"] else "yok",
                 d2["n_bit_identical"], d2["n_pairs"],
                 d2["abs_delta_mean"], d2["abs_delta_max"]))

    # -------------------------------------------- mimari farklarla kiyaslama
    try:
        f5 = load_json(C.RESULTS_DIR / "F5_cv.json")
        means = {n: m["aggregate"]["f1"]["mean"] for n, m in f5["models"].items()}
        spread = max(means.values()) - min(means.values())
        gat = out["configs"].get("hybrid_gat", {})
        out["comparison_to_architecture"] = {
            "cv_model_means": means,
            "largest_gap_between_models": float(spread),
            "largest_replication_error_with_gat": gat.get("abs_delta_max"),
            "replication_error_exceeds_architecture_gap":
                bool(gat.get("abs_delta_max", 0) > spread),
        }
        print("\n  Tablo 13'te en iyi ve en kotu konfigurasyon arasindaki fark : %.4f"
              % spread)
        print("  GAT'li bir kosunun kendi tekrarindan en buyuk sapmasi        : %.4f"
              % gat.get("abs_delta_max", float("nan")))
        if gat.get("abs_delta_max", 0) > spread:
            print("  -> TEK bir kosunun tekrar hatasi, mimariler arasi en buyuk")
            print("     farktan BUYUK. Tek kosuya dayanan mimari sonuc gecersizdir.")
    except FileNotFoundError:
        pass

    save_json(out, OUT)
    print("\nkaydedildi: %s" % OUT)


if __name__ == "__main__":
    main()
