"""
F32 — Mimari yelpaze ile replikasyon hatasinin karsilastirmasi.

NICIN AYRI BIR DOSYA
--------------------
Makale su cumleyi kuruyor: "bes konfigurasyonun en iyi-en kotu farki tum
kosularda 0.0282, dejenere kosular haric 0.0075; tek bir kosunun
replikasyon hatasi 0.0414." Bu uc sayinin ikisini elle hesaplamistim ve
HICBIR sonuc dosyasinda gecmiyorlardi. Kendi protokolum tam olarak bunu
engellemek icin var: metne giren her sayi bir sonuc dosyasindan gelmeli.

F15 kendi `comparison_to_architecture` bolumunu tasiyor ama o IKI tohumlu
CV ortalamalarina bakiyor (en buyuk fark 0.0205). Birincil analiz bes
tohuma gectigi icin o karsilastirma eskidi. F15'i degistirmek yerine
-- islenmis bir sonuc dosyasini sonradan duzenlemek kotu pratik --
turetilmis buyuklukleri burada yeniden hesaplayip ayri bir dosyaya
yaziyorum.

EGITIM YOK: yalnizca mevcut kosulari okur.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stanet import config as C
from stanet.utils import init_console, protocol_stamp, save_json

init_console()

PRIMARY = os.environ.get("F32_CV", "F5_cv_5seed")
COLLAPSE_THRESHOLD = 0.30
WEIGHT_LEARNING = {"stanet_gated", "hybrid_crossatt"}
CONTROL = "lstm_only"


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    cv = json.loads((C.RESULTS_DIR / f"{PRIMARY}.json").read_text(encoding="utf-8"))
    rep = json.loads((C.RESULTS_DIR / "F15_reproducibility.json")
                     .read_text(encoding="utf-8"))
    stamp = protocol_stamp()
    if cv["protocol"]["hash"] != stamp["hash"]:
        sys.exit(f"protokol uyusmuyor: {cv['protocol']['hash']} != {stamp['hash']}")

    per = {}
    for name, m in cv["models"].items():
        vals = [r["f1"] for r in m["runs"]]
        keep = [x for x in vals if x >= COLLAPSE_THRESHOLD]
        per[name] = {
            "n": len(vals),
            "f1_mean": mean(vals),
            "f1_sd": sd(vals),
            "f1_min": min(vals),
            "n_collapsed": len(vals) - len(keep),
            "f1_mean_excl_degenerate": mean(keep) if keep else None,
            "f1_sd_excl_degenerate": sd(keep) if keep else None,
            "learns_weight": name in WEIGHT_LEARNING,
        }

    fus = [n for n in per if n != CONTROL]
    span_all = (max(per[n]["f1_mean"] for n in fus)
                - min(per[n]["f1_mean"] for n in fus))
    span_keep = (max(per[n]["f1_mean_excl_degenerate"] for n in fus)
                 - min(per[n]["f1_mean_excl_degenerate"] for n in fus))

    # replikasyon hatasi: GATConv tasiyan konfigurasyonun en buyuk ikiz farki
    gat = [c for c in rep["configs"].values() if c["has_gatconv"]]
    rep_max = max(c["abs_delta_max"] for c in gat)
    rep_mean = max(c["abs_delta_mean"] for c in gat)

    cw = sum(per[n]["n_collapsed"] for n in per if per[n]["learns_weight"])
    nw = sum(per[n]["n"] for n in per if per[n]["learns_weight"])
    cf = sum(per[n]["n_collapsed"] for n in per if not per[n]["learns_weight"])
    nf = sum(per[n]["n"] for n in per if not per[n]["learns_weight"])

    order_all = sorted(fus + [CONTROL], key=lambda n: -per[n]["f1_mean"])
    order_keep = sorted(fus + [CONTROL],
                        key=lambda n: -per[n]["f1_mean_excl_degenerate"])

    out = {
        "protocol": stamp,
        "source": {"cv": PRIMARY, "replication": "F15_reproducibility"},
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "models": per,
        "fusion_span_all_runs": span_all,
        "fusion_span_excluding_degenerate": span_keep,
        "replication_error_max": rep_max,
        "replication_error_mean": rep_mean,
        "replication_over_span_all": rep_max / span_all,
        "replication_over_span_excluding_degenerate": rep_max / span_keep,
        "sd_range_excluding_degenerate": [
            min(per[n]["f1_sd_excl_degenerate"] for n in per),
            max(per[n]["f1_sd_excl_degenerate"] for n in per)],
        "collapse": {"weight_learning": [cw, nw], "fixed_weight": [cf, nf],
                     "total": [cw + cf, nw + nf]},
        "ranking_all_runs": order_all,
        "ranking_excluding_degenerate": order_keep,
        "ranking_changes_when_degenerate_excluded": order_all != order_keep,
        "note": ("Dejenere kosular cikarildiginda agirlik ogrenen operatorler "
                 "en iyi ikiye cikiyor; yirmi beste bir basarisiz kosu her "
                 "birini birincilikten sonlulua tasiyor."),
    }
    save_json(out, C.RESULTS_DIR / "F32_arch_vs_replication.json")

    print(f"birincil CV: {PRIMARY}  (protokol {stamp['hash']})")
    print(f"\n{'model':<20}{'F1':>10}{'sd':>9}{'cokme':>7}{'haric F1':>11}")
    for n in order_all:
        d = per[n]
        print(f"  {n:<18}{d['f1_mean']:>10.4f}{d['f1_sd']:>9.4f}"
              f"{d['n_collapsed']:>4}/{d['n']:<3}{d['f1_mean_excl_degenerate']:>10.4f}")
    print(f"\nfuzyon yelpazesi, tum kosular   : {span_all:.4f}")
    print(f"fuzyon yelpazesi, dejenere haric: {span_keep:.4f}")
    print(f"replikasyon hatasi (GATConv)    : {rep_max:.4f}")
    print(f"  -> yelpazenin {rep_max / span_keep:.1f} kati")
    print(f"cokme: ogrenen {cw}/{nw} | sabit {cf}/{nf}")
    print(f"siralama dejenere haric degisiyor mu: "
          f"{out['ranking_changes_when_degenerate_excluded']}")


if __name__ == "__main__":
    main()
