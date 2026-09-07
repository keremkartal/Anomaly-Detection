"""
Bir PDF'teki sayilari results/*.json ile karsilastirir.

`_verify_numbers.py` LaTeX kaynagini denetler; bu betik ise **derlenmis bir
PDF'i** denetler. Kullanim amaci: baska biri tarafindan guncellenmis bir
surumun (or. MDPI sablonundaki kopya) gercekten guncel sayilari tasiyip
tasimadigini kontrol etmek.

  python makale/_verify_pdf.py "C:/yol/main.pdf"
"""
import json
import re
import sys
from pathlib import Path

import fitz

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
RES = HERE.parent / "results"


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def norm(t):
    """PDF'ten gelen metni normalize et: satir sonu tireleri, bosluklar."""
    t = t.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"-\s*\n\s*", "", t)      # satir sonu bolunmesi
    t = re.sub(r"\s+", " ", t)
    return t


def main(pdf_path):
    doc = fitz.open(pdf_path)
    text = norm("".join(p.get_text() for p in doc))
    print("=" * 78)
    print(f"PDF DOGRULAMASI — {Path(pdf_path).name}  ({doc.page_count} sayfa)")
    print("=" * 78 + "\n")

    f1b, f2, f3, f4, f5, f6, f7 = (load(n) for n in (
        "F1b_fair", "F2_graph", "F3_fusion", "F4_benchmark",
        "F5_cv", "F6_efficiency", "F7_cluster_stats"))

    checks = []

    def chk(label, value, dec=4, group=""):
        if value is None:
            return
        s = f"{value:.{dec}f}"
        checks.append((group, label, s, s in text))

    # ---------------------------------------------------------- guncel sayilar
    if f1b:
        for k, lbl in (("hybrid_lstm_gat", "hibrit F1"), ("lstm_only", "LSTM-only F1"),
                       ("lightgbm", "LightGBM F1"), ("xgboost", "XGBoost F1"),
                       ("random_forest", "RandomForest F1")):
            if k in f1b["models"]:
                a = f1b["models"][k]["aggregate"]["f1"]
                chk(lbl, a["mean"], group="Tablo 9")
                chk(lbl + " std", a.get("std"), group="Tablo 9")
        chk("kural F1", f1b["models"]["rule_best"]["single"]["f1"], group="Tablo 9")
        for k, v in f1b.get("comparisons", {}).items():
            b = v["bootstrap_diff"]
            chk(f"{k} GA alt", b["lo"], group="Tablo 10")
            chk(f"{k} GA ust", b["hi"], group="Tablo 10")

    if f4:
        for k, v in f4["configs"].items():
            chk(f"{k} F1", v["aggregate"]["f1"]["mean"], group="Tablo 11")
            chk(f"{k} std", v["aggregate"]["f1"]["std"], group="Tablo 11")

    if f3:
        for k, v in f3["variants"].items():
            chk(f"{k} F1", v["aggregate"]["f1"]["mean"], group="Tablo 12")
            chk(f"{k} std", v["aggregate"]["f1"]["std"], group="Tablo 12")
        chk("gated en kotu seed", f3["worst_seed_f1"]["gated"], group="Tablo 12")

    if f5:
        for k, v in f5["models"].items():
            chk(f"{k} kosu F1", v["aggregate"]["f1"]["mean"], group="Tablo 13")
            chk(f"{k} std", v["aggregate"]["f1"]["std"], group="Tablo 13")
        for t, v in f5.get("karar_B", {}).items():
            if isinstance(v, dict) and "hybrid_f1" in v:
                chk(f"{t} hibrit", v["hybrid_f1"], group="Tablo 14")
                chk(f"{t} lstm", v["lstm_only_f1"], group="Tablo 14")

    if f2:
        for k, v in f2["runs"].items():
            chk(f"{k} F1", v["f1_mean"], group="Tablo 15")

    if f6:
        m = (f6["models"].get("STANet (weighted sum, GAT)")
             or f6["models"].get("STANet (gated, GAT)"))
        if m:
            for k, lbl in (("cuda_single_window", "GPU p50"),
                           ("cuda_single_window_cached", "GPU cached"),
                           ("cpu_single_window", "CPU p50"),
                           ("cpu_single_window_cached", "CPU cached")):
                if k in m:
                    chk(lbl, m[k]["p50_ms"], 2, group="Tablo 16/17 (gecikme)")

    groups = {}
    for g, lbl, val, ok in checks:
        groups.setdefault(g, []).append((lbl, val, ok))

    print("GUNCEL SAYILAR PDF'TE VAR MI\n")
    total_ok = total = 0
    for g in groups:
        ok = [c for c in groups[g] if c[2]]
        bad = [c for c in groups[g] if not c[2]]
        total_ok += len(ok); total += len(groups[g])
        mark = "OK  " if not bad else "EKSIK"
        print(f"  [{mark}] {g:22s} {len(ok):>3}/{len(groups[g]):<3}")
        for lbl, val, _ in bad:
            print(f"          eksik: {lbl:34s} {val}")
    print(f"\n  TOPLAM: {total_ok}/{total}\n")

    # ---------------------------------------------------------- eskimis sayilar
    STALE = {
        "0.8836": "v1 hibrit F1 (yeni: 0.8713)",
        "0.0033": "v1 hibrit std (yeni: 0.0188)",
        "0.8484": "v1 weighted_sum F1 (yeni: 0.8713)",
        "0.0138": "v1 weighted_sum std (yeni: 0.0188)",
        "0.6811": "v1 gated F1 (yeni: 0.6953)",
        "0.3366": "v1 gated std (yeni: 0.3488)",
        "0.8513": "v1 Tablo 15 hibrit (yeni: 0.8713 / 0.6953)",
        "0.0833": "v1 gated cokme (yeni: 0.0715)",
        "0.8837": "v1 CV LSTM-only (yeni: 0.8839)",
        "0.8674": "v1 CV weighted_sum (yeni: 0.8831)",
        "0.8736": "v1 CV gated (yeni: 0.8908)",
        "0.0338": "v1 gated ECE (yeni: 0.0579)",
        "0.8489": "v1 LSTM-only F1 (yeni: 0.8455)",
        "0.0261": "v1 LSTM-only std (yeni: 0.0190)",
        "0.8758": "v1 GRU F1 (yeni: 0.8701)",
        "0.8565": "v1 wavenet F1 (yeni: 0.8535)",
        "0.3659": "v1 adaptive F1 (yeni: 0.4657)",
        "0.5883": "v1 GAT-only AUC (yeni: 0.5910)",
        "0.7266": "v1 merged GAT-only AUC (yeni: 0.7115)",
        "0.8141": "v1 merged hibrit (yeni: 0.8413)",
        "0.1558": "v1 hiz asimi hibrit (yeni: 0.1469)",
        "0.9080": "v1 uzun duraklama hibrit (yeni: 0.9400)",
    }
    print("ESKIMIS (v1) SAYILAR HALA VAR MI\n")
    found = [(v, d) for v, d in STALE.items() if v in text]
    if found:
        print(f"  [DIKKAT] {len(found)} eskimis deger bulundu:")
        for v, desc in found:
            n = text.count(v)
            print(f"      {v}  x{n}   {desc}")
    else:
        print("  [OK]   eskimis v1 degeri yok")
    print()

    # ---------------------------------------------------------- yapisal ekler
    print("YAPISAL EKLER VAR MI\n")
    STRUCT = [
        ("Protocol Stamping", "protocol stamp"),
        ("Unit of Statistical Analysis", "istatistiksel birim bolumu"),
        ("trip level|trip-level", "trip duzeyi ifadesi"),
        ("Holm", "Holm duzeltmesi"),
        ("cluster bootstrap", "kume bootstrap"),
        ("permutation", "permutasyon testi"),
        ("Scope of the Contribution", "kapsam bolumu"),
        ("keremkartal/Anomaly-Detection", "depo adresi"),
        ("22637361", "Zenodo surum DOI"),
        ("14 trips", "14 trip ifadesi"),
    ]
    for pat, desc in STRUCT:
        hit = re.search(pat, text, re.IGNORECASE)
        print(f"  [{'OK  ' if hit else 'YOK '}] {desc}")
    print()

    # ---------------------------------------------------------- yer tutucu
    print("YER TUTUCU / TAMAMLANMAMIS IS\n")
    # "PENDING" buyuk harfle ve kelime sinirinda aranir; aksi halde
    # "depending" gibi kelimeler yanlis pozitif verir.
    PH = [(r"PENDING", 0), (r"to be inserted", re.IGNORECASE),
          (r"TBD", 0), (r"reconciliation is pending", re.IGNORECASE),
          (r"must be verified from the publisher", re.IGNORECASE),
          (r"XXX", 0), (r"Cross-experiment inconsistencies", re.IGNORECASE)]
    hits = [p for p, fl in PH if re.search(p, text, fl)]
    if hits:
        print(f"  [DIKKAT] {hits}")
    else:
        print("  [OK]   yer tutucu yok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "main_revised.pdf"))
