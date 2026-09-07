"""
Makale dogrulayicisi.

Bes bagimsiz kontrol:

  A. PROTOKOL TUTARLILIGI — tum sonuc JSON'lari ayni protokol ozetini
     ve ayni seed listesini tasiyor mu? (v1'in ana kusuru buydu)
  B. PAYLASILAN KOSU        — dort tabloda gecen `lstm+gat+weighted_sum`
     konfigurasyonu her yerde AYNI sayiyi mi veriyor?
  C. SAYI ESLESMESI         — .tex icindeki degerler JSON'larda var mi?
  D. FIGUR TAZELIGI         — figurler okuduklari JSON'lardan sonra mi uretilmis?
  E. YER TUTUCU             — tamamlanmamis is ifadesi kalmis mi?

Cikis kodu 0 = hepsi gecti, 1 = en az bir kontrol basarisiz.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
here = Path(__file__).parent
res = here.parent / "results"
figdir = res / "figures"
TEX = here / "main_revised.tex"
tex = TEX.read_text(encoding="utf-8")

OK, FAIL = "  [OK]  ", "  [HATA]"
failures = []


def load(name):
    p = res / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


NAMES = ("F1b_fair", "F2_graph", "F3_fusion", "F4_benchmark",
         "F5_cv", "F6_efficiency", "F7_cluster_stats")
J = {n: load(n) for n in NAMES}
missing = [n for n, v in J.items() if v is None]
if missing:
    print(f"UYARI: bulunamayan sonuc dosyalari: {missing}\n")


# ======================================================== A. PROTOKOL
def check_protocol():
    print("A. PROTOKOL TUTARLILIGI")
    stamps = {n: v.get("protocol", {}).get("hash")
              for n, v in J.items() if v is not None}
    have = {n: h for n, h in stamps.items() if h}
    if not have:
        failures.append("hicbir JSON protokol damgasi tasimiyor")
        print(f"{FAIL} hicbir JSON protokol damgasi tasimiyor")
        return
    uniq = set(have.values())
    for n, h in sorted(stamps.items()):
        print(f"    {n:20s} {h or '— (damga yok)'}")
    if len(uniq) == 1 and not [n for n, h in stamps.items() if not h]:
        print(f"{OK} tum sonuclar tek protokolde: {uniq.pop()}")
    else:
        msg = f"farkli/eksik protokol damgasi: {sorted(uniq)}"
        failures.append(msg)
        print(f"{FAIL} {msg}")

    # seed listeleri
    seeds = {n: tuple(v.get("config", {}).get("seeds", []))
             for n, v in J.items() if v is not None and "config" in v}
    seeds = {n: s for n, s in seeds.items() if s}
    su = set(seeds.values())
    if len(su) <= 1:
        print(f"{OK} seed listeleri uyumlu: {sorted(su)[0] if su else '—'}")
    else:
        # F5 CV'de seed sayisi bilerek farkli (fold x seed tasarimi)
        non_cv = {n: s for n, s in seeds.items() if n != "F5_cv"}
        if len(set(non_cv.values())) <= 1:
            print(f"{OK} sabit-bolme seed listeleri uyumlu "
                  f"{sorted(set(non_cv.values()))[0]}; F5 (CV) bilerek farkli "
                  f"{seeds.get('F5_cv')}")
        else:
            msg = f"seed listeleri uyusmuyor: {seeds}"
            failures.append(msg)
            print(f"{FAIL} {msg}")
    print()


# ======================================================== B. PAYLASILAN KOSU
def check_shared_run():
    print("B. PAYLASILAN KOSU — lstm+gat+weighted_sum dort tabloda")
    vals = {}
    if J["F1b_fair"]:
        vals["Tablo 9/10  F1b hybrid_lstm_gat"] = \
            J["F1b_fair"]["models"]["hybrid_lstm_gat"]["aggregate"]["f1"]["mean"]
    if J["F4_benchmark"] and "T:lstm" in J["F4_benchmark"]["configs"]:
        vals["Tablo 11    F4 T:lstm"] = \
            J["F4_benchmark"]["configs"]["T:lstm"]["aggregate"]["f1"]["mean"]
    if J["F3_fusion"] and "weighted_sum" in J["F3_fusion"]["variants"]:
        vals["Tablo 12    F3 weighted_sum"] = \
            J["F3_fusion"]["variants"]["weighted_sum"]["aggregate"]["f1"]["mean"]
    if J["F2_graph"] and "original/weighted_sum" in J["F2_graph"]["runs"]:
        vals["Tablo 15    F2 trip-instance/weighted_sum"] = \
            J["F2_graph"]["runs"]["original/weighted_sum"]["f1_mean"]

    for k, v in vals.items():
        print(f"    {k:42s} {v:.6f}")
    if len(vals) < 2:
        print("    (karsilastirmak icin yeterli dosya yok)\n")
        return
    spread = max(vals.values()) - min(vals.values())
    if spread < 1e-9:
        print(f"{OK} dordu de birebir ayni (fark {spread:.2e})")
    else:
        msg = (f"paylasilan kosu farkli degerler veriyor, fark {spread:.6f} "
               f"— runstore devre disi mi?")
        failures.append(msg)
        print(f"{FAIL} {msg}")
    print()


# ======================================================== C. SAYILAR
def check_numbers():
    print("C. SAYI ESLESMESI (.tex <- results/*.json)")
    checks = []

    def chk(label, value, decimals=4):
        if value is None:
            return
        s = f"{value:.{decimals}f}"
        checks.append((label, s, s in tex))

    f1b, f3, f4, f5, f6, f2 = (J[n] for n in
                               ("F1b_fair", "F3_fusion", "F4_benchmark",
                                "F5_cv", "F6_efficiency", "F2_graph"))

    if f1b:
        for k, lbl in (("hybrid_lstm_gat", "hibrit F1"),
                       ("lightgbm", "LightGBM F1"), ("xgboost", "XGBoost F1"),
                       ("random_forest", "RandomForest F1"),
                       ("lstm_only", "LSTM-only F1")):
            if k in f1b["models"]:
                a = f1b["models"][k]["aggregate"]["f1"]
                chk(lbl, a["mean"]); chk(lbl + " std", a.get("std"))
        if "rule_best" in f1b["models"]:
            chk("kural F1", f1b["models"]["rule_best"]["single"]["f1"])

    if f3:
        for k, v in f3["variants"].items():
            a = v["aggregate"]["f1"]
            chk(f"F3 {k} F1", a["mean"]); chk(f"F3 {k} std", a["std"])
        if "worst_seed_f1" in f3:
            chk("F3 gated en kotu seed", f3["worst_seed_f1"]["gated"])

    if f4:
        for k, v in f4["configs"].items():
            chk(f"F4 {k} F1", v["aggregate"]["f1"]["mean"])
            chk(f"F4 {k} std", v["aggregate"]["f1"]["std"])

    if f5:
        for k, v in f5["models"].items():
            chk(f"F5 {k} kosu-ort F1", v["aggregate"]["f1"]["mean"])
            chk(f"F5 {k} std", v["aggregate"]["f1"]["std"])
            chk(f"F5 {k} en kotu kosu", v["worst_run_f1"])
            chk(f"F5 {k} havuz F1", v["pooled"]["f1"])
            chk(f"F5 {k} ECE", v["calibration"]["ece"])
            chk(f"F5 {k} Brier", v["calibration"]["brier"])
        for t, v in f5.get("karar_B", {}).items():
            if not isinstance(v, dict) or "hybrid_f1" not in v:
                continue
            chk(f"F5 {t} hibrit", v["hybrid_f1"])
            chk(f"F5 {t} lstm", v["lstm_only_f1"])
            cb = v.get("cluster_bootstrap", {})
            if cb:
                chk(f"F5 {t} trip GA alt", cb["lo"])
                chk(f"F5 {t} trip GA ust", cb["hi"])
        # CV fuzyon ablasyonu: gated'in digerlerine karsi trip duzeyi farklari
        for k, v in (f5.get("fusion_ablation_cv", {})
                     .get("gated_vs_others_trip_level", {}).items()):
            cb = v["cluster_bootstrap"]
            chk(f"F5 CV gated vs {k} fark", cb["observed_diff"])
            chk(f"F5 CV gated vs {k} GA alt", cb["lo"])
            chk(f"F5 CV gated vs {k} GA ust", cb["hi"])

    if f6:
        m = (f6["models"].get("STANet (weighted sum, GAT)")
             or f6["models"].get("STANet (gated, GAT)"))
        if m:
            for k, lbl in (("cuda_single_window", "F6 GPU p50"),
                           ("cuda_single_window_cached", "F6 GPU cached"),
                           ("cpu_single_window", "F6 CPU p50"),
                           ("cpu_single_window_cached", "F6 CPU cached")):
                if k in m:
                    chk(lbl, m[k]["p50_ms"], 2)

    if f2:
        for key, v in f2["runs"].items():
            chk(f"F2 {key} F1", v["f1_mean"])
            chk(f"F2 {key} std", v["f1_std"])

    if J["F7_cluster_stats"]:
        ref = J["F7_cluster_stats"]["families"]["baselines"]["reference_ci"]
        chk("F7 referans trip GA alt", ref["lo"])
        chk("F7 referans trip GA ust", ref["hi"])

    ok = [c for c in checks if c[2]]
    bad = [c for c in checks if not c[2]]
    print(f"    kontrol edilen : {len(checks)}")
    print(f"    metinde bulunan: {len(ok)}")
    if bad:
        print(f"{FAIL} metinde BULUNAMAYAN {len(bad)} deger:")
        for lbl, val, _ in bad:
            print(f"      {lbl:38s} {val}")
        failures.append(f"{len(bad)} sayi .tex ile JSON arasinda uyusmuyor")
    else:
        print(f"{OK} tum sayilar eslesiyor")
    print()


# ======================================================== D. FIGURLER
FIG_SOURCES = {
    "fig_F1_baselines.png":   ["F1b_fair"],
    "fig_F1_roc_pr.png":      ["F1b_fair"],
    "fig_F3_fusion_ablation.png": ["F3_fusion"],
    "fig_F3_gate.png":        ["F3_fusion"],
    "fig_F4_benchmark.png":   ["F4_benchmark"],
    "fig_F5_subtype.png":     ["F5_cv"],
    "fig_F5_calibration.png": ["F5_cv"],
    "fig_F6_efficiency.png":  ["F6_efficiency"],
}


def check_figures():
    print("D. FIGUR TAZELIGI (figur, okudugu JSON'dan sonra mi uretildi)")
    stale, absent = [], []
    for fig, srcs in FIG_SOURCES.items():
        fp = figdir / fig
        if not fp.exists():
            absent.append(fig)
            continue
        ft = fp.stat().st_mtime
        for s in srcs:
            sp = res / f"{s}.json"
            if sp.exists() and sp.stat().st_mtime > ft:
                stale.append((fig, s))
    if absent:
        print(f"    uretilmemis figur: {absent}")
    if stale:
        print(f"{FAIL} kaynagindan ESKI figurler:")
        for f, s in stale:
            print(f"      {f} < {s}.json")
        failures.append(f"{len(stale)} figur kaynagindan eski — f9'u yeniden kosun")
    elif not absent:
        print(f"{OK} tum figurler kaynaklarindan yeni")
    print()


# ======================================================== E. YER TUTUCU
PLACEHOLDERS = [
    r"PENDING-REPO",
    r"PENDING-DOI",
    r"\[repository URL",
    r"\[DOI to be",
    r"to be inserted",
    r"to be determined",
    r"reconciliation is pending",
    r"TBD",
    r"XXX",
    r"TODO",
    r"belirlenmeli",
    r"must be verified from the publisher record",
]


def check_placeholders():
    print("E. YER TUTUCU / TAMAMLANMAMIS IS IFADESI")
    hits = []
    for pat in PLACEHOLDERS:
        for m in re.finditer(pat, tex, re.IGNORECASE):
            line = tex[:m.start()].count("\n") + 1
            ctx = tex[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
            hits.append((line, pat, ctx.strip()))
    if hits:
        print(f"{FAIL} {len(hits)} yer tutucu bulundu:")
        for line, pat, ctx in hits:
            print(f"      satir {line:4d} [{pat}]  ...{ctx}...")
        failures.append(f"{len(hits)} yer tutucu .tex icinde duruyor")
    else:
        print(f"{OK} yer tutucu yok")
    print()


if __name__ == "__main__":
    print("=" * 78)
    print(f"MAKALE DOGRULAMASI — {TEX.name}")
    print("=" * 78 + "\n")
    check_protocol()
    check_shared_run()
    check_numbers()
    check_figures()
    check_placeholders()
    print("=" * 78)
    if failures:
        print(f"SONUC: {len(failures)} KONTROL BASARISIZ")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SONUC: TUM KONTROLLER GECTI")
    sys.exit(0)
