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
import pathlib
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


# FAZ B dosyalari da denetime dahil: metne giren her p degeri ve her sayi
# bunlardan birinde gecmek zorunda. F10_repeated_cv tam kosum bitince
# listeye girer; pilot simdiden burada cunku makalede aniliyor.
NAMES = ("F1b_fair", "F2_graph", "F3_fusion", "F4_benchmark",
         "F5_cv", "F5_cv_5seed", "F6_efficiency", "F7_cluster_stats",
         "F8_effects_subtypes",
         "F10_pilot", "F10_repeated_cv", "F11_classic_cv",
         "F11_classic_cv_5seed", "F12_classic_vs_deep_5seed",
         "F14_graph_controls_5seed", "F18_inductive_5seed",
         "F22_imputation_5seed", "F32_arch_vs_replication",
         "F12_classic_vs_deep", "F13_prevalence", "F14_graph_controls",
         "F15_reproducibility", "F16_equivalence", "F18_inductive",
         "F19_noise", "F20_window_stride", "F21_baseline_search",
         "F22_imputation", "F23_data_audit", "F24_model_audit",
         "F25_pooled_seed", "F26_repeated_analysis",
         "F27_new_data_audit", "F29_collapse_mechanism",
         "F28_hamburg_knn_n0_5_a5", "F28_kocaeli_knn_n0_5_a5",
         "F28_hamburg_baseline_n0_5_a5", "F28_kocaeli_baseline_n0_5_a5",
         "F30_leak_diagnosis", "F31_markov")
J = {n: load(n) for n in NAMES}
missing = [n for n, v in J.items() if v is None]
if missing:
    print(f"UYARI: bulunamayan sonuc dosyalari: {missing}\n")


# ======================================================== A. PROTOKOL
def check_protocol():
    print("A. PROTOKOL TUTARLILIGI")
    # Egitim YAPMAYAN sonuc dosyalarinin protokol damgasi olmaz ve olmamalidir:
    # ham veriyi sayan bir denetimin epoch butcesiyle ya da cuDNN ayariyla
    # isi yoktur. Onlari damga karsilastirmasindan muaf tutuyoruz.
    # Egitim YAPMAYAN denetimler (F23 veri, F24 mimari) damga tasimaz.
    # F20 ve F21 ise BILEREK birden cok protokolde kosar: pencere/adim ve
    # ogrenme orani damganin icindedir, o yuzden her arama noktasi kendi
    # ozetini alir. Ikisi de `baseline_protocol` altinda hangi damgadan
    # tureidigini soyler ve asagida ayrica denetlenir.
    NO_PROTOCOL = {"F23_data_audit", "F24_model_audit", "F27_new_data_audit",
                   "F29_collapse_mechanism", "F30_leak_diagnosis", "F31_markov",
                   "F28_hamburg_knn_n0_5_a5", "F28_kocaeli_knn_n0_5_a5",
                   "F28_hamburg_baseline_n0_5_a5",
                   "F28_kocaeli_baseline_n0_5_a5",
                   "F20_window_stride", "F21_baseline_search"}
    stamps = {n: v.get("protocol", {}).get("hash")
              for n, v in J.items()
              if v is not None and n not in NO_PROTOCOL}
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

    # Cok protokollu deneyler: her biri KENDI damgasini uretir ama hepsinin
    # TABANI ana protokol olmali; degilse tarama baska bir zeminden baslamis
    # demektir ve sonuclari ana tablolarla karsilastirilamaz.
    base = {n: (J[n] or {}).get("baseline_protocol", {}).get("hash")
            for n in ("F20_window_stride", "F21_baseline_search")
            if J.get(n) is not None}
    # NOT: yukaridaki basari dalinda `uniq.pop()` kumeyi bosaltiyor,
    # o yuzden damgayi `have`den okuyoruz.
    main_hash = sorted(set(have.values()))[0] if have else None
    for n, h in base.items():
        if h != main_hash:
            msg = f"{n} tabani {h}, ana protokol {main_hash}"
            failures.append(msg)
            print(f"{FAIL} {msg}")
    if base:
        print(f"{OK} cok protokollu deneyler ana damgadan turuyor: "
              f"{', '.join(sorted(base))}")

    # seed listeleri
    seeds = {n: tuple(v.get("config", {}).get("seeds", []))
             for n, v in J.items()
             if v is not None and "config" in v and n not in NO_PROTOCOL}
    seeds = {n: s for n, s in seeds.items() if s}
    su = set(seeds.values())
    if len(su) <= 1:
        print(f"{OK} seed listeleri uyumlu: {sorted(su)[0] if su else '—'}")
    else:
        # Capraz dogrulama deneylerinde seed sayisi BILEREK farklidir: orada
        # gozlem birimi fold x seed'dir, sabit bolmede ise yalnizca seed.
        # Ayrimin kendisi bir tasarim karari, tutarsizlik degil — ama CV
        # dosyalari kendi aralarinda, sabit-bolme dosyalari da kendi
        # aralarinda ayni seed listesini kullanmak zorunda.
        # Bes tohumlu surumler ayri bir tohum setinde (42-46); kendi
        # aralarinda tutarli olmalari yeterli, iki tohumlu arsivle
        # ayni olmalari gerekmiyor.
        CV_FILES = {"F5_cv", "F10_repeated_cv", "F10_pilot", "F11_classic_cv",
                    "F12_classic_vs_deep", "F13_prevalence", "F14_graph_controls",
                    "F18_inductive", "F19_noise", "F20_window_stride",
                    "F22_imputation", "F25_pooled_seed", "F26_repeated_analysis"}
        non_cv = {n: s for n, s in seeds.items() if n not in CV_FILES}
        cv = {n: s for n, s in seeds.items() if n in CV_FILES}
        if len(set(non_cv.values())) <= 1 and len(set(cv.values())) <= 1:
            print(f"{OK} sabit-bolme seed listeleri uyumlu "
                  f"{sorted(set(non_cv.values()))[0]}; capraz dogrulama "
                  f"dosyalari kendi aralarinda uyumlu "
                  f"{sorted(set(cv.values()))[0] if cv else '—'}")
        else:
            msg = f"seed listeleri uyusmuyor: {seeds}"
            failures.append(msg)
            print(f"{FAIL} {msg}")

        # Hakem 3.7: cokmenin gorudugu tohum CV'ye DAHIL olmak zorunda.
        # Makale bu iddiayi yaziyor, dolayisiyla kontrol edilebilir olmali.
        prim = J.get("F5_cv_5seed")
        if prim:
            got = sorted(prim.get("config", {}).get("seeds", []))
            want = [42, 43, 44, 45, 46]
            if got == want:
                print(f"{OK} birincil CV bes tohumda ve 45 dahil "
                      f"({', '.join(map(str, got))})")
            else:
                msg = (f"birincil CV tohum listesi {got}, beklenen {want} "
                       f"(hakem 3.7 seed 45'i CV'de istiyor)")
                failures.append(msg)
                print(f"{FAIL} {msg}")
        else:
            msg = "F5_cv_5seed yok — tab:cv'nin kaynagi bulunamadi"
            failures.append(msg)
            print(f"{FAIL} {msg}")
    print()


# ============================================ B2. CIKTI <-> KOSU DEPOSU
def check_store_consistency():
    """Deney JSON'lari kosu deposuyla ayni sayiyi mi soyluyor.

    Ayni deneyi es zamanli iki kez baslatmak, iki surecin ayni anahtarlara
    yazmasina ve her birinin kendi bellek degerlerini tutmasina yol acar.
    Sonuc: tablo bir sayi, arsiv baska bir sayi gosterir. Bu kontrol onu
    yakalar.
    """
    print("B2. CIKTI <-> KOSU DEPOSU (tablo sayilari arsivle ayni mi)")
    import glob as _glob
    KEY = {("lstm", "gat", "weighted_sum"): "hybrid_gat",
           ("lstm", "gat", "gated"): "stanet_gated",
           ("lstm", "gat", "concat"): "hybrid_concat",
           ("lstm", "gat", "cross_attention"): "hybrid_crossatt",
           ("lstm", "none", "weighted_sum"): "lstm_only"}
    store = {}
    for f in _glob.glob(str(res / "_runs" / "0902cd97ec17" / "*cv5*.json")):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        sp = d.get("spec", {})
        if sp.get("graph") != "trip_instance" or sp.get("ctrl"):
            continue
        if sp.get("rep") is not None:
            continue
        name = KEY.get((sp.get("temporal"), sp.get("spatial"), sp.get("fusion")))
        if not name:
            continue
        store[(name, sp.get("fold"), d.get("seed"))] = d["metrics"]["f1"]

    bad, checked = [], 0
    src = J.get("F5_cv_5seed")
    if not src:
        print(f"{FAIL} F5_cv_5seed yok")
        failures.append("F5_cv_5seed yok (B2)")
        print()
        return
    for m, v in src.get("models", {}).items():
        for r in v.get("runs", []):
            k = (m, r.get("fold"), r.get("seed"))
            if k not in store:
                continue
            checked += 1
            if abs(store[k] - r["f1"]) > 1e-9:
                bad.append((k, r["f1"], store[k]))
    print(f"    karsilastirilan kosu : {checked}")
    if bad:
        msg = (f"{len(bad)} kosu deposuyla uyusmuyor — ayni deney es zamanli "
               f"iki kez mi kostu?")
        failures.append(msg)
        print(f"{FAIL} {msg}")
        for k, a, b in bad[:5]:
            print(f"      {k[0]} fold{k[1]} s{k[2]}: tablo {a:.6f} / depo {b:.6f}")
    else:
        print(f"{OK}   her kosu kaydi kosu deposuyla birebir")
    print()


# ========================================== F. ARSIV SURUMU <-> MAKALE
def check_archive_version():
    """CITATION.cff surumu makalede anilan snapshot ile ayni mi."""
    print("F. ARSIV SURUMU (CITATION.cff <-> makale)")
    cff = res.parent / "CITATION.cff"
    if not cff.exists():
        print(f"{FAIL} CITATION.cff yok")
        failures.append("CITATION.cff yok")
        print()
        return
    txt = cff.read_text(encoding="utf-8")
    m = re.search(r"^version:\s*(\S+)", txt, re.M)
    cff_ver = m.group(1) if m else None
    m2 = re.search(r"\\texttt\{v([\d.]+)\}", tex)
    tex_ver = m2.group(1) if m2 else None
    print(f"    CITATION.cff : v{cff_ver}")
    print(f"    makale       : v{tex_ver}")
    if cff_ver == tex_ver:
        print(f"{OK}   surumler ayni")
    else:
        # Ayrisma MESRU, ama sebebi yazili olmali.
        note = "version DOI is minted when that release is cut" in txt
        if note:
            print(f"{OK}   ayrisik ama CITATION.cff sebebini yaziyor: "
                  f"v{cff_ver} release'i henuz kesilmedi")
            print(f"         >> GONDERIM ONCESI: release'i kes, DOI'yi bas, "
                  f"makalede v{tex_ver} -> v{cff_ver}")
        else:
            msg = (f"surum ayrisik ve sebebi yazili degil: "
                   f"CITATION.cff v{cff_ver}, makale v{tex_ver}")
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

    # --- veri karti: ham CSV'den uretilen sayilar (F23). Bunlar daha once
    # elle yazilmisti ve kaynaksizdi; artik denetleniyorlar.
    da = J.get("F23_data_audit")
    if da:
        r = da["records"]
        for label, v in (("veri: uretilen kayit", r["generated"]),
                         ("veri: kalan kayit", r["retained"]),
                         ("veri: silinen kayit", r["removed_union"]),
                         ("veri: hiz disi", r["speed_out_of_range"]),
                         ("veri: egim disi", r["gradient_out_of_range"])):
            txt = f"{v:,}".replace(",", "{,}")
            checks.append((label, txt, txt in tex))
        for t, d in da["subtypes"].items():
            for stage in ("generated", "cleaned"):
                txt = f"{d[stage]:,}".replace(",", "{,}")
                checks.append((f"veri: tip {t} {stage}", txt, txt in tex))
        rs = da["rule_sufficiency"]["subtype1"]
        for label, v in (("veri: tip1 kurali saglayan", rs["records_satisfying_rule"]),
                         ("veri: tip1 etiketli", rs["labelled_subtype1"]),
                         ("veri: tip1 etiketsiz", rs["labelled_none"])):
            txt = f"{v:,}".replace(",", "{,}")
            checks.append((label, txt, txt in tex))

    # --- mimari: modeller kurulup olculdu (F24). Denklem 15 / Tablo 5
    # uyusmazligi bu sinifti; artik tablo sayilari koddan denetleniyor.
    ma = J.get("F24_model_audit")
    if ma:
        gr = ma["graph"]
        for label, v in (("mimari: dugum", gr["num_nodes"]),
                         ("mimari: kenar", gr["num_edges"]),
                         ("mimari: tekil yol", gr["num_unique_osm"])):
            txt = f"{v:,}".replace(",", "{,}")
            checks.append((label, txt, txt in tex))
        ref = ma["models"]["gated"]
        for label, v in (("mimari: zamansal parametre", ref["temporal_params"]),
                         ("mimari: uzamsal parametre", ref["spatial_params"])):
            txt = f"{v:,}".replace(",", "{,}")
            checks.append((label, txt, txt in tex))
        for f, d in ma["models"].items():
            fp = d["fusion_params"]
            if fp is None or fp < 2:      # 0 ve 1 metinde her yerde gecer
                continue
            txt = f"{fp:,}".replace(",", "{,}")
            checks.append((f"mimari: fuzyon {f}", txt, txt in tex))
        if not ma.get("gate_is_two_layer") or not ma.get("gate_has_tanh"):
            failures.append("kapi kodda iki katmanli/tanh degil — Denklem 15 gecersiz")

    # tab:cv artik BES tohumlu kosudan besleniyor (hakem 3.7). Iki tohumlu
    # surum arsiv; sayilari metinde aranmaz.
    f1b, f3, f4, f6, f2 = (J[n] for n in
                           ("F1b_fair", "F3_fusion", "F4_benchmark",
                            "F6_efficiency", "F2_graph"))
    f5 = J.get("F5_cv_5seed") or J["F5_cv"]

    # F32: mimari yelpaze <-> replikasyon hatasi. Bu dort sayi makalenin
    # merkez argumanini tasiyor ve elle hesaplanmis halleri bir sonuc
    # dosyasinda GECMIYORDU; artik etiketli olarak denetleniyor.
    f32 = J.get("F32_arch_vs_replication")
    if f32:
        chk("F32 fuzyon yelpazesi (tum)", f32["fusion_span_all_runs"])
        chk("F32 fuzyon yelpazesi (dejenere haric)",
            f32["fusion_span_excluding_degenerate"])
        chk("F32 replikasyon hatasi", f32["replication_error_max"])
        cw, nw = f32["collapse"]["weight_learning"]
        cf, nf = f32["collapse"]["fixed_weight"]
        for lbl, txt in ((f"F32 cokme ogrenen {cw}/{nw}", f"{cw}/{nw}"),
                         (f"F32 cokme sabit {cf}/{nf}", f"{cf}/{nf}")):
            checks.append((lbl, txt, txt in tex))
        if not f32["ranking_changes_when_degenerate_excluded"]:
            failures.append("F32: siralama dejenere kosular cikarilinca "
                            "degismiyor — makale degistigini soyluyor")

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
        # A2: etiket safligi tavani artik YALNIZCA egitim verisinden.
        # Metindeki yuzde degerleri bu JSON alaniyla eslesmeli; eski
        # (test etiketli) degerlerin metne geri sizmasini bu kontrol engeller.
        for tag, v in f2["graphs"].items():
            amb = v.get("ambiguity", {})
            if "theoretical_max_accuracy" in amb:
                chk(f"F2 {tag} etiket tavani %",
                    amb["theoretical_max_accuracy"] * 100, 2)

    if J["F7_cluster_stats"]:
        ref = J["F7_cluster_stats"]["families"]["baselines"]["reference_ci"]
        chk("F7 referans trip GA alt", ref["lo"])
        chk("F7 referans trip GA ust", ref["hi"])

    # F8: alt-tip tablosu ve etki boyutlari (A3, A5)
    f8 = J.get("F8_effects_subtypes")
    if f8:
        for name, v in f8["subtypes"].items():
            en = v["name_en"]
            chk(f"F8 {en} hibrit P", v["hybrid"]["precision"])
            chk(f"F8 {en} hibrit R", v["hybrid"]["recall"])
            chk(f"F8 {en} hibrit F1", v["hybrid"]["f1"])
            chk(f"F8 {en} lstm F1", v["lstm_only"]["f1"])
            chk(f"F8 {en} delta", v["delta_f1"])
            if not v["sparse"]:
                cb = v["cluster_bootstrap"]
                chk(f"F8 {en} GA alt", cb["lo"])
                chk(f"F8 {en} GA ust", cb["hi"])
        for pair, v in f8["effects"].items():
            e = v["effect_size"]
            chk(f"F8 {pair} ort fark", e["mean_delta"])
            chk(f"F8 {pair} sd", e["sd_delta"])
            chk(f"F8 {pair} Cohen d", e["cohens_d"], 2)

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
    "fig_F17_collapse.png": ["F17_gate_collapse", "F29_collapse_mechanism"],
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


# =================================================== D2. FIGUR OKUNABILIRLIGI
# Bir figur kagida girerken kuculurse icindeki yazi da kuculur. Tazelik
# kontrolu (D) bunu gormez: figur guncel olabilir ama okunamayacak kadar
# kucuk basilabilir. Bu kontrol, her figurun URETILDIGI genislik ile
# BASILACAGI genisligi karsilastirir.
#
# IEEEtran journal: sutun 252 pt, iki sutun 516 pt (olculdu). Olcek 1'e
# yakinsa kaynaktaki punto kagitta aynen gorunur; 0,9'un altina dusen her
# figurde yazi orantili olarak kuculur.
COL_PT, TEXT_PT, PT_PER_IN = 252.0, 516.0, 72.27
MIN_SCALE = 0.90
SMALLEST_SOURCE_PT = 6.5      # f9 betiklerindeki en kucuk punto
MIN_PRINTED_PT = 6.0          # bunun altinda basili yazi okunmaz


def check_figure_scale():
    print("D2. FIGUR OKUNABILIRLIGI (basili olcek ve punto)")
    try:
        from PIL import Image
    except ImportError:
        print("    Pillow yok, atlandi\n")
        return
    pat = re.compile(
        r"\\begin\{(figure\*?)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}", re.S)
    inc = re.compile(
        r"\\includegraphics\[width=([0-9.]+)\\(linewidth|textwidth)\]\{([^}]+)\}")
    bad, checked = [], 0
    for m in pat.finditer(tex):
        env, body = m.group(1), m.group(2)
        for im in inc.finditer(body):
            frac, unit, fn = float(im.group(1)), im.group(2), im.group(3)
            fp = pathlib.Path(TEX).parent / fn
            if not fp.exists():
                continue
            base = COL_PT if env == "figure" else TEXT_PT
            printed_in = frac * base / PT_PER_IN
            img = Image.open(fp)
            dpi = float(img.info.get("dpi", (200, 200))[0]) or 200.0
            scale = printed_in / (img.size[0] / dpi)
            checked += 1
            if scale < MIN_SCALE or SMALLEST_SOURCE_PT * scale < MIN_PRINTED_PT:
                bad.append((fn, env, scale, SMALLEST_SOURCE_PT * scale))
    print(f"    kontrol edilen figur: {checked}")
    if bad:
        print(f"{FAIL} basili olcekte kuculen figurler:")
        for fn, env, sc, pt in bad:
            print(f"      {fn:28s} {env:8s} olcek {sc:.3f} -> "
                  f"en kucuk yazi {pt:.1f} pt")
        failures.append(f"{len(bad)} figur basili olcekte okunmuyor — "
                        f"son genislikte uretin ya da figure* yapin")
    else:
        print(f"{OK} tum figurler basilacaklari genislikte uretilmis")
    print()

# ======================================================== E2. ESKIMIS IFADE
# Sayi kontrolu (C) bir degerin metinde BULUNDUGUNU dogrular; metinde kalmis
# ESKI bir ifadeyi yakalamaz. Tablo 11'in basligi "three seeds" derken tablo
# bes tohumluydu ve C kontrolu bunu gormedi. Asagidakiler o bosluğu kapatir.
STALE_PHRASES = [
    # E-3: cokmenin "kayboldugu" iddiasi. tab:collapse: birlesik grafta
    # 3/50, trip-instance 1/50 -- yani kaybolmuyor, DAHA SIK. Bu cumle
    # bir kez farkli kelimelerle geri geldigi icin uc varyasyon birden.
    (r"disappears when that branch is made more informative",
     "cokme birlesik grafta DAHA SIK (3/50 vs 1/50)"),
    (r"disappears when the spatial branch is made",
     "cokme kaybolmuyor -- tab:collapse"),
    (r"localizes the collapse",
     "cokme bir grafa lokalize DEGIL; operator sinifina ait"),
    # E-3: iki tohumluk fuzyon yelpazesi
    (r"configurations span \$0\.0205\$ F1",
     "yelpaze 5 tohumda 0.0582 / cokmeler haric 0.0095"),
    (r"baselines that match the hybrid model",
     "klasik model ANLAMLI kazaniyor: +0.0409, Holm p=0.0002"),
    (r"crossed with two seeds",
     "birincil CV bes tohum (hakem 3.7)"),
    (r"Encoder benchmark, three seeds", "Tablo 11 basligi: benchmark 5 tohum"),
    (r"three seeds for the encoder", "kodlayici benchmark 5 tohum"),
    (r"five for the fusion ablation, three", "tohum sayilari artik esit"),
    (r"Cross-experiment inconsistencies", "v1 ozet cumlesi"),
    (r"run-level reconciliation", "v1 ifadesi"),
    # A2: test etiketleriyle hesaplanmis eski tavanlar
    (r"95\.13", "eski etiket tavani (test etiketli) — yeni: 94.58"),
    (r"93\.45", "eski etiket tavani (test etiketli) — yeni: 92.84"),
    (r"the primary setting of the accuracy tables", "graf artik secilmiyor"),
    # FAZ C/D: yapilan isler icin "yapilmadi" diyen cumleler
    (r"sensitivity analysis over these defaults was not performed",
     "imputasyon duyarliligi YAPILDI (tab:imputation)"),
    (r"over one city network", "artik iki ag var (sec:independent_network)"),
    (r"It occurred in none of ten cross-validated runs",
     "500 kosuda 8 cokme var (tab:collapse)"),
    (r"disappears on a graph construction where", "graf gerekcesi gecersiz"),
]


def check_stale_phrases():
    print("E2. ESKIMIS IFADE")
    hits = []
    for pat, desc in STALE_PHRASES:
        for m in re.finditer(pat, tex, re.IGNORECASE):
            line = tex[:m.start()].count("\n") + 1
            hits.append((line, desc, m.group(0)))
    if hits:
        print(f"{FAIL} {len(hits)} eskimis ifade:")
        for line, desc, txt in hits:
            print(f"      satir {line:4d}  {desc}: \"{txt}\"")
        failures.append(f"{len(hits)} eskimis ifade .tex icinde")
    else:
        print(f"{OK} eskimis ifade yok")
    print()


# ====================================================== E5. ASIRI IDDIA DILI
# Makalenin merkezi bulgusu "bu tasarim bu buyuklukteki farklari cozemiyor".
# Ustunluk ya da kesinlik iddia eden bir kelime metne geri sizdiginde kendi
# bulgumuzla celisiriz. Bunlari bir kez temizledik; kontrol geri gelmelerini
# engelliyor.
#
# Bir isaret su durumlarda GECERLIDIR:
#   * yakininda cozulebilir bir sonuca atif varsa (Holm, p =, interval, CI)
#   * baskasinin iddiasini aktardigi belliyse (reported, earlier version,
#     previous version, claimed)
#   * `% IDDIA-OK` yorumu varsa
OVERCLAIM = [
    (r"\boutperform(s|ed|ing)?\b", "ustunluk"),
    (r"\b(significantly|markedly|substantially)\s+(better|worse|superior)", "asiri nitelik"),
    (r"\b(clearly|obviously|evidently|undoubtedly)\b", "kesinlik zarfi"),
    (r"\b(proves|proven)\b", "ispat"),
    (r"\b(superior|inferior)\s+to\b", "ustunluk"),
    (r"\bstate[- ]of[- ]the[- ]art\b", "SOTA"),
    (r"\bconfirms?\s+(that|the)\b", "dogrulama"),
]
OVERCLAIM_OK = [
    r"Holm", r"p\s*=", r"p\s*<", r"interval", r"CI\b", r"\\ref\{sec:equivalence",
    r"reported", r"earlier version", r"previous version", r"claimed",
    r"% IDDIA-OK", r"would", r"cannot", r"does not", r"not resolvable",
]


def check_overclaim():
    print("E5. ASIRI IDDIA DILI (ustunluk iddialari cozulebilir mi)")
    lines = tex.split("\n")
    flagged, supported = [], 0
    for i, line in enumerate(lines):
        if line.strip().startswith("%"):
            continue
        for pat, lab in OVERCLAIM:
            m = re.search(pat, line, re.IGNORECASE)
            if not m:
                continue
            window = "\n".join(lines[max(0, i - 1):i + 3])
            if any(re.search(ok, window, re.IGNORECASE) for ok in OVERCLAIM_OK):
                supported += 1
            else:
                flagged.append((i + 1, lab, m.group(0), line.strip()[:95]))
            break
    print(f"    gerekceli ustunluk ifadesi : {supported}")
    if flagged:
        print(f"{FAIL} {len(flagged)} gerekcesiz asiri iddia:")
        for ln, lab, hit, ctx in flagged:
            print(f"      satir {ln:4d} [{lab}] \"{hit}\"  {ctx}")
        failures.append(f"{len(flagged)} asiri iddia cozulebilir bir sonuca baglanmamis")
    else:
        print(f"{OK} ustunluk iddialarinin hepsi gerekceli")
    print()

# ==================================================== E4. ESDEGERLIK DILI
# Anlamsiz bir p degerinden esdegerlik cikarmak, bu makalenin duzeltmek
# zorunda kaldigi hatalardan biriydi ve metne geri sizmasi kolaydir: bir
# karsilastirmayi "fark yok" diye kisaltmak dogal gelir. Asagidaki kaliplar
# yoklugu IDDIA eden ifadelerdir; her biri ya esdegerlik bolumune atif
# yapmali, ya bir TOST sonucunun yaninda gecmeli, ya da acik yorumla
# gerekcelendirilmeli.
EQUIVALENCE_CLAIMS = [
    r"statistically indistinguishable",
    r"\bindistinguishable\b",
    r"no measurable (benefit|difference|effect|gain)",
    r"(shows|show|showed) no (benefit|difference|effect)",
    r"performs? the same",
    r"identical performance",
    r"there is no difference",
    r"are equivalent",
    r"is equivalent to (the|a|another) (model|operator|encoder|variant)",
]
# Bir iddiayi gecerli kilan isaretler (ayni satirda ya da komsu satirda)
EQUIVALENCE_SUPPORT = [
    r"sec:equivalence", r"TOST", r"two one-sided",
    r"p_\{\\mathrm\{TOST\}\}", r"% ESDEGERLIK-OK",
    # Esdegerlik sinirini ADLANDIRAN bir cumle, tanim geregi teste baglidir:
    # delta olmadan esdegerlik ifadesi kurulamaz.
    r"\\delta", r"equivalence bound",
]


def check_equivalence_language():
    print("E4. ESDEGERLIK DILI (yokluk iddialari gerekceli mi)")
    lines = tex.split("\n")
    # Esdegerlik bolumunun KENDISI bu ifadeleri tanimlamak icin kullaniyor
    # ("...not a finding that two methods perform the same"). Orada gecen bir
    # kalip iddia degil tanimdir, bu yuzden once her satirin hangi alt bolumde
    # oldugunu buluyoruz ve o bolumu muaf tutuyoruz.
    in_equiv, cur = [False] * len(lines), False
    for i, line in enumerate(lines):
        if re.search(r"\\(sub)*section\{", line):
            cur = False
        if "label{sec:equivalence}" in line:
            cur = True
        in_equiv[i] = cur

    flagged, supported = [], 0
    for i, line in enumerate(lines):
        for pat in EQUIVALENCE_CLAIMS:
            m = re.search(pat, line, re.IGNORECASE)
            if not m:
                continue
            if in_equiv[i]:
                supported += 1
                break
            # iddianin gecerliligini ayni ve komsu iki satirda ariyoruz:
            # gerekce genellikle ayni cumlede ya da hemen ardindan gelir
            window = "\n".join(lines[max(0, i - 1):i + 3])
            if any(re.search(sp, window, re.IGNORECASE) for sp in EQUIVALENCE_SUPPORT):
                supported += 1
            else:
                flagged.append((i + 1, m.group(0), line.strip()[:110]))
            break
    print(f"    gerekceli yokluk iddiasi : {supported}")
    if flagged:
        print(f"{FAIL} {len(flagged)} gerekcesiz esdegerlik/yokluk iddiasi:")
        for ln, hit, ctx in flagged:
            print(f"      satir {ln:4d} [{hit}]  {ctx}")
        failures.append(f"{len(flagged)} esdegerlik iddiasi TOST'a baglanmamis")
    else:
        print(f"{OK} yokluk iddialarinin hepsi bir esdegerlik testine bagli")
    print()

# ======================================================== E3. IDDIA DENETIMI
# C kontrolu bir SAYININ metinde bulundugunu dogrular. Metinde kalmis bir
# IDDIAYI gormez. Tablo 11'in basligi "three seeds" derken tablo bes tohumluydu
# ve C bunu kacirdi (E2 o tip ifadeler icin). Bu kontrol bir adim oteye gider:
# metindeki her p degeri bir sonuc dosyasinda gercekten var mi?
#
# Amac ikinci tur revizyonu engellemek: yeni sayilar geldiginde metinde kalan
# eski p degerleri ve anlamlilik iddialari burada yakalanir.

def _collect_pvalues():
    """Sonuc dosyalarindaki tum p degerlerini topla."""
    found = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if (k in ("p_value", "p_holm", "p_raw", "p_corrected", "p_tost",
                          "p_lower", "p_upper", "p_naive_INVALID")
                        and isinstance(v, (int, float))):
                    found.add(round(float(v), 4))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for name, j in J.items():
        if j is not None:
            walk(j)
    return found


def check_claims():
    """
    Metindeki her p degeri bir sonuc dosyasindan geliyor mu?

    Iki istisna var:
      * `p = 0.2` gibi dropout oranlari p degeri degildir — satirinda
        "Dropout" gecen eslesmeler atlanir.
      * Geri cekilmis eski iddialar bilerek anilir. Bunlar .tex icinde
        `% ESKI-DEGER` yorumuyla isaretlenir; isaretli satirlar atlanir.
    """
    print("E3. IDDIA DENETIMI (metindeki p degerleri sonuc dosyalarinda var mi)")
    known = _collect_pvalues()
    if not known:
        print("    (sonuc dosyalarinda p degeri bulunamadi, kontrol atlandi)\n")
        return

    lines = tex.split("\n")
    cited, skipped = [], 0
    for m in re.finditer(r"p\s*(=|<)\s*([01]\.[0-9]+)", tex):
        idx = tex[:m.start()].count("\n")
        line_txt = lines[idx]
        if "Dropout" in line_txt or "dropout" in line_txt:
            skipped += 1
            continue
        if "ESKI-DEGER" in line_txt:
            skipped += 1
            continue
        cited.append((idx + 1, m.group(1), m.group(2), m.group(0)))

    orphan = []
    for line, op, txt, raw in cited:
        val = float(txt)
        if op == "<":
            # "p < 0.001" bir ust sinirdir: sonuc dosyalarinda bu sinirin
            # ALTINDA bir deger bulunmasi yeterli, birebir eslesme aranmaz.
            ok = any(k < val for k in known)
        else:
            dec = len(txt.split(".")[1])      # metinde kac ondalik yazilmis
            ok = any(round(k, dec) == round(val, dec) for k in known)
        if not ok:
            orphan.append((line, txt, raw))

    print(f"    metindeki p degeri : {len(cited)}  (atlanan {skipped})")
    print(f"    sonuc dosyalarinda : {len(cited) - len(orphan)}")
    if orphan:
        print(f"{FAIL} {len(orphan)} p degeri hicbir sonuc dosyasinda yok:")
        for line, txt, raw in orphan:
            ctx = lines[line - 1].strip()[:88]
            print(f"      satir {line:4d}  {raw:12s}  {ctx}")
        failures.append(f"{len(orphan)} p degeri sonuc dosyalariyla eslesmiyor")
    else:
        print(f"{OK} her p degeri bir sonuc dosyasindan geliyor")
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
    check_store_consistency()
    check_archive_version()
    check_numbers()
    check_figures()
    check_figure_scale()
    check_stale_phrases()
    check_claims()
    check_equivalence_language()
    check_overclaim()
    check_placeholders()
    print("=" * 78)
    if failures:
        print(f"SONUC: {len(failures)} KONTROL BASARISIZ")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SONUC: TUM KONTROLLER GECTI")
    sys.exit(0)
