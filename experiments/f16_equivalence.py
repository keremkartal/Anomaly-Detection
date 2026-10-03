"""
F16 — "Fark yok" diyen her iddia icin ESDEGERLIK testi (degerlendirme: esdegerlik iddiasi).

Isaretlenen sorun: anlamsizlik ne faydanin yokluğunu ne de esdegerligi
kurar; esdegerlik iddiasi kendi testini gerektirir.

Makalenin ana bulgusu "hicbir fuzyon operatoru digerlerinden ayirt
edilemiyor" ve "zamansal kodlayicilar birbirinden ayirt edilemiyor". Bunlar
anlamsiz p degerlerine dayaniyordu; anlamsizlik "fark yok" demek degildir,
"bu veriyle karar veremiyoruz" demektir. Aradaki farki ancak TOST gosterir.

Her karsilastirma UC sonuctan birini alir:

  FARKLI     fark siniri asiyor ve sifirdan ayrilabiliyor
  ESDEGER    %90 guven araligi tamamen (-delta, +delta) icinde
  KARARSIZ   ne biri ne digeri — veri yetersiz. Bunu boyle yazacagiz.

IKI AILE, IKI BELIRSIZLIK MODELI
--------------------------------
  (a) CAPRAZ DOGRULAMA (F5): egitim kumeleri fold'lar arasinda ORTUSUR.
      Nadeau-Bengio duzeltmesi uygulanir. Cikarim yeni bolumlemelere
      genellenir.
  (b) SABIT BOLME (F3, F4): her tohum AYNI egitim kumesini gorur. Ortusme
      duzeltmesi burada anlamsizdir cunku ortusme %100'dur; belirsizlik
      YALNIZCA tohum degiskenligidir ve cikarim yalnizca "ayni bolmede
      baska bir tohum" evrenine genellenir. Bunu cikti dosyasina da
      yaziyoruz ki makalede yanlis genelleme yapilmasin.

KONTROL: uzamsal operatorler (F4 S:*) ayirt edilebilir olmali. Onlarda TOST
esdegerlik ILAN ETMEMELI. Bu, testin kor olmadiginin kanitidir.

KULLANIM
  python experiments/f16_equivalence.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.cluster_stats import holm_correction, nadeau_bengio, tost_equivalence
from stanet.stats import exact_two_sample
from stanet.utils import init_console, load_json, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F16_equivalence.json"

DELTA = 0.02
COLLAPSE = 0.30          # F1 bunun altinda ise kosu cokmus sayilir
DELTA_NOTE = ("0.02 F1. Gerekce: capraz dogrulamada kosular arasi standart "
              "sapma 0.031-0.054 arasindadir ve ayni kosunun GPU tekrarinda "
              "olculen ortalama mutlak sapma 0.0195'tir (F15). Bu iki sayinin "
              "altinda kalan bir mimari farki 'gercek' saymak olcum "
              "gurultusunu yorumlamak olur. Sinir karsilastirmalar "
              "yapilmadan once sabitlendi.")


def verdict(nb_or_t, tost, holm=None, holm_exact=None, collapsed=False):
    """Bir karsilastirmanin sonucu.

    Iki testten HERHANGI BIRI Holm duzeltmesinden sonra anlamliysa "FARKLI"
    denir; boylece eslestirme varsayimina bagimli kalmayiz. Taraflardan biri
    cokme yasiyorsa ortalama fark anlamli bir ozet olmadigi icin ayrica
    isaretlenir — bu durumda bulgu farkin BUYUKLUGU degil COKMENIN KENDISIDIR.
    """
    sig = (holm or {}).get("significant")
    if sig is None:
        sig = nb_or_t.get("significant")
    sig_ex = (holm_exact or {}).get("significant", False)
    if tost and tost["equivalent"]:
        return "ESDEGER"
    if sig or sig_ex:
        return "FARKLI (cokme)" if collapsed else "FARKLI"
    return "KARARSIZ (cokme)" if collapsed else "KARARSIZ"


def paired_t(diffs, alpha=0.05):
    """Sabit bolme: ortusme duzeltmesi yok, belirsizlik yalnizca tohum."""
    from scipy import stats
    d = np.asarray(diffs, float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(d.mean() / se) if se > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), n - 1))
    crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    return {"n": n, "mean_diff": float(d.mean()), "se": se, "t": t, "p_value": p,
            "ci_lo": float(d.mean() - crit * se), "ci_hi": float(d.mean() + crit * se),
            "significant": bool(p < alpha),
            "inference_scope": "ayni bolmede baska tohumlar; yeni veriye genellemez"}


def family_cv(reference=None):
    """F5: fold x tohum eslestirmesi, Nadeau-Bengio + TOST."""
    f5 = load_json(C.RESULTS_DIR / "F5_cv.json")
    tab = {}
    for name, m in f5["models"].items():
        tab[name] = {(r["fold"], r["seed"]): float(r["f1"]) for r in m["runs"]}
    fi = f5["fold_info"]
    n_tr = int(np.mean([x["trips"]["train"] for x in fi]))
    n_te = int(np.mean([x["trips"]["test"] for x in fi]))

    rows, pv = {}, {}
    for a, b in _pairs(tab, reference):
        keys = sorted(set(tab[a]) & set(tab[b]))
        diffs = [tab[a][k] - tab[b][k] for k in keys]
        nb = nadeau_bengio(diffs, n_train=n_tr, n_test=n_te)
        ts = tost_equivalence(diffs, DELTA, n_train=n_tr, n_test=n_te)
        key = a + "_vs_" + b
        rows[key] = {"n_paired": len(keys), "test": nb, "tost": ts}
        pv[key] = nb["p_corrected"]
    for k, v in holm_correction(pv).items():
        rows[k]["holm"] = v
    for k, v in rows.items():
        v["verdict"] = verdict(v["test"], v["tost"], v.get("holm"))
    return {"source": "F5_cv.json", "uncertainty": "nadeau_bengio",
            "reference": reference, "family_size": len(pv),
            "n_train_trips": n_tr, "n_test_trips": n_te, "comparisons": rows}


def _pairs(names, reference):
    """Karsilastirma AILESI.

    `reference` verilirse aile "referans vs her alternatif" olur; makalenin
    onceden belirlenmis iddia yapisi budur ve aile boyu k-1'dir. Referans
    verilmezse tum ciftler alinir (kesifsel; aile boyu k(k-1)/2'ye ciktigi
    icin Holm cok daha muhafazakar olur).
    """
    if reference is None:
        return list(itertools.combinations(sorted(names), 2))
    return [(reference, b) for b in sorted(names) if b != reference]


def family_fixed(path, getter, label, subset=None, reference=None):
    """F3/F4: sabit bolme, tohum bazinda eslestirme, ortusme duzeltmesi YOK."""
    d = load_json(C.RESULTS_DIR / path)
    tab = getter(d)
    if subset:
        tab = {k: v for k, v in tab.items() if k in subset}
    rows, pv, pv_ex = {}, {}, {}
    collapse = {k: int(sum(1 for x in v if x < COLLAPSE)) for k, v in tab.items()}
    for a, b in _pairs(tab, reference):
        n = min(len(tab[a]), len(tab[b]))
        diffs = [tab[a][i] - tab[b][i] for i in range(n)]
        pt = paired_t(diffs)
        ts = tost_equivalence(diffs, DELTA)        # ortusme yok -> naif SE dogru
        # Tohum numarasi iki mimariyi ESLESTIRMEZ; ayrica cokme yasayan
        # konfigurasyonlarda dagilim iki tepeli oldugu icin ortalama farkin
        # t-testi yaniltici. Varsayimsiz tam permutasyon testi de raporlanir.
        ex = exact_two_sample(tab[a][:n], tab[b][:n])
        key = a + "_vs_" + b
        rows[key] = {"n_paired": n, "test": pt, "tost": ts, "exact_perm": ex,
                     "n_collapsed": {a: collapse[a], b: collapse[b]}}
        pv[key] = pt["p_value"]
        pv_ex[key] = ex["p_value"]
    for k, v in holm_correction(pv).items():
        rows[k]["holm"] = v
    for k, v in holm_correction(pv_ex).items():
        rows[k]["holm_exact"] = v
    for k, v in rows.items():
        v["verdict"] = verdict(v["test"], v["tost"], v.get("holm"),
                               v.get("holm_exact"),
                               collapsed=any(n > 0 for n in v["n_collapsed"].values()))
    return {"source": path, "uncertainty": "paired_t_seeds_only",
            "label": label, "reference": reference,
            "family_size": len(pv), "n_collapsed_runs": collapse,
            "comparisons": rows}


def show(title, fam):
    print("\n" + "=" * 100)
    print(title)
    print("belirsizlik: %s" % fam["uncertainty"])
    print("=" * 100)
    print("  %-36s %9s %22s %8s %9s %9s %16s"
          % ("karsilastirma", "ort fark", "%95 GA", "Holm p", "Holm tam",
             "p_TOST", "sonuc"))
    for k, v in sorted(fam["comparisons"].items(),
                       key=lambda kv: -abs(kv[1]["test"]["mean_diff"])):
        t, ts = v["test"], v["tost"]
        print("  %-36s %+9.4f [%+.4f, %+.4f] %8.4f %9s %9.4f %16s"
              % (k, t["mean_diff"], t["ci_lo"], t["ci_hi"],
                 v.get("holm", {}).get("p_holm", float("nan")),
                 ("%.4f" % v["holm_exact"]["p_holm"]) if "holm_exact" in v else "-",
                 ts["p_tost"], v["verdict"]))


def main():
    stamp = protocol_stamp()
    print("protokol %s | esdegerlik siniri delta = %.3f F1" % (stamp["hash"], DELTA))
    print(DELTA_NOTE + "\n")

    out = {"protocol": stamp, "delta": DELTA, "delta_rationale": DELTA_NOTE,
           "families": {}}

    runs_f3 = lambda d: {k: [r["f1"] for r in v["runs"]]
                         for k, v in d["variants"].items()}
    runs_f4 = lambda d: {k: [r["f1"] for r in v["runs"]]
                         for k, v in d["configs"].items()}

    # ---- BIRINCIL AILELER: onerilen yapilandirma vs her alternatif ----------
    cv = family_cv(reference="stanet_gated")
    out["families"]["cv_reference"] = cv
    show("CAPRAZ DOGRULAMA — onerilen (stanet_gated) vs alternatifler "
         "[BIRINCIL, aile 4]", cv)

    f3 = family_fixed("F3_fusion.json", runs_f3,
                      "fuzyon operatorleri, sabit bolme, 5 tohum",
                      reference="gated")
    out["families"]["fixed_fusion_reference"] = f3
    show("SABIT BOLME — onerilen kapi vs diger fuzyonlar [BIRINCIL, aile 3]", f3)

    temporal = ["T:lstm", "T:gru", "T:stgcn", "T:wavenet", "T:transformer"]
    f4t = family_fixed("F4_benchmark.json", runs_f4, "zamansal kodlayicilar",
                       subset=temporal, reference="T:lstm")
    out["families"]["fixed_temporal_reference"] = f4t
    show("SABIT BOLME — LSTM vs diger zamansal kodlayicilar [BIRINCIL, aile 4]", f4t)

    spatial = ["T:lstm", "S:cheb", "S:diffusion", "S:adaptive", "S:none"]
    f4s = family_fixed("F4_benchmark.json", runs_f4,
                       "uzamsal operatorler (KONTROL: burada esdegerlik CIKMAMALI)",
                       subset=spatial, reference="T:lstm")
    out["families"]["fixed_spatial_control"] = f4s
    show("SABIT BOLME — GAT vs diger uzamsal operatorler [KONTROL, aile 4]", f4s)

    # ---- KESIFSEL: tum ciftler. Aile buyudugu icin Holm cok daha sert. -----
    cv_all = family_cv()
    out["families"]["cv_all_pairs_exploratory"] = cv_all
    show("CAPRAZ DOGRULAMA — tum ciftler [KESIFSEL, aile 10]", cv_all)

    f3_all = family_fixed("F3_fusion.json", runs_f3,
                          "fuzyon, tum ciftler")
    out["families"]["fixed_fusion_all_pairs_exploratory"] = f3_all
    show("SABIT BOLME — dort fuzyon, tum ciftler [KESIFSEL, aile 6]", f3_all)

    # Zamansal eksende tum ciftler: makalenin "LSTM, GRU ve Transformer
    # birbirinden 0,0015 F1 icinde" ifadesi bir ESDEGERLIK imasi tasiyor ve
    # o ima referans ailesinde degil, GRU-Transformer ciftinde sinaniyor.
    f4t_all = family_fixed("F4_benchmark.json", runs_f4,
                           "zamansal kodlayicilar, tum ciftler",
                           subset=temporal)
    out["families"]["fixed_temporal_all_pairs_exploratory"] = f4t_all
    show("SABIT BOLME — zamansal kodlayicilar, tum ciftler [KESIFSEL, aile 10]",
         f4t_all)

    # ---------------------------------------------------------------- ozet
    print("\n" + "=" * 100)
    print("OZET — kac karsilastirma hangi sonuca dustu")
    print("=" * 100)
    tally = {}
    for fname, fam in out["families"].items():
        c = {}
        for v in fam["comparisons"].values():
            c[v["verdict"]] = c.get(v["verdict"], 0) + 1
        tally[fname] = c
        print("  %-26s %s" % (fname, ", ".join("%s %d" % (k, v)
                                               for k, v in sorted(c.items()))))
    out["tally"] = tally

    ctrl = tally.get("fixed_spatial_control", {})
    n_diff = sum(v for k, v in ctrl.items() if k.startswith("FARKLI"))
    fam = out["families"]["fixed_spatial_control"]["comparisons"]
    survivors = [k for k, v in fam.items() if v["verdict"].startswith("FARKLI")]
    print("\n  KONTROL AILESI (GAT vs dort alternatif)")
    print("  Holm duzeltmesinden sonra ayirt edilebilen: %d/4 -> %s"
          % (n_diff, ", ".join(survivors) if survivors else "yok"))
    if n_diff:
        print("  Test kor degil: en az bir gercek fark yakalaniyor.")
    print("  ANCAK: geri kalan uc karsilastirmada fark YOK demiyoruz, "
          "KARAR VEREMIYORUZ diyoruz.")
    print("  Bes tohumla sabit bolmede tam permutasyon testinin ulasabilecegi "
          "en kucuk p degeri 0.0079'dur;")
    print("  dort karsilastirmalik bir ailede Holm bunu 0.0317'ye tasir. "
          "Yani sabit bolme tasarimi")
    print("  ancak TAM AYRISMA gosteren farklari ayirt edebilir. Bu, "
          "cikarimi capraz dogrulamaya")
    print("  tasima gerekcesidir (F10).")
    out["control_passed"] = bool(n_diff)
    out["power_note"] = (
        "Sabit bolme + 5 tohum: tam permutasyon testinin en kucuk iki yonlu p "
        "degeri 2/252 = 0.0079. Dort karsilastirmalik ailede Holm sonrasi en "
        "kucuk erisilebilir deger 0.0317. Dolayisiyla bu tasarim yalnizca iki "
        "kumenin TAMAMEN ayristigi farklari anlamli bulabilir; daha kucuk ama "
        "gercek farklar icin 'karar veremiyoruz' dogru ifadedir.")

    save_json(out, OUT)
    print("\nkaydedildi: %s" % OUT)


if __name__ == "__main__":
    main()
