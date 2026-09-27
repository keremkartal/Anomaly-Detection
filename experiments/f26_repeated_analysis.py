"""
F26 — Tekrarli capraz dogrulamanin tam analizi (500 kosu).

F10 kosulari uretti ve eslestirilmis karsilastirmalari verdi. Burada geri
kalan dort soru cevaplaniyor:

  1. VARYANS BILESENLERI. F1'deki degiskenligin ne kadari bolumlemeden, ne
     kadari fold'dan, ne kadari tohumdan geliyor? Bu, "kac tekrar, kac tohum
     gerekir" sorusunun ve makaledeki belirsizlik iddiasinin dayanagi.

  2. ESDEGERLIK. Anlamsiz her karsilastirma icin TOST: fark onceden
     belirlenmis sinirin icinde mi, yoksa karar veremiyor muyuz?

  3. COKME. Hangi operatorler cokuyor, hangi grafta, hangi oranda? Makale
     cokmeyi kapiya ozgu ve grafa bagli anlatiyor; 500 kosu bunu sinayacak.

  4. GRAF FAKTORU. Iki graf kurgusu arasindaki fark, ayni (konfig, tekrar,
     fold, tohum) hucrelerinde eslestirilerek olculuyor. A2 karariyla graf
     SECILMIYOR; onceden belirlenmis bir faktor ve etkisi raporlanmali.

VARYANS BILESENLERI NASIL
-------------------------
Her konfigurasyon icin F1_{r,f,s} degerleri uzerinde dengeli bir uc yonlu
ayrisma: tekrar (r), fold (f) ve tohum (s) ana etkileri artik varyansla
birlikte. Fold etkisi burada "hangi trip'ler teste dustu" demektir ve en
buyuk bilesen olmasi beklenir; tohum bileseni modelin kendi kararsizligidir.
Cokme yasayan kosular bilesenleri sisirdigi icin analiz cokmeli ve cokmesiz
olarak iki kez yapilir.

KULLANIM
  python experiments/f26_repeated_analysis.py
"""
import itertools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.cluster_stats import holm_correction, nadeau_bengio, tost_equivalence
from stanet.utils import init_console, load_json, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F26_repeated_analysis.json"
SRC = C.RESULTS_DIR / "F10_repeated_cv.json"

COLLAPSE = 0.30
DELTA = 0.02          # F16 ile ayni sinir, ayni gerekce (F15'teki tekrar hatasi)
KEY = re.compile(r"r(\d+)_f(\d+)_s(\d+)")


def parse(per_run):
    """{(graf, konfig)}[(r, f, s)] = F1"""
    out = {}
    for graph, cfgs in per_run.items():
        for name, runs in cfgs.items():
            for k, v in runs.items():
                m = KEY.match(k)
                r, f, s = (int(x) for x in m.groups())
                out.setdefault((graph, name), {})[(r, f, s)] = float(v)
    return out


def variance_components(cells):
    """Dengeli uc yonlu ayrisma: tekrar, fold, tohum, artik.

    Her bilesen, o faktorun duzey ortalamalarinin genel ortalamadan
    sapmalarinin kareli ortalamasidir; artik geri kalandir. Bu bir REML
    kestirimi degil, betimsel bir ayrisma -- amac hangi faktorun oynadigini
    gostermek, varyans parametresi kestirmek degil.
    """
    if len(cells) < 4:
        return None
    ks = list(cells)
    v = np.array([cells[k] for k in ks])
    grand = float(v.mean())
    comp = {}
    for i, fac in enumerate(("repeat", "fold", "seed")):
        levels = {}
        for k in ks:
            levels.setdefault(k[i], []).append(cells[k])
        means = np.array([np.mean(x) for x in levels.values()])
        comp[fac] = float(((means - grand) ** 2).mean())
    total = float(v.var())
    explained = sum(comp.values())
    comp["residual"] = float(max(total - explained, 0.0))
    comp["total"] = total
    comp["sd_total"] = float(v.std(ddof=1))
    denom = explained + comp["residual"]
    comp["share"] = {k: (comp[k] / denom if denom > 0 else 0.0)
                     for k in ("repeat", "fold", "seed", "residual")}
    return comp


def main():
    if not SRC.exists():
        print("F10 sonucu yok: %s" % SRC)
        return 1
    src = load_json(SRC)
    cells = parse(src["per_run"])
    design = src["design"]
    graphs = design["graphs"]
    configs = design["configs"]
    stamp = protocol_stamp()

    n_tr, n_te = 59, 18          # trip sayilari (5-fold, 90 trip)
    out = {"protocol": stamp, "source": SRC.name, "design": design,
           "delta": DELTA, "collapse_threshold": COLLAPSE,
           "variance_components": {}, "collapse": {},
           "comparisons": {}, "graph_effect": {}}

    # ------------------------------------------------------------ 1. varyans
    print("=" * 96)
    print("1. VARYANS BILESENLERI — degiskenlik nereden geliyor")
    print("=" * 96)
    print("  %-14s %-18s %8s %9s %9s %9s %9s"
          % ("graf", "konfig", "sd", "bolumleme", "fold", "tohum", "artik"))
    for graph in graphs:
        for name in configs:
            c = cells.get((graph, name))
            if not c:
                continue
            for tag, sel in (("all", c),
                             ("no_collapse", {k: v for k, v in c.items()
                                              if v >= COLLAPSE})):
                vc = variance_components(sel)
                if vc is None:
                    continue
                out["variance_components"].setdefault(graph, {}) \
                    .setdefault(name, {})[tag] = vc
            vc = out["variance_components"][graph][name]["all"]
            sh = vc["share"]
            print("  %-14s %-18s %8.4f %8.1f%% %8.1f%% %8.1f%% %8.1f%%"
                  % (graph, name, vc["sd_total"], 100 * sh["repeat"],
                     100 * sh["fold"], 100 * sh["seed"], 100 * sh["residual"]))

    # cokmesiz hali, yalnizca coken konfigurasyonlar icin
    print("\n  cokmeler cikarilinca (yalnizca etkilenen konfigurasyonlar):")
    for graph in graphs:
        for name in configs:
            d = out["variance_components"].get(graph, {}).get(name, {})
            if "no_collapse" not in d or d["no_collapse"]["sd_total"] == d["all"]["sd_total"]:
                continue
            a, b = d["all"], d["no_collapse"]
            print("  %-14s %-18s sd %.4f -> %.4f  (fold payi %.0f%% -> %.0f%%)"
                  % (graph, name, a["sd_total"], b["sd_total"],
                     100 * a["share"]["fold"], 100 * b["share"]["fold"]))

    # ------------------------------------------------------------ 2. cokme
    print("\n" + "=" * 96)
    print("2. COKME — F1 < %.2f" % COLLAPSE)
    print("=" * 96)
    print("  %-18s %-16s %-16s %s" % ("konfig", "trip_instance", "osm_merged", "toplam"))
    learns_weight = {"stanet_gated", "hybrid_crossatt"}
    for name in configs:
        row, tot, n_tot = [], 0, 0
        for graph in graphs:
            c = cells.get((graph, name), {})
            k = sum(1 for v in c.values() if v < COLLAPSE)
            row.append("%d/%d" % (k, len(c)))
            tot += k
            n_tot += len(c)
        mark = "  <- karisim agirligi ogreniyor" if name in learns_weight else ""
        print("  %-18s %-16s %-16s %d/%d%s"
              % (name, row[0], row[1], tot, n_tot, mark))
        out["collapse"][name] = {
            "by_graph": {g: {"n": sum(1 for v in cells.get((g, name), {}).values()
                                      if v < COLLAPSE),
                             "of": len(cells.get((g, name), {}))} for g in graphs},
            "total": tot, "of": n_tot,
            "learns_mixing_weight": name in learns_weight}
    lw = sum(out["collapse"][n]["total"] for n in configs if n in learns_weight)
    lw_of = sum(out["collapse"][n]["of"] for n in configs if n in learns_weight)
    fx = sum(out["collapse"][n]["total"] for n in configs if n not in learns_weight)
    fx_of = sum(out["collapse"][n]["of"] for n in configs if n not in learns_weight)
    print("\n  ogrenilen karisim agirligi olanlar : %d/%d (%%%.1f)"
          % (lw, lw_of, 100 * lw / max(lw_of, 1)))
    print("  sabit operator ve kontrol          : %d/%d (%%%.1f)"
          % (fx, fx_of, 100 * fx / max(fx_of, 1)))
    out["collapse_summary"] = {
        "learned_weight": {"n": lw, "of": lw_of, "rate": lw / max(lw_of, 1)},
        "fixed_or_control": {"n": fx, "of": fx_of, "rate": fx / max(fx_of, 1)},
    }

    # ------------------------------------------------------------ 3. TOST
    print("\n" + "=" * 96)
    print("3. KARSILASTIRMALAR — Nadeau-Bengio + TOST (delta = %.3f)" % DELTA)
    print("=" * 96)
    for graph in graphs:
        rows, pv = {}, {}
        for a, b in itertools.combinations(configs, 2):
            ca, cb = cells.get((graph, a)), cells.get((graph, b))
            if not ca or not cb:
                continue
            ks = sorted(set(ca) & set(cb))
            diffs = [ca[k] - cb[k] for k in ks]
            nb = nadeau_bengio(diffs, n_train=n_tr, n_test=n_te)
            ts = tost_equivalence(diffs, DELTA, n_train=n_tr, n_test=n_te)
            key = a + "_vs_" + b
            rows[key] = {"n_paired": len(ks), "nadeau_bengio": nb, "tost": ts}
            pv[key] = nb["p_corrected"]
        for k, v in holm_correction(pv).items():
            rows[k]["holm"] = v
        for k, v in rows.items():
            nb, ts = v["nadeau_bengio"], v["tost"]
            sig = v.get("holm", {}).get("significant")
            v["verdict"] = ("ESDEGER" if ts["equivalent"]
                            else ("FARKLI" if sig else "KARARSIZ"))
        out["comparisons"][graph] = rows
        print("\n  GRAF %s   (aile %d karsilastirma)" % (graph, len(rows)))
        print("  %-36s %9s %22s %8s %9s %10s"
              % ("karsilastirma", "ort fark", "NB %95 GA", "Holm p", "p_TOST", "sonuc"))
        for k, v in sorted(rows.items(),
                           key=lambda kv: -abs(kv[1]["nadeau_bengio"]["mean_diff"])):
            nb, ts = v["nadeau_bengio"], v["tost"]
            print("  %-36s %+9.4f [%+.4f, %+.4f] %8.4f %9.4f %10s"
                  % (k, nb["mean_diff"], nb["ci_lo"], nb["ci_hi"],
                     v.get("holm", {}).get("p_holm", float("nan")),
                     ts["p_tost"], v["verdict"]))

    # ------------------------------------------------------------ 4. graf
    print("\n" + "=" * 96)
    print("4. GRAF FAKTORU — ayni hucrede osm_merged eksi trip_instance")
    print("=" * 96)
    print("  %-18s %9s %22s %8s %9s %10s"
          % ("konfig", "ort fark", "NB %95 GA", "NB p", "p_TOST", "sonuc"))
    pv2, rows2 = {}, {}
    for name in configs:
        a, b = cells.get((graphs[1], name)), cells.get((graphs[0], name))
        if not a or not b:
            continue
        ks = sorted(set(a) & set(b))
        diffs = [a[k] - b[k] for k in ks]
        if len(set(diffs)) == 1 and diffs[0] == 0.0:
            rows2[name] = {"identical": True, "n_paired": len(ks)}
            print("  %-18s %9s  %-22s %8s %9s %10s"
                  % (name, "0.0000", "(uzamsal dal yok)", "—", "—", "AYNI"))
            continue
        nb = nadeau_bengio(diffs, n_train=n_tr, n_test=n_te)
        ts = tost_equivalence(diffs, DELTA, n_train=n_tr, n_test=n_te)
        rows2[name] = {"n_paired": len(ks), "nadeau_bengio": nb, "tost": ts}
        pv2[name] = nb["p_corrected"]
    for k, v in (holm_correction(pv2).items() if pv2 else []):
        rows2[k]["holm"] = v
    for name, v in rows2.items():
        if v.get("identical"):
            continue
        nb, ts = v["nadeau_bengio"], v["tost"]
        sig = v.get("holm", {}).get("significant")
        v["verdict"] = ("ESDEGER" if ts["equivalent"]
                        else ("FARKLI" if sig else "KARARSIZ"))
        print("  %-18s %+9.4f [%+.4f, %+.4f] %8.4f %9.4f %10s"
              % (name, nb["mean_diff"], nb["ci_lo"], nb["ci_hi"],
                 nb["p_corrected"], ts["p_tost"], v["verdict"]))
    out["graph_effect"] = rows2

    save_json(out, OUT)
    print("\nkaydedildi: %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
