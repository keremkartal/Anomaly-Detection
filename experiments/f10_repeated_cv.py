"""
F10 — Tekrarli gruplu capraz dogrulama (Hakem 3, madde 3).

Hakem, cikarimin tekrarli gruplu bolumlemelere ve paylasilan egitim
tohumlarina genisletilmesini; belirsizlik tahmininin de tohum
degiskenligini ve ortusen egitim kumelerini hesaba katmasini istiyor.

Uc parcali cevap:

  1. TEKRARLI BOLUMLEME. `GroupKFold` deterministiktir; F5 tek bir
     bolumleme uzerinde calisiyordu. Burada R farkli bolumleme uretilir
     (`build_cv_folds(repeat=r)`), her biri trip gruplamasini korur.

  2. PAYLASILAN TOHUM. Karsilastirilan tum konfigurasyonlar AYNI
     (tekrar, fold, tohum) uclusunde kosulur; farklar eslestirilmis olarak
     alinir. Boylece bir modelin "sansli bolumleme" avantaji elenir.

  3. ORTUSME DUZELTMESI. Tekrarli CV'de egitim kumeleri ortusur; siradan
     t-testi varyansi kucuk tahmin eder. Nadeau-Bengio duzeltmesi uygulanir
     (`stanet.cluster_stats.nadeau_bengio`).

Graf kurgusu A2 karariyla SECILMIYOR: iki kurgu da onceden belirlenmis
deney faktorudur, bu yuzden her sey iki grafta kosulur.

KULLANIM
  python experiments/f10_repeated_cv.py --pilot     # kucuk dogrulama kosusu
  python experiments/f10_repeated_cv.py             # tam kosum
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stanet import config as C
from stanet.cluster_stats import nadeau_bengio
from stanet.data import build_cv_folds
from stanet.graph import build_graph, map_segments
from stanet.runstore import get_or_train, make_spec
from stanet.utils import get_device, init_console, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F10_repeated_cv.json"

CONFIGS = {
    "hybrid_gat":      dict(temporal="lstm", spatial="gat",  fusion="weighted_sum"),
    "stanet_gated":    dict(temporal="lstm", spatial="gat",  fusion="gated"),
    "hybrid_concat":   dict(temporal="lstm", spatial="gat",  fusion="concat"),
    "hybrid_crossatt": dict(temporal="lstm", spatial="gat",  fusion="cross_attention"),
    "lstm_only":       dict(temporal="lstm", spatial="none", fusion="weighted_sum"),
}

# eslestirilmis karsilastirmalar: (A, B) -> A - B farki raporlanir
PAIRS = [("hybrid_gat", "lstm_only"),
         ("stanet_gated", "lstm_only"),
         ("stanet_gated", "hybrid_gat"),
         ("stanet_gated", "hybrid_concat"),
         ("stanet_gated", "hybrid_crossatt")]

PILOT = dict(repeats=[0, 1], seeds=[42],
             configs=["hybrid_gat", "lstm_only"], graphs=[False])
FULL = dict(repeats=[0, 1, 2, 3, 4], seeds=[42, 43],
            configs=list(CONFIGS), graphs=[False, True])


def run(plan, device):
    stamp = protocol_stamp()
    graphs = {m: build_graph(merge_by_osm=m) for m in plan["graphs"]}
    n_total = (len(plan["repeats"]) * C.N_FOLDS * len(plan["seeds"])
               * len(plan["configs"]) * len(plan["graphs"]))
    print(f"protokol {stamp['hash']} | {len(plan['repeats'])} tekrar x "
          f"{C.N_FOLDS} fold x {len(plan['seeds'])} tohum x "
          f"{len(plan['configs'])} konfig x {len(plan['graphs'])} graf "
          f"= {n_total} kosu\n", flush=True)

    # runs[(graf, konfig)][(tekrar, fold, tohum)] = F1
    runs, meta = {}, {}
    done = 0
    for merge in plan["graphs"]:
        g = graphs[merge]
        gtag = "osm_merged" if merge else "trip_instance"
        for r in plan["repeats"]:
            folds = build_cv_folds(n_folds=C.N_FOLDS, legacy=False, repeat=r)
            idx = {}
            for f in folds:
                k = f.info["fold"]
                idx[k] = {sp: map_segments(getattr(f, f"meta_{sp}"), g)[0]
                          for sp in ("train", "val", "test")}
                meta[(r, k)] = {"n_train_trips": f.info["trips"]["train"],
                                "n_test_trips": f.info["trips"]["test"]}
            for name in plan["configs"]:
                for f in folds:
                    k = f.info["fold"]
                    for seed in plan["seeds"]:
                        # r=0 ORIJINAL bolumlemedir (GroupKFold). Anahtara
                        # `rep0` yazmiyoruz ki F5'in onbellekteki kosulari
                        # aynen kullanilsin: boylece tekrarli CV tablosunun
                        # birinci dilimi Tablo 13 ile TANIM GEREGI ayni olur,
                        # "ayni model farkli tabloda farkli sayi" sorunu
                        # yeniden dogmaz.
                        rep_kw = {} if r == 0 else {"rep": r}
                        spec = make_spec(merge_by_osm=merge, fold=k, cv=C.N_FOLDS,
                                         **rep_kw, **CONFIGS[name])
                        res = get_or_train(spec, seed, f, idx[k], g, device,
                                           verbose=False)
                        runs.setdefault((gtag, name), {})[(r, k, seed)] = \
                            float(res["metrics"]["f1"])
                        done += 1
                        tag = " [onbellek]" if res["cached"] else ""
                        print(f"  [{done:4d}/{n_total}] {gtag:13s} {name:16s} "
                              f"r{r} f{k} s{seed}: F1={res['metrics']['f1']:.4f}{tag}",
                              flush=True)
    return runs, meta, stamp


def analyse(runs, meta, plan, stamp):
    out = {"protocol": stamp, "design": {
        "repeats": plan["repeats"], "n_folds": C.N_FOLDS,
        "seeds": plan["seeds"], "configs": plan["configs"],
        "graphs": ["osm_merged" if m else "trip_instance" for m in plan["graphs"]],
        "note": "graf kurgusu secilmez; onceden belirlenmis faktordur"},
        "per_run": {}, "comparisons": {}}

    for (gtag, name), d in runs.items():
        out["per_run"].setdefault(gtag, {})[name] = {
            f"r{r}_f{k}_s{s}": v for (r, k, s), v in sorted(d.items())}

    # her graf icin ozet ve eslestirilmis karsilastirmalar
    for gtag in {g for g, _ in runs}:
        print("\n" + "=" * 84)
        print(f"GRAF: {gtag}")
        print("=" * 84)
        present = [n for (g, n) in runs if g == gtag]
        print(f"\n  {'konfig':18s} {'F1 (ort+-sd)':>18s} {'en kotu':>9s} {'kosu':>6s}")
        summ = {}
        for name in plan["configs"]:
            if name not in present:
                continue
            v = np.array(list(runs[(gtag, name)].values()))
            summ[name] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                          "min": float(v.min()), "n": int(len(v))}
            print(f"  {name:18s} {v.mean():.4f}+-{v.std(ddof=1):.4f}  "
                  f"{v.min():>9.4f} {len(v):>6d}")
        out.setdefault("summary", {})[gtag] = summ

        # eslestirilmis farklar + Nadeau-Bengio
        ntr = int(np.mean([m["n_train_trips"] for m in meta.values()]))
        nte = int(np.mean([m["n_test_trips"] for m in meta.values()]))
        print(f"\n  ESLESTIRILMIS KARSILASTIRMA (Nadeau-Bengio, "
              f"egitim {ntr} / test {nte} trip)")
        print(f"  {'karsilastirma':30s} {'ort fark':>9s} {'NB %95 GA':>22s} "
              f"{'NB p':>8s} {'naif p':>8s} {'genisleme':>10s}")
        for a, b in PAIRS:
            if (gtag, a) not in runs or (gtag, b) not in runs:
                continue
            keys = sorted(set(runs[(gtag, a)]) & set(runs[(gtag, b)]))
            if len(keys) < 2:
                continue
            diffs = [runs[(gtag, a)][k] - runs[(gtag, b)][k] for k in keys]
            nb = nadeau_bengio(diffs, n_train=ntr, n_test=nte)
            out["comparisons"].setdefault(gtag, {})[f"{a}_vs_{b}"] = {
                "n_paired": len(keys), "diffs": diffs, "nadeau_bengio": nb}
            print(f"  {a + ' vs ' + b:30s} {nb['mean_diff']:+9.4f} "
                  f"[{nb['ci_lo']:+.4f}, {nb['ci_hi']:+.4f}] "
                  f"{nb['p_corrected']:>8.4f} {nb['p_naive_INVALID']:>8.4f} "
                  f"{nb['ci_widening']:>9.2f}x")
    return out


def main():
    pilot = "--pilot" in sys.argv
    plan = PILOT if pilot else FULL
    print(("PILOT" if pilot else "TAM") + " KOSUM\n")
    device = get_device()
    runs, meta, stamp = run(plan, device)
    out = analyse(runs, meta, plan, stamp)
    out["pilot"] = pilot
    path = (C.RESULTS_DIR / "F10_pilot.json") if pilot else OUT
    save_json(out, path)
    print(f"\nkaydedildi: {path}")


if __name__ == "__main__":
    main()
