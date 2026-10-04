"""
F30 — Sizinti neden bir sehirde etkisiz, otekinde yikici?

GOZLEM
------
`tunnel` graf dugum ozniteligi her iki sehirde de anomali 9'u kusursuz
belirliyor (P(anomali 9 | tunnel) = 1,0000). Ama uzamsal dalli modellere
eklendiginde etkisi tamamen farkli:

  Kocaeli  ortalama sisme -0,0005   (hicbir sey degismiyor)
  Hamburg  ortalama sisme -0,0723   (kapili model 0,8664 -> 0,6435)

ACIKLAMA — olculdu, varsayilmadi
--------------------------------
Anomali 9 "sinyal donmasi"; her iki sehirde de hiz kanalindan KUSURSUZ
gorulebiliyor: P(ardisik hiz farki sifir | anomali 9) = 1,0000. Yani zamansal
dal bu sinifi graf olmadan da yakalayabilir.

Fark TERS kosulda:

  Kocaeli  P(anomali 9 | donmus hiz) = 0,7686   tip 9, anomalilerin %60,4'u
  Hamburg  P(anomali 9 | donmus hiz) = 0,1044   tip 9, anomalilerin %5,1'i

Kocaeli'de donma guclu bir gostergedir ve zamansal dal zaten bu sinifi
aliyordur; `tunnel` ozniteligi FAZLALIK, eklemek bir sey degistirmez.
Hamburg'da donma zayif bir gostergedir ve `tunnel`, zamansal dalin sahip
olmadigi bir bilgiyi tasir: NADIR bir sinifin tek kusursuz gostergesi.
Kapi iste o zaman uzamsal dala cekiliyor, zamansal baskinliga kacamiyor ve
kosu cokuyor (bkz. F29).

SONUC
-----
Graf ozniteliginden sizan bir etiket, ayni etiket diger dalin girdilerinden
de gorulebiliyorsa ZARARSIZ; gorulemiyorsa ZARARLI -- ve zarari sisirilmis
dogruluk olarak degil, EGITIM KARARSIZLIGI olarak ortaya cikiyor.

KULLANIM
  python experiments/f30_leak_diagnosis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from stanet import config as C
from stanet import datasets as D
from stanet.utils import init_console, load_json, save_json

init_console()
OUT = C.RESULTS_DIR / "F30_leak_diagnosis.json"
PAIRS = [("kocaeli", "kocaeli_knn_n0_5_a5", "F28_kocaeli_knn_n0_5_a5.json"),
         ("hamburg", "hamburg_knn_n0_5_a5", "F28_hamburg_knn_n0_5_a5.json")]
FREEZE_TOL = 1e-9


def kinematics(path):
    """Tip 9'un hiz kanalindan gorulebilirligi."""
    d = pd.read_csv(path, usecols=["trip_id", "time", "speed", "anomaly", "tunnel"])
    d = d.sort_values(["trip_id", "time"]).reset_index(drop=True)
    a9 = (d.anomaly == 9).values
    anom = (d.anomaly != 0).values
    t1 = (d.tunnel == 1).values
    dv = d.groupby("trip_id")["speed"].diff().abs().fillna(1.0).values
    frozen = dv < FREEZE_TOL
    return {
        "n_rows": int(len(d)), "anomaly_rate": float(anom.mean()),
        "type9_share_of_anomalies": float(a9.sum() / max(anom.sum(), 1)),
        "p_tunnel_implies_type9": float((t1 & a9).sum() / max(t1.sum(), 1)),
        "p_frozen_given_type9": float(frozen[a9].mean()),
        "p_frozen_given_not_type9": float(frozen[~a9].mean()),
        "p_type9_given_frozen": float((a9 & frozen).sum() / max(frozen.sum(), 1)),
    }


def main():
    av = D.available()
    out = {"freeze_tolerance": FREEZE_TOL, "cities": {}}
    for city, dsname, resfile in PAIRS:
        rec = {"dataset": dsname, "kinematics": kinematics(av[dsname]["trajectory"])}
        f28 = C.RESULTS_DIR / resfile
        if f28.exists():
            j = load_json(f28)["datasets"][dsname]
            rec["leak_cost"] = j["leak_cost"]
            rec["f1"] = {
                tag: {k: v["f1_mean"] for k, v in j["settings"][tag]["models"].items()}
                for tag in ("tunnel_in", "tunnel_out")}
            rec["sd"] = {
                tag: {k: v["f1_sd"] for k, v in j["settings"][tag]["models"].items()}
                for tag in ("tunnel_in", "tunnel_out")}
            rec["collapsed"] = {
                tag: {k: v["n_collapsed"] for k, v in j["settings"][tag]["models"].items()}
                for tag in ("tunnel_in", "tunnel_out")}
        out["cities"][city] = rec

    print("=" * 92)
    print("TIP 9'UN ZAMANSAL DALDAN GORULEBILIRLIGI")
    print("=" * 92)
    print("  %-10s %10s %12s %16s %16s" % ("sehir", "anom.", "tip9 payi",
                                           "P(don|tip9)", "P(tip9|don)"))
    for city, rec in out["cities"].items():
        k = rec["kinematics"]
        print("  %-10s %9.1f%% %11.1f%% %16.4f %16.4f"
              % (city, 100 * k["anomaly_rate"], 100 * k["type9_share_of_anomalies"],
                 k["p_frozen_given_type9"], k["p_type9_given_frozen"]))

    print("\n" + "=" * 92)
    print("SIZINTININ BEDELI (tunnel_in eksi tunnel_out)")
    print("=" * 92)
    print("  %-10s %22s %22s" % ("sehir", "uzamsal dalli ort.", "kontrol"))
    for city, rec in out["cities"].items():
        lc = rec.get("leak_cost")
        if lc:
            print("  %-10s %22.4f %22.4f"
                  % (city, lc["mean_spatial"], lc["control"]))

    print("\n  en cok etkilenen konfigurasyon:")
    for city, rec in out["cities"].items():
        lc = rec.get("leak_cost")
        if not lc:
            continue
        worst = min(lc["per_config"].items(), key=lambda kv: kv[1])
        f1 = rec["f1"]
        sd = rec["sd"]
        print("    %-10s %-16s %.4f -> %.4f  (sd %.4f -> %.4f, cokme %d -> %d)"
              % (city, worst[0], f1["tunnel_in"][worst[0]], f1["tunnel_out"][worst[0]],
                 sd["tunnel_in"][worst[0]], sd["tunnel_out"][worst[0]],
                 rec["collapsed"]["tunnel_in"][worst[0]],
                 rec["collapsed"]["tunnel_out"][worst[0]]))

    kk = out["cities"]["kocaeli"]["kinematics"]
    hh = out["cities"]["hamburg"]["kinematics"]
    out["verdict"] = {
        "statement": ("graf ozniteliginden sizan etiket, ayni etiket diger "
                      "dalin girdilerinden de gorulebiliyorsa zararsiz; "
                      "gorulemiyorsa zararli, ve zarar dogruluk sismesi degil "
                      "egitim kararsizligi olarak gorunur"),
        "redundancy_kocaeli": kk["p_type9_given_frozen"],
        "redundancy_hamburg": hh["p_type9_given_frozen"],
    }
    print("\n  SONUC: %s" % out["verdict"]["statement"])
    save_json(out, OUT)
    print("\nkaydedildi: %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
