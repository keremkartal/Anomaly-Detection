"""
F27 — Revizyon veri kumelerinin denetimi (36 yorunge + 4 graf).

Yeni veri geldiginde ilk is onu kosmak degil, ne oldugunu olcmek. Bu betik
uc soruyu cevapliyor ve sonuclari denetlenebilir bir dosyaya yaziyor.

1. SIZINTI. `tunnel` bir graf dugum ozniteligi. Yeni kumelerde `tunnel == 1`
   olan satirlarin etiketi her zaman 9 mu? Oyleyse grafa erisen model
   anomalilerin ne kadarini bedava aliyor?

2. YAYGINLIK EKSENI. Klasor adlarindaki %1 / %5 / %15 hedefleri gercek
   anomali oranina yansiyor mu?

3. GURULTU EKSENI. Gurultu carpani (0,5 / 1,0 / 1,5) olculebilir bir fark
   yaratiyor mu? Olcut: normal satirlarda ardisik hiz farkinin standart
   sapmasi ve ivmenin standart sapmasi.

Ayrica alt tip dagilimlari ve zorunlu on isleme filtresinin kac satir
sildigi raporlanir.

KULLANIM
  python experiments/f27_new_data_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from stanet import config as C
from stanet import datasets as D
from stanet.datasets import clean_new_dataset
from stanet.utils import init_console, save_json

init_console()
OUT = C.RESULTS_DIR / "F27_new_data_audit.json"
COLS = ["trip_id", "speed", "accel", "anomaly", "maxspeed", "tunnel"]


def audit_one(path):
    d = pd.read_csv(path, usecols=COLS)
    raw_n = len(d)
    c = clean_new_dataset(d)
    cl = c.attrs["cleaning"]

    t1, a9 = c.tunnel == 1, c.anomaly == 9
    anom = c.anomaly != 0
    n_anom = int(anom.sum())
    normal = c[~anom]
    # gurultu olcutu: ardisik hiz farkinin trip ici standart sapmasi
    dv = normal.groupby("trip_id")["speed"].apply(lambda s: s.diff().std())

    over = c.speed > 1.10 * c.maxspeed
    return {
        "rows_raw": int(raw_n), "rows_clean": int(len(c)),
        "cleaning": cl,
        "n_trips": int(c.trip_id.nunique()),
        "anomaly_rate": float(anom.mean()),
        "subtypes": {str(k): int(v) for k, v in
                     sorted(c.anomaly.value_counts().items())},
        "leak": {
            "n_tunnel_rows": int(t1.sum()),
            "p_anomaly9_given_tunnel": float((t1 & a9).sum() / max(int(t1.sum()), 1)),
            "p_tunnel_given_anomaly9": float((t1 & a9).sum() / max(int(a9.sum()), 1)),
            "type9_share_of_anomalies": float(int(a9.sum()) / max(n_anom, 1)),
            "tunnel_share_of_anomalies": float(int((t1 & anom).sum()) / max(n_anom, 1)),
        },
        "label_rule": {
            "n_over_limit": int(over.sum()),
            "p_anomalous_given_over_limit": float((over & anom).sum()
                                                  / max(int(over.sum()), 1)),
            "p_type1_given_over_limit": float((over & (c.anomaly == 1)).sum()
                                              / max(int(over.sum()), 1)),
        },
        "noise_proxy": {
            "speed_diff_sd_mean": float(dv.mean()),
            "accel_sd": float(normal.accel.std()),
            "speed_sd": float(normal.speed.std()),
        },
    }


def main():
    av = D.available()
    new = {k: v for k, v in av.items() if v["new_format"]}
    print("denetlenecek veri kumesi: %d\n" % len(new), flush=True)
    out = {"n_datasets": len(new), "datasets": {}}

    print("%-28s %8s %6s %7s %8s %9s %9s"
          % ("kume", "satir", "trip", "anom%", "tip9/an", "P(9|tun)", "dv sd"))
    for name in sorted(new):
        a = audit_one(new[name]["trajectory"])
        a["meta"] = {k: str(v) for k, v in new[name].items()}
        out["datasets"][name] = a
        print("%-28s %8d %6d %6.1f%% %7.1f%% %9.4f %9.4f"
              % (name, a["rows_clean"], a["n_trips"], 100 * a["anomaly_rate"],
                 100 * a["leak"]["type9_share_of_anomalies"],
                 a["leak"]["p_anomaly9_given_tunnel"],
                 a["noise_proxy"]["speed_diff_sd_mean"]), flush=True)

    # ------------------------------------------------------------ 1. sizinti
    p = [a["leak"]["p_anomaly9_given_tunnel"] for a in out["datasets"].values()]
    sh = {c: [a["leak"]["tunnel_share_of_anomalies"]
              for k, a in out["datasets"].items() if a["meta"]["city"] == c]
          for c in ("kocaeli", "hamburg")}
    out["summary_leak"] = {
        "p_anomaly9_given_tunnel": {"min": float(min(p)), "median": float(np.median(p)),
                                    "max": float(max(p))},
        "tunnel_share_of_anomalies": {c: {"min": float(min(v)), "max": float(max(v))}
                                      for c, v in sh.items()},
        "verdict": ("tunnel bir graf dugum ozniteligidir; bu baginti grafa "
                    "erisen her modele anomalilerin bir kismini bedava verir"),
    }
    print("\n" + "=" * 92)
    print("1. SIZINTI")
    print("  P(anomali=9 | tunnel=1): min %.4f  medyan %.4f  maks %.4f"
          % (min(p), np.median(p), max(p)))
    for c, v in sh.items():
        print("  %-8s anomalilerin %%%.1f-%%%.1f'i tunel satirinda"
              % (c, 100 * min(v), 100 * max(v)))

    # ------------------------------------------------------------ 2. yayginlik
    print("\n2. YAYGINLIK EKSENI (hedef -> gerceklesen)")
    prev = {}
    for k, a in out["datasets"].items():
        prev.setdefault(a["meta"]["prevalence"], []).append(a["anomaly_rate"])
    for tgt in ("1", "5", "15"):
        v = prev.get(tgt, [])
        if v:
            print("  hedef %%%-3s -> gerceklesen %%%.1f - %%%.1f (ort %%%.1f)"
                  % (tgt, 100 * min(v), 100 * max(v), 100 * np.mean(v)))
    out["summary_prevalence"] = {t: {"min": float(min(v)), "max": float(max(v)),
                                     "mean": float(np.mean(v))}
                                 for t, v in prev.items() if v}

    # ------------------------------------------------------------ 3. gurultu
    print("\n3. GURULTU EKSENI (carpan -> olculen degiskenlik)")
    noi = {}
    for k, a in out["datasets"].items():
        noi.setdefault(a["meta"]["noise"], []).append(
            (a["noise_proxy"]["speed_diff_sd_mean"], a["noise_proxy"]["accel_sd"]))
    rows = []
    for tgt in ("0_5", "1_0", "1_5"):
        v = noi.get(tgt, [])
        if v:
            dv = float(np.mean([x[0] for x in v]))
            ac = float(np.mean([x[1] for x in v]))
            rows.append((tgt, dv, ac))
            print("  carpan %-4s -> dv sd %.4f | ivme sd %.4f" % (tgt, dv, ac))
    if len(rows) == 3:
        spread = max(r[1] for r in rows) - min(r[1] for r in rows)
        within = float(np.mean([np.std([x[0] for x in noi[t]]) for t in noi]))
        out["summary_noise"] = {
            "by_level": {r[0]: {"speed_diff_sd": r[1], "accel_sd": r[2]} for r in rows},
            "between_level_spread": spread,
            "within_level_sd": within,
            "axis_effective": bool(spread > 2 * within),
        }
        print("  seviyeler arasi yayilim %.4f | seviye ici sd %.4f -> eksen %s"
              % (spread, within,
                 "CALISIYOR" if spread > 2 * within else "CALISMIYOR"))

    save_json(out, OUT)
    print("\nkaydedildi: %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
