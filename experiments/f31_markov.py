"""
F31 — Markov gecis davranisini YAYINLANAN VERIDEN geri cikar (madde 1.6).

HAKEMIN ISTEGI
--------------
"additional information about how transition probabilities are selected or
calibrated would improve reproducibility"

Makale dort kinematik durum (sabit hiz, hizlanma, yavaslama, durma) arasinda
trafik rejimine kosullu bir Markov zinciri tarif ediyor ama gecis
olasiliklarini vermiyor. Uretim parametreleri bize ulasmadi.

BU BETIGIN YAPTIGI
------------------
Parametreleri degil, GERCEKLESEN gecis davranisini olcer. Yayinlanan
yorungelerden durum dizisi cikarilir ve rejim basina ampirik gecis matrisi
hesaplanir. Bu, hakemin istediginin tekrarlanabilirlik kismini karsilar:
okur, uretilen verinin nasil davrandigini sayiyla gorebilir.

NE KARSILAMAZ
-------------
Bu matrisler uretecin PARAMETRELERI degildir. Uretec parametresi ile
gerceklesen davranis arasinda uc fark var: (i) anomali senaryolari Markov
surecini devre disi birakiyor, bu yuzden yalnizca anomalisiz satirlar
kullanilir; (ii) hiz limitleri ve ivme sinirlari gecisleri kisitliyor;
(iii) sensor gurultusu durum siniflandirmasina karisiyor. Metinde bu acikca
yazilmali.

DURUM TANIMLARI
---------------
  stop        v < 1,0 km/h          (makalenin uzun duraklama esigiyle ayni)
  accelerate  a > +EPS m/s^2
  decelerate  a < -EPS m/s^2
  constant    |a| <= EPS

EPS = 0,10 m/s^2. Secim duyarliligi ayrica raporlanir, cunku "sabit hiz"
durumunun payi EPS'e gore degisir -- ve bu calismada sasirtici derecede
kucuk cikiyor.

KULLANIM
  python experiments/f31_markov.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from stanet import config as C
from stanet.utils import init_console, save_json

init_console()
OUT = C.RESULTS_DIR / "F31_markov.json"
STATES = ["stop", "decelerate", "constant", "accelerate"]
EPS = 0.10
EPS_SWEEP = [0.05, 0.10, 0.25, 0.50]
STOP_SPEED = 1.0
REGIMES = {"free-flow": (1, 3), "transition": (4, 6), "congested": (7, 9)}


def classify(speed, accel, eps):
    st = np.full(len(speed), 2, dtype=int)          # constant
    st[accel > eps] = 3                              # accelerate
    st[accel < -eps] = 1                             # decelerate
    st[speed < STOP_SPEED] = 0                       # stop (oncelikli)
    return st


def transitions(df, eps):
    """Rejim basina (4x4) sayim matrisi; gecisler trip ICINDE ve ardisik."""
    out = {k: np.zeros((4, 4), dtype=np.int64) for k in REGIMES}
    for _, g in df.groupby("trip_id", sort=False):
        g = g.sort_values("time")
        st = classify(g.speed.values, g.accel.values, eps)
        tr = g.traffic.values
        dt = np.diff(g.time.values)
        ok = dt == 1                                  # yalnizca ardisik saniyeler
        a, b = st[:-1][ok], st[1:][ok]
        reg = tr[:-1][ok]
        for name, (lo, hi) in REGIMES.items():
            m = (reg >= lo) & (reg <= hi)
            if m.any():
                np.add.at(out[name], (a[m], b[m]), 1)
    return out


def main():
    d = pd.read_csv(C.RAW_TRAJECTORY,
                    usecols=["trip_id", "time", "speed", "accel", "traffic", "anomaly"])
    d = d[d["speed"].between(*C.SPEED_RANGE)]
    normal = d[d.anomaly == 0].copy()
    print("yorunge satiri %d | anomalisiz %d | trip %d"
          % (len(d), len(normal), normal.trip_id.nunique()), flush=True)

    out = {"source": str(C.RAW_TRAJECTORY.name), "eps": EPS,
           "stop_speed_kmh": STOP_SPEED, "states": STATES,
           "regimes": {k: list(v) for k, v in REGIMES.items()},
           "excludes_anomalous_rows": True,
           "caveat": ("bunlar urerecin parametreleri degil, GERCEKLESEN gecis "
                      "davranisidir; anomali senaryolari, hiz limitleri ve "
                      "sensor gurultusu araya girer"),
           "regime_speed_mean": {}, "matrices": {}, "state_shares": {},
           "eps_sensitivity": {}}

    for name, (lo, hi) in REGIMES.items():
        m = normal.traffic.between(lo, hi)
        out["regime_speed_mean"][name] = float(normal.loc[m, "speed"].mean())
        out["matrices"][name] = {}

    cnt = transitions(normal, EPS)
    print("\nREJIM BASINA GECIS OLASILIKLARI (satir: su an, sutun: sonraki)")
    for name in REGIMES:
        M = cnt[name]
        tot = M.sum()
        P = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        occ = M.sum(axis=1) / max(tot, 1)
        out["matrices"][name] = {
            STATES[i]: {STATES[j]: float(P[i, j]) for j in range(4)}
            for i in range(4)}
        out["state_shares"][name] = {STATES[i]: float(occ[i]) for i in range(4)}
        out["matrices"][name]["_n_transitions"] = int(tot)
        print("\n  %s  (ort hiz %.1f km/h, %s gecis)"
              % (name, out["regime_speed_mean"][name], "{:,}".format(int(tot))))
        print("    %-12s %8s %8s %8s %8s   pay" % ("", *STATES))
        for i, s in enumerate(STATES):
            print("    %-12s %8.4f %8.4f %8.4f %8.4f   %.4f"
                  % (s, P[i, 0], P[i, 1], P[i, 2], P[i, 3], occ[i]))

    # EPS duyarliligi: "sabit hiz" durumu ne kadar kullaniliyor
    print("\nEPS DUYARLILIGI — 'sabit hiz' durumunun payi")
    for e in EPS_SWEEP:
        c2 = transitions(normal, e)
        shares = {}
        for name in REGIMES:
            M = c2[name]
            shares[name] = float(M.sum(axis=1)[2] / max(M.sum(), 1))
        out["eps_sensitivity"]["%.2f" % e] = shares
        print("  eps=%.2f  " % e + "  ".join("%s %.4f" % (k, v)
                                             for k, v in shares.items()))

    allshare = {s: float(sum(out["state_shares"][r][s] *
                             out["matrices"][r]["_n_transitions"] for r in REGIMES) /
                         sum(out["matrices"][r]["_n_transitions"] for r in REGIMES))
                for s in STATES}
    out["state_shares"]["overall"] = allshare
    print("\n  tum rejimler: " + "  ".join("%s %.4f" % (k, v)
                                           for k, v in allshare.items()))
    save_json(out, OUT)
    print("\nkaydedildi: %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
