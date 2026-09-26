"""
F23 — Veri kartindaki her sayinin HAM VERIDEN denetimi.

Makaledeki sayilarin dogrulayiciya baglanma kurali sonuc dosyalarindan
geliyordu, ama veri kartindaki sayilarin (kayit sayisi, alt tip satirlari,
temizleme kayiplari) kaynagi yoktu: elle yazilmislardi. Bu betik onlari ham
CSV'den yeniden uretir ve `F23_data_audit.json` icine yazar; boylece veri
kartindaki sayilar da denetlenebilir hale gelir.

DENETIM SIRASINDA BULUNANLAR
----------------------------
  * Tablo 6'nin satir sayilari TEMIZLEME ONCESI; ayni paragraftaki pencere
    orani temizleme SONRASI. Uc farkli asama, uc farkli evren.
  * Egim filtresi 8.186 satir siliyor; iki kurali birden ihlal eden tek satir
    yuzunden birlesim 8.245.
  * Hiz filtresinin sildigi 60 satirin 55'i tip 6. Alt tip 66'dan 11'e
    dusuyor: uretici "imkansiz hiz" enjekte ediyor, temizleyici siliyor.
  * Tip 1 kurali GEREK ama YETER degil: kurali saglayan 1.400 temiz satirin
    1.353'u tip 1, 9'u tip 6, 38'i etiketsiz.

KULLANIM
  python experiments/f23_data_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from stanet import config as C
from stanet.utils import init_console, save_json

init_console()
OUT = C.RESULTS_DIR / "F23_data_audit.json"

SPEED_RANGE = (0.0, 200.0)
GRADE_RANGE = (-50.0, 50.0)
GRADE_COL = "percent"
SPEED_LIMIT_FACTOR = 1.10


def main():
    raw = pd.read_csv(C.RAW_TRAJECTORY)
    ok_v = raw["speed"].between(*SPEED_RANGE)
    ok_g = raw[GRADE_COL].between(*GRADE_RANGE)
    clean = raw[ok_v & ok_g]

    out = {
        "source": str(C.RAW_TRAJECTORY.name),
        "cleaning_rule": {"speed_kmh": list(SPEED_RANGE),
                          "gradient_percent": list(GRADE_RANGE),
                          "gradient_column": GRADE_COL},
        "records": {
            "generated": int(len(raw)),
            "speed_out_of_range": int((~ok_v).sum()),
            "gradient_out_of_range": int((~ok_g).sum()),
            "violating_both": int(((~ok_v) & (~ok_g)).sum()),
            "removed_union": int((~(ok_v & ok_g)).sum()),
            "removed_fraction": float((~(ok_v & ok_g)).mean()),
            "retained": int(len(clean)),
        },
        "anomaly_rate": {
            "generated": float((raw["anomaly"] != 0).mean()),
            "cleaned": float((clean["anomaly"] != 0).mean()),
        },
        "subtypes": {},
        "cleaning_conflict": {},
        "rule_sufficiency": {},
    }

    for t in range(1, 10):
        g = int((raw["anomaly"] == t).sum())
        c = int((clean["anomaly"] == t).sum())
        out["subtypes"][str(t)] = {
            "generated": g, "cleaned": c, "removed": g - c,
            "removed_fraction": float((g - c) / g) if g else 0.0,
            "name": C.ANOMALY_NAMES.get(t, str(t)),
        }

    # ------------------------------------------- uretim/temizleme catismasi
    removed_by_speed = raw.loc[~ok_v, "anomaly"].value_counts().to_dict()
    worst = max(out["subtypes"].items(),
                key=lambda kv: kv[1]["removed_fraction"])
    t6 = raw["anomaly"] == 6
    out["cleaning_conflict"] = {
        "removed_by_speed_filter_by_subtype": {str(k): int(v)
                                               for k, v in removed_by_speed.items()},
        "worst_hit_subtype": worst[0],
        "worst_hit": worst[1],
        "subtype6_speed_range_kmh": [float(raw.loc[t6, "speed"].min()),
                                     float(raw.loc[t6, "speed"].max())],
        "note": ("uretici imkansiz hiz enjekte ediyor, temizleyici imkansiz "
                 "hizi siliyor; iki kural ayni kayitlar uzerinde calisiyor"),
    }

    # ------------------------------------------- tip 1 kuralinin yeterliligi
    over = clean["speed"] > SPEED_LIMIT_FACTOR * clean["maxspeed"]
    lab1 = clean["anomaly"] == 1
    others = clean.loc[over & ~lab1, "anomaly"].value_counts().to_dict()
    out["rule_sufficiency"]["subtype1"] = {
        "rule": "v_t > %.2f * v_max" % SPEED_LIMIT_FACTOR,
        "records_satisfying_rule": int(over.sum()),
        "labelled_subtype1": int((over & lab1).sum()),
        "labelled_other": {str(k): int(v) for k, v in others.items()},
        "labelled_none": int(((clean["anomaly"] == 0) & over).sum()),
        "subtype1_not_satisfying_rule": int((lab1 & ~over).sum()),
        "excess_if_rule_applied_literally":
            float((over.sum() - (over & lab1).sum()) / max(int((over & lab1).sum()), 1)),
        "note": ("kural GEREK ama YETER degil; eksik kosul yayinlanan veriden "
                 "geri cikarilamadi"),
    }

    # ------------------------------------------- ekrana ozet
    r = out["records"]
    print("KAYITLAR")
    print("  uretilen %d | hiz disi %d | egim disi %d | ikisi birden %d"
          % (r["generated"], r["speed_out_of_range"],
             r["gradient_out_of_range"], r["violating_both"]))
    print("  silinen (birlesim) %d (%.2f%%) -> kalan %d"
          % (r["removed_union"], 100 * r["removed_fraction"], r["retained"]))
    print("\nANOMALI ORANI  uretilen %.4f  temizlenmis %.4f"
          % (out["anomaly_rate"]["generated"], out["anomaly_rate"]["cleaned"]))
    print("\nALT TIPLER")
    print("  %-3s %-26s %8s %8s %9s" % ("no", "ad", "uretilen", "temiz", "kayip"))
    for t, d in out["subtypes"].items():
        flag = "  <<<" if d["removed_fraction"] > 0.2 else ""
        print("  %-3s %-26s %8d %8d %8.1f%%%s"
              % (t, d["name"][:26], d["generated"], d["cleaned"],
                 100 * d["removed_fraction"], flag))
    cc = out["cleaning_conflict"]
    print("\nKURAL CATISMASI")
    print("  hiz filtresinin sildikleri alt tipe gore: %s"
          % cc["removed_by_speed_filter_by_subtype"])
    print("  en cok etkilenen: tip %s, %%%.0f kayip, hiz araligi %.1f-%.1f km/h"
          % (cc["worst_hit_subtype"], 100 * cc["worst_hit"]["removed_fraction"],
             cc["subtype6_speed_range_kmh"][0], cc["subtype6_speed_range_kmh"][1]))
    rs = out["rule_sufficiency"]["subtype1"]
    print("\nTIP 1 KURALININ YETERLILIGI")
    print("  kurali saglayan %d | tip1 %d | baska tip %s | etiketsiz %d"
          % (rs["records_satisfying_rule"], rs["labelled_subtype1"],
             rs["labelled_other"], rs["labelled_none"]))
    print("  tip1 olup kurali saglamayan: %d (0 olmali)"
          % rs["subtype1_not_satisfying_rule"])
    print("  kural birebir uygulanirsa fazladan pozitif: %%%.1f"
          % (100 * rs["excess_if_rule_applied_literally"]))

    save_json(out, OUT)
    print("\nkaydedildi: %s" % OUT)


if __name__ == "__main__":
    main()
