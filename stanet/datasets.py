"""
Veri kumesi kaydi — revizyon icin uretilen 36 yorunge kumesi.

NEDEN AYRI BIR MODUL
--------------------
Simdiye kadar tek bir veri kumesi vardi ve yollari `config.py` sabitti.
Revizyon icin iki sehir, iki imputasyon stratejisi, uc gurultu ve uc
yayginlik ayari uretildi. Bunlarin sonuclari KESINLIKLE ayni tabloya
karisamaz, bu yuzden secilen veri kumesinin adi protokol damgasina girer:
veri degisirse damga degisir, onbellek gecersizlesir, kosular yeniden
yapilir. Pencere/adim taramasinda kullanilan mekanizmanin aynisi.

ZORUNLU TEMIZLEME
-----------------
Yeni kumelerin README'si bir on isleme filtresi sart kosuyor. Filtre burada
`clean_new_dataset` olarak birebir uygulanir, ama iki notla:

  * `tunnel == 1 & anomaly != 9` satirlarini silmek, zaten %99,9 olan bir
    bagintiyi %100'e tamamliyor. `tunnel` bir GRAF DUGUM OZNITELIGI oldugu
    icin bu, uzamsal dala sizan deterministik bir ipucu demek. Filtreyi
    uyguluyoruz (verinin sahipleri sart kosmus) ama `drop_tunnel_feature`
    secenegi grafta o ozniteligi kaldirarak SIZINTISIZ kontrolu mumkun
    kiliyor. Ikisi birlikte raporlanmali.
  * `anomaly == 0 & speed > 1.1*maxspeed` satirlarini silmek, etiket
    kuralini tanim geregi tam dogru hale getiriyor. Bu da raporlanmali.

Olculen sizinti buyuklugu (bkz. F27): Kocaeli'de anomalilerin ~%60'i
yalnizca `tunnel` ozniteliginden okunabiliyor; Hamburg'da ~%1.
"""
from pathlib import Path

from . import config as C

ROOT = C.ROOT
NEW = (ROOT / "yeniveriseti" / "DATASETS_REVİSİON-20261003T203912Z-1-001"
       / "DATASETS_REVİSİON")
TEMPORAL = NEW / "TEMPORAL"
SPATIAL = NEW / "SPATIAL"

CITIES = ("kocaeli", "hamburg")
IMPUTATIONS = ("baseline", "knn")
NOISES = ("0_5", "1_0", "1_5")
PREVALENCES = ("1", "5", "15")


def _paths(city, imputation, noise, prevalence):
    """Bir (sehir, imputasyon, gurultu, yayginlik) hucresinin iki dosyasi."""
    if imputation == "baseline":
        tdir = TEMPORAL / f"{city}_baseline_temporal_data" / \
            f"{city}_baseline_noise_{noise}_anomaly_{prevalence}"
        sfile = SPATIAL / f"{city}_spatial_data" / \
            f"{city}_spatial_data_BASELINE" / f"{city}_GNN_DATA_BASELINE.csv"
    else:
        tdir = TEMPORAL / f"{city}_temporal_data" / \
            f"{city}_noise_{noise}_anomaly_{prevalence}"
        sfile = SPATIAL / f"{city}_spatial_data" / \
            f"{city}_spatial_data_KNN" / f"{city}_GNN_DATA_KNN.csv"
    return tdir / "sentetic_data.csv", sfile


def registry() -> dict:
    """{ad: {trajectory, graph, city, imputation, noise, prevalence}}"""
    out = {"original": {
        "trajectory": C.DATA_DIR / "lstm_data.csv",
        "graph": C.DATA_DIR / "GNN_DATA.csv",
        "city": "kocaeli", "imputation": "original",
        "noise": None, "prevalence": None, "new_format": False}}
    for city in CITIES:
        for imp in IMPUTATIONS:
            for n in NOISES:
                for p in PREVALENCES:
                    t, s = _paths(city, imp, n, p)
                    out[f"{city}_{imp}_n{n}_a{p}"] = {
                        "trajectory": t, "graph": s, "city": city,
                        "imputation": imp, "noise": n, "prevalence": p,
                        "new_format": True}
    return out


def available() -> dict:
    """Diskte gercekten bulunan kumeler."""
    return {k: v for k, v in registry().items()
            if Path(v["trajectory"]).exists() and Path(v["graph"]).exists()}


def use(name: str, drop_tunnel_feature: bool = False):
    """
    Secilen veri kumesine gecer. `config` sabitlerini degistirir, bu yuzden
    protokol damgasi da degisir ve kosular ayri onbellege yazilir.

    `drop_tunnel_feature=True` -> `tunnel` graf dugum ozniteliklerinden
    cikarilir. Yeni kumelerde `tunnel == 1` olan her satirin etiketi 9
    oldugu icin bu oznitelik uzamsal dala neredeyse mukemmel bir ipucu
    sizdiriyor; sizintisiz kontrol bu secenekle kosulur.
    """
    reg = registry()
    if name not in reg:
        raise KeyError(f"bilinmeyen veri kumesi: {name}")
    d = reg[name]
    C.RAW_TRAJECTORY = Path(d["trajectory"])
    C.GNN_DATA = Path(d["graph"])
    C.DATASET_NAME = name
    C.DATASET_NEW_FORMAT = bool(d["new_format"])
    C.DROP_TUNNEL_FEATURE = bool(drop_tunnel_feature)
    C.GNN_BIN_COLS = ([c for c in C.GNN_BIN_COLS_ALL if c != "tunnel"]
                      if drop_tunnel_feature else list(C.GNN_BIN_COLS_ALL))
    return d


def clean_new_dataset(df):
    """
    Yeni kumelerin README'sinde sart kosulan on isleme.

    Birebir uygulanir; kac satirin hangi kuraldan dustugu `attrs` icine
    yazilir ki makalede raporlanabilsin.
    """
    n0 = len(df)
    df = df.copy()
    # Sutunlarin bir kismi yoksa (ornegin denetim betigi yalnizca gerekli
    # sutunlari okuyorsa) o adim atlanir; filtrenin geri kalani calisir.
    if "bearing" in df.columns:
        df["bearing"] = df["bearing"] % 360
    if "percent" in df.columns:
        df["percent"] = df["percent"].clip(0, 100)

    n = len(df)
    df = df[df["speed"] >= 0]
    d_speed = n - len(df)

    n = len(df)
    df = df[~((df["anomaly"] == 0) & (df["speed"] > df["maxspeed"] * 1.1))]
    d_label = n - len(df)

    d_tunnel = 0
    if "tunnel" in df.columns:
        n = len(df)
        df = df[~((df["tunnel"] == 1) & (df["anomaly"] != 9))]
        d_tunnel = n - len(df)

    n = len(df)
    df = df[~(df["accel"].abs() > 15)]
    d_accel = n - len(df)

    df.attrs["cleaning"] = {
        "initial": int(n0), "remaining": int(len(df)),
        "dropped_negative_speed": int(d_speed),
        "dropped_unlabelled_over_limit": int(d_label),
        "dropped_tunnel_mismatch": int(d_tunnel),
        "dropped_extreme_accel": int(d_accel),
        "note": ("yeni kumelerin README'sinde sart kosulan filtre; "
                 "tunnel ve etiket kurallari karsi ornekleri SILIYOR, "
                 "duzeltmiyor"),
    }
    return df
