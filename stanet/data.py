"""
Trajektori veri hattı.

İki mod:
  legacy=True   -> orijinal makale hattını birebir üretir (scaler TÜM veriye fit;
                   sızıntılı, sadece mevcut checkpoint'leri doğrulamak için)
  legacy=False  -> scaler yalnızca train trip'lerine fit edilir (düzeltilmiş)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import config as C


# --------------------------------------------------------------------------- yükleme
def load_raw() -> pd.DataFrame:
    """Ham trajektori CSV'sini okur ve makale §IV-C temizlik filtrelerini uygular."""
    df = pd.read_csv(C.RAW_TRAJECTORY)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])

    n0 = len(df)
    lo, hi = C.SPEED_RANGE
    df = df[(df["speed"] >= lo) & (df["speed"] <= hi)].copy()
    n_speed = n0 - len(df)

    lo, hi = C.PERCENT_RANGE
    df = df[(df["percent"] >= lo) & (df["percent"] <= hi)].copy()
    n_percent = n0 - n_speed - len(df)

    df.attrs["cleaning"] = {"initial": n0, "dropped_speed": n_speed,
                            "dropped_percent": n_percent, "remaining": len(df)}

    # yön açısını sürekli temsile çevir
    df["bearing"] = df["bearing"] % 360
    rad = np.radians(df["bearing"])
    df["bearing_sin"] = np.sin(rad)
    df["bearing_cos"] = np.cos(rad)

    df["anomaly_type"] = df["anomaly"].astype(int)
    df["anomaly_binary"] = (df["anomaly"] > 0).astype(int)
    return df


# --------------------------------------------------------------------------- pencereleme
def build_windows(df: pd.DataFrame):
    """
    Trip bazında kayan pencere. Etiket ve meta pencerenin SON adımından alınır
    (orijinal hatla aynı).

    Döner: X (N,tau,4), y (N,), meta (N,3), atype (N,)
    """
    X, y, meta, atype = [], [], [], []
    for _, g in df.groupby("trip_id"):
        g = g.sort_values("time")
        feats = g[C.FEATURE_COLS].values
        targ = g[C.TARGET_COL].values
        mets = g[C.META_COLS].values
        types = g["anomaly_type"].values
        n = len(g)
        for i in range(0, n - C.SEQUENCE_LENGTH + 1, C.STEP_SIZE):
            last = i + C.SEQUENCE_LENGTH - 1
            X.append(feats[i:i + C.SEQUENCE_LENGTH])
            y.append(targ[last])
            meta.append(mets[last])
            atype.append(types[last])
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            np.asarray(meta),
            np.asarray(atype, dtype=np.int64))


# --------------------------------------------------------------------------- split
def split_trips(unique_trips, seed=C.SPLIT_SEED):
    """Makale ile birebir aynı 70/15/15 trip bölmesi."""
    train, temp = train_test_split(unique_trips, test_size=C.TEST_SIZE,
                                   random_state=seed, shuffle=True)
    val, test = train_test_split(temp, test_size=C.VAL_TEST_SPLIT,
                                 random_state=seed, shuffle=True)
    return train, val, test


@dataclass
class Dataset:
    X_train: np.ndarray; y_train: np.ndarray; meta_train: np.ndarray; type_train: np.ndarray
    X_val: np.ndarray;   y_val: np.ndarray;   meta_val: np.ndarray;   type_val: np.ndarray
    X_test: np.ndarray;  y_test: np.ndarray;  meta_test: np.ndarray;  type_test: np.ndarray
    scalers: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def pos_weight(self) -> float:
        pos = float(self.y_train.sum())
        return (len(self.y_train) - pos) / pos


def _fit_scalers(speed, accel):
    ss = MinMaxScaler().fit(speed.reshape(-1, 1))
    sa = StandardScaler().fit(accel.reshape(-1, 1))
    return {"speed": ss, "accel": sa}


def _apply_scalers(df, scalers):
    df = df.copy()
    df["speed_norm"] = scalers["speed"].transform(df["speed"].values.reshape(-1, 1))
    df["accel_norm"] = scalers["accel"].transform(df["accel"].values.reshape(-1, 1))
    return df


def build_dataset(legacy: bool = False, seed: int = C.SPLIT_SEED) -> Dataset:
    """
    legacy=True  : scaler tüm veriye fit (orijinal, sızıntılı)
    legacy=False : scaler yalnızca train trip'lerine fit (düzeltilmiş)
    """
    df = load_raw()
    cleaning = df.attrs.get("cleaning", {})
    trips = np.unique(df["trip_id"].values)
    tr_trips, va_trips, te_trips = split_trips(trips, seed=seed)

    if legacy:
        scalers = _fit_scalers(df["speed"].values, df["accel"].values)
    else:
        m = df["trip_id"].isin(tr_trips)
        scalers = _fit_scalers(df.loc[m, "speed"].values, df.loc[m, "accel"].values)

    df = _apply_scalers(df, scalers)
    X, y, meta, atype = build_windows(df)

    tr = np.isin(meta[:, 0], tr_trips)
    va = np.isin(meta[:, 0], va_trips)
    te = np.isin(meta[:, 0], te_trips)

    ds = Dataset(
        X[tr], y[tr], meta[tr], atype[tr],
        X[va], y[va], meta[va], atype[va],
        X[te], y[te], meta[te], atype[te],
        scalers=scalers,
        info={
            "legacy": legacy, "split_seed": seed, "cleaning": cleaning,
            "n_sequences": int(len(y)), "anomaly_ratio_all": float(y.mean()),
            "trips": {"train": len(tr_trips), "val": len(va_trips), "test": len(te_trips)},
            "sequences": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
            "anomaly_ratio": {"train": float(y[tr].mean()), "val": float(y[va].mean()),
                              "test": float(y[te].mean())},
        },
    )
    return ds


# --------------------------------------------------------------------------- Group K-fold
def build_cv_folds(n_folds: int = C.N_FOLDS, legacy: bool = False):
    """
    Trip-bazlı K-fold. Her fold için scaler YALNIZCA o fold'un train'ine fit edilir.
    Döner: fold listesi; her eleman bir Dataset (val = test fold'un yarısı).
    """
    df_raw = load_raw()
    cleaning = df_raw.attrs.get("cleaning", {})
    trips_all = df_raw["trip_id"].values
    unique_trips = np.unique(trips_all)

    gkf = GroupKFold(n_splits=n_folds)
    dummy = np.zeros(len(unique_trips))
    folds = []

    for k, (tr_idx, te_idx) in enumerate(gkf.split(dummy, groups=unique_trips)):
        te_trips = unique_trips[te_idx]
        tr_pool = unique_trips[tr_idx]
        # train havuzundan val ayır (fold'a göre deterministik)
        tr_trips, va_trips = train_test_split(
            tr_pool, test_size=0.1765, random_state=C.SPLIT_SEED + k, shuffle=True)

        if legacy:
            scalers = _fit_scalers(df_raw["speed"].values, df_raw["accel"].values)
        else:
            m = df_raw["trip_id"].isin(tr_trips)
            scalers = _fit_scalers(df_raw.loc[m, "speed"].values, df_raw.loc[m, "accel"].values)

        df = _apply_scalers(df_raw, scalers)
        X, y, meta, atype = build_windows(df)
        tr = np.isin(meta[:, 0], tr_trips)
        va = np.isin(meta[:, 0], va_trips)
        te = np.isin(meta[:, 0], te_trips)

        folds.append(Dataset(
            X[tr], y[tr], meta[tr], atype[tr],
            X[va], y[va], meta[va], atype[va],
            X[te], y[te], meta[te], atype[te],
            scalers=scalers,
            info={"fold": k, "legacy": legacy, "cleaning": cleaning,
                  "trips": {"train": len(tr_trips), "val": len(va_trips), "test": len(te_trips)},
                  "sequences": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
                  "anomaly_ratio": {"train": float(y[tr].mean()), "val": float(y[va].mean()),
                                    "test": float(y[te].mean())}},
        ))
    return folds


# --------------------------------------------------------------------------- yardımcı
def inverse_kinematics(X: np.ndarray, scalers: dict):
    """Normalize pencerelerden gerçek hız/ivme değerlerini geri çıkarır (kural baseline için)."""
    flat_s = X[..., 0].reshape(-1, 1)
    flat_a = X[..., 1].reshape(-1, 1)
    speed = scalers["speed"].inverse_transform(flat_s).reshape(X.shape[:-1])
    accel = scalers["accel"].inverse_transform(flat_a).reshape(X.shape[:-1])
    return speed, accel


def type_distribution(y: np.ndarray, atype: np.ndarray) -> dict:
    """Pozitif örneklerin alt-tip dağılımı."""
    pos = atype[y == 1]
    u, c = np.unique(pos, return_counts=True)
    return {C.ANOMALY_NAMES.get(int(k), str(k)): int(v) for k, v in zip(u, c)}
