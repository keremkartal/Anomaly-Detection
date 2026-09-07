"""
Öğrenmesiz ve klasik makine öğrenmesi baseline'ları (F1).

Hepsi HAM kinematik değerler üzerinde çalışır (normalize edilmemiş), böylece
scaler sızıntısı tartışması bu baseline'ları etkilemez.
"""
import numpy as np

from . import config as C


# --------------------------------------------------------------------- kural
def rule_predictions(speed, accel, bearing_sin, bearing_cos, variant="best"):
    """
    Makale Tablo 6'daki üretim eşiklerinden türetilmiş sıfır-parametreli kural.
    `speed`, `accel` : (N, tau) ham değerler.

    Pencere etiketi son adımdan alındığı için kurallar da son adıma uygulanır;
    süreklilik gerektiren tipler için pencere geneli istatistiği kullanılır.
    """
    s_last, a_last = speed[:, -1], accel[:, -1]

    r_dwell = s_last < 1.0                       # tip 2: uzun duraklama
    r_accel = np.abs(a_last) > 3.0               # tip 3/7: anormal ivme
    r_freeze = speed[:, -5:].std(axis=1) < 1e-6  # tip 9: sinyal donması
    dtheta = np.abs(np.arctan2(bearing_sin[:, -1], bearing_cos[:, -1])
                    - np.arctan2(bearing_sin[:, -2], bearing_cos[:, -2]))
    dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
    r_head = dtheta > np.radians(10)             # tip 5: anormal yön değişimi

    variants = {
        "dwell_only": r_dwell,
        "dwell_accel": r_dwell | r_accel,
        "best": r_dwell | r_accel | r_freeze,
        "all_rules": r_dwell | r_accel | r_freeze | r_head,
    }
    return {k: v.astype(int) for k, v in variants.items()}[variant], variants


# --------------------------------------------------------------------- öznitelik
def window_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Pencere agregatları — ağaç modelleri için.
    X_raw: (N, tau, 4) ham [speed, accel, bearing_sin, bearing_cos]
    Döner: (N, 4*9 + 2) öznitelik
    """
    feats = []
    for c in range(X_raw.shape[2]):
        ch = X_raw[:, :, c]
        feats += [ch.mean(1), ch.std(1), ch.min(1), ch.max(1),
                  ch[:, -1], ch[:, 0], ch[:, -1] - ch[:, 0],
                  np.median(ch, axis=1), ch.max(1) - ch.min(1)]
    # son 10 adımın hız/ivme davranışı (süreklilik sinyali)
    feats += [X_raw[:, -10:, 0].mean(1), X_raw[:, -10:, 1].std(1)]
    return np.column_stack(feats).astype(np.float32)


def feature_names() -> list:
    ch_names = ["speed", "accel", "bsin", "bcos"]
    stats = ["mean", "std", "min", "max", "last", "first", "delta", "median", "range"]
    names = [f"{c}_{s}" for c in ch_names for s in stats]
    return names + ["speed_last10_mean", "accel_last10_std"]
