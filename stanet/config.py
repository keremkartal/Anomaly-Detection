"""Merkezi konfigürasyon. Tüm yollar ve hiperparametreler burada."""
from pathlib import Path

# ---------------------------------------------------------------- yollar
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "veriseti"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "_cache"

for _d in (RESULTS_DIR, FIG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_TRAJECTORY = DATA_DIR / "lstm_data.csv"
GNN_DATA = DATA_DIR / "GNN_DATA.csv"
LEGACY_SPLITS = DATA_DIR / "dataset_splits.npz"
LEGACY_HYBRID_CKPT = DATA_DIR / "best_f1_hybrid_model.pth"
LEGACY_LSTM_CKPT = DATA_DIR / "best_lstm_only_model.pth"
LEGACY_GNN_CKPT = DATA_DIR / "best_gnn_only_model.pth"

# ---------------------------------------------------------------- ön işleme
SPEED_RANGE = (0.0, 200.0)      # makale §IV-C temizlik filtresi
PERCENT_RANGE = (-50.0, 50.0)
SEQUENCE_LENGTH = 50            # tau
STEP_SIZE = 10
FEATURE_COLS = ["speed_norm", "accel_norm", "bearing_sin", "bearing_cos"]
META_COLS = ["trip_id", "segment_id", "time"]
TARGET_COL = "anomaly_binary"

# GNN düğüm öznitelikleri (makale Tablo 4 sırası)
GNN_NUM_COLS = ["length_log", "max_speed", "rarity_score",
                "betweenness", "closeness", "node_degree", "lanes"]
GNN_BIN_COLS = ["tunnel", "traffic_light"]

# ---------------------------------------------------------------- split
SPLIT_SEED = 42                 # orijinal makale ile aynı
TEST_SIZE = 0.30                # train_test_split birinci aşama
VAL_TEST_SPLIT = 0.50           # temp -> val/test
N_FOLDS = 5                     # F5 Group K-fold

# ---------------------------------------------------------------- eğitim
import os as _os

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
# Duman testi için: STANET_MAX_EPOCHS=2 STANET_PATIENCE=1 STANET_SEEDS=42
MAX_EPOCHS = int(_os.environ.get("STANET_MAX_EPOCHS", 50))    # makale protokolü
PATIENCE = int(_os.environ.get("STANET_PATIENCE", 15))        # makale protokolü
GRAD_CLIP = 1.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
DROPOUT = 0.2
THRESHOLD = 0.50                # varsayılan; F5'te val üzerinde seçilecek

# mimari
LSTM_INPUT_DIM = 4
LSTM_HIDDEN_DIM = 128
LSTM_LAYERS = 2
GAT_HIDDEN_DIM = 32
GAT_OUTPUT_DIM = 64
GAT_HEADS = 2
FUSION_DIM = 64

SEEDS = [int(s) for s in _os.environ.get("STANET_SEEDS", "42,43,44,45,46").split(",")]

# cuDNN determinizmi. KAPALI (gerekce: stanet.utils.set_seed dokumantasyonu).
# Bu deger protokol damgasina girer; degistirilirse tum onbellek gecersiz olur.
DETERMINISTIC = _os.environ.get("STANET_DETERMINISTIC", "0") == "1"

# Paylasilan kosu deposu — ayni konfigurasyon+seed+protokol bir kez egitilir.
RUNS_DIR = RESULTS_DIR / "_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
N_FOLDS = int(_os.environ.get("STANET_FOLDS", N_FOLDS))

# ---------------------------------------------------------------- anomali taksonomisi
# makale Tablo 6; eşikler veriden ampirik olarak doğrulandı
ANOMALY_NAMES = {
    0: "normal",
    1: "hiz_asimi",
    2: "uzun_duraklama",
    3: "anormal_ivme",
    4: "kaza",
    5: "anormal_yon_degisimi",
    6: "gps_hiz_artefakti",
    7: "gps_ivme_artefakti",
    8: "gps_yon_artefakti",
    9: "sinyal_donmasi",
}
# uzamsal dalın v_max bilgisi taşıdığı için katkı sağlaması beklenen tip
SPATIAL_RELEVANT_TYPES = [1]
