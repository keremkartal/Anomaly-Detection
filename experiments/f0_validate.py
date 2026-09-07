"""
F0 kabul testi: yeni kod tabanı, mevcut checkpoint'lerden makaledeki
test sayılarını yeniden üretiyor mu?

Beklenen (makale Tablo 12):
  GAT (Spatial Only)   acc .7456  P .0461  R .3439  F1 .0813  AUC .5828
  LSTM (Temporal Only) acc .9869  P .7937  R .8089  F1 .8013  AUC .8807
  STANet (Proposed)    acc .9883  P .7988  R .8599  F1 .8282  AUC .9814*
  *notebook çıktısı 0.9639 — makale ile tutarsız (bkz. yolharitasi §4 R-06)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from stanet import config as C
from stanet.data import build_dataset, type_distribution
from stanet.evaluate import compute_metrics, metrics_by_type, predict
from stanet.graph import build_graph, map_segments
from stanet.models import FusionHybrid, GATOnly, LSTMOnly
from stanet.train import SeqDataset, make_loaders
from stanet.utils import count_parameters, get_device, init_console, save_json
from torch.utils.data import DataLoader

init_console()

PAPER = {
    "gat_only":  dict(accuracy=.7456, precision=.0461, recall=.3439, f1=.0813, roc_auc=.5828),
    "lstm_only": dict(accuracy=.9869, precision=.7937, recall=.8089, f1=.8013, roc_auc=.8807),
    "stanet":    dict(accuracy=.9883, precision=.7988, recall=.8599, f1=.8282, roc_auc=.9639),
}


def main():
    device = get_device()
    print(f"cihaz: {device}\n")

    ds = build_dataset(legacy=True)          # orijinal (sızıntılı) hat
    graph = build_graph(merge_by_osm=False)  # orijinal graf
    idx_test, missing = map_segments(ds.meta_test, graph)
    print(f"test segment eşleşmesi: {len(idx_test)-missing}/{len(idx_test)} "
          f"(eşleşmeyen {missing})")

    node_x, edge_index = graph.to_tensors(device)
    loader = DataLoader(SeqDataset(ds.X_test, ds.y_test, idx_test),
                        batch_size=C.BATCH_SIZE, shuffle=False)

    results = {"dataset_info": ds.info, "graph_info": graph.info,
               "test_type_distribution": type_distribution(ds.y_test, ds.type_test),
               "models": {}}

    specs = [
        ("stanet", FusionHybrid(fusion_type="gated", num_node_features=graph.num_features,
                                legacy_head=True), C.LEGACY_HYBRID_CKPT),
        ("lstm_only", LSTMOnly(), C.LEGACY_LSTM_CKPT),
        ("gat_only", GATOnly(num_node_features=graph.num_features, hidden_dim=64),
         C.LEGACY_GNN_CKPT),
    ]

    # Bu bolum, ILK SURUMUN checkpoint'lerini yukleyip o zamanki sayilari
    # yeniden uretmeye yarar. Checkpoint'ler depoya dahil degildir (egitim
    # ciktisi, kaynak degil), bu yuzden temiz bir klonda atlanir. Veri ve graf
    # butunlugu kontrolleri checkpoint'lerden bagimsizdir ve her zaman calisir.
    missing = [str(c) for _, _, c in specs if not Path(c).exists()]
    if missing:
        print("\n" + "=" * 70)
        print("ESKI CHECKPOINT'LER YOK — legacy dogrulama bolumu atlaniyor.")
        print("Bu BEKLENEN durumdur; checkpoint'ler depoya dahil degil.")
        print("Veri ve graf kontrolleri yukarida tamamlandi.")
        print("Makalenin sonuclari icin README'deki F3/F4/F2/F1b/F5 sirasini")
        print("izleyin; o betikler checkpoint'e ihtiyac duymaz.")
        print("=" * 70)
        results["legacy_validation"] = {"skipped": True, "missing": missing}
        save_json(results, C.RESULTS_DIR / "F0_validation.json")
        print(f"\nkaydedildi: {C.RESULTS_DIR / 'F0_validation.json'}")
        return

    for name, model, ckpt in specs:
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd)          # strict=True — uyumsuzluk hata verir
        model = model.to(device).eval()

        yt, yp, gw = predict(model, loader, node_x, edge_index, device)
        m = compute_metrics(yt, yp)
        m["params"] = count_parameters(model)

        if not np.all(np.isnan(gw)):
            m["gate"] = {
                "min": float(np.nanmin(gw)), "max": float(np.nanmax(gw)),
                "mean": float(np.nanmean(gw)), "std": float(np.nanstd(gw)),
                "mean_normal": float(np.nanmean(gw[yt == 0])),
                "mean_anomaly": float(np.nanmean(gw[yt == 1])),
                "frac_above_half": float((gw > 0.5).mean()),
            }
        m["by_type"] = metrics_by_type(yt, yp, ds.type_test)
        results["models"][name] = m

        print(f"\n=== {name} ===")
        exp = PAPER[name]
        print(f"{'metrik':10s} {'yeni kod':>10s} {'makale':>10s} {'fark':>9s}")
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            d = m[k] - exp[k]
            flag = "OK" if abs(d) < 0.002 else "FARK"
            print(f"{k:10s} {m[k]:10.4f} {exp[k]:10.4f} {d:+9.4f}  {flag}")
        print(f"parametre: {m['params']['total']:,}")
        if "gate" in m:
            g = m["gate"]
            print(f"gate: [{g['min']:.3f}, {g['max']:.3f}] ort={g['mean']:.3f} "
                  f"normal={g['mean_normal']:.3f} anomali={g['mean_anomaly']:.3f}")

    save_json(results, C.RESULTS_DIR / "F0_validation.json")
    print(f"\nkaydedildi: {C.RESULTS_DIR / 'F0_validation.json'}")


if __name__ == "__main__":
    main()
