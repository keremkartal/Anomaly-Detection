"""
F6 — Verimlilik ve edge profili (Hakem 2, Madde 5).

"The authors need to report information such as the number of model parameters,
single-window inference time, and performance on CPUs or edge devices."

Ölçülenler:
  - parametre sayısı (toplam + dal bazında)
  - tek pencere (batch=1) gecikmesi: GPU ve CPU, p50/p95/p99
  - throughput (pencere/sn) farklı batch boyutlarında
  - bellek kullanımı
  - GRAF ÖNBELLEKLEME etkisi: GAT forward'ı her batch'te tüm düğümler üzerinde
    koşuyor; düğüm gömmeleri çıkarımda BİR KEZ hesaplanıp saklanabilir
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from stanet import config as C
from stanet.graph import build_graph
from stanet.hybrid import GenericHybrid
from stanet.utils import (get_device, init_console, protocol_stamp, save_json,
                          set_seed)

init_console()
# v2: kodlayici satirlari F4 benchmark ile AYNI fuzyonu (weighted_sum) kullanir,
# boylece gecikme tablosu dogruluk tablosuyla ayni konfigurasyonlari olcer.
# Gated varyant, fuzyon operatorunun gecikme maliyetini gostermek icin ayrica
# olculur (F3'un birincil karsilastirmasi).
CONFIGS = [
    ("STANet (weighted sum, GAT)", dict(temporal="lstm", spatial="gat",
                                        fusion="weighted_sum")),
    ("STANet (gated, GAT)", dict(temporal="lstm", spatial="gat", fusion="gated")),
    ("concat fusion", dict(temporal="lstm", spatial="gat", fusion="concat")),
    ("cross-attention fusion", dict(temporal="lstm", spatial="gat",
                                    fusion="cross_attention")),
    ("LSTM-only", dict(temporal="lstm", spatial="none", fusion="weighted_sum")),
    ("GRU temporal", dict(temporal="gru", spatial="gat", fusion="weighted_sum")),
    ("STGCN temporal", dict(temporal="stgcn_temporal", spatial="gat",
                            fusion="weighted_sum")),
    ("WaveNet temporal", dict(temporal="wavenet_temporal", spatial="gat",
                              fusion="weighted_sum")),
    ("Transformer temporal", dict(temporal="transformer_temporal", spatial="gat",
                                  fusion="weighted_sum")),
    ("DCRNN diffusion", dict(temporal="lstm", spatial="diffusion",
                             fusion="weighted_sum")),
    ("STGCN cheb", dict(temporal="lstm", spatial="cheb", fusion="weighted_sum")),
    ("GraphWaveNet adaptive", dict(temporal="lstm", spatial="adaptive",
                                   fusion="weighted_sum")),
]
N_WARMUP, N_REPEAT = 50, 500


@torch.no_grad()
def latency(model, x, nx, ei, si, device, repeats=N_REPEAT):
    model.eval()
    for _ in range(N_WARMUP):
        model(x, nx, ei, si)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x, nx, ei, si)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ts)
    return {"mean_ms": float(a.mean()), "std_ms": float(a.std()),
            "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "min_ms": float(a.min())}


@torch.no_grad()
def latency_cached(model, x, nx, ei, si, device, repeats=N_REPEAT):
    """Düğüm gömmeleri önceden hesaplanmış hâl (gerçekçi dağıtım senaryosu)."""
    model.eval()
    emb = model.spatial(nx, ei, model._cache)          # bir kez

    def step():
        e_t = model.temporal(x)
        e_s = emb[si]
        p_t = torch.relu(model.project_t(e_t))
        p_s = torch.relu(model.project_s(e_s))
        if model.fusion_type == "gated":
            w = model.attention_gate(torch.cat([p_t, p_s], 1))
            fused = w * p_t + (1 - w) * p_s
        elif model.fusion_type == "concat":
            fused = torch.cat([p_t, p_s], 1)
        elif model.fusion_type == "weighted_sum":
            s = torch.sigmoid(model.static_weight)
            fused = s * p_t + (1 - s) * p_s
        else:
            q = p_t.unsqueeze(1)
            kv = torch.stack([p_t, p_s], 1)
            fused = model.cross_attn(q, kv, kv)[0].squeeze(1)
        return model.classifier(fused)

    for _ in range(N_WARMUP):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ts)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


@torch.no_grad()
def throughput(model, nx, ei, device, batch_sizes=(1, 32, 64, 256)):
    model.eval()
    out = {}
    for b in batch_sizes:
        x = torch.randn(b, C.SEQUENCE_LENGTH, 4, device=device)
        si = torch.randint(0, nx.size(0), (b,), device=device)
        for _ in range(10):
            model(x, nx, ei, si)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 30
        for _ in range(n):
            model(x, nx, ei, si)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out[f"batch_{b}"] = {"windows_per_sec": float(b * n / dt),
                             "ms_per_batch": float(dt / n * 1000)}
    return out


MERGE = False    # v2: birincil ayar (trip-instance). Onceki surumde gecikme
                 # YANLISLIKLA 628 dugumlu birlestirilmis grafta olculmustu;
                 # dogruluk tablolari ise 4.678 dugumlu grafi kullaniyordu.
                 # Buyuk graf uzamsal dalin maliyetini artirir, dolayisiyla
                 # eski sayilar birincil konfigurasyonu oldugundan ucuz
                 # gosteriyordu.


def main():
    graph = build_graph(merge_by_osm=MERGE)
    results = {"protocol": protocol_stamp(),
               "config": {"seeds": list(C.SEEDS), "merge_by_osm": MERGE,
                          "graph_tag": "osm_merged" if MERGE else "trip_instance"},
               "graph": graph.info, "n_warmup": N_WARMUP, "n_repeat": N_REPEAT,
               "torch_version": torch.__version__, "models": {}}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    results["gpu"] = gpu_name
    print(f"GPU: {gpu_name} | torch {torch.__version__}")
    print(f"graf: {graph.info['num_nodes']} dugum, {graph.info['num_edges']} kenar\n")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.insert(0, torch.device("cuda"))

    for label, cfg in CONFIGS:
        set_seed(42)
        entry = {"config": cfg}
        for device in devices:
            model = GenericHybrid(num_node_features=graph.num_features,
                                  num_nodes=graph.num_nodes, **cfg).to(device)
            nx_, ei_ = graph.to_tensors(device)
            x1 = torch.randn(1, C.SEQUENCE_LENGTH, 4, device=device)
            si1 = torch.randint(0, graph.num_nodes, (1,), device=device)

            if "params" not in entry:
                entry["params_total"] = sum(p.numel() for p in model.parameters())
                entry["params_by_branch"] = model.branch_parameters()
                entry["params_fusion"] = model.fusion_parameters()

            d = device.type
            rep = N_REPEAT if d == "cuda" else max(60, N_REPEAT // 8)
            entry[f"{d}_single_window"] = latency(model, x1, nx_, ei_, si1, device, rep)
            entry[f"{d}_single_window_cached"] = latency_cached(model, x1, nx_, ei_,
                                                                si1, device, rep)
            entry[f"{d}_throughput"] = throughput(model, nx_, ei_, device)
            if d == "cuda":
                torch.cuda.reset_peak_memory_stats()
                model(torch.randn(64, C.SEQUENCE_LENGTH, 4, device=device), nx_, ei_,
                      torch.randint(0, graph.num_nodes, (64,), device=device))
                torch.cuda.synchronize()
                entry["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["models"][label] = entry
        g = entry.get("cuda_single_window", {})
        gc = entry.get("cuda_single_window_cached", {})
        c = entry["cpu_single_window"]
        cc = entry["cpu_single_window_cached"]
        print(f"{label:24s} {entry['params_total']:>9,}p | "
              f"GPU {g.get('p50_ms', float('nan')):6.2f}ms "
              f"(onbellekli {gc.get('p50_ms', float('nan')):5.2f}) | "
              f"CPU {c['p50_ms']:7.2f}ms (onbellekli {cc['p50_ms']:6.2f})", flush=True)

    # ---------------------------------------------------- ozet
    print("\n" + "=" * 78)
    print("ONBELLEKLEME KAZANCI (tek pencere, p50)")
    print("=" * 78)
    for label, e in results["models"].items():
        for d in ("cuda", "cpu"):
            k, kc = f"{d}_single_window", f"{d}_single_window_cached"
            if k in e:
                sp = e[k]["p50_ms"] / max(e[kc]["p50_ms"], 1e-9)
                e[f"{d}_cache_speedup"] = float(sp)
                print(f"  {label:24s} {d.upper():4s} {e[k]['p50_ms']:7.2f} -> "
                      f"{e[kc]['p50_ms']:6.2f} ms  ({sp:.1f}x)")

    save_json(results, C.RESULTS_DIR / "F6_efficiency.json")
    print(f"\nkaydedildi: {C.RESULTS_DIR / 'F6_efficiency.json'}")


if __name__ == "__main__":
    main()
