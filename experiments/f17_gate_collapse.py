"""
F17 — Kapi cokmesinin MEKANIZMASI (degerlendirme: cokme neden oluyor).

Makale "onerilen ornek-bazli skaler kapi bes tohumdan birinde cokuyor"
diyor (F3, tohum 45: F1 = 0,0715) ve cokmenin, kapinin tek basina neredeyse
hic sinyal tasimayan uzamsal dala kilitlenmesiyle olustugunu ONE SURUYOR.
Bu bir yorumdu; egitim seyrinden gosterilmemisti. Burada gosteriliyor.

NE OLCULUYOR
------------
Her epoch sonunda, dogrulama kumesi uzerinde `eval()` ve `no_grad()` altinda:

  * kapi degerlerinin dagilimi (min, ortalama, maks, standart sapma)
  * kapinin uc noktalara yigilmasi: g < 0,02 ve g > 0,98 oranlari
  * siniflar arasi ayrim: normal pencerelerin ortalama kapisi ile anomali
    pencerelerinin ortalama kapisi arasindaki fark
  * dogrulama F1 ve kaybi

Ayrica her epoch'ta, KIRPMADAN ONCEKI gradyan normunun ortalamasi ve
kirpilan yigin orani kaydedilir. Cokme bir gradyan patlamasiyla mi yoksa
sessiz bir doygunlukla mi geliyor sorusunu bu ayirir.

Olcum egitimi DEGISTIRMEZ: `eval()` + `no_grad()` altinda rastgelelik
tuketilmez, gradyan normu zaten hesaplanan bir degerdir. Yine de bu kosular
ONBELLEKTE AYRI bir anahtara yazilir (`probe=gate`), boylece makaledeki F3
degerleri oldugu gibi kalir. Iki kosu arasindaki fark GPU determinizmsizligi
kadardir ve F15'te olculmustur.

IKI GRAF
--------
F3, cokmenin trip-orneklemeli grafta olustugunu ve OSM-birlestirilmis grafta
kaybolduğunu bildirdi. Iki graf da kosulur: eger cokme yalnizca uzamsal dalin
zayif oldugu grafta oluyorsa, mekanizma iddiasi dogrulanmis olur.

KULLANIM
  python experiments/f17_gate_collapse.py
  python experiments/f17_gate_collapse.py --seeds 45      # yalnizca coken tohum
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from stanet import config as C
from stanet.data import build_dataset
from stanet.evaluate import compute_metrics, pick_threshold, predict
from stanet.graph import build_graph, map_segments
from stanet.runstore import _build, _dirs, make_spec, spec_key
from stanet.train import train_model
from stanet.utils import (get_device, init_console, load_json, protocol_stamp,
                          save_json, set_seed)

init_console()
OUT = C.RESULTS_DIR / "F17_gate_collapse.json"
COLLAPSE = 0.30
FUSIONS = ["gated"]                       # kapisi olan tek operator
GRAPHS = [False, True]                    # trip_instance, osm_merged
EXTREME_LO, EXTREME_HI = 0.02, 0.98


def gate_probe(model, loader, node_x, edge_index, device):
    """Epoch sonu kapi dagilimi. eval() + no_grad() cagiran taraftadir."""
    was_training = model.training
    model.eval()
    gates, labels = [], []
    for x_b, y_b, i_b in loader:
        x_b, i_b = x_b.to(device), i_b.to(device)
        _, g = model(x_b, node_x, edge_index, i_b)
        if g is None:
            model.train(was_training)
            return {}
        gates.append(np.asarray(g.detach().cpu()).ravel())
        labels.append(np.asarray(y_b).ravel())
    model.train(was_training)
    g = np.concatenate(gates)
    y = np.concatenate(labels)
    if np.all(np.isnan(g)):
        return {}
    out = {
        "gate_min": float(np.nanmin(g)), "gate_max": float(np.nanmax(g)),
        "gate_mean": float(np.nanmean(g)), "gate_sd": float(np.nanstd(g)),
        "gate_frac_below_%.2f" % EXTREME_LO: float(np.nanmean(g < EXTREME_LO)),
        "gate_frac_above_%.2f" % EXTREME_HI: float(np.nanmean(g > EXTREME_HI)),
    }
    if (y == 1).any() and (y == 0).any():
        out["gate_mean_normal"] = float(np.nanmean(g[y == 0]))
        out["gate_mean_anomaly"] = float(np.nanmean(g[y == 1]))
        out["gate_class_delta"] = out["gate_mean_anomaly"] - out["gate_mean_normal"]
    return out


def instrumented_run(spec, seed, ds, node_idx, graph, device):
    """Olculu kosu; kendi onbellek anahtarini kullanir, F3'u bozmaz."""
    d, _ = _dirs()
    key = spec_key(spec, seed)
    jf = d / (key + ".json")
    if jf.exists():
        meta = load_json(jf)
        print("  [onbellek] %s: F1=%.4f" % (key, meta["metrics"]["f1"]), flush=True)
        return meta["metrics"], meta["history"]

    set_seed(seed)
    model = _build(spec, graph)
    out = train_model(model, ds, node_idx, graph, seed=seed, device=device,
                      select_by="val_f1", verbose=False, tag=key,
                      probe=gate_probe)
    nx_, ei_ = graph.to_tensors(device)
    yv, pv, _ = predict(out["model"], out["loaders"]["val"], nx_, ei_, device)
    yt, pt, gw = predict(out["model"], out["loaders"]["test"], nx_, ei_, device)
    th = pick_threshold(yv, pv)
    m = compute_metrics(yt, pt, th)
    m.update({"threshold_val": float(th), "best_epoch": out["best_epoch"],
              "epochs_run": len(out["history"]), "seed": seed,
              "total_params": out["params"]["total"]})
    save_json({"spec": spec, "seed": seed, "protocol": protocol_stamp(),
               "metrics": m, "history": out["history"]}, jf)
    print("  %s: F1=%.4f (ep %d/%d)"
          % (key, m["f1"], m["best_epoch"], m["epochs_run"]), flush=True)
    del model, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return m, out_history_safe(jf)


def out_history_safe(jf):
    return load_json(jf)["history"]


def describe(hist, metrics, f3_f1=None):
    """Bir kosunun cokup cokmedigini ve kapinin ne yaptigini ozetler."""
    collapsed = metrics["f1"] < COLLAPSE
    lo_key = "gate_frac_below_%.2f" % EXTREME_LO
    hi_key = "gate_frac_above_%.2f" % EXTREME_HI
    have = [h for h in hist if "gate_mean" in h]
    d = {"collapsed": bool(collapsed), "f1": float(metrics["f1"]),
         "best_epoch": metrics["best_epoch"], "epochs_run": metrics["epochs_run"],
         "f1_in_F3": f3_f1, "n_epochs_with_gate": len(have)}
    if have:
        first, last = have[0], have[-1]
        d.update({
            "gate_mean_first_epoch": first["gate_mean"],
            "gate_mean_last_epoch": last["gate_mean"],
            "gate_sd_last_epoch": last["gate_sd"],
            "gate_extreme_frac_last": (last.get(lo_key, 0.0) + last.get(hi_key, 0.0)),
            "gate_class_delta_last": last.get("gate_class_delta"),
            # kapi hangi epoch'ta uc noktaya yigildi
            "epoch_gate_saturated": next(
                (h["epoch"] for h in have
                 if (h.get(lo_key, 0.0) + h.get(hi_key, 0.0)) > 0.95), None),
        })
    gn = [h["grad_norm_mean"] for h in hist if h.get("grad_norm_mean") is not None]
    if gn:
        d.update({"grad_norm_first": float(gn[0]), "grad_norm_last": float(gn[-1]),
                  "grad_norm_max": float(np.max(gn)),
                  "grad_clipped_frac_last": hist[-1].get("grad_clipped_frac")})
    return d


def main():
    seeds = C.SEEDS
    if "--seeds" in sys.argv:
        seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
    device = get_device()
    stamp = protocol_stamp()
    ds = build_dataset(legacy=False)
    print("protokol %s | cihaz %s | tohumlar %s\n" % (stamp["hash"], device, seeds),
          flush=True)

    try:
        f3 = load_json(C.RESULTS_DIR / "F3_fusion.json")
        f3_seeds = f3["config"]["seeds"]
        f3_f1 = {s: f3["variants"]["gated"]["runs"][i]["f1"]
                 for i, s in enumerate(f3_seeds)}
    except Exception:
        f3_f1 = {}

    results = {"protocol": stamp, "seeds": seeds,
               "measurement": {"where": "validation set, eval() + no_grad()",
                               "extremes": [EXTREME_LO, EXTREME_HI],
                               "collapse_threshold": COLLAPSE,
                               "note": ("olcum egitimi degistirmez; kosular ayri "
                                        "onbellek anahtarina yazilir, F3 degerleri "
                                        "korunur")},
               "runs": {}}

    for merge in GRAPHS:
        graph = build_graph(merge_by_osm=merge)
        gtag = "osm_merged" if merge else "trip_instance"
        node_idx = {sp: map_segments(getattr(ds, "meta_" + sp), graph)[0]
                    for sp in ("train", "val", "test")}
        print("GRAF %s — %d dugum, %d kenar"
              % (gtag, graph.num_nodes, int(graph.edge_index.shape[1])), flush=True)
        for fusion in FUSIONS:
            for seed in seeds:
                spec = make_spec(merge_by_osm=merge, temporal="lstm", spatial="gat",
                                 fusion=fusion, probe="gate")
                m, hist = instrumented_run(spec, seed, ds, node_idx, graph, device)
                results["runs"].setdefault(gtag, {}).setdefault(fusion, {})[str(seed)] = {
                    "summary": describe(hist, m,
                                        f3_f1.get(seed) if not merge else None),
                    "history": hist,
                }
        print(flush=True)

    # ------------------------------------------------------------------ ozet
    print("=" * 104)
    print("KAPI COKMESI — epoch bazinda olculmus")
    print("=" * 104)
    print("  %-13s %5s %7s %8s %9s %9s %10s %9s %9s"
          % ("graf", "tohum", "F1", "cokme", "kapi ilk", "kapi son",
             "uc yigilma", "doygun ep", "gr.norm"))
    for gtag, per_f in results["runs"].items():
        for fusion, per_s in per_f.items():
            for seed, r in sorted(per_s.items(), key=lambda kv: int(kv[0])):
                s = r["summary"]
                print("  %-13s %5s %7.4f %8s %9s %9s %10s %9s %9s"
                      % (gtag, seed, s["f1"], "EVET" if s["collapsed"] else "-",
                         _f(s.get("gate_mean_first_epoch")),
                         _f(s.get("gate_mean_last_epoch")),
                         _f(s.get("gate_extreme_frac_last")),
                         s.get("epoch_gate_saturated") or "-",
                         _f(s.get("grad_norm_last"))))

    coll = [(g, s) for g, pf in results["runs"].items() for f, ps in pf.items()
            for s, r in ps.items() if r["summary"]["collapsed"]]
    results["collapsed_runs"] = [{"graph": g, "seed": s} for g, s in coll]
    print("\n  coken kosular: %s" % (", ".join("%s/s%s" % (g, s) for g, s in coll)
                                     or "yok"))
    by_graph = {}
    for g, pf in results["runs"].items():
        n = sum(1 for f, ps in pf.items() for r in ps.values()
                if r["summary"]["collapsed"])
        by_graph[g] = {"n_collapsed": n,
                       "n_runs": sum(len(ps) for ps in pf.values())}
        print("  %-13s %d/%d kosu cokuyor" % (g, n, by_graph[g]["n_runs"]))
    results["collapse_by_graph"] = by_graph

    save_json(results, OUT)
    print("\nkaydedildi: %s" % OUT)


def _f(v, fmt="%.4f"):
    return "-" if v is None else (fmt % v)


if __name__ == "__main__":
    main()
