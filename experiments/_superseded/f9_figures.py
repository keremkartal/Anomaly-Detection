"""
F9 — Makale figürlerinin yeniden üretimi.

Tüm figürler TEK sonuç setinden üretilir; makaledeki mevcut figür/sayı
karışıklığı (iki farklı koşunun sayılarının aynı tabloda olması) böylece biter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)

from stanet import config as C
from stanet.utils import init_console, load_json

init_console()
plt.rcParams.update({"font.size": 11, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})
PALETTE = {"stanet": "#2E5EAA", "concat": "#D1495B", "lstm": "#00798C",
           "rule": "#8B8B8B", "ml": "#EDAE49", "other": "#66A182"}


def save(fig, name):
    p = C.FIG_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {p.name}")
    return p


def fig_baseline_comparison():
    """F1: kural / klasik ML / derin model karşılaştırması."""
    r = load_json(C.RESULTS_DIR / "F1_zemin.json")["models"]
    order = [("rule_dwell_only", "Kural: hız<1", "rule"),
             ("rule_dwell_accel", "Kural: hız+ivme", "rule"),
             ("rule_best", "Kural: hız+ivme+donma", "rule"),
             ("logistic_reg", "Lojistik regresyon", "ml"),
             ("random_forest", "Random Forest", "ml"),
             ("xgboost", "XGBoost", "ml"),
             ("lightgbm", "LightGBM", "ml"),
             ("stanet", "STANet (önerilen)", "stanet")]
    names = [o[1] for o in order]
    f1 = [r[o[0]]["f1"] for o in order]
    cols = [PALETTE[o[2]] for o in order]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    b = ax.barh(names, f1, color=cols, edgecolor="black", linewidth=0.5)
    ax.bar_label(b, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("F1-Score (test seti, eşik validation'da seçildi)")
    ax.set_xlim(0, 1.0)
    ax.axvline(r["stanet"]["f1"], color=PALETTE["stanet"], ls="--", lw=1, alpha=0.6)
    ax.set_title("Zemin ölçümü: öğrenmesiz kurallar ve klasik ML karşısında STANet")
    ax.invert_yaxis()
    return save(fig, "fig_F1_baselines.png")


def fig_roc_pr():
    """F1: ROC ve PR eğrileri."""
    d = np.load(C.CACHE_DIR / "F1_predictions.npz")
    y = d["y_test"]
    keys = [("stanet", "STANet", PALETTE["stanet"]),
            ("lightgbm", "LightGBM", PALETTE["ml"]),
            ("xgboost", "XGBoost", PALETTE["other"]),
            ("random_forest", "Random Forest", PALETTE["concat"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, lbl, c in keys:
        if k not in d:
            continue
        p = d[k]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=c, lw=1.8,
                     label=f"{lbl} (AUC={np.trapz(tpr, fpr):.4f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, color=c, lw=1.8,
                     label=f"{lbl} (AP={average_precision_score(y, p):.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    axes[1].axhline(y.mean(), color="gray", ls=":", lw=1,
                    label=f"rastgele ({y.mean():.3f})")
    for a in axes:
        a.legend(fontsize=8, loc="lower left" if a is axes[1] else "lower right")
    fig.tight_layout()
    return save(fig, "fig_F1_roc_pr.png")


def fig_fusion_ablation():
    """F3: füzyon ablasyonu, 5 seed mean±std."""
    p = C.RESULTS_DIR / "F3_fusion.json"
    if not p.exists():
        return None
    r = load_json(p)
    labels = {"gated": "Gated\n(önerilen)", "concat": "Concatenation",
              "weighted_sum": "Weighted sum", "cross_attention": "Cross-attention"}
    order = r["ranking"]
    means = [r["variants"][f]["aggregate"]["f1"]["mean"] for f in order]
    stds = [r["variants"][f]["aggregate"]["f1"]["std"] for f in order]
    cols = [PALETTE["stanet"] if f == "gated" else PALETTE["other"] for f in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(order))
    axes[0].bar(x, means, yerr=stds, capsize=5, color=cols,
                edgecolor="black", linewidth=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m + s + 0.008, f"{m:.4f}", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[f] for f in order], fontsize=9)
    axes[0].set_ylabel("F1-Score")
    axes[0].set_title(f"Füzyon ablasyonu ({len(r['config']['seeds'])} seed, ort±std)")
    axes[0].set_ylim(0, max(np.array(means) + np.array(stds)) * 1.18)

    for f in order:
        v = [run["f1"] for run in r["variants"][f]["runs"]]
        axes[1].scatter([labels[f]] * len(v), v, s=34, alpha=0.75,
                        color=PALETTE["stanet"] if f == "gated" else PALETTE["other"])
    axes[1].set_ylabel("F1-Score")
    axes[1].set_title("Seed başına dağılım")
    axes[1].tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    return save(fig, "fig_F3_fusion_ablation.png")


def fig_gate_distribution():
    """F3: gate değerlerinin sınıfa göre dağılımı — 'adaptif mi?' sorusu."""
    p = C.RESULTS_DIR / "F3_fusion.json"
    if not p.exists():
        return None
    r = load_json(p)
    runs = r["variants"]["gated"]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    seeds = [x["seed"] for x in runs]
    lo = [x["gate"]["min"] for x in runs]
    hi = [x["gate"]["max"] for x in runs]
    mn = [x["gate"]["mean_normal"] for x in runs]
    ma = [x["gate"]["mean_anomaly"] for x in runs]

    xs = np.arange(len(seeds))
    axes[0].vlines(xs, lo, hi, color="gray", lw=6, alpha=0.35, label="gözlenen aralık")
    axes[0].scatter(xs, mn, color=PALETTE["lstm"], s=48, zorder=3, label="normal (ort)")
    axes[0].scatter(xs, ma, color=PALETTE["concat"], s=48, zorder=3, label="anomali (ort)")
    axes[0].axhline(0.5, color="k", ls="--", lw=0.9, alpha=0.6,
                    label="0,5 (eşit ağırlık)")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    axes[0].set_ylabel(r"$g_{gate}$")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Modality gate: aralık ve sınıf ortalamaları")
    axes[0].legend(fontsize=8, loc="lower right")

    delta = np.array(ma) - np.array(mn)
    axes[1].bar(xs, delta, color=PALETTE["stanet"], edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="k", lw=0.9)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    axes[1].set_ylabel(r"$g_{gate}$(anomali) − $g_{gate}$(normal)")
    axes[1].set_title("Sınıfa bağlı kayma (adaptiflik göstergesi)")
    for i, dv in enumerate(delta):
        axes[1].text(i, dv, f"{dv:+.4f}", ha="center",
                     va="bottom" if dv >= 0 else "top", fontsize=8)
    fig.tight_layout()
    return save(fig, "fig_F3_gate.png")


def fig_benchmark():
    """F4: mimari benchmark (yeni sema: res['configs'])."""
    p = C.RESULTS_DIR / "F4_benchmark.json"
    if not p.exists():
        return None
    r = load_json(p)
    rows = []
    for name, v in r["configs"].items():
        a = v["aggregate"]
        rows.append((name, a["f1"]["mean"], a["f1"]["std"], v["origin"], v["axis"]))
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cols = [PALETTE["stanet"] if n == "T:lstm" else
            (PALETTE["ml"] if ax_ == "temporal" else PALETTE["other"])
            for n, _, _, _, ax_ in rows]
    b = ax.barh([f"{n}" for n, _, _, _, _ in rows], [x[1] for x in rows],
                xerr=[x[2] for x in rows], capsize=4, color=cols,
                edgecolor="black", linewidth=0.5)
    ax.bar_label(b, fmt="%.4f", padding=3, fontsize=8)
    for i, (_, _, _, org, _) in enumerate(rows):
        ax.text(0.01, i, f"  {org}", va="center", fontsize=7.5, color="white")
    ax.set_xlabel("F1-Score (3 seed, ort±std)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Mimari benchmark: STGCN / DCRNN / Graph WaveNet cekirdek operatorleri")
    fig.tight_layout()
    return save(fig, "fig_F4_benchmark.png")


def fig_subtype():
    """F5: alt-tip stratifikasyonu — uzamsal katkının olduğu yer."""
    p = C.RESULTS_DIR / "F5_cv.json"
    if not p.exists():
        return None
    r = load_json(p)
    kb = r["karar_B"]
    types = [k for k in kb if k != "verdict"]
    if not types:
        return None
    x = np.arange(len(types))
    st = [kb[t]["hybrid_f1"] for t in types]
    ls = [kb[t]["lstm_only_f1"] for t in types]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.36
    b1 = ax.bar(x - w / 2, st, w, label="Hibrit (uzamsal dal var)",
                color=PALETTE["stanet"], edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, ls, w, label="LSTM-only (uzamsal dal yok)",
                color=PALETTE["lstm"], edgecolor="black", linewidth=0.5)
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={kb[t]['n_pos']})" for t in types], fontsize=9)
    ax.set_ylabel("F1-Score (5-fold havuzlanmış)")
    ax.set_title("Uzamsal bağlam hangi anomali tipinde katkı sağlıyor?")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return save(fig, "fig_F5_subtype.png")


def fig_calibration():
    """F5: kalibrasyon eğrisi (makalede hiç yoktu)."""
    p = C.RESULTS_DIR / "F5_cv.json"
    if not p.exists():
        return None
    r = load_json(p)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="mükemmel kalibrasyon")
    for name, col in (("hybrid_gat", PALETTE["stanet"]),
                      ("lstm_only", PALETTE["lstm"])):
        if name not in r["models"]:
            continue
        cal = r["models"][name]["calibration"]
        cur = cal["reliability"]
        ax.plot([c["confidence"] for c in cur], [c["accuracy"] for c in cur],
                "o-", color=col, lw=1.6, ms=5,
                label=f"{name} (Brier={cal['brier']:.4f}, ECE={cal['ece']:.4f})")
    ax.set(xlabel="Ortalama tahmin olasılığı", ylabel="Gözlenen anomali oranı",
           title="Kalibrasyon (reliability diagram)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return save(fig, "fig_F5_calibration.png")


def fig_efficiency():
    """F6: gecikme ve önbellekleme kazancı."""
    p = C.RESULTS_DIR / "F6_efficiency.json"
    if not p.exists():
        return None
    r = load_json(p)["models"]
    names = list(r)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(names))
    w = 0.36
    gpu = [r[n].get("cuda_single_window", {}).get("p50_ms", np.nan) for n in names]
    gpuc = [r[n].get("cuda_single_window_cached", {}).get("p50_ms", np.nan) for n in names]
    axes[0].bar(x - w / 2, gpu, w, label="tam forward", color=PALETTE["concat"])
    axes[0].bar(x + w / 2, gpuc, w, label="graf önbellekli", color=PALETTE["stanet"])
    axes[0].set_ylabel("tek pencere gecikmesi p50 (ms)")
    axes[0].set_title("GPU")
    cpu = [r[n]["cpu_single_window"]["p50_ms"] for n in names]
    cpuc = [r[n]["cpu_single_window_cached"]["p50_ms"] for n in names]
    axes[1].bar(x - w / 2, cpu, w, label="tam forward", color=PALETTE["concat"])
    axes[1].bar(x + w / 2, cpuc, w, label="graf önbellekli", color=PALETTE["stanet"])
    axes[1].set_title("CPU (edge senaryosu)")
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
        a.legend(fontsize=8)
        a.set_yscale("log")
    fig.tight_layout()
    return save(fig, "fig_F6_efficiency.png")


def main():
    print("figürler üretiliyor...\n")
    made = []
    for fn in (fig_baseline_comparison, fig_roc_pr, fig_fusion_ablation,
               fig_gate_distribution, fig_benchmark, fig_subtype,
               fig_calibration, fig_efficiency):
        try:
            p = fn()
            if p:
                made.append(p.name)
        except FileNotFoundError as e:
            print(f"  atlandı ({fn.__name__}): {e}")
        except Exception as e:
            print(f"  HATA ({fn.__name__}): {type(e).__name__}: {e}")
    print(f"\n{len(made)} figür üretildi -> {C.FIG_DIR}")


if __name__ == "__main__":
    main()
