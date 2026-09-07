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
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
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
    """
    Tablo 9 ile AYNI kaynaktan (F1b_fair.json) uretilir.

    v1'de bu figur `F1_zemin.json`'dan uretiliyordu: tek kosu, sizintili
    scaler'li eski hat. Tablo 9 ise `F1b_fair.json`'dan geliyordu. Ayni
    modeller iki yerde farkli degerler gosterdi; hakem bunu isaretledi.
    Artik tek kaynak var ve cok-seed std hata cubugu olarak ciziliyor.
    """
    src = C.RESULTS_DIR / "F1b_fair.json"
    d = load_json(src)
    r = d["models"]
    order = [("rule_dwell_only", "Rule: v<1", "rule"),
             ("rule_dwell_accel", "Rule: v<1 or |a|>3", "rule"),
             ("rule_best", "Rule: 3 thresholds", "rule"),
             ("random_forest", "Random Forest", "ml"),
             ("xgboost", "XGBoost", "ml"),
             ("lightgbm", "LightGBM", "ml"),
             ("lstm_only", "LSTM only (no spatial)", "lstm"),
             ("hybrid_lstm_gat", "Hybrid LSTM-GAT", "stanet")]
    order = [o for o in order if o[0] in r]
    names = [o[1] for o in order]
    f1 = [r[o[0]]["aggregate"]["f1"]["mean"] for o in order]
    err = [r[o[0]]["aggregate"]["f1"].get("std", 0.0) for o in order]
    cols = [PALETTE[o[2]] for o in order]
    ref = r["hybrid_lstm_gat"]["aggregate"]["f1"]["mean"]
    n_seed = len(d["config"]["seeds"])

    fig, ax = plt.subplots(figsize=(8, 4.4))
    b = ax.barh(names, f1, xerr=err, color=cols, edgecolor="black",
                linewidth=0.5, error_kw={"ecolor": "black", "capsize": 3, "lw": 1})
    ax.bar_label(b, labels=[f"{v:.4f}" for v in f1], padding=14, fontsize=9)
    ax.set_xlabel(f"F1-Score (test set, threshold selected on validation; "
                  f"mean $\pm$ s.d. over {n_seed} seeds)")
    ax.set_xlim(0, 1.05)
    ax.axvline(ref, color=PALETTE["stanet"], ls="--", lw=1, alpha=0.6)
    ax.set_title("Reference points: rule-based and classical models vs. the hybrid")
    ax.invert_yaxis()
    return save(fig, "fig_F1_baselines.png")


def fig_roc_pr():
    """ROC ve PR egrileri — kaynak F1b (Tablo 9 ile ayni kosular)."""
    d = np.load(C.CACHE_DIR / "F1b_predictions.npz")
    y = d["y_test"]
    keys = [("hybrid_lstm_gat_median", "Hybrid LSTM-GAT", PALETTE["stanet"]),
            ("lstm_only_median", "LSTM only", PALETTE["lstm"]),
            ("lightgbm_median", "LightGBM", PALETTE["ml"]),
            ("xgboost_median", "XGBoost", PALETTE["other"]),
            ("random_forest_median", "Random Forest", PALETTE["concat"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, lbl, c in keys:
        if k not in d.files:
            continue
        pr_ = d[k]
        fpr, tpr, _ = roc_curve(y, pr_)
        axes[0].plot(fpr, tpr, color=c, lw=1.8,
                     label=f"{lbl} (AUC={auc(fpr, tpr):.4f})")
        pr, rc, _ = precision_recall_curve(y, pr_)
        axes[1].plot(rc, pr, color=c, lw=1.8,
                     label=f"{lbl} (AP={average_precision_score(y, pr_):.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    axes[1].axhline(y.mean(), color="gray", ls=":", lw=1,
                    label=f"chance ({y.mean():.3f})")
    for a in axes:
        a.legend(fontsize=8, loc="lower left" if a is axes[1] else "lower right")
    fig.suptitle("Median-seed predictions from the same runs reported in Table 9",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    return save(fig, "fig_F1_roc_pr.png")


def fig_fusion_ablation():
    """
    Fuzyon ablasyonu — SABIT BOLME ve CAPRAZ DOGRULAMA yan yana.

    Makalenin merkezi bulgusu bu figurde gorunur olmali: gated operator
    sabit bolmede sonuncu ve bir tohumda cokuyor, bes bolme uzerinde ise
    birinci ve hic cokmuyor. Siralamanin kendisi bolmeye bagli.
    """
    p3 = C.RESULTS_DIR / "F3_fusion.json"
    p5 = C.RESULTS_DIR / "F5_cv.json"
    if not p3.exists():
        return None
    r3 = load_json(p3)
    labels = {"gated": "Gated\n(proposed)", "concat": "Concatenation",
              "weighted_sum": "Weighted sum", "cross_attention": "Cross-attention"}
    cv_key = {"gated": "stanet_gated", "concat": "hybrid_concat",
              "weighted_sum": "hybrid_gat", "cross_attention": "hybrid_crossatt"}
    order = r3["ranking"]
    col = lambda f: PALETTE["stanet"] if f == "gated" else PALETTE["other"]

    r5 = load_json(p5) if p5.exists() else None
    ncol = 2 if r5 else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.0 * ncol, 4.6), squeeze=False)
    axes = axes[0]

    def panel(ax, values, title, sub):
        x = np.arange(len(order))
        means = [float(np.mean(values[f])) for f in order]
        stds = [float(np.std(values[f], ddof=1)) for f in order]
        ax.bar(x, means, yerr=stds, capsize=5, color=[col(f) for f in order],
               edgecolor="black", linewidth=0.5, alpha=0.85, zorder=2)
        for i, f in enumerate(order):
            v = values[f]
            ax.scatter(np.full(len(v), i) + np.linspace(-0.14, 0.14, len(v)), v,
                       s=22, color="black", alpha=0.55, zorder=3)
            ax.text(i, means[i] + stds[i] + 0.03, f"{means[i]:.4f}",
                    ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[f] for f in order], fontsize=9)
        ax.set_ylabel("F1-Score")
        ax.set_title(f"{title}\n{sub}", fontsize=10.5)
        ax.set_ylim(0, 1.18)

    v3 = {f: [run["f1"] for run in r3["variants"][f]["runs"]] for f in order}
    panel(axes[0], v3, "Fixed partition",
          f"{len(r3['config']['seeds'])} seeds, 14 trips  "
          f"(gated ranks {r3['gated_rank']}/4)")

    if r5:
        v5 = {f: [run["f1"] for run in r5["models"][cv_key[f]]["runs"]]
              for f in order if cv_key[f] in r5["models"]}
        if len(v5) == len(order):
            rank = r5["fusion_ablation_cv"]["gated_rank"]
            n = r5["fusion_ablation_cv"]["n_runs_per_variant"]
            panel(axes[1], v5, "Trip-grouped cross-validation",
                  f"{n} runs (5 folds x 2 seeds), 90 trips  "
                  f"(gated ranks {rank}/4)")
            axes[1].set_ylabel("")

    fig.suptitle("The fusion ranking is partition-dependent: bars are means, "
                 "dots are individual runs", fontsize=10, y=1.00)
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
    axes[0].vlines(xs, lo, hi, color="gray", lw=6, alpha=0.35, label="observed range")
    axes[0].scatter(xs, mn, color=PALETTE["lstm"], s=48, zorder=3, label="normal (mean)")
    axes[0].scatter(xs, ma, color=PALETTE["concat"], s=48, zorder=3, label="anomalous (mean)")
    axes[0].axhline(0.5, color="k", ls="--", lw=0.9, alpha=0.6,
                    label="0.5 (equal weight)")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    axes[0].set_ylabel(r"$g_{gate}$")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Modality gate: range and class means")
    axes[0].legend(fontsize=8, loc="lower right")

    delta = np.array(ma) - np.array(mn)
    axes[1].bar(xs, delta, color=PALETTE["stanet"], edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="k", lw=0.9)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    axes[1].set_ylabel(r"$g_{gate}$(anomalous) $-$ $g_{gate}$(normal)")
    axes[1].set_title("Class-conditional shift (adaptivity indicator)")
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
    ax.set_xlabel("F1-Score (3 seeds, mean$\pm$sd)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Encoder benchmark: STGCN / DCRNN / Graph WaveNet core operators")
    fig.tight_layout()
    return save(fig, "fig_F4_benchmark.png")


def fig_subtype():
    """F5: alt-tip stratifikasyonu — uzamsal katkının olduğu yer."""
    p = C.RESULTS_DIR / "F5_cv.json"
    if not p.exists():
        return None
    r = load_json(p)
    kb = r["karar_B"]
    # v2: karar_B artik "verdict" ve "cluster_info" gibi alt-tip olmayan
    # anahtarlar da tasiyor; yalnizca gercek alt-tip girdilerini al.
    types = [k for k, v in kb.items()
             if isinstance(v, dict) and "hybrid_f1" in v]
    if not types:
        return None
    x = np.arange(len(types))
    st = [kb[t]["hybrid_f1"] for t in types]
    ls = [kb[t]["lstm_only_f1"] for t in types]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.36
    b1 = ax.bar(x - w / 2, st, w, label="Hybrid (with spatial branch)",
                color=PALETTE["stanet"], edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, ls, w, label="LSTM-only (no spatial branch)",
                color=PALETTE["lstm"], edgecolor="black", linewidth=0.5)
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    # Turkce alt-tip adlarini Ingilizceye cevir (makale Ingilizce)
    EN = {"hiz_asimi": "Speed-limit violation",
          "uzun_duraklama": "Abnormal dwell",
          "sinyal_donmasi": "Signal freeze",
          "anormal_ivme": "Abnormal acceleration",
          "kaza": "Crash",
          "anormal_yon_degisimi": "Abnormal heading change"}
    ax.set_xticklabels(
        [f"{EN.get(t, t.replace('_', ' '))}\n(n={kb[t]['n_pos']}, "
         f"Holm $p$={kb[t]['holm']['p_holm']:.3f})" for t in types], fontsize=9)
    ax.set_ylabel("F1-Score (5-fold pooled)")
    ax.set_title("Where does spatial context contribute? "
                 "(trip-level tests over 90 trips, Holm-corrected)", fontsize=11)
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
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    # v2: dort fuzyonun tamami + kontrol, Ingilizce etiketlerle
    SERIES = [("hybrid_gat", "Weighted-sum hybrid", PALETTE["stanet"]),
              ("stanet_gated", "Gated hybrid", PALETTE["concat"]),
              ("hybrid_concat", "Concatenation hybrid", PALETTE["ml"]),
              ("hybrid_crossatt", "Cross-attention hybrid", PALETTE["other"]),
              ("lstm_only", "LSTM only (control)", PALETTE["lstm"])]
    for name, lbl, col in SERIES:
        if name not in r["models"]:
            continue
        cal = r["models"][name]["calibration"]
        cur = cal["reliability"]
        ax.plot([c["confidence"] for c in cur], [c["accuracy"] for c in cur],
                "o-", color=col, lw=1.6, ms=4.5,
                label=f"{lbl} (Brier {cal['brier']:.4f}, ECE {cal['ece']:.4f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed anomaly frequency",
           title="Calibration (pooled cross-validated predictions)")
    ax.legend(fontsize=7.5, loc="upper left")
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
    axes[0].bar(x - w / 2, gpu, w, label="full forward", color=PALETTE["concat"])
    axes[0].bar(x + w / 2, gpuc, w, label="graph-cached", color=PALETTE["stanet"])
    axes[0].set_ylabel("single-window latency p50 (ms)")
    axes[0].set_title("GPU")
    cpu = [r[n]["cpu_single_window"]["p50_ms"] for n in names]
    cpuc = [r[n]["cpu_single_window_cached"]["p50_ms"] for n in names]
    axes[1].bar(x - w / 2, cpu, w, label="full forward", color=PALETTE["concat"])
    axes[1].bar(x + w / 2, cpuc, w, label="graph-cached", color=PALETTE["stanet"])
    axes[1].set_title("CPU (edge scenario)")
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

    # LaTeX kaynak dizinine kopyala. Figürler iki yerde tutulduğu için
    # elle kopyalamak, düzeltmeye çalıştığımız senkron hatasının aynısıdır;
    # bu yüzden kopyalama üretimin bir parçası.
    import shutil
    tex_dir = C.ROOT / "makale"
    if tex_dir.is_dir():
        for name in made:
            shutil.copy2(C.FIG_DIR / name, tex_dir / name)
        print(f"{len(made)} figür kopyalandı -> {tex_dir}")


if __name__ == "__main__":
    main()
