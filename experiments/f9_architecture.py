"""
Mimari figuru (Sekil 1) — yeniden uretilir.

NEDEN: Onceki surumden devralinan raster figur "fixed threshold 0.50" ve
tek bir fuzyon operatoru gosteriyordu. Revize makale esigi VALIDATION
uzerinde seciyor ve fuzyon blogunu deneysel degisken olarak ele aliyor.
Figur ile metin celisiyordu; figur metne gore yeniden cizildi.

Cikti: makale/stanet_framework_q1.png (ayni dosya adi, ayni en-boy orani)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from stanet.utils import init_console

init_console()
OUT = Path(__file__).resolve().parent.parent / "makale" / "stanet_framework_q1.png"

BLUE = ("#2563eb", "#eff6ff")
GREEN = ("#047857", "#ecfdf5")
PURPLE = ("#7c3aed", "#f5f3ff")
CRIMSON = ("#be123c", "#fff1f2")
SLATE = ("#475569", "#f8fafc")
INK = "#0f172a"


def box(ax, x, y, w, h, style, title, lines, title_size=15, body_size=11,
        pad=0.007, header=False):
    edge, face = style
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.020",
        linewidth=2.2, edgecolor=edge, facecolor=face, zorder=2))
    cx = x + w / 2
    if header:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=3)
        return
    ty = y + h - 0.030
    ax.text(cx, ty, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK, zorder=3)
    n = len(lines)
    if not n:
        return
    top = ty - 0.034
    step = (top - y - 0.016) / max(n, 1)
    for i, ln in enumerate(lines):
        ax.text(cx, top - (i + 0.55) * step, ln, ha="center", va="center",
                fontsize=body_size, color=INK, zorder=3)


def arrow(ax, p0, p1, color="#2563eb", rad=0.0, lw=2.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=18, linewidth=lw,
        color=color, zorder=1,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    ax.set_xlim(0, 1); ax.set_ylim(0.02, 1); ax.axis("off")

    C1X, C1W = 0.015, 0.280          # sutun 1
    C2X, C2W = 0.362, 0.272          # sutun 2
    C3X, C3W = 0.700, 0.285          # sutun 3

    # ------------------------------------------------------------ basliklar
    box(ax, C1X, 0.930, C1W, 0.055, BLUE, "1) Data Inputs", [], header=True)
    box(ax, C2X, 0.930, C2W, 0.055, GREEN, "2) Parallel Encoders", [], header=True)
    box(ax, C3X, 0.930, C3W, 0.055, PURPLE,
        "3) Fusion, Prediction & Evidence", [], header=True, title_size=13.5)

    # ------------------------------------------------------------ sutun 1
    box(ax, C1X, 0.690, C1W, 0.195, BLUE, "Trajectory Window", [
        r"$\mathbf{X} \in \mathbb{R}^{50 \times 4}$",
        r"$[\,v_t,\ a_t,\ \sin\theta_t,\ \cos\theta_t\,]$",
        r"$\tau = 50$ s window, stride 10",
    ])
    box(ax, C1X, 0.435, C1W, 0.215, BLUE, "OSM Road Graph", [
        r"$\mathcal{G} = (\mathcal{V}, \mathcal{E})$, 17 node features",
        "trip-instance: 4,678 nodes / 53,660 edges",
        "osm-merged: 628 nodes / 699 edges",
        "both constructions evaluated",
    ], body_size=10)

    # ------------------------------------------------------------ sutun 2
    box(ax, C2X, 0.690, C2W, 0.195, GREEN, "Temporal Encoder", [
        r"2 layers $\cdot$ hidden 128 $\cdot$ dropout 0.2",
        r"$\mathbf{h}_{temp} \in \mathbb{R}^{128}$",
        "benchmarked against GRU, gated TCN,",
        "dilated causal conv., Transformer",
    ], body_size=10)
    box(ax, C2X, 0.435, C2W, 0.215, GREEN, "Spatial Encoder", [
        r"GAT $17 \rightarrow 64$ (2 heads) $\rightarrow 64$ (1 head)",
        r"$\alpha_{ij}$: graph-neighborhood attention",
        r"$\mathbf{h}_{spat} \in \mathbb{R}^{64}$",
        "benchmarked against Chebyshev, diffusion,",
        "adaptive adjacency, and no spatial branch",
    ], body_size=10)

    # ------------------------------------------------------------ sutun 3
    box(ax, C3X, 0.760, C3W, 0.125, PURPLE, "Shared-Space Projection", [
        r"temporal $128 \rightarrow 64$   $\cdot$   spatial $64 \rightarrow 64$",
        r"$\mathbf{h}'_{temp},\ \mathbf{h}'_{spat} \in \mathbb{R}^{64}$",
    ], body_size=10.5)
    box(ax, C3X, 0.520, C3W, 0.200, PURPLE, "Fusion Operator", [
        r"$\mathbf{gated}$:  $\mathbf{z} = g\,\mathbf{h}'_{temp} + (1-g)\,\mathbf{h}'_{spat}$",
        r"$g = \sigma(\mathbf{W}_g[\mathbf{h}'_{temp} \Vert \mathbf{h}'_{spat}] + b_g)$",
        r"$\mathbf{weighted\ sum}$: one learnable scalar",
        r"$\mathbf{concatenation}$   $\cdot$   $\mathbf{cross}$-$\mathbf{attention}$",
        "the experimental variable of this study",
    ], body_size=10)
    box(ax, C3X, 0.345, C3W, 0.135, CRIMSON, "Prediction Head", [
        r"MLP $64 \rightarrow 32 \rightarrow 1$ $\cdot$ dropout 0.2",
        r"sigmoid anomaly probability $\hat{y}$",
    ], body_size=10.5)
    box(ax, C3X, 0.100, C3W, 0.205, CRIMSON, "Reported Evidence", [
        r"fixed split: mean $\pm$ s.d. over 5 seeds",
        "cross-validation: 5 folds x 2 seeds",
        "trip-level cluster bootstrap + permutation",
        "Holm-corrected family-wise error rate",
    ], body_size=10)

    # ------------------------------------------------------------ protokol
    box(ax, C1X, 0.100, C2X + C2W - C1X, 0.290, SLATE,
        "Training and Decision Protocol", [
        "identical for every variant compared in this paper",
        r"weighted BCE, $w_{pos} \approx 11.65$ from the training split only"
        r"   $\cdot$   Adam, lr $= 10^{-3}$   $\cdot$   batch 64",
        r"ReduceLROnPlateau   $\cdot$   max 50 epochs, patience 15   "
        r"$\cdot$   checkpoint and early stopping on validation F1",
        r"threshold $\mathbf{selected\ on\ validation}$ (argmax validation F1), "
        r"then applied unchanged to the test split",
        "",
        "every result file carries a protocol hash; a configuration that appears",
        "in several tables is trained once and read by all of them",
    ], title_size=14, body_size=10)

    # ------------------------------------------------------------ oklar
    g1 = (C1X + C1W, C2X)
    g2 = (C2X + C2W, C3X)
    arrow(ax, (g1[0] + 0.006, 0.788), (g1[1] - 0.006, 0.788))
    arrow(ax, (g1[0] + 0.006, 0.543), (g1[1] - 0.006, 0.543))
    arrow(ax, (g2[0] + 0.006, 0.788), (g2[1] - 0.006, 0.845), color="#047857", rad=-0.12)
    arrow(ax, (g2[0] + 0.006, 0.600), (g2[1] - 0.006, 0.800), color="#047857", rad=-0.16)
    cx3 = C3X + C3W / 2
    arrow(ax, (cx3, 0.755), (cx3, 0.727), color="#7c3aed")
    arrow(ax, (cx3, 0.515), (cx3, 0.487), color="#7c3aed")
    arrow(ax, (cx3, 0.340), (cx3, 0.312), color="#be123c")
    arrow(ax, (g2[0] + 0.006, 0.235), (g2[1] - 0.006, 0.235), color="#64748b", lw=1.6)

    fig.text(0.5, 0.045,
             r"Terminology: $\alpha_{ij}$ weights GAT neighbors within the graph; "
             r"$g$ weights the complete temporal and spatial embeddings against "
             r"each other. The two are distinct mechanisms.",
             ha="center", fontsize=11, color="#334155")

    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"kaydedildi: {OUT}")


if __name__ == "__main__":
    main()