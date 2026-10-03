"""
F29 — kapi cokmesinin mekanizma figuru (degerlendirme notu: cokme nasil
oluyor, egitim egrisi ve kapi dagilimiyla gosterin).

NE OLCULDU, NE BULUNDU
----------------------
Ilk okumada cokmeyi "kapi sinif ayrimini ters yone suruyor" diye acikladik.
On kosunun tamami incelenince bu YANLIS cikti: `osm_merged` grafinda tohum 45
ve 46'nin ikinci epoch'taki sinif ayrimi -0,190 ve -0,338, yani coken
kosununkinden (-0,001) cok daha negatif, ama ikisi de 0,86 F1'e ulasiyor.
Isaret ayirt edici degil.

Ayirt edici olan sey KAPININ KACISI: fuzyon kapisinin ortalamasi 0,5'i gecip
temsili zamansal dal agirlikli hale getirebiliyor mu? On kosunun dokuzunda
geciyor (4-19. epoch arasi) ve hepsi normal F1'e ulasiyor. Gecemeyen tek kosu
0,381'de kaliyor ve F1'i 0,08'de donuyor — uzamsal dal tek basina neredeyse
hic sinyal tasimadigi icin orada kalmak cokme demek.

Figur bunu uc panelde gosterir; ucuncu panel iliskiyi on kosunun tamaminda
ozetler.

KULLANIM
  python experiments/f29_collapse_figure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stanet import config as C
from stanet.utils import init_console, load_json, save_json

init_console()
SRC = C.RESULTS_DIR / "F17_gate_collapse.json"
OUT_JSON = C.RESULTS_DIR / "F29_collapse_mechanism.json"
GRAPH = "trip_instance"
ESCAPE = 0.5

COL_IN, FULL_IN = 3.42, 6.78
plt.rcParams.update({
    "font.size": 8, "figure.dpi": 150,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.linewidth": 0.7, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "savefig.bbox": "standard",
})
RED, BLUE, GREY = "#C0392B", "#2E5EAA", "#9AA5B1"


def series(hist, key):
    e = [h["epoch"] for h in hist if h.get(key) is not None]
    v = [h[key] for h in hist if h.get(key) is not None]
    return np.array(e), np.array(v)


def main():
    d = load_json(SRC)
    runs = d["runs"][GRAPH]["gated"]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_IN, 2.35))

    for ax, key, title, ylab in (
            (axes[0], "val_f1", "(a) Validation F1", "F1"),
            (axes[1], "gate_mean", "(b) Mean gate value", r"$\bar{g}$")):
        for seed, rec in sorted(runs.items()):
            e, v = series(rec["history"], key)
            bad = rec["summary"]["collapsed"]
            ax.plot(e, v, color=RED if bad else GREY,
                    lw=1.6 if bad else 1.0, zorder=3 if bad else 1,
                    label=("seed %s (collapsed)" % seed) if bad else None)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
    axes[1].axhline(ESCAPE, color=BLUE, lw=1.0, ls="--", zorder=2)
    axes[1].set_ylim(0, 1)
    axes[1].text(0.97, 0.52, "temporal-dominant", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=6.4, color=BLUE)
    axes[0].legend(loc="upper left", framealpha=0.9, borderpad=0.3)

    # (c) butun kosular: kapinin ulastigi en yuksek deger ile nihai F1
    recs = []
    for g, per in d["runs"].items():
        for seed, rec in sorted(per["gated"].items()):
            h = rec["history"]
            gm = [(x["epoch"], x["gate_mean"]) for x in h
                  if x.get("gate_mean") is not None]
            if not gm:
                continue
            recs.append({
                "graph": g, "seed": seed,
                "max_gate": float(max(v for _, v in gm)),
                "epoch_crossed": next((e for e, v in gm if v > ESCAPE), None),
                "f1": float(rec["summary"]["f1"]),
                "collapsed": bool(rec["summary"]["collapsed"]),
                "epochs_run": rec["summary"]["epochs_run"],
            })
    axes[2].scatter([r["max_gate"] for r in recs], [r["f1"] for r in recs],
                    c=[RED if r["collapsed"] else BLUE for r in recs],
                    s=26, zorder=3, edgecolor="white", linewidth=0.5)
    axes[2].axvline(ESCAPE, color=BLUE, lw=1.0, ls="--", zorder=2)
    axes[2].set_title("(c) Escape determines outcome")
    axes[2].set_xlabel(r"highest $\bar{g}$ reached")
    axes[2].set_ylabel("final F1")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].annotate("never crosses", xy=(0.381, 0.076), xytext=(0.10, 0.30),
                 fontsize=6.4, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    fig.tight_layout(pad=0.35)
    out = C.FIG_DIR / "fig_F17_collapse.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("kaydedildi: %s" % out)
    (C.ROOT / "makale" / out.name).write_bytes(out.read_bytes())

    crossed_ok = sum(1 for r in recs if r["epoch_crossed"] is not None
                     and not r["collapsed"])
    nocross_bad = sum(1 for r in recs if r["epoch_crossed"] is None
                      and r["collapsed"])
    ep = [r["epoch_crossed"] for r in recs if r["epoch_crossed"] is not None]
    summary = {
        "escape_threshold": ESCAPE, "n_runs": len(recs),
        "n_crossed_and_healthy": crossed_ok,
        "n_never_crossed_and_collapsed": nocross_bad,
        "crossing_epoch_min": int(min(ep)), "crossing_epoch_max": int(max(ep)),
        "collapsed_max_gate": [r["max_gate"] for r in recs if r["collapsed"]],
        "healthy_min_max_gate": float(min(r["max_gate"] for r in recs
                                          if not r["collapsed"])),
        "verdict": ("cokme, kapinin 0,5'i gecip temsili zamansal dal agirlikli "
                    "hale getirememesidir; sinif ayriminin isareti ayirt edici "
                    "degildir"),
        "runs": recs,
    }
    print("\n  %-13s %-4s %7s %10s %8s" % ("graf", "tohum", "maks g", "gecis ep", "F1"))
    for r in recs:
        print("  %-13s %-4s %7.3f %10s %8.4f%s"
              % (r["graph"], r["seed"], r["max_gate"],
                 r["epoch_crossed"], r["f1"],
                 "   <- COKME" if r["collapsed"] else ""))
    print("\n  0,5'i gecen ve cokmeyen : %d/%d" % (crossed_ok, len(recs)))
    print("  gecemeyen ve coken      : %d/%d" % (nocross_bad, len(recs)))
    print("  gecis epoch araligi     : %d-%d" % (min(ep), max(ep)))
    print("  cokmeyen kosularda en dusuk tepe: %.3f" % summary["healthy_min_max_gate"])
    save_json(summary, OUT_JSON)
    print("\nkaydedildi: %s" % OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
