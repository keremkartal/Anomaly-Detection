# Superseded experiment scripts

These two scripts belong to the first version of the study. They are kept for
provenance and are **not** part of the current pipeline. Do not use their
output.

| Script | Superseded by | Why |
|---|---|---|
| `f1_baseline.py` | `f1b_fair_baseline.py` | Compared an old checkpoint (gated fusion, scaler fitted before the split, single run) against single-run classical models. Neither side was a fair comparison. Its output, `F1_zemin.json`, is archived under `results/_v1_arsiv/`. |
| `f9_figures.py` | `f9_figures_en.py` | Turkish axis labels, and it built the baseline figure from `F1_zemin.json` while the corresponding table came from `F1b_fair.json` — the two disagreed. That mismatch is documented in Section VI-A of the paper. |

The current pipeline is `f0` … `f7`, `f9_figures_en.py` and `f9_architecture.py`.
See the reproduction order in the top-level `README.md`.
