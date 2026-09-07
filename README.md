# STANet — Controlled Evaluation of Hybrid LSTM–GAT Traffic Anomaly Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22637361.svg)](https://doi.org/10.5281/zenodo.22637361)

Reproduction package for the paper *"How Much Does Fusion Design Matter?
Partition, Seed, and Cluster Effects in the Evaluation of Hybrid LSTM–GAT
Traffic Anomaly Detectors."*

This repository contains **everything needed to reproduce every number in the
paper**: the data, the pipeline, the experiment scripts, the raw result files,
the figure generator, and a verification script that checks the manuscript's
numbers against the result files.

---

## What this study reports

The task is binary anomaly detection on 50-step vehicle-trajectory windows,
with a road-network graph supplying spatial context. One component is varied at
a time while the task, data partition, optimization budget, checkpoint
criterion, and threshold rule are held fixed. Every result is a mean over
multiple random seeds.

| Axis | Variants compared |
|---|---|
| Temporal encoder | LSTM, GRU, STGCN gated TCN, Graph WaveNet dilated conv, Transformer |
| Spatial encoder | GAT, Chebyshev (STGCN), diffusion (DCRNN), adaptive adjacency (Graph WaveNet), none |
| Fusion operator | sample-wise scalar gate, fixed concatenation, static weighted sum, cross-attention |
| Graph construction | trip-instance nodes (4 678) vs. OSM-merged physical roads (628) |
| Reference models | XGBoost, LightGBM, Random Forest, zero-parameter threshold rules |

Findings are reported with trip-level cluster bootstrap confidence intervals,
trip-level permutation tests, and Holm correction — **not** window-level tests,
because consecutive windows overlap by 80 % and are not independent.

---

## Quick start

```bash
git clone https://github.com/keremkartal/Anomaly-Detection.git
cd Anomaly-Detection

# PyTorch first (build-specific wheels)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.7.0
pip install -r requirements.txt

# sanity check: data loads, graph builds, protocol hash prints
python experiments/f0_validate.py
```

A CUDA GPU is recommended (an NVIDIA GPU with 4 GB is enough). Everything runs
on CPU as well, roughly 15× slower.

---

## Reproducing the paper

Run in this order. Each script writes a JSON into `results/` and prints a
summary table.

| # | Command | What it produces | Runs | Approx. GPU time |
|---|---|---|---|---|
| F0 | `python experiments/f0_validate.py` | data/graph integrity report | – | < 1 min |
| F3 | `python experiments/f3_fusion.py` | fusion ablation, fixed split | 20 | ~50 min |
| F4 | `python experiments/f4_benchmark.py` | 9-configuration encoder benchmark | 45 | ~1 h 50 min |
| F2 | `python experiments/f2_graph.py` | graph-construction ablation | 30 | ~50 min |
| F1b | `python experiments/f1b_fair_baseline.py` | classical ML + rule baselines | – | ~5 min (CPU) |
| F5 | `python experiments/f5_cv.py` | 5-fold trip-grouped CV, fusion ablation under CV, calibration | 50 | ~2 h |
| F7 | `python experiments/f7_cluster_stats.py` | trip-level statistics for all families | – | ~2 min (CPU) |
| F6 | `python experiments/f6_efficiency.py` | latency and parameter counts | – | ~5 min |
| F9 | `python experiments/f9_figures_en.py` | eight result figures, copied into `makale/` | – | < 1 min |
| F9b | `python experiments/f9_architecture.py` | the architecture diagram (Fig. 1) | – | < 1 min |

Total: ~5.5 h on an NVIDIA RTX-class GPU.

Runs are cached (see below), so re-running a script after a crash resumes
rather than retraining.

Finally, verify the manuscript against the results:

```bash
python makale/_verify_numbers.py
```

---

## Two mechanisms that guarantee the tables agree

Version 1 of this study reported the **same configuration** with two different
scores in two different tables (0.8836 and 0.8484). Two causes were found:
some experiments had run with `cudnn.deterministic=True` and others with
`False` with no record of which, and two code paths instantiated
architecturally identical models under different parameter names, which draws
different initial weights at the same seed. Version 2 makes that failure
structurally impossible.

### 1. Protocol stamp

`stanet.utils.protocol_stamp()` returns the full protocol — seeds, epoch
budget, patience, batch size, learning rate, scheduler, dropout, checkpoint
criterion, threshold rule, cuDNN flags, sequence length, stride, split seed,
PyTorch and CUDA versions — plus a 12-character hash. Every result JSON carries
it. Scripts that import another script's results **assert** that the hashes
match and refuse to run otherwise:

```python
assert f4["protocol"]["hash"] == results["protocol"]["hash"]
```

### 2. Shared run store

`stanet.runstore` caches each `(configuration, seed, protocol)` triple under
`results/_runs/<protocol_hash>/`. The configuration `lstm + gat + weighted_sum`
appears in four tables of the paper; all four **read the same cached run**, so
they report bit-identical numbers rather than hoping two independent runs
agree. Changing any protocol field changes the hash, which invalidates the
cache and forces retraining — silent mixing cannot occur.

```
results/_runs/0902cd97ec17/
  lstm-gat-weighted_sum__trip_instance__s42.json   <- metrics + protocol
  lstm-gat-weighted_sum__trip_instance__s42.npz    <- test/val probabilities
  ...
```

---

## Repository layout

```
stanet/                  library
  config.py              all paths and hyperparameters, one place
  data.py                loading, cleaning, windowing, trip-grouped splits, CV folds
  graph.py               road-network graph construction (both variants)
  models.py              original model definitions (legacy checkpoint compatible)
  encoders.py            temporal and spatial encoders for the benchmark
  hybrid.py              GenericHybrid: temporal x spatial x fusion
  train.py               single shared training loop for every variant
  evaluate.py            metrics, threshold selection, prediction
  stats.py               window-level bootstrap and McNemar (kept for comparison)
  cluster_stats.py       trip-level cluster bootstrap, permutation, Holm
  runstore.py            shared run cache keyed by protocol hash
  baselines.py           threshold rules and window feature extraction
  utils.py               seeding, device, JSON I/O, protocol stamp

experiments/             one script per experiment, F0-F9
  _superseded/           version-1 scripts, kept for provenance only
results/                 result JSONs (committed), figures, run cache (ignored)
  _v1_arsiv/             version-1 results plus a note on what changed and why
veriseti/                dataset (see veriseti/README.md)
makale/                  LaTeX source, figures, data card, verification script
```

---

## Dataset

The trajectories are **synthetic**. They were generated by a physics-informed
simulation over a road network derived from OpenStreetMap (Kocaeli, Türkiye);
anomalies were injected by known rules and the ground-truth labels come from
those injection records. There is no underlying real-world trajectory corpus.

| | |
|---|---|
| Trajectory rows | 293 168 |
| Trips | 90 |
| Windows (τ = 50, stride 10) | 28 090 |
| Road-network rows | 4 678 |
| Unique physical roads (`osm_segment_id`) | 628 |
| Anomaly rate | 7.9 % (train) |

Full documentation, including the imputation shares for lane counts and speed
limits and the known limitations, is in **`makale/VERI_KARTI.md`** (Turkish)
and summarized in the paper's data section.

Two structural properties matter when reading any result here:

- **Windows are not independent.** Stride 10 on length-50 windows means
  consecutive windows share 40 of 50 steps — 80 % overlap. All statistical
  tests in this repository therefore use the **trip** as the unit.
- **Only 90 trips exist.** The fixed test split contains **14 trips**.
  Trip-level confidence intervals on the fixed split are correspondingly wide;
  the cross-validated analysis, in which all 90 trips receive a test
  prediction, is the more reliable evidence and is treated as primary.

---

## Citation

See `CITATION.cff`.

- Version DOI (this snapshot, v2.0.0): [10.5281/zenodo.22637361](https://doi.org/10.5281/zenodo.22637361)
- Concept DOI (always the latest version): [10.5281/zenodo.22637360](https://doi.org/10.5281/zenodo.22637360)

The version DOI is the one to cite when reproducing the numbers in the paper; it pins the exact code and data used.

## License

Source code: MIT (`LICENSE`).
Road-network component of the dataset: derived from OpenStreetMap,
© OpenStreetMap contributors, ODbL v1.0.
