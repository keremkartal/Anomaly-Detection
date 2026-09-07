# Dataset

**These trajectories are synthetic.** They were produced by a physics-informed
Markov-chain simulation over a road network derived from OpenStreetMap
(Kocaeli, Türkiye). Anomalies were injected during simulation by known rules;
the ground-truth labels come from those injection records. There is no
underlying real-world trajectory corpus.

Full data card (Turkish): `../makale/VERI_KARTI.md`

## Files used by the pipeline

| File | Size | Role |
|---|---|---|
| `lstm_data.csv` | 30 MB | trajectory table — 293 168 rows, 90 trips, 1 s resolution |
| `GNN_DATA.csv` | 2.5 MB | road-network table — 4 678 rows, 628 unique `osm_segment_id` |

`stanet/data.py` and `stanet/graph.py` read only these two files. Everything
else is derived at run time.

## Files NOT required (and not tracked in git)

`lstm_data_cleaned_step1.csv`, `lstm_data_cleaned_step2.csv`,
`lstm_data_normalized.csv`, `dataset_splits.npz`, `*.pth`, `*.pt` are
intermediates and checkpoints from the original exploratory notebook. The
current pipeline regenerates every one of them from the two source CSVs. They
are kept locally only so that the legacy path (`build_dataset(legacy=True)`,
which reproduces the original leaky preprocessing for comparison) can be run.

## Columns

`lstm_data.csv`: `trip_id, segment_id, time, speed, accel, bearing, percent,
traffic, dwel_duration, dwel_flag, anomaly, maxspeed`

`GNN_DATA.csv` node features used by the spatial branch: `length_log,
max_speed, rarity_score, betweenness, closeness, node_degree, lanes` plus
binary `tunnel, traffic_light`.

## Two structural properties to keep in mind

1. **`segment_id` is trip-specific.** No `segment_id` appears in two trips. The
   same physical road (`osm_segment_id`) is repeated on average 7.4 times as
   separate rows. The "4 678 nodes, average degree 11.47" figure comes from
   this repetition; the merged physical network has **628 nodes and average
   degree 1.11**. Both graph variants are evaluated (experiment F2).

2. **Windows overlap by 80 %.** Windows are cut with length τ = 50 and stride
   10, so consecutive windows share 40 steps, and all windows of a trip share a
   driver, a vehicle and a route. Window-level statistical tests overstate the
   effective sample size. Every test in this repository uses the **trip** as
   the unit (`stanet/cluster_stats.py`).

## License

Road-network component derived from OpenStreetMap,
© OpenStreetMap contributors, licensed ODbL v1.0
(https://www.openstreetmap.org/copyright). Redistribution of derivatives must
comply with ODbL. The simulated trajectories themselves are released with the
repository under the terms in `../LICENSE`.
