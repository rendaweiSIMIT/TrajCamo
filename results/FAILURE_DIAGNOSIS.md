# Per-Video Failure Diagnosis (MoCA-Mask test, K=8, min_move=4.0)

Goal: for each video, identify whether the trajectory pipeline has any
chance of recovering the camouflaged target, and at which stage it
fails when it does. Columns:

* **N tot**: trajectories returned by CoTracker3 (grid=48 → 2304).
* **N vis**: trajectories visible in ≥20% of frames.
* **N start-in-GT**: trajectories whose frame-0 position falls inside any GT mask.
* **N on-target**: visible trajectories whose ≥50% of visible positions
  fall inside some GT mask (the trajectories that *should* form the
  target cluster).
* **N dynamic**: visible trajectories that survive `min_move_px=4` filter.
* **N on-target ∩ dynamic**: target trajectories that also pass the
  dynamic filter (these are the ones available to the clusterer).
* **Cluster recoverable**: yes if K-means finds a cluster with >50% of
  its points being on-target — i.e. the cluster IS a target cluster.
* **Best purity**: max fraction of on-target points across all K clusters.

| Video | N tot | N vis | start-in-GT | on-target | dynamic | on-target ∩ dynamic | Cluster recoverable | Best purity |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| arctic_fox | 2304 | 2297 | 281 | 263 | 2174 | 263 | `no` | 0.150 |
| arctic_fox_3 | 2304 | 838 | 79 | 0 | 678 | 0 | `no` | 0.000 |
| black_cat_1 | 2304 | 1859 | 140 | 90 | 1716 | 90 | `no` | 0.072 |
| copperhead_snake | 2304 | 2036 | 131 | 128 | 2002 | 128 | `no` | 0.288 |
| flower_crab_spider_0 | 2304 | 2253 | 82 | 33 | 96 | 2 | `no` | 0.000 |
| flower_crab_spider_1 | 2304 | 2277 | 91 | 66 | 1637 | 66 | `yes` | 1.000 |
| flower_crab_spider_2 | 2304 | 2293 | 82 | 70 | 147 | 45 | `yes` | 1.000 |
| hedgehog_3 | 2304 | 2202 | 357 | 270 | 159 | 106 | `yes` | 1.000 |
| ibex | 2304 | 1838 | 30 | 15 | 1517 | 15 | `no` | 0.011 |
| mongoose | 2304 | 2304 | 56 | 60 | 25 | 0 | `no` | 0.000 |
| moth | 2304 | 2297 | 37 | 31 | 56 | 11 | `yes` | 1.000 |
| pygmy_seahorse_0 | 2304 | 1836 | 45 | 37 | 1647 | 37 | `no` | 0.028 |
| rusty_spotted_cat_0 | 2304 | 2303 | 32 | 32 | 41 | 7 | `no` | 0.000 |
| sand_cat_0 | 2304 | 2166 | 149 | 99 | 1748 | 99 | `yes` | 0.625 |
| snow_leopard_10 | 2304 | 2304 | 32 | 32 | 68 | 0 | `no` | 0.000 |
| stick_insect_1 | 2304 | 2048 | 181 | 112 | 1944 | 112 | `no` | 0.091 |

## Failure mode summary

* Total test videos: 16
* Recoverable (best cluster purity ≥ 0.5): **5**
* Failure mode A — *trajectory grid too sparse for tiny target* (< 10 on-target trajectories from CoTracker3 grid): **1**
* Failure mode B — *dynamic filter removes target* (≥10 on-target visible but < 5 survive dynamic filter): **3**
* Failure mode C — *clustering merges target with background* (target trajectories present but not cleanly clustered): **7**

**Failure A (sparse grid)**: arctic_fox_3

**Failure B (filter)**: flower_crab_spider_0, mongoose, snow_leopard_10

**Failure C (clustering)**: arctic_fox, black_cat_1, copperhead_snake, ibex, pygmy_seahorse_0, rusty_spotted_cat_0, stick_insect_1