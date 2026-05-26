# TrajCamo Main Experiment Report (MoCA-Mask test, 16 videos)

**Pipeline:** CoTracker3 trajectory pre-cache + raw-velocity K-means clustering (K=8) + centroid-neighborhood prompt sampling + SAM 3 base propagation. **Agentic MLLM loop is not yet applied in this run** — these numbers establish the trajectory-grouping pipeline's standalone strength (without language reasoning).

Two cluster-selection rules are evaluated:

* **Oracle**: pick the cluster with highest IoU against the GT union   (upper bound for what an MLLM agent could achieve on cluster selection alone).
* **Heuristic**: pick the cluster with highest within-cluster consistency   score (no GT, simulates a no-language deployment).

## Per-video metrics

### Selection rule: `oracle`

| Video | T | n_traj | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ |
|---|---:|---:|---:|---:|---:|---:|
| arctic_fox | 30 | 2174 | 0.539 | 0.0090 | 0.737 | 0.939 |
| arctic_fox_3 | 35 | 678 | 0.000 | 0.7869 | 0.143 | 0.239 |
| black_cat_1 | 73 | 1716 | 0.294 | 0.0149 | 0.579 | 0.763 |
| copperhead_snake | 83 | 2002 | 0.933 | 0.0100 | 0.784 | 0.847 |
| flower_crab_spider_0 | 40 | 96 | 0.000 | 0.0564 | 0.432 | 0.844 |
| flower_crab_spider_1 | 44 | 1637 | 0.977 | 0.0029 | 0.949 | 0.995 |
| flower_crab_spider_2 | 30 | 147 | 0.970 | 0.0036 | 0.946 | 0.996 |
| hedgehog_3 | 25 | 159 | 0.734 | 0.0228 | 0.784 | 0.899 |
| ibex | 25 | 1517 | 0.975 | 0.0308 | 0.522 | 0.292 |
| mongoose | 10 | 2304 | 0.000 | 0.0154 | 0.472 | 0.806 |
| moth | 106 | 56 | 0.832 | 0.0022 | 0.851 | 0.953 |
| pygmy_seahorse_0 | 20 | 1647 | 0.932 | 0.3365 | 0.347 | 0.258 |
| rusty_spotted_cat_0 | 20 | 41 | 0.000 | 0.0240 | 0.466 | 0.815 |
| sand_cat_0 | 10 | 1748 | 0.932 | 0.0064 | 0.915 | 0.985 |
| snow_leopard_10 | 155 | 68 | 0.000 | 0.0010 | 0.495 | 0.872 |
| stick_insect_1 | 39 | 1944 | 0.000 | 0.0124 | 0.480 | 0.812 |
| **MEAN** | | | **0.507** | **0.0835** | **0.619** | **0.770** |

### Selection rule: `heuristic`

| Video | T | n_traj | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ |
|---|---:|---:|---:|---:|---:|---:|
| arctic_fox | 30 | 2174 | 0.539 | 0.0090 | 0.737 | 0.939 |
| arctic_fox_3 | 35 | 678 | 0.000 | 0.4193 | 0.314 | 0.288 |
| black_cat_1 | 73 | 1716 | 0.294 | 0.0149 | 0.579 | 0.763 |
| copperhead_snake | 83 | 2002 | 0.933 | 0.0100 | 0.784 | 0.847 |
| flower_crab_spider_0 | 40 | 96 | 0.000 | 0.0594 | 0.429 | 0.810 |
| flower_crab_spider_1 | 44 | 1637 | 0.000 | 0.0390 | 0.453 | 0.968 |
| flower_crab_spider_2 | 30 | 147 | 0.000 | 0.0399 | 0.441 | 0.920 |
| hedgehog_3 | 25 | 159 | 0.000 | 0.0899 | 0.428 | 0.625 |
| ibex | 25 | 1517 | 0.975 | 0.0308 | 0.522 | 0.292 |
| mongoose | 10 | 2304 | 0.000 | 0.0114 | 0.478 | 0.929 |
| moth | 106 | 56 | 0.000 | 0.0089 | 0.484 | 0.716 |
| pygmy_seahorse_0 | 20 | 1647 | 0.932 | 0.3365 | 0.347 | 0.258 |
| rusty_spotted_cat_0 | 20 | 41 | 0.000 | 0.0240 | 0.466 | 0.815 |
| sand_cat_0 | 10 | 1748 | 0.000 | 0.0440 | 0.462 | 0.650 |
| snow_leopard_10 | 155 | 68 | 0.000 | 0.0023 | 0.490 | 0.695 |
| stick_insect_1 | 39 | 1944 | 0.000 | 0.0124 | 0.480 | 0.812 |
| **MEAN** | | | **0.230** | **0.0720** | **0.493** | **0.708** |

## Comparison against SLT-Net (CVPR 2022)

| Method | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ |
|---|---:|---:|---:|---:|
| SLT-Net (paper) | 0.357 | 0.03 | 0.656 | 0.785 |
| **TrajCamo — oracle** | 0.507 | 0.0835 | 0.619 | 0.770 |
| ↳ Δ vs SLT-Net | +0.150 | -0.053 | -0.037 | -0.015 |
| **TrajCamo — heuristic** | 0.230 | 0.0720 | 0.493 | 0.708 |
| ↳ Δ vs SLT-Net | -0.127 | -0.042 | -0.163 | -0.077 |

## Reading the numbers

* This is the **trajectory-grouping pipeline alone** — no MLLM agent, no
  language input, no learned signature encoder. Just CoTracker3 + K-means
  on raw velocity + centroid-sample to SAM 3.
* The **oracle** row is an upper bound: it tells us the maximum accuracy
  obtainable if the cluster-selection step were perfect.
* The **heuristic** row is the realistic no-language number, comparable
  to SLT-Net (which is also non-language).
* Adding (i) a learned kinematic signature encoder, (ii) the agentic MLLM
  loop with `Add-Pos` / `Add-Neg` corrections, and (iii) language-conditioned
  cluster selection are the remaining sources of accuracy gain in the
  full TrajCamo system.
