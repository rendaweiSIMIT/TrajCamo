# TrajCamo Ablation Studies — MoCA-Mask test split (16 videos)

All ablations use the same Phase-2 pipeline (CoTracker3 trajectories + K-means + SAM 3 propagation), changing one knob at a time from the base configuration:  `feature=raw_velocity, K=8, n_prompt_points=12, min_move_px=4.0, prompt_frames=0`.

Each cell reports the 16-video mean for that metric. The `Δ vs SLT` column shows `(F_w MAE S_α E_φ)` deltas vs SLT-Net's published numbers (`F_w=0.357 / MAE=0.030 / S_α=0.656 / E_φ=0.785`); positive delta means we beat SLT-Net (for MAE delta is sign-flipped so positive is still better).

**SLT-Net baseline (CVPR 2022):**  F_w_β = 0.357, MAE = 0.03, S_α = 0.656, E_φ = 0.785

## A. Feature ablation

| Tag | Setting | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ | Δ vs SLT (F/M/S/E) |
|---|---|---:|---:|---:|---:|---|
| `feat_coherence` | coherence / K=8 / n=12 / mv=4.0 / frm=0  [oracle] | 0.337 | 0.1841 | 0.514 | 0.531 | -0.020 / -0.154 / -0.142 / -0.254 |
| `feat_coherence` | coherence / K=8 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.126 | 0.1256 | 0.453 | 0.528 | -0.231 / -0.096 / -0.203 / -0.257 |
| `feat_position` | position / K=8 / n=12 / mv=4.0 / frm=0  [oracle] | 0.340 | 0.0525 | 0.576 | 0.674 | -0.017 / -0.022 / -0.080 / -0.111 |
| `feat_position` | position / K=8 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.208 | 0.0915 | 0.526 | 0.606 | -0.149 / -0.061 / -0.130 / -0.179 |
| `feat_raw_speed` | raw_speed / K=8 / n=12 / mv=4.0 / frm=0  [oracle] | 0.333 | 0.0382 | 0.580 | 0.698 | -0.024 / -0.008 / -0.076 / -0.087 |
| `feat_raw_speed` | raw_speed / K=8 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.215 | 0.0465 | 0.499 | 0.592 | -0.142 / -0.016 / -0.157 / -0.193 |
| `feat_raw_velocity` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0  [oracle] | 0.507 | 0.0829 | 0.620 | 0.745 | +0.150 / -0.053 / -0.036 / -0.040 |
| `feat_raw_velocity` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.230 | 0.0717 | 0.494 | 0.675 | -0.127 / -0.042 / -0.162 / -0.110 |

## B. K (# clusters) sweep

| Tag | Setting | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ | Δ vs SLT (F/M/S/E) |
|---|---|---:|---:|---:|---:|---|
| `K_12` | raw_velocity / K=12 / n=12 / mv=4.0 / frm=0  [oracle] | 0.412 | 0.0383 | 0.600 | 0.723 | +0.055 / -0.008 / -0.056 / -0.062 |
| `K_12` | raw_velocity / K=12 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.172 | 0.1006 | 0.462 | 0.612 | -0.185 / -0.071 / -0.194 / -0.173 |
| `K_16` | raw_velocity / K=16 / n=12 / mv=4.0 / frm=0  [oracle] | 0.470 | 0.0378 | 0.620 | 0.750 | +0.113 / -0.008 / -0.036 / -0.035 |
| `K_16` | raw_velocity / K=16 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.172 | 0.1036 | 0.458 | 0.695 | -0.185 / -0.074 / -0.198 / -0.090 |
| `K_24` | raw_velocity / K=24 / n=12 / mv=4.0 / frm=0  [oracle] | 0.581 | 0.0373 | 0.665 | 0.744 | +0.224 / -0.007 / +0.009 / -0.041 |
| `K_24` | raw_velocity / K=24 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.202 | 0.0486 | 0.499 | 0.719 | -0.155 / -0.019 / -0.157 / -0.066 |
| `K_4` | raw_velocity / K=4 / n=12 / mv=4.0 / frm=0  [oracle] | 0.387 | 0.0504 | 0.584 | 0.682 | +0.030 / -0.020 / -0.072 / -0.103 |
| `K_4` | raw_velocity / K=4 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.172 | 0.0462 | 0.487 | 0.647 | -0.185 / -0.016 / -0.169 / -0.138 |
| `K_6` | raw_velocity / K=6 / n=12 / mv=4.0 / frm=0  [oracle] | 0.402 | 0.0864 | 0.566 | 0.674 | +0.045 / -0.056 / -0.090 / -0.111 |
| `K_6` | raw_velocity / K=6 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.230 | 0.0706 | 0.497 | 0.645 | -0.127 / -0.041 / -0.159 / -0.140 |

## C. n_prompt_points sweep

| Tag | Setting | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ | Δ vs SLT (F/M/S/E) |
|---|---|---:|---:|---:|---:|---|
| `nprompt_1` | raw_velocity / K=8 / n=1 / mv=4.0 / frm=0  [oracle] | 0.458 | 0.1267 | 0.603 | 0.726 | +0.101 / -0.097 / -0.053 / -0.059 |
| `nprompt_1` | raw_velocity / K=8 / n=1 / mv=4.0 / frm=0  [heuristic] | 0.224 | 0.1592 | 0.475 | 0.578 | -0.133 / -0.129 / -0.181 / -0.207 |
| `nprompt_16` | raw_velocity / K=8 / n=16 / mv=4.0 / frm=0  [oracle] | 0.521 | 0.1056 | 0.615 | 0.728 | +0.164 / -0.076 / -0.041 / -0.057 |
| `nprompt_16` | raw_velocity / K=8 / n=16 / mv=4.0 / frm=0  [heuristic] | 0.240 | 0.1335 | 0.472 | 0.612 | -0.117 / -0.104 / -0.184 / -0.173 |
| `nprompt_24` | raw_velocity / K=8 / n=24 / mv=4.0 / frm=0  [oracle] | 0.531 | 0.1126 | 0.616 | 0.718 | +0.174 / -0.083 / -0.040 / -0.067 |
| `nprompt_24` | raw_velocity / K=8 / n=24 / mv=4.0 / frm=0  [heuristic] | 0.251 | 0.1297 | 0.478 | 0.612 | -0.106 / -0.100 / -0.178 / -0.173 |
| `nprompt_4` | raw_velocity / K=8 / n=4 / mv=4.0 / frm=0  [oracle] | 0.310 | 0.0163 | 0.611 | 0.765 | -0.047 / +0.014 / -0.045 / -0.020 |
| `nprompt_4` | raw_velocity / K=8 / n=4 / mv=4.0 / frm=0  [heuristic] | 0.075 | 0.0232 | 0.496 | 0.691 | -0.282 / +0.007 / -0.160 / -0.094 |
| `nprompt_8` | raw_velocity / K=8 / n=8 / mv=4.0 / frm=0  [oracle] | 0.423 | 0.0129 | 0.647 | 0.832 | +0.066 / +0.017 / -0.009 / +0.047 |
| `nprompt_8` | raw_velocity / K=8 / n=8 / mv=4.0 / frm=0  [heuristic] | 0.157 | 0.0235 | 0.521 | 0.728 | -0.200 / +0.007 / -0.135 / -0.057 |

## D. min_move_px (dynamic-filter threshold) sweep

| Tag | Setting | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ | Δ vs SLT (F/M/S/E) |
|---|---|---:|---:|---:|---:|---|
| `minmove_0` | raw_velocity / K=8 / n=12 / mv=0.0 / frm=0  [oracle] | 0.509 | 0.1016 | 0.585 | 0.727 | +0.152 / -0.072 / -0.071 / -0.058 |
| `minmove_0` | raw_velocity / K=8 / n=12 / mv=0.0 / frm=0  [heuristic] | 0.230 | 0.0706 | 0.497 | 0.607 | -0.127 / -0.041 / -0.159 / -0.178 |
| `minmove_16` | raw_velocity / K=8 / n=12 / mv=16.0 / frm=0  [oracle] | 0.340 | 0.0343 | 0.599 | 0.763 | -0.017 / -0.004 / -0.057 / -0.022 |
| `minmove_16` | raw_velocity / K=8 / n=12 / mv=16.0 / frm=0  [heuristic] | 0.110 | 0.0249 | 0.518 | 0.686 | -0.247 / +0.005 / -0.138 / -0.099 |
| `minmove_2` | raw_velocity / K=8 / n=12 / mv=2.0 / frm=0  [oracle] | 0.631 | 0.0980 | 0.648 | 0.701 | +0.274 / -0.068 / -0.008 / -0.084 |
| `minmove_2` | raw_velocity / K=8 / n=12 / mv=2.0 / frm=0  [heuristic] | 0.288 | 0.0774 | 0.505 | 0.624 | -0.069 / -0.047 / -0.151 / -0.161 |
| `minmove_32` | raw_velocity / K=8 / n=12 / mv=32.0 / frm=0  [oracle] | 0.379 | 0.0366 | 0.611 | 0.763 | +0.022 / -0.007 / -0.045 / -0.022 |
| `minmove_32` | raw_velocity / K=8 / n=12 / mv=32.0 / frm=0  [heuristic] | 0.094 | 0.0250 | 0.513 | 0.672 | -0.263 / +0.005 / -0.143 / -0.113 |
| `minmove_8` | raw_velocity / K=8 / n=12 / mv=8.0 / frm=0  [oracle] | 0.464 | 0.0543 | 0.599 | 0.740 | +0.107 / -0.024 / -0.057 / -0.045 |
| `minmove_8` | raw_velocity / K=8 / n=12 / mv=8.0 / frm=0  [heuristic] | 0.230 | 0.0939 | 0.497 | 0.658 | -0.127 / -0.064 / -0.159 / -0.127 |

## E. Multi-frame prompting

| Tag | Setting | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ | Δ vs SLT (F/M/S/E) |
|---|---|---:|---:|---:|---:|---|
| `frames_0` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0  [oracle] | 0.507 | 0.0829 | 0.620 | 0.745 | +0.150 / -0.053 / -0.036 / -0.040 |
| `frames_0` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.230 | 0.0717 | 0.494 | 0.675 | -0.127 / -0.042 / -0.162 / -0.110 |
| `frames_0_last` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0,last  [oracle] | 0.440 | 0.0841 | 0.603 | 0.765 | +0.083 / -0.054 / -0.053 / -0.020 |
| `frames_0_last` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0,last  [heuristic] | 0.163 | 0.0676 | 0.480 | 0.649 | -0.194 / -0.038 / -0.176 / -0.136 |
| `frames_0_mid_last` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0,mid,last  [oracle] | 0.308 | 0.0523 | 0.577 | 0.746 | -0.049 / -0.022 / -0.079 / -0.039 |
| `frames_0_mid_last` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0,mid,last  [heuristic] | 0.080 | 0.0573 | 0.466 | 0.665 | -0.277 / -0.027 / -0.190 / -0.120 |
| `frames_0_only` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0  [oracle] | 0.507 | 0.0829 | 0.620 | 0.745 | +0.150 / -0.053 / -0.036 / -0.040 |
| `frames_0_only` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.230 | 0.0717 | 0.494 | 0.675 | -0.127 / -0.042 / -0.162 / -0.110 |
| `frames_quartiles` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0,T/4,T/2,3T/4,last  [oracle] | 0.247 | 0.0455 | 0.565 | 0.734 | -0.110 / -0.015 / -0.091 / -0.051 |
| `frames_quartiles` | raw_velocity / K=8 / n=12 / mv=4.0 / frm=0,T/4,T/2,3T/4,last  [heuristic] | 0.028 | 0.0693 | 0.451 | 0.652 | -0.329 / -0.039 / -0.205 / -0.133 |

## F. Learned signature encoder

| Tag | Setting | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ | Δ vs SLT (F/M/S/E) |
|---|---|---:|---:|---:|---:|---|
| `learned_encoder` | ? / K=8 / n=12 / mv=4.0 / frm=0  [oracle] | 0.281 | 0.0221 | 0.601 | 0.722 | -0.076 / +0.008 / -0.055 / -0.063 |
| `learned_encoder` | ? / K=8 / n=12 / mv=4.0 / frm=0  [heuristic] | 0.059 | 0.0342 | 0.487 | 0.632 | -0.298 / -0.004 / -0.169 / -0.153 |

## Best oracle configuration: `minmove_2` (F_w_β = 0.631)

| Video | T | n_traj | F_w_β | MAE | S_α | E_φ |
|---|---:|---:|---:|---:|---:|---:|
| arctic_fox | 30 | 2270 | 0.538 | 0.0090 | 0.737 | 0.939 |
| arctic_fox_3 | 35 | 789 | 0.000 | 0.7869 | 0.143 | 0.239 |
| black_cat_1 | 73 | 1801 | 0.294 | 0.0149 | 0.579 | 0.763 |
| copperhead_snake | 83 | 2036 | 0.933 | 0.0100 | 0.784 | 0.847 |
| flower_crab_spider_0 | 40 | 433 | 0.978 | 0.1118 | 0.680 | 0.419 |
| flower_crab_spider_1 | 44 | 2135 | 0.977 | 0.0029 | 0.949 | 0.995 |
| flower_crab_spider_2 | 30 | 514 | 0.971 | 0.0289 | 0.814 | 0.773 |
| hedgehog_3 | 25 | 440 | 0.728 | 0.0222 | 0.785 | 0.936 |
| ibex | 25 | 1737 | 0.975 | 0.0308 | 0.522 | 0.292 |
| mongoose | 10 | 1410 | 0.082 | 0.1870 | 0.396 | 0.274 |
| moth | 106 | 282 | 0.833 | 0.0022 | 0.852 | 0.953 |
| pygmy_seahorse_0 | 20 | 1799 | 0.932 | 0.3365 | 0.347 | 0.258 |
| rusty_spotted_cat_0 | 20 | 153 | 0.937 | 0.0022 | 0.917 | 0.993 |
| sand_cat_0 | 10 | 1972 | 0.912 | 0.0076 | 0.895 | 0.972 |
| snow_leopard_10 | 155 | 312 | 0.000 | 0.0021 | 0.490 | 0.758 |
| stick_insect_1 | 39 | 2036 | 0.000 | 0.0124 | 0.480 | 0.812 |
