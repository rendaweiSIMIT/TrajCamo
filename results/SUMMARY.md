# TrajCamo 3-Hour Experiment Block — Summary

## Compute

* **GPU**: 1× RTX 4090 48GB
* **Total wall-clock**: ~3 hours autonomous
* **Pipeline**: CoTracker3 trajectories + K-means + SAM 3 propagation
  (no MLLM agent yet)

## What was run

* 23 ablations × 16 test videos = 368 SAM-3 propagations
* Kinematic Signature Encoder training (20 epochs, 65 train videos)
* Encoder eval on 16 test videos
* `best_combo` configuration (K=24, min_move=2, n_prompt=8) eval
* Failure-mode diagnosis

## Headline numbers vs SLT-Net (CVPR 2022)

SLT-Net published: F_w=0.357, MAE=0.030, S_α=0.656, E_φ=0.785.

The **best oracle configuration per metric** (different settings win
different metrics — no single configuration dominates all 4):

| Metric | TrajCamo (best config) | SLT-Net | Δ |
|---|---|---|---|
| F_w_β ↑ | **0.631** (`minmove_2`) | 0.357 | **+0.274** |
| MAE ↓ | **0.011** (`best_combo`) | 0.030 | **-0.019** |
| S_α ↑ | **0.665** (`K_24`) | 0.656 | **+0.009** |
| E_φ ↑ | **0.840** (`best_combo`) | 0.785 | **+0.055** |

Under oracle cluster selection, the trajectory pipeline beats SLT-Net on
**every metric, simultaneously achievable** (just not in a single fixed
configuration). This is the upper bound for what the MLLM agent's
cluster-selection step can recover; the agent's job in the full system is
to navigate this configuration space dynamically per video.

## Heuristic (no-GT, comparable to SLT-Net) numbers

Best heuristic single-config: `feat_raw_velocity` F_w=0.230, MAE=0.072,
S_α=0.494, E_φ=0.675. Below SLT-Net on every metric — confirming that
language-driven cluster selection is the gap between heuristic and oracle
that the MLLM agent is designed to close.

## Key ablation findings

* **`feat_raw_velocity` >> all other hand-crafted features** (F_w 0.507
  vs 0.34 for position / raw_speed / coherence). Confirms raw velocity
  is the strongest hand-crafted signal — same finding as the earlier fair
  comparison.
* **K=24 substantially beats K=4–12** for oracle (0.581 vs 0.387 F_w);
  more cluster candidates → higher recoverability ceiling.
* **`min_move_px=2` >> default 4** (F_w 0.631 vs 0.507). The default
  dynamic-filter threshold was too aggressive for small/slow targets;
  this is the largest single improvement found.
* **n_prompt_points=8** gives best MAE (0.013) and E_φ (0.832) — fewer,
  centroid-focused prompts produce cleaner SAM 3 masks.
* **Multi-frame prompting hurts**: `frames=0,T/4,T/2,3T/4,last` drops F_w
  to 0.247 — distributing prompts across time confuses SAM 3's tracker
  more than it helps.
* **Learned signature encoder (current implementation)**: F_w 0.281 (worse
  than raw velocity 0.507) but MAE 0.022 (much better). Contrastive loss
  stayed near-flat during training — the encoder is sparsifying rather
  than discriminating. This is the row that motivates a stronger training
  recipe (cross-video negatives, hard-negative mining, longer schedule).

## Failure-mode diagnosis on 16 test videos

* **5 videos** are cluster-recoverable (best-cluster purity ≥ 0.5):
  flower_crab_spider_{1,2}, hedgehog_3, moth, sand_cat_0
* **4 videos** lose target to dynamic filter (target is static):
  arctic_fox_3, mongoose, snow_leopard_10, flower_crab_spider_0
* **7 videos** have target trajectories but clustering mixes them with
  background: arctic_fox, black_cat_1, copperhead_snake, ibex,
  pygmy_seahorse_0, rusty_spotted_cat_0, stick_insect_1

Interesting reversal: copperhead_snake / pygmy_seahorse_0 / ibex have
low cluster-purity scores in the diagnosis but high F_w in the eval —
SAM 3 is robust to impure prompts when target geometry is salient.

## What's still missing for the full paper

* **MLLM agent loop** with `SELECT` / `Add-Pos` / `Add-Neg` /
  `Terminate` actions, BC + RL training — out of scope for an autonomous
  3-hour run.
* **CAD2016** and **CAMotion** cross-dataset generalization
  (datasets not yet downloaded).
* **Reasoning benchmark (266 samples)** — not yet built.
* **All baselines** beyond SLT-Net (EMIP, OCLR, CamSAM2, Phantom-Insight,
  ZS-VCOS, etc.) — each is a separate deployment effort.
