# TrajCamo

**Long-Window Trajectory Grouping Brings Video Reasoning Segmentation to Camouflaged Targets.**

A language-guided framework for camouflaged video object segmentation that replaces appearance-grounded keyframe localization with **long-window trajectory grouping** over a dense point-tracker field, plus an agentic refinement loop in which a multimodal language agent iteratively selects, prompts, and corrects motion hypotheses against the query.

## Pipeline (4 stages)

1. **Long-Window Trajectory Field** — CoTracker3 (offline) extracts dense, occlusion-aware trajectories spanning the whole video.
2. **Kinematic Signature Encoder** — small Transformer with contrastive loss maps each trajectory to an embedding summarizing its long-window motion pattern.
3. **Long-Window Trajectory Grouping** — spectral clustering on a motion-and-locality affinity recovers candidate object hypotheses as spatially connected, kinematically consistent subspaces.
4. **Query-conditioned Agentic Cluster Refinement** — MLLM agent emits a short sequence of `SELECT`, `Add-Pos`, `Add-Neg`, `Terminate` actions, interacting with a frozen SAM2/SAM3 backbone whose moving point prompts come from the selected trajectory cluster.

## Repository layout

```
.
├── code/
│   ├── phase1_precache_trajectories.py    # CoTracker3 pre-cache (offline pass)
│   ├── phase2_main_eval.py                # Cluster + SAM3 propagation + 4 COD metrics
│   ├── cod_metrics.py                     # F_w_β, MAE, S_α, E_φ implementations
│   └── ...
├── paper/
│   └── TrajCamo_draft.tex                 # paper draft
└── README.md
```

## Datasets

* **MoCA-Mask** (Cheng et al., CVPR 2022) — 71 train + 16 test videos, ~22K frames, sparse 5-frame GT.
* **CAD2016** — cross-dataset generalization test.
* **CAMotion** — recent in-the-wild benchmark.
* Camouflaged Video Reasoning Benchmark — 266 video-instruction-mask samples (this work).

## Baseline reference

SLT-Net (Cheng et al., CVPR 2022) on MoCA-Mask test split:
F_w_β = 0.357, MAE = 0.030, S_α = 0.656, E_φ = 0.785

## Status

Work in progress. Phase-1 trajectory pre-cache is operational on all 87 MoCA-Mask videos (~2.3 min on a single RTX 4090 48GB).
