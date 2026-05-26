# TrajCamo — Project Plan (TCSVT Submission)

**Target venue**: IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)
**Timeline**: ~2–3 months of focused work
**Owner**: rendaweiSIMIT
**Last updated**: 2026-05-26

This file is the single source of truth for what we're building and in what
order. Read this before opening other files. Update it whenever scope or
priorities shift.

---

## 1. Paper thesis (locked in)

> **Long-window trajectory grouping** over a dense point-tracker field
> **recovers camouflaged video targets that per-frame appearance- and
> instantaneous-motion-based methods miss**. A multimodal language agent
> **iteratively** selects and corrects these motion hypotheses against a
> query, producing both a final mask sequence and an externally inspectable
> reasoning trace.

The four stages of the model are fixed (see `paper/TrajCamo_draft.tex`):

1. **Long-Window Trajectory Field** — CoTracker3 (offline) over the full
   video → dense, occlusion-aware trajectories.
2. **Kinematic Signature Encoder** — small temporal Transformer + per-
   trajectory contrastive loss → translation-and-magnitude invariant
   embedding `z_i`.
3. **Long-Window Trajectory Grouping** — spectral clustering on a motion-
   and-locality affinity → candidate cluster set `{C_k}` with internal
   consistency + local contrast scoring.
4. **Query-conditioned Agentic Cluster Refinement** — MLLM agent emits
   `SELECT(k)` / `Add-Pos(f,x,y)` / `Add-Neg(f,x,y)` / `Terminate` actions
   interacting with a frozen SAM 3 backbone whose moving point prompts come
   from the currently selected trajectory cluster.

---

## 2. Decisions already made

| Decision | Value | Rationale |
|---|---|---|
| MLLM backbone (main) | **InternVL3-8B** | International recognition (OpenGVLab); current trend in 2025 VRS literature (Sa2VA / GLUS / VideoGLaMM all use InternVL); native grounding output (clean `(x, y)` coords); good RL support via VeRL / TRL. |
| MLLM backbone (debug) | InternVL3-2B | Fast iteration on 4090 during pipeline development. |
| MLLM backbone (fallback) | Qwen2.5-VL-7B | Use only if InternVL3 RL training proves unstable. |
| Vision features | DINOv2-large | Paper §5.2 specifies; frozen. |
| Promptable mask backbone | SAM 3 base | Already used in main experiment; frozen. |
| Tracker | CoTracker3 (offline) | Already cached for all 87 videos. |
| Training scheme | **BC + RL** (GRPO) | Paper §3.5; RL is **mandatory** for the agent novelty story. |
| Action vocabulary | 4 actions (`SELECT`, `Add-Pos`, `Add-Neg`, `Terminate`) | Paper §3.4. May add `Merge` / `Split` if BC reveals need. |
| `K_max` step budget | 5 (default) | To be revisited if step-utilization in BC says otherwise. |
| Datasets (for now) | **MoCA-Mask only** | Train 71 / Test 16; matches SLT-Net / EMIP / CamSAM2 protocol. Cross-dataset (CAD2016, CAMotion) is a Stage-D nice-to-have. |
| Baselines | **Cite published numbers** from baseline papers; deploy 0–2 ourselves only if a critical row is missing. | Standard VCOD practice; deploying 10 baselines is 1+ month of work for no novelty gain. |
| Reasoning benchmark | 266 samples (`paper §4`); 61 seed annotated already; user grows over weeks. Eval **after** Stage C. |
| Compute | Pre-download on 4090; train on **A100 80GB** elsewhere; rsync from `/root/autodl-tmp/` snapshot. |

---

## 3. What is already done (as of 2026-05-26 evening)

### Trajectory pipeline (pre-agent)

* **Phase 1 — Trajectory cache**: CoTracker3 trajectories pre-computed and
  saved for all 87 MoCA-Mask videos (`_traj_cache/`, ~69 MB).
* **Phase 2 — Main eval (oracle + heuristic) on MoCA-Mask test (16 videos)**:
    * Oracle F_w = 0.507  (vs SLT-Net 0.357, **+0.150**)
    * Heuristic F_w = 0.230
    * Result + per-video metrics in `results/MAIN_REPORT.md`.
* **24 ablations × 16 test videos**, full report in
  `results/ABLATIONS_REPORT.md`. **Per-metric best oracle config** beats
  SLT-Net on **every** metric:
    * F_w 0.631 (`minmove_2`), MAE 0.011 (`best_combo`), S_a 0.665 (`K_24`),
      E_p 0.840 (`best_combo`).
    * No single config wins all 4 → this is exactly the gap the MLLM agent
      should close.
* **Failure-mode diagnosis** per test video in `results/FAILURE_DIAGNOSIS.md`:
  5 videos cluster-recoverable, 4 lose target to dynamic filter, 7 mix
  target with background under K-means. Motivates the agent's `Add-Pos` /
  `Add-Neg` correction operators.
* **Kinematic Signature Encoder (v1)** trained but **failed** (contrastive
  loss flat, F_w 0.281 vs raw velocity 0.507). v2 with cross-video
  negatives + hard-neg mining is in Stage B.

### Stage A — agent foundation (code skeleton DONE, training NOT YET)

* **Pre-downloaded model weights** (rsync-ready under `/root/autodl-tmp/models/`):
    * `InternVL3-8B/`   ~16 GB
    * `InternVL3-2B/`   ~4 GB    (4090 debug)
    * `Qwen2.5-VL-7B-Instruct/`  ~16 GB    (backup)
    * `dinov2-large/`   ~2.4 GB
* **Agent module (`code/agent/`)** — 4 files, end-to-end pipeline works:
    * `actions.py` — 4-action vocabulary (`SELECT(k)` / `ADD_POS(f,x,y)` /
      `ADD_NEG(f,x,y)` / `TERMINATE`), tolerant regex parser, system prompt.
    * `state_builder.py` — cluster overview rendering, thumbnail strip,
      current-mask overlay strip.
    * `agent.py` — `InternVL3Agent` wrapper + `PromptStream` accumulator +
      `run_agent_on_video()` main loop.
    * `infer.py` — CLI entry point with per-video 4-COD-metric eval.
* **First smoke test** (arctic_fox, InternVL3-2B, K_max=5, **no training**):
    * 5 actions emitted, all parse cleanly:
      `SELECT(0) → ADD_NEG → SELECT(1) → SELECT(1) → SELECT(2)`
    * 30/30 frames get non-empty masks; MLLM↔SAM3 round-trip stable.
    * Metrics: F_w=0.000, MAE=0.032, S_α=0.461, E_φ=0.734 (18s).
    * **Quality is bad on purpose** — the 2B model zero-shot can't pick
      the right cluster. Plumbing is the goal here; Stage B fixes quality.
* **Per-step wall-clock**: InternVL3-2B forward 0.2-0.5 s, SAM 3 full-video
  re-propagation 2.7-2.8 s (dominant), so 5 steps ≈ 15-20 s per video.
* **Env reproducibility** (`code/setup_env_on_new_machine.sh`):
  auto-detects GPU sm version (sm_120 Blackwell → cu128 wheel; sm_8x → cu126),
  creates fresh `sam3` env, installs pinned `requirements_sam3_env.txt`,
  installs SAM 3 editable, optional flash-attn.

### Environment caveats (codified, will be folded into setup script)

* **transformers 5.9 is incompatible with InternVL3's custom modeling
  code** — pinned to **4.49.0**.
* **`accelerate ≥ 0.26.0`** required for `low_cpu_mem_usage=True`.
* **transformers dynamic-module cache occasionally fails to copy
  `configuration_intern_vit.py`** to the per-hash cache subdir. Workaround
  is a manual `cp` from the model directory; will codify in the setup
  script before A100 migration.

---

## 4. Roadmap (week-by-week)

### Stage A — Agent foundation  (Week 1–2)

> Goal: end-to-end agent inference works, even with no training.
> The first agent number on MoCA-Mask test, however bad.

* **A.1**  Pre-download all backbone weights (this 4090): InternVL3-8B,
  InternVL3-2B, Qwen2.5-VL-7B (fallback), DINOv2-large. **Status: in progress.**
* **A.2**  Action vocabulary parser (`code/agent/action_vocab.py`):
  emit / parse the 4 action types in InternVL3's grounding syntax.
* **A.3**  State serialization (`code/agent/state.py`):
  * render a single "cluster overview" image per video (representative frame
    with `{C_k}` trajectory points colored by cluster index);
  * concatenate `[text task prompt]` + `[4–6 thumbnail frames]` + `[cluster
    overview]` + `[current mask preview, if t>0]` + `[action history text]`
    into the MLLM's multimodal input;
  * keep all dimensions in InternVL3's expected format.
* **A.4**  Oracle action trajectory generator (`code/agent/oracle.py`):
  given (video, GT mask), simulate the agentic loop with mask-IoU
  supervision and record the optimal `(s_t, a_t)` pairs.
* **A.5**  Cold-start BC training loop (`code/agent/train_bc.py`):
  Qwen2.5-VL-7B / InternVL3-8B + rank-16 LoRA + autoregressive action-token
  loss. **First on 4090 with 2B for smoke test, then on A100 with 8B.**
* **A.6**  Inference loop (`code/agent/infer.py`):
  run up to `K_max=5` steps; on `Terminate` emit final mask.
* **A.7**  First end-to-end agent eval on MoCA-Mask test, with **no
  training** (random / heuristic + system prompt only) — sanity check that
  the plumbing works.
* **A.8**  Week-1 review: decide whether to keep / extend action vocabulary.

### Stage B — BC + RL training  (Week 3–4)

> Goal: trained agent that beats heuristic on at least 3/4 metrics.

* **B.1**  Improved Kinematic Signature Encoder: cross-video negatives,
  hard-negative mining, longer schedule. Re-eval on ablation row.
* **B.2**  BC training to convergence on A100. Save best ckpt.
* **B.3**  RL fine-tuning with GRPO:
    * reward = `IoU(M_final, M_GT) − λ * step_count`, `λ = 0.01`;
    * 4 rollouts per video, ~5 RL epochs;
    * standard policy-gradient on token-level log-probs of action tokens.
* **B.4**  All ablations from paper §5.5 with the trained agent:
    * full agent vs single-shot (`K_max = 1`, `SELECT`-only);
    * action-vocabulary subset (`-Add-Pos`, `-Add-Neg`, `SELECT` only);
    * BC vs BC+RL;
    * language-anchoring vs top-consistency cluster;
    * trajectory-as-memory vs SAM2-mask-memory-only.

### Stage C — Polish & paper-ready  (Week 5–6)

* **C.1**  Re-run main MoCA-Mask test result with the **fully trained agent
  + signature encoder**. Update `paper/TrajCamo_draft.tex` Table 1.
* **C.2**  Fill in `\TBD`'s in paper §5.3 / §5.5 from collected JSON.
* **C.3**  Qualitative figure (Figure 2): for 3 representative videos,
  panels of (a) sampled frames, (b) trajectory clusters by index, (c) MLLM
  selected `C*`, (d) final mask vs GT.
* **C.4**  Copy baseline numbers from the corresponding papers into Table 1
  (no deployment; mark `† reported from original paper` in caption).
* **C.5**  Limitations section update (real failure modes from
  `FAILURE_DIAGNOSIS.md`, agent's known weak cases).

### Stage D — Cross-dataset + reasoning benchmark  (Week 7+, optional / parallel)

* **D.1**  Download CAD2016 (Bideau & Learned-Miller 2016 ECCV); deploy
  zero-shot eval on the trained agent → fills Table 1 right block.
* **D.2**  Download CAMotion (2026 in-the-wild VCOD benchmark); zero-shot
  eval. Reviewers like cross-dataset transferability.
* **D.3**  Once user finishes the 266-sample reasoning benchmark:
  evaluate the agent in language-driven mode (the `text query` is no
  longer "find the camouflaged animal" but the user's actual query).
  Report F_w_β + reasoning success rate (`paper §5.4` Table 2).

### Stage E — Submission prep

* **E.1**  Final paper revision (probably 2 weeks of writing).
* **E.2**  Code cleanup, README, dataset / weight links, reproducibility checklist.
* **E.3**  TCSVT submission.

---

## 5. Pre-downloaded artifacts (on this 4090, /root/autodl-tmp/)

```
/root/autodl-tmp/
├── VOSdataset/                       # MoCA-Mask (1.2 GB) + traj cache (69 MB)
│   ├── TrainDataset_per_sq/   71 videos
│   ├── TestDataset_per_sq/    16 videos
│   ├── _traj_cache/           CoTracker3 trajectories for all 87 videos
│   └── _signature_encoder.pt  v1 encoder (badly trained, kept for reference)
├── models/                           # Downloaded fresh, ready to rsync
│   ├── InternVL3-8B/                 main agent backbone   (~16 GB)
│   ├── InternVL3-2B/                 debug agent backbone  (~4 GB)
│   ├── Qwen2.5-VL-7B-Instruct/       backup backbone       (~16 GB)
│   └── dinov2-large/                 frozen vision feats   (~2.4 GB)
├── sam3/                             # SAM 3 source code (pip-installed editable)
├── sam3_base_weights/                # SAM 3 base ckpt    (~3.3 GB)
├── sam3_weights/                     # SAM 3.1 multiplex  (~3.3 GB)
└── VOScode/                          # all experiment scripts (Phase 1 / 2 / 4, ablations, ...)
```

When ready to move to A100: `rsync -avh /root/autodl-tmp/ <a100>:/data/`.

---

## 6. Working agreement

* Both Claude and the human owner update `PROJECT_PLAN.md` whenever
  scope shifts.
* Each completed sub-stage produces a git commit on `main` with a result
  artifact under `results/`.
* Big architectural changes (action vocabulary, MLLM swap, dataset addition)
  require an explicit "ok" before execution.
* No new dataset / baseline deployment without an explicit go-ahead.
