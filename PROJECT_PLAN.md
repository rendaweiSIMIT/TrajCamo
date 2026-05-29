# TrajCamo — Project Plan (TCSVT Submission)

**Target venue**: IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)
**Timeline**: ~2–3 months of focused work
**Owner**: rendaweiSIMIT
**Last updated**: 2026-05-29

> This file is the single source of truth for what we're building, in what
> order, and where we currently are. **Read this before opening other files.**
> Update the checkboxes whenever a step finishes; update sections 1-2 only
> when scope or decisions actually shift.

---

## 1. Paper thesis (locked)

> **Long-window trajectory grouping** over a dense point-tracker field
> **recovers camouflaged video targets that per-frame appearance- and
> instantaneous-motion-based methods miss**. A multimodal language agent
> **iteratively** selects and corrects these motion hypotheses against a
> query, producing both a final mask sequence and an externally inspectable
> reasoning trace.

The four pipeline stages are fixed (see `paper/TrajCamo_draft.tex`):

1. **Long-Window Trajectory Field** — CoTracker3 offline
2. **Kinematic Signature Encoder** — small temporal Transformer + contrastive loss
3. **Long-Window Trajectory Grouping** — spectral clustering on motion-and-locality affinity
4. **Query-conditioned Agentic Cluster Refinement** — MLLM agent emits `SELECT(k)` / `Add-Pos(f,x,y)` / `Add-Neg(f,x,y)` / `Terminate` actions interacting with frozen SAM 3

---

## 2. Decisions already made (don't revisit without explicit "ok")

| Decision | Value |
|---|---|
| MLLM backbone (main) | **InternVL3-8B** |
| MLLM backbone (debug) | InternVL3-2B (small, fast iteration) |
| MLLM backbone (backup) | Qwen2.5-VL-7B |
| Vision features | DINOv2-large (frozen) |
| Mask backbone | SAM 3 base (frozen) |
| Training scheme | **BC + RL (GRPO)** — RL is mandatory |
| Action vocabulary | 4 actions: `SELECT`, `Add-Pos`, `Add-Neg`, `Terminate` |
| `K_max` step budget | 5 (default) |
| Datasets (now) | **MoCA-Mask only** (71 train / 16 test) |
| Cross-dataset (later) | CAD2016 + CAMotion (Stage D) |
| Baselines | **Cite published numbers**, no self-deployment |
| Reasoning benchmark | 266 samples; user annotates over weeks |
| Compute | Single GPU — A100 80GB or RTX PRO 6000 Blackwell 96GB |

---

## 3. Progress checklist

Legend:  ✅ done  ·  🟡 in progress  ·  ☐ pending

### Stage 0 — Foundation (one-shot, no need to revisit)

- ✅ MoCA-Mask dataset downloaded (`VOSdataset/`, 71 train + 16 test)
- ✅ CoTracker3 trajectory pre-cache for all 87 videos (`_traj_cache/`, ~69 MB)
- ✅ Model weights downloaded to `/root/autodl-tmp/models/`:
  InternVL3-8B / InternVL3-2B / Qwen2.5-VL-7B / DINOv2-large
- ✅ Env reproducibility script (`code/setup_env_on_new_machine.sh`)
- ✅ Migrated to RTX PRO 6000 Blackwell 96GB; env rebuilt
  (torch+cu128, transformers 4.49, peft 0.19, trl 0.17, flash-attn 2.8.3)

### Stage A — Agent foundation (Week 1-2)

Goal: end-to-end agent inference + first BC-trained agent number on MoCA-Mask test.

**A.1 Trajectory pipeline (sanity check before agent)**
- ✅ Phase-2 main eval on test set:  oracle F_w **0.507** vs SLT-Net 0.357 (+0.150)
- ✅ 24 ablations × 16 test videos (`results/ABLATIONS_REPORT.md`)
- ✅ Per-video failure-mode diagnosis (`results/FAILURE_DIAGNOSIS.md`)

**A.2 Agent code skeleton**
- ✅ `code/agent/actions.py` — 4-action vocabulary + tolerant parser + system prompt
- ✅ `code/agent/state_builder.py` — cluster overview, thumbnails, mask-overlay images
- ✅ `code/agent/agent.py` — `InternVL3Agent` + `PromptStream` + `run_agent_on_video`
- ✅ `code/agent/infer.py` — CLI with 4-COD-metric eval
- ✅ End-to-end smoke (arctic_fox, InternVL3-2B, NO training):
  5 actions all parse, MLLM↔SAM3 round-trip stable, F_w=0.0 (expected)

**A.3 Oracle action generator** (`code/agent/oracle.py`)
- 🟡 Implemented: greedy `SELECT(argmax-IoU cluster)` → `Add-Pos/Add-Neg` corrections at worst-IoU frame → `Terminate` at IoU ≥ 0.85 or K_max=5
- 🟡 **Running on all 71 train videos right now** — outputs go to `agent_outputs/oracle/`
- ☐ Inspect oracle quality: action-length distribution, final-IoU histogram, percentage of videos converging to IoU ≥ 0.85

**A.4 BC (behavior cloning) training loop** (`code/agent/train_bc.py`)
- ☐ Build HuggingFace `Dataset` from oracle's `index.jsonl`
- ☐ Multimodal data collator (text + 2-3 PIL images per sample)
- ☐ Wrap InternVL3 with rank-16 LoRA on attention + MLP projections
- ☐ TRL `SFTTrainer` with autoregressive loss on **action tokens only**
- ☐ Smoke train on InternVL3-2B + 5 videos for 1 epoch (verify gradient flow)
- ☐ Full train on InternVL3-8B + 71 videos for 30 epochs (~3-4h on Blackwell)
- ☐ Save best LoRA adapter to `VOScode/agent_outputs/bc_ckpt/`

**A.5 First trained-agent eval on test set**
- ☐ Load BC-trained agent, run on all 16 MoCA-Mask test videos
- ☐ Compare BC-agent vs untrained-agent vs phase-2-heuristic on F_w/MAE/S_a/E_p
- ☐ Save per-video metrics to `results/agent_bc_first.json`
- ☐ **Hard gate for Stage B**: BC-agent F_w ≥ heuristic-agent F_w (= 0.230).
  If lower, debug data quality / loss masking / LoRA target modules before RL.

**A.6 Stage A review**
- ☐ Decide whether action vocab is sufficient (or add `Merge` / `Split`)
- ☐ Decide whether per-step BC supervision is sufficient or needs auxiliary losses

### Stage B — Full BC + RL training (Week 3-4)

Goal: trained agent that beats heuristic on at least 3/4 COD metrics on MoCA-Mask test.

**B.1 Improved Kinematic Signature Encoder (v2)**
- ☐ Add cross-video negatives + hard-negative mining
- ☐ Train v2; re-eval on the encoder-ablation row (currently F_w 0.281 v1)
- ☐ If v2 beats raw-velocity (0.507 oracle), bake into the agent pipeline

**B.2 BC training, hardened**
- ☐ Multi-epoch full run with logged train/val loss and per-epoch IoU on a held-out val split
- ☐ Save best checkpoint by val IoU

**B.3 RL fine-tuning with GRPO**
- ☐ Reward = `IoU(M_final, M_GT) − λ · step_count`, `λ = 0.01`
- ☐ 4 rollouts per video, ~5 RL epochs
- ☐ TRL `GRPOTrainer` configured for multimodal inputs (custom rollout function)
- ☐ Save RL-tuned adapter to `VOScode/agent_outputs/rl_ckpt/`

**B.4 Agent ablations (paper §5.5)**
- ☐ Full agent vs single-shot (`K_max=1`, SELECT-only)
- ☐ Action-vocab subsets: drop `Add-Pos`, drop `Add-Neg`, both dropped
- ☐ BC-only vs BC+RL
- ☐ Language-anchored cluster vs top-consistency cluster
- ☐ Trajectory-as-memory vs SAM2-mask-memory-only
- ☐ Compile all into `results/AGENT_ABLATIONS.md`

### Stage C — Polish & paper-ready (Week 5-6)

- ☐ **C.1** Re-run main MoCA-Mask test result with fully trained agent
- ☐ **C.2** Update `paper/TrajCamo_draft.tex` Table 1 with real numbers (no more `\TBD`)
- ☐ **C.3** Fill all `\TBD`s in §5.5 ablation tables from collected JSON
- ☐ **C.4** Qualitative figure (`paper/figures/qualitative.pdf`):
  3 videos × 4 panels (frames / trajectory clusters / agent's C* / final mask vs GT)
- ☐ **C.5** Copy baseline numbers from each baseline's published paper into Table 1
- ☐ **C.6** Limitations section update from `results/FAILURE_DIAGNOSIS.md`

### Stage D — Cross-dataset + reasoning benchmark (Week 7+, optional)

- ☐ **D.1** Download CAD2016; zero-shot eval; add to Table 1 right block
- ☐ **D.2** Download CAMotion; zero-shot eval
- ☐ **D.3** Once user finishes annotating 266 reasoning samples:
  zero-shot agent eval with language queries; report F_w_β + RSR (paper §5.4)

### Stage E — Submission prep

- ☐ **E.1** Final paper revision (~2 weeks of writing)
- ☐ **E.2** Code cleanup, README, reproducibility checklist
- ☐ **E.3** TCSVT submission

---

## 4. File / directory layout on the working machine

```
/root/autodl-tmp/
├── VOSdataset/                          MoCA-Mask
│   ├── TrainDataset_per_sq/             71 videos
│   ├── TestDataset_per_sq/              16 videos
│   ├── _traj_cache/                     CoTracker3 trajectories (87 videos)
│   └── _signature_encoder.pt            v1 encoder (broken, kept for record)
├── models/                              MLLM + vision backbones (~39 GB)
│   ├── InternVL3-8B/                    main agent backbone
│   ├── InternVL3-2B/                    debug agent backbone
│   ├── Qwen2.5-VL-7B-Instruct/          backup backbone
│   └── dinov2-large/                    frozen vision features
├── sam3/                                SAM 3 source code (editable install)
├── sam3_base_weights/                   SAM 3 ckpt (~3.3 GB)
├── sam3_weights/                        SAM 3.1 multiplex (~3.3 GB)
├── VOScode/                             all experiment scripts
│   ├── phase1_precache_trajectories.py
│   ├── phase2_main_eval.py
│   ├── phase4_train_signature_encoder.py
│   ├── ablation_runner.py
│   ├── compile_ablation_report.py
│   ├── diagnose_failures.py
│   ├── cod_metrics.py
│   └── agent/                           agent module
│       ├── actions.py
│       ├── state_builder.py
│       ├── agent.py
│       ├── oracle.py                    A.3
│       └── infer.py
└── TrajCamo_repo/                       this git repo, mirrors to GitHub `main`
    ├── code/                            mirrors of VOScode/*.py
    ├── paper/                           LaTeX draft
    ├── results/                         JSON + markdown reports
    └── PROJECT_PLAN.md                  THIS FILE
```

When migrating to another machine: `rsync -avh /root/autodl-tmp/ <target>:/data/`,
then run `bash code/setup_env_on_new_machine.sh`.

---

## 5. Working agreement

* Update the **checklist in §3** whenever any step finishes (just flip ☐ → ✅).
* Each completed sub-stage produces one git commit on `main` with a result
  artifact under `results/`.
* Big architectural changes (action vocab, MLLM swap, dataset addition)
  require an explicit "ok" from the owner before execution.
* No new dataset / baseline deployment without an explicit go-ahead.
