"""
Stage B.4 — Agent ablation runner.

Evaluates several agent configurations on the 16 MoCA-Mask test videos and
writes per-config aggregate metrics + per-video metrics to a JSON report.

Ablation axes:
  • LoRA adapter:  no-LoRA  |  BC-only  |  BC+RL
  • Step budget:   K_max ∈ {1, 3, 5}
  • Action vocab:  full 4 actions  |  drop ADD_POS  |  drop ADD_NEG
                                   |  drop both (SELECT-only)

The 5 main ablation runs (paper §5.5):
  R1  RL agent + full vocab, K_max=5      ←  full system (best)
  R2  RL agent + SELECT-only, K_max=1     ←  single-shot baseline
  R3  RL agent + full vocab - ADD_POS, K_max=5
  R4  RL agent + full vocab - ADD_NEG, K_max=5
  R5  BC agent + full vocab, K_max=5      ←  ablation of RL stage

Usage:
  python run_agent_ablations.py \
      --base /root/autodl-tmp/models/InternVL3-8B \
      --bc_lora /root/autodl-tmp/VOScode/agent_outputs/bc_8b/lora_final \
      --rl_lora /root/autodl-tmp/VOScode/agent_outputs/grpo_8b/lora_best \
      --out /root/autodl-tmp/VOScode/agent_outputs/ablations
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import InternVL3Agent, run_agent_on_video
from cod_metrics import e_phi, f_beta_w, mae, s_alpha


def build_sam3():
    from sam3.model_builder import build_sam3_video_model
    m = build_sam3_video_model(
        checkpoint_path="/root/autodl-tmp/sam3_base_weights/sam3.pt",
        load_from_HF=False,
    )
    p = m.tracker
    p.backbone = m.detector.backbone
    return p


def eval_one_video(masks, frame_names, gt_dir, orig_h, orig_w) -> dict:
    per_frame = []
    for i, fname in enumerate(frame_names):
        gp = gt_dir / f"{fname}.png"
        if not gp.exists():
            continue
        m = masks.get(i)
        if m is None:
            mb = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            mb = m
            if mb.shape != (orig_h, orig_w):
                mb = cv2.resize(mb.astype(np.uint8), (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)
        gt = np.array(Image.open(gp))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = (gt > 0).astype(np.float64)
        pr = (mb > 0).astype(np.float64)
        per_frame.append(dict(
            frame=fname,
            F_w=f_beta_w(pr, gt), MAE=mae(pr, gt),
            S_a=s_alpha(pr, gt), E_p=e_phi(pr, gt),
        ))
    if not per_frame:
        return {"error": "no GT frames"}
    return {k: float(np.mean([m[k] for m in per_frame]))
            for k in ("F_w", "MAE", "S_a", "E_p")} | {"n_frames": len(per_frame)}


def run_one_ablation(
    cfg: dict, mllm: InternVL3Agent, sam3, test_root: Path, traj_root: Path,
    args,
) -> dict:
    """Run one ablation config on all test videos. Returns aggregate + per-video."""
    name = cfg["name"]
    K_max = cfg["K_max"]
    forbid = set(cfg.get("forbid", []))
    print(f"\n=========== {name} (K_max={K_max}, forbid={forbid or 'none'}) ===========",
          flush=True)
    rows = []
    videos = sorted([d for d in test_root.iterdir() if d.is_dir()])
    for vd in videos:
        cache = traj_root / f"{vd.name}.npz"
        if not cache.exists():
            continue
        t0 = time.time()
        res = run_agent_on_video(
            mllm, sam3, vd, cache,
            K=args.K, K_max=K_max,
            verbose=False,
            forbid_actions=forbid or None,
        )
        m = eval_one_video(res["masks"], res["frame_names"],
                           vd / "GT", res["orig_h"], res["orig_w"])
        elapsed = time.time() - t0
        if "error" not in m:
            print(f"  {vd.name:>24}: F_w={m['F_w']:.3f} "
                  f"MAE={m['MAE']:.4f} ({elapsed:.0f}s, {res['n_steps']} steps, "
                  f"hist={res['history']})", flush=True)
        rows.append({
            "name": vd.name, "metrics": m, "n_steps": res["n_steps"],
            "history": res["history"], "seconds": round(elapsed, 1),
        })
    # Aggregate
    agg = {}
    for k in ("F_w", "MAE", "S_a", "E_p"):
        vals = [r["metrics"][k] for r in rows if "error" not in r["metrics"]]
        if vals:
            agg[k] = float(np.mean(vals))
    agg["n_videos"] = len(rows)
    print(f"\n  >> {name} aggregate: "
          + " ".join(f"{k}={agg[k]:.3f}" for k in ("F_w","MAE","S_a","E_p") if k in agg),
          flush=True)
    return {"config": cfg, "aggregate": agg, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/InternVL3-8B")
    ap.add_argument("--bc_lora",
                    default="/root/autodl-tmp/VOScode/agent_outputs/bc_8b/lora_final")
    ap.add_argument("--rl_lora",
                    default="/root/autodl-tmp/VOScode/agent_outputs/grpo_8b/lora_best")
    ap.add_argument("--no_lora", action="store_true",
                    help="also run a no-LoRA / untrained baseline")
    ap.add_argument("--out", type=Path,
                    default=Path("/root/autodl-tmp/VOScode/agent_outputs/ablations"))
    ap.add_argument("--test_root", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/TestDataset_per_sq"))
    ap.add_argument("--traj_root", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_traj_cache/TestDataset_per_sq"))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated config names to run (default: all)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Build SAM 3 once
    print(f"[init] loading SAM 3", flush=True)
    sam3 = build_sam3()

    # The 5 main configs (RL-anchored). When --no_lora is on, we also add 2.
    configs = [
        # (name, lora, K_max, forbid)
        dict(name="R1_rl_full_K5",       lora=args.rl_lora, K_max=5, forbid=[]),
        dict(name="R2_rl_select_only_K1",lora=args.rl_lora, K_max=1, forbid=["ADD_POS","ADD_NEG"]),
        dict(name="R3_rl_no_addpos_K5",  lora=args.rl_lora, K_max=5, forbid=["ADD_POS"]),
        dict(name="R4_rl_no_addneg_K5",  lora=args.rl_lora, K_max=5, forbid=["ADD_NEG"]),
        dict(name="R5_bc_full_K5",       lora=args.bc_lora, K_max=5, forbid=[]),
    ]
    if args.no_lora:
        configs.append(dict(name="R0_nolora_full_K5", lora=None, K_max=5, forbid=[]))

    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c["name"] in wanted]
        print(f"[filter] running {[c['name'] for c in configs]}", flush=True)

    # Run each config — re-init the MLLM each time so LoRA changes apply.
    all_results = []
    for cfg in configs:
        lora = cfg["lora"]
        print(f"\n[init] loading MLLM {args.base} + LoRA {lora}", flush=True)
        mllm = InternVL3Agent(args.base, lora_path=lora)
        try:
            res = run_one_ablation(cfg, mllm, sam3, args.test_root,
                                    args.traj_root, args)
            all_results.append(res)
            with open(args.out / f"{cfg['name']}.json", "w") as f:
                json.dump(res, f, indent=2, default=str)
        finally:
            # Free MLLM memory before next config
            del mllm
            import torch
            torch.cuda.empty_cache()

    # Combined report
    report_path = args.out / "ablations_combined.json"
    with open(report_path, "w") as f:
        json.dump([{"config": r["config"], "aggregate": r["aggregate"]}
                   for r in all_results], f, indent=2)
    print(f"\n[done] {len(all_results)} configs run. "
          f"Combined: {report_path}", flush=True)

    # Pretty print summary table
    print(f"\n{'config':<30} {'F_w':>8} {'MAE':>8} {'S_a':>8} {'E_p':>8}")
    print("-" * 70)
    for r in all_results:
        a = r["aggregate"]
        print(f"{r['config']['name']:<30} "
              f"{a.get('F_w', 0):>8.3f} {a.get('MAE', 0):>8.4f} "
              f"{a.get('S_a', 0):>8.3f} {a.get('E_p', 0):>8.3f}")


if __name__ == "__main__":
    main()
