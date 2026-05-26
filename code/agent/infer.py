"""
CLI entry point: end-to-end agent inference on one (or all) MoCA-Mask test videos.

Usage:
    python infer.py --video arctic_fox --mllm /root/autodl-tmp/models/InternVL3-2B
    python infer.py --all
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
    p = m.tracker; p.backbone = m.detector.backbone
    return p


def eval_one_video(masks, frame_names, gt_dir, orig_h, orig_w) -> dict:
    """Compute per-frame F_w, MAE, S_a, E_p on frames where GT exists."""
    per_frame = []
    for i, fname in enumerate(frame_names):
        gt_path = gt_dir / f"{fname}.png"
        if not gt_path.exists():
            continue
        m = masks.get(i)
        if m is None:
            mb = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            mb = m
            if mb.shape != (orig_h, orig_w):
                mb = cv2.resize(mb.astype(np.uint8), (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)
        gt = np.array(Image.open(gt_path))
        if gt.ndim == 3: gt = gt[..., 0]
        gt = (gt > 0).astype(np.float64)
        pr = (mb > 0).astype(np.float64)
        per_frame.append({
            "frame": fname,
            "F_w": f_beta_w(pr, gt), "MAE": mae(pr, gt),
            "S_a": s_alpha(pr, gt), "E_p": e_phi(pr, gt),
        })
    if not per_frame:
        return {"error": "no GT frames"}
    agg = {k: float(np.mean([m[k] for m in per_frame]))
           for k in ("F_w", "MAE", "S_a", "E_p")}
    agg["n_frames"] = len(per_frame)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mllm", default="/root/autodl-tmp/models/InternVL3-2B",
                    help="path to InternVL3 model dir")
    ap.add_argument("--video", default=None,
                    help="single video name (e.g. arctic_fox)")
    ap.add_argument("--all", action="store_true",
                    help="run all 16 test videos")
    ap.add_argument("--dataset", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset"))
    ap.add_argument("--traj_cache", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_traj_cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("/root/autodl-tmp/VOScode/agent_outputs"))
    ap.add_argument("--K_max", type=int, default=5)
    ap.add_argument("--K", type=int, default=8)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Build models
    print(f"[init] loading SAM 3 ...", flush=True)
    sam3 = build_sam3()
    print(f"[init] loading MLLM from {args.mllm} ...", flush=True)
    mllm = InternVL3Agent(args.mllm)
    print(f"[init] ready.", flush=True)

    test_root = args.dataset / "TestDataset_per_sq"
    if args.video:
        videos = [test_root / args.video]
        if not videos[0].exists():
            print(f"[error] video {args.video} not found")
            return
    elif args.all:
        videos = sorted([d for d in test_root.iterdir() if d.is_dir()])
    else:
        print("[error] specify --video <name> or --all")
        return

    all_rows = []
    for vd in videos:
        name = vd.name
        cache = args.traj_cache / "TestDataset_per_sq" / f"{name}.npz"
        if not cache.exists():
            print(f"  [skip] {name}: no cache")
            continue
        print(f"\n=== {name} ===")
        t0 = time.time()
        res = run_agent_on_video(
            mllm, sam3, vd, cache,
            K=args.K, K_max=args.K_max, verbose=True,
        )
        metrics = eval_one_video(res["masks"], res["frame_names"],
                                 vd / "GT", res["orig_h"], res["orig_w"])
        res["metrics"] = metrics
        res["seconds"] = round(time.time() - t0, 1)
        all_rows.append({k: v for k, v in res.items() if k != "masks"})

        if "error" not in metrics:
            print(f"  metrics: F_w={metrics['F_w']:.3f} MAE={metrics['MAE']:.4f} "
                  f"S_a={metrics['S_a']:.3f} E_p={metrics['E_p']:.3f}  "
                  f"({res['seconds']}s, {res['n_steps']} steps)")
        else:
            print(f"  metrics: {metrics}")
        print(f"  history: {res['history']}")

    # Aggregate
    if all_rows:
        vals = {k: [] for k in ("F_w", "MAE", "S_a", "E_p")}
        for r in all_rows:
            m = r["metrics"]
            if "error" in m: continue
            for k in vals: vals[k].append(m[k])
        if all(len(vals[k]) for k in vals):
            print(f"\n=== aggregate over {len(vals['F_w'])} videos ===")
            for k in vals:
                print(f"  {k}: {np.mean(vals[k]):.3f}")

    with open(args.out / "agent_results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"\n[written] {args.out / 'agent_results.json'}")


if __name__ == "__main__":
    main()
