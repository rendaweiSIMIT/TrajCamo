"""
Phase 1: Pre-cache CoTracker3 trajectories for all 87 MoCA-Mask videos
(71 train + 16 test). Caches to /root/autodl-tmp/VOSdataset/_traj_cache/.

Each output .npz contains:
    tracks:      (T, N, 2) float32 — point positions in PADDED (target_h, target_w)
    visibility:  (T, N)    bool    — per-frame visibility flags
    frame_names: list of str       — original JPEG filenames (with extension stripped)
    grid_size:   int               — N = grid_size**2 before pruning
    orig_h, orig_w: int            — original video dims
    new_h, new_w:   int            — resized dims (before padding)
    target_h, target_w: int        — padded dims (used by CoTracker3)
    scale:       float             — orig → resized scale

Usage:
    python phase1_precache_trajectories.py
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# Send torch.hub cache to data disk to spare system disk (CoTracker3 weights already there from earlier)
os.environ.setdefault("TORCH_HOME", "/root/.cache/torch")  # keep weights where torch.hub put them


def load_video(img_dir: Path, target_h: int = 384, target_w: int = 512):
    paths = sorted(
        img_dir.glob("*.jpg"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    frame_names = [p.stem for p in paths]
    if not paths:
        return None
    frames_orig = [np.array(Image.open(p).convert("RGB")) for p in paths]
    H0, W0 = frames_orig[0].shape[:2]
    scale = min(target_h / H0, target_w / W0)
    new_h, new_w = int(H0 * scale), int(W0 * scale)
    resized = [cv2.resize(f, (new_w, new_h)) for f in frames_orig]
    padded = [
        cv2.copyMakeBorder(r, 0, target_h - new_h, 0, target_w - new_w,
                           cv2.BORDER_CONSTANT, value=0)
        for r in resized
    ]
    arr = np.stack(padded)  # (T, H, W, 3) uint8
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
    return dict(
        video=video,
        frame_names=frame_names,
        orig_h=H0, orig_w=W0,
        new_h=new_h, new_w=new_w,
        target_h=target_h, target_w=target_w,
        scale=scale,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/VOSdataset"))
    ap.add_argument("--cache",   type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_traj_cache"))
    ap.add_argument("--grid",    type=int, default=48)
    ap.add_argument("--target_h", type=int, default=384)
    ap.add_argument("--target_w", type=int, default=512)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Process only these video names (debug)")
    args = ap.parse_args()

    print(f"[init] loading CoTracker3 (offline) ...", flush=True)
    model = torch.hub.load(
        "facebookresearch/co-tracker", "cotracker3_offline"
    ).cuda().eval()
    print(f"[init] ready, grid={args.grid}", flush=True)

    args.cache.mkdir(parents=True, exist_ok=True)
    splits = ["TrainDataset_per_sq", "TestDataset_per_sq"]
    all_videos = []
    for split in splits:
        root = args.dataset / split
        if not root.exists():
            continue
        for v in sorted(root.iterdir()):
            if v.is_dir() and (v / "Imgs").exists():
                if args.only and v.name not in args.only:
                    continue
                all_videos.append((split, v))
    print(f"[init] {len(all_videos)} videos to process", flush=True)

    total_t = time.time()
    n_done = 0
    n_skip = 0
    for split, vd in all_videos:
        out_dir = args.cache / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{vd.name}.npz"
        if out_path.exists():
            n_skip += 1
            continue
        t0 = time.time()
        info = load_video(vd / "Imgs", args.target_h, args.target_w)
        if info is None or info["video"].shape[1] == 0:
            print(f"  [skip-empty] {split}/{vd.name}", flush=True)
            continue
        try:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tracks, vis = model(info["video"], grid_size=args.grid)
            tracks = tracks[0].float().cpu().numpy()  # (T, N, 2)
            vis_np = vis[0].float().cpu().numpy().astype(bool)  # (T, N)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [error] {split}/{vd.name}: {e}", flush=True)
            del info
            torch.cuda.empty_cache()
            continue
        # Cast to float32 (small enough) — float16 loses precision for trajectory coords
        np.savez_compressed(
            out_path,
            tracks=tracks.astype(np.float32),
            visibility=vis_np,
            frame_names=np.array(info["frame_names"]),
            grid_size=args.grid,
            orig_h=info["orig_h"], orig_w=info["orig_w"],
            new_h=info["new_h"], new_w=info["new_w"],
            target_h=info["target_h"], target_w=info["target_w"],
            scale=info["scale"],
        )
        size_mb = out_path.stat().st_size / 1e6
        n_done += 1
        elapsed = time.time() - t0
        T = info["video"].shape[1]
        print(
            f"  [done] {split}/{vd.name:<32} T={T:>4} N={tracks.shape[1]} "
            f"{elapsed:>5.1f}s  {size_mb:>5.1f}MB",
            flush=True,
        )
        del info, tracks, vis_np
        torch.cuda.empty_cache()

    print(
        f"\n[summary] processed {n_done} new, skipped {n_skip} cached, "
        f"total wall-clock {time.time()-total_t:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
