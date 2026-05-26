"""
Phase 2: TrajCamo main evaluation on MoCA-Mask test split (16 videos).

Pipeline per test video:
  1. Load cached CoTracker3 trajectories.
  2. Compute per-trajectory features (raw velocity time series, L2-normalized).
  3. K-means cluster trajectories (K=8 default).
  4. Pick cluster by two rules:
        - oracle:    max IoU(dilated-cluster-region, GT-union)
        - heuristic: max consistency score (within-cluster variance ratio)
  5. Sample ~12 representative trajectory points from the selected cluster
     at frame 0 (FPS). Feed as positive point prompts to SAM 3 base tracker.
  6. Propagate through video, save mask sequence.
  7. Compute F_w, MAE, S_alpha, E_phi per frame, aggregate.

Reports: per-video metrics + dataset means.

Comparison baseline (SLT-Net, CVPR 2022, published numbers on MoCA-Mask test):
    S_alpha = 0.656, F_w_beta = 0.357, MAE = 0.030, E_phi = 0.785

Usage:
    python phase2_main_eval.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from cod_metrics import mae, f_beta_w, s_alpha, e_phi


# ---------------------------------------------------------------------------
# Feature extraction (raw velocity — winner from fair comparison)
# ---------------------------------------------------------------------------
def feat_raw_velocity(tracks: np.ndarray) -> np.ndarray:
    v = np.diff(tracks, axis=0)                  # (T-1, N, 2)
    f = v.transpose(1, 0, 2).reshape(v.shape[1], -1)   # (N, 2*(T-1))
    return normalize(f, axis=1)


# ---------------------------------------------------------------------------
# Cluster utilities
# ---------------------------------------------------------------------------
def cluster_to_region_padded(tracks_padded: np.ndarray, mask: np.ndarray,
                              H: int, W: int, dilate_px: int = 21) -> np.ndarray:
    region = np.zeros((H, W), dtype=np.uint8)
    pts = tracks_padded[:, mask].reshape(-1, 2)
    xs = pts[:, 0].round().astype(int).clip(0, W - 1)
    ys = pts[:, 1].round().astype(int).clip(0, H - 1)
    region[ys, xs] = 1
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        region = cv2.dilate(region, k, iterations=1)
    return region.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return 0.0 if u == 0 else float((a & b).sum() / u)


def cluster_consistency(features: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Within-cluster variance explained by cluster mean: 1 - normalized variance."""
    sel = labels == k
    if sel.sum() < 2:
        return 0.0
    fs = features[sel]
    mu = fs.mean(axis=0, keepdims=True)
    within = ((fs - mu) ** 2).sum(axis=1).mean()
    # total variance baseline
    overall_mu = features.mean(axis=0, keepdims=True)
    total = ((features - overall_mu) ** 2).sum(axis=1).mean()
    return float(1.0 - within / (total + 1e-8))


def fps_sample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Farthest-point sampling — return indices."""
    if len(points) <= n:
        return np.arange(len(points))
    rng = np.random.RandomState(seed)
    idxs = [rng.randint(len(points))]
    dists = np.linalg.norm(points - points[idxs[0]], axis=1)
    for _ in range(n - 1):
        next_idx = int(np.argmax(dists))
        idxs.append(next_idx)
        new_d = np.linalg.norm(points - points[next_idx], axis=1)
        dists = np.minimum(dists, new_d)
    return np.array(idxs)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_trajectory_cache(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return dict(
        tracks=data["tracks"],
        visibility=data["visibility"],
        frame_names=list(data["frame_names"]),
        grid_size=int(data["grid_size"]),
        orig_h=int(data["orig_h"]), orig_w=int(data["orig_w"]),
        new_h=int(data["new_h"]),   new_w=int(data["new_w"]),
        target_h=int(data["target_h"]), target_w=int(data["target_w"]),
        scale=float(data["scale"]),
    )


def load_gt_union(gt_dir: Path, target_h: int, target_w: int,
                  new_h: int, new_w: int) -> np.ndarray:
    """Union of all GT masks, scaled into padded (target_h, target_w)."""
    gt = np.zeros((target_h, target_w), dtype=np.uint8)
    for p in sorted(gt_dir.glob("*.png")):
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[..., 0]
        m_resized = cv2.resize(
            (m > 0).astype(np.uint8), (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )
        gt[:new_h, :new_w] |= m_resized
    return gt.astype(bool)


def load_gt_binary_at(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# SAM 3 base predictor wrapper
# ---------------------------------------------------------------------------
def build_sam3_tracker(ckpt_path: Path):
    from sam3.model_builder import build_sam3_video_model
    print(f"[init] building SAM 3 base ...", flush=True)
    sam3_model = build_sam3_video_model(
        checkpoint_path=str(ckpt_path),
        load_from_HF=False,
    )
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    print(f"[init] SAM 3 ready.", flush=True)
    return predictor


def sam3_propagate_from_points(predictor, imgs_dir: Path, point_prompts_xy01,
                                num_frames: int):
    """
    Add positive point prompts (in relative [0,1]) at frame 0 to SAM 3 tracker
    and propagate. Returns dict {frame_idx_in_imgs_dir: binary_mask (H, W)}.
    """
    state = predictor.init_state(video_path=str(imgs_dir))
    pts = torch.tensor(point_prompts_xy01, dtype=torch.float32)
    labels = torch.ones(len(point_prompts_xy01), dtype=torch.int32)
    predictor.add_new_points(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=pts,
        labels=labels,
        clear_old_points=True,
    )
    predictor.propagate_in_video_preflight(state, run_mem_encoder=True)

    per_frame = {}
    iterator = predictor.propagate_in_video(
        state, start_frame_idx=0, max_frame_num_to_track=num_frames, reverse=False,
    )
    for frame_idx, obj_ids, _, video_res_masks, _ in iterator:
        if len(obj_ids) == 0:
            per_frame[frame_idx] = None
        else:
            mask = (video_res_masks[0] > 0).cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            per_frame[frame_idx] = mask.astype(np.uint8)
    return per_frame


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------
def evaluate_video(predictor, video_dir: Path, traj_cache: Path,
                   out_dir: Path, K: int = 8, n_prompt_points: int = 12):
    name = video_dir.name
    imgs_dir = video_dir / "Imgs"
    gt_dir = video_dir / "GT"
    cache_path = traj_cache / "TestDataset_per_sq" / f"{name}.npz"
    if not cache_path.exists():
        return {"name": name, "error": "no trajectory cache"}

    info = load_trajectory_cache(cache_path)
    tracks_p = info["tracks"]          # (T, N, 2) in padded coords
    vis_p = info["visibility"]          # (T, N) bool
    frame_names = info["frame_names"]
    target_h, target_w = info["target_h"], info["target_w"]
    new_h, new_w = info["new_h"], info["new_w"]
    orig_h, orig_w = info["orig_h"], info["orig_w"]
    scale = info["scale"]

    # Drop trajectories rarely visible
    keep = vis_p.mean(axis=0) >= 0.2
    tracks_v = tracks_p[:, keep]                 # (T, Nv, 2)
    vis_v = vis_p[:, keep]

    # Filter dynamic trajectories: total spatial range > MIN_MOVE_PX.
    # Static background trajectories carry no motion information, and after L2
    # normalization their feature vectors collapse to random unit directions,
    # which causes K-means to merge them all into one huge cluster.
    MIN_MOVE_PX = 4.0   # in padded pixel coords (image is 384x512)
    pos_range = tracks_v.max(axis=0) - tracks_v.min(axis=0)   # (Nv, 2)
    movement = np.linalg.norm(pos_range, axis=1)              # (Nv,)
    dynamic = movement > MIN_MOVE_PX
    if dynamic.sum() < K * 5:
        # too few dynamic — fall back to all visible
        dynamic = np.ones_like(dynamic, dtype=bool)
    tracks_k = tracks_v[:, dynamic]
    vis_k = vis_v[:, dynamic]

    # Features + cluster (only on dynamic subset)
    F = feat_raw_velocity(tracks_k)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(F)
    labels = km.labels_

    # GT union for oracle scoring
    gt_union_padded = load_gt_union(gt_dir, target_h, target_w, new_h, new_w)

    # Score each cluster
    cluster_info = []
    for k in range(K):
        sel = labels == k
        if sel.sum() < 5:
            continue
        region = cluster_to_region_padded(tracks_k, sel, target_h, target_w)
        cluster_iou_oracle = iou(region, gt_union_padded)
        consistency = cluster_consistency(F, labels, k)
        cluster_info.append(dict(
            k=int(k), size=int(sel.sum()),
            iou_oracle=cluster_iou_oracle,
            consistency=consistency,
        ))

    if not cluster_info:
        return {"name": name, "error": "no valid clusters"}

    # Two selection rules
    best_oracle = max(cluster_info, key=lambda c: c["iou_oracle"])
    best_heuristic = max(cluster_info, key=lambda c: c["consistency"])

    results = {"name": name, "n_frames": len(frame_names),
               "n_trajectories": int(tracks_k.shape[1]),
               "cluster_info": cluster_info}

    # Run SAM 3 propagation for each selected cluster
    for label_name, chosen in [("oracle", best_oracle), ("heuristic", best_heuristic)]:
        chosen_k = chosen["k"]
        sel = labels == chosen_k
        # Get visible cluster points on frame 0 in PADDED coords
        f0_visible = vis_k[0] & sel
        f0_pts_padded = tracks_k[0, f0_visible]  # (Nv, 2)
        if len(f0_pts_padded) == 0:
            # fall back to any visible frame
            for t in range(tracks_k.shape[0]):
                ft_visible = vis_k[t] & sel
                if ft_visible.sum() >= n_prompt_points:
                    f0_pts_padded = tracks_k[t, ft_visible]
                    break
        if len(f0_pts_padded) == 0:
            results[label_name] = {"error": "no visible cluster points"}
            continue

        # Use cluster CENTROID + a tight neighborhood of its closest points.
        # This avoids feeding SAM3 a set of points scattered across the
        # whole frame (which happens when a cluster degenerates into a
        # large mega-cluster).
        centroid = f0_pts_padded.mean(axis=0)
        dists = np.linalg.norm(f0_pts_padded - centroid, axis=1)
        order = np.argsort(dists)
        # Median radius gives a robust local-cluster bound; cap at min of {n_prompt_points, half} points
        n_take = min(n_prompt_points, max(3, len(f0_pts_padded) // 4))
        prompt_pts_padded = f0_pts_padded[order[:n_take]]
        # Discard any prompt that's > 2× median dist away (outlier guard)
        med_d = np.median(dists)
        kept_mask = np.linalg.norm(prompt_pts_padded - centroid, axis=1) <= max(med_d * 2, 30.0)
        prompt_pts_padded = prompt_pts_padded[kept_mask]
        if len(prompt_pts_padded) == 0:
            prompt_pts_padded = centroid[None, :]

        # Map PADDED -> ORIGINAL coords -> [0,1] relative
        # Padded coords are in (new_h, new_w) area within (target_h, target_w); padding is bottom/right.
        # Original coords = padded coords / scale, clipped to (orig_w, orig_h).
        prompt_pts_orig = prompt_pts_padded / scale
        prompt_pts_orig[:, 0] = prompt_pts_orig[:, 0].clip(0, orig_w - 1)
        prompt_pts_orig[:, 1] = prompt_pts_orig[:, 1].clip(0, orig_h - 1)
        prompt_pts_rel = np.stack(
            [prompt_pts_orig[:, 0] / orig_w, prompt_pts_orig[:, 1] / orig_h], axis=1,
        )

        # Run SAM 3 propagation
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_frame_masks = sam3_propagate_from_points(
                    predictor, imgs_dir, prompt_pts_rel.tolist(),
                    num_frames=len(frame_names),
                )
        except Exception as e:
            import traceback; traceback.print_exc()
            results[label_name] = {"error": str(e)}
            continue

        # Save masks + compute metrics
        m_dir = out_dir / name / label_name / "masks"
        m_dir.mkdir(parents=True, exist_ok=True)

        per_frame_metrics = []
        for i, fname in enumerate(frame_names):
            mask = per_frame_masks.get(i)
            if mask is None:
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)
            # Save indexed mask
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(
                m_dir / f"{fname}.png"
            )
            gt_path = gt_dir / f"{fname}.png"
            if not gt_path.exists():
                continue
            gt = load_gt_binary_at(gt_path).astype(np.float64)
            pr = (mask > 0).astype(np.float64)
            per_frame_metrics.append({
                "frame": fname,
                "F_w": f_beta_w(pr, gt),
                "MAE": mae(pr, gt),
                "S_a": s_alpha(pr, gt),
                "E_p": e_phi(pr, gt),
            })

        if per_frame_metrics:
            agg = {k: float(np.mean([m[k] for m in per_frame_metrics]))
                   for k in ("F_w", "MAE", "S_a", "E_p")}
            agg["n_frames"] = len(per_frame_metrics)
        else:
            agg = {"error": "no GT frames"}
        results[label_name] = {
            "selected_cluster": chosen,
            "n_prompt_points": len(prompt_pts_rel),
            "metrics": agg,
        }
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/VOSdataset"))
    ap.add_argument("--traj_cache", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_traj_cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("/root/autodl-tmp/VOScode/main_outputs"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/root/autodl-tmp/sam3_base_weights/sam3.pt"))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_prompt_points", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    predictor = build_sam3_tracker(args.ckpt)

    test_root = args.dataset / "TestDataset_per_sq"
    videos = sorted([d for d in test_root.iterdir() if d.is_dir()])
    if args.only:
        videos = [v for v in videos if v.name in set(args.only)]

    all_results = []
    print(f"\n[start] evaluating {len(videos)} test videos\n", flush=True)
    for v in videos:
        t0 = time.time()
        res = evaluate_video(predictor, v, args.traj_cache, args.out,
                             K=args.K, n_prompt_points=args.n_prompt_points)
        res["seconds"] = round(time.time() - t0, 1)
        all_results.append(res)
        # Pretty-print one line per video
        def fmt(d):
            if not isinstance(d, dict) or "metrics" not in d:
                return "ERR"
            m = d["metrics"]
            if "error" in m:
                return "ERR"
            return f"F_w={m['F_w']:.3f} MAE={m['MAE']:.4f} S_a={m['S_a']:.3f} E_p={m['E_p']:.3f}"
        print(
            f"  {res['name']:<28} T={res.get('n_frames','?'):>4} "
            f"({res['seconds']:>5.1f}s)  "
            f"oracle:{fmt(res.get('oracle',{}))}   "
            f"heuristic:{fmt(res.get('heuristic',{}))}",
            flush=True,
        )

    # Aggregate
    print("\n========= aggregate (MoCA-Mask test, 16 videos) =========", flush=True)
    for sel in ("oracle", "heuristic"):
        vals = {k: [] for k in ("F_w", "MAE", "S_a", "E_p")}
        for r in all_results:
            d = r.get(sel, {})
            if isinstance(d, dict) and "metrics" in d and "error" not in d["metrics"]:
                for k in vals:
                    vals[k].append(d["metrics"][k])
        if all(len(vals[k]) for k in vals):
            print(
                f"  {sel:<10}  F_w={np.mean(vals['F_w']):.3f}  "
                f"MAE={np.mean(vals['MAE']):.4f}  "
                f"S_a={np.mean(vals['S_a']):.3f}  "
                f"E_p={np.mean(vals['E_p']):.3f}",
                flush=True,
            )
    print(
        f"\n  SLT-Net (paper)  F_w=0.357  MAE=0.0300  S_a=0.656  E_p=0.785",
        flush=True,
    )

    with open(args.out / "main_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[written] {args.out / 'main_results.json'}", flush=True)


if __name__ == "__main__":
    main()
