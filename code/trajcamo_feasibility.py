"""
Feasibility test for the TrajCamo thesis: does motion COHERENCE (not magnitude)
of long-window point trajectories separate camouflaged targets from background?

For each VOSdataset video:
  1. Extract dense trajectories with CoTracker3 (offline mode).
  2. Visualize the trajectory field on the first frame.
  3. Build two feature representations per trajectory:
       (a) magnitude   : mean & std of speed only
       (b) coherence   : full velocity time-series + spectral profile (FFT)
  4. K-means cluster on each (K=5).
  5. For each cluster, build a region mask (KDE of trajectory positions
     dilated) and compute IoU against the GT first-frame mask.
  6. Report best-cluster IoU for both feature sets — this tells us:
     - whether CoTracker3 tracks camouflaged pixels at all
     - whether coherence beats magnitude (the paper's central thesis)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Video loader
# ---------------------------------------------------------------------------
def load_video(img_dir: Path, target_h: int = 384, target_w: int = 512):
    paths = sorted(img_dir.glob("*.jpg"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    frames_orig = [np.array(Image.open(p).convert("RGB")) for p in paths]
    H0, W0 = frames_orig[0].shape[:2]

    # Resize maintaining aspect ratio, then pad to (target_h, target_w)
    scale = min(target_h / H0, target_w / W0)
    new_h, new_w = int(H0 * scale), int(W0 * scale)
    resized = [cv2.resize(f, (new_w, new_h)) for f in frames_orig]
    padded = [cv2.copyMakeBorder(r, 0, target_h - new_h, 0, target_w - new_w,
                                 cv2.BORDER_CONSTANT, value=0) for r in resized]
    arr = np.stack(padded)  # (T, H, W, 3) uint8

    video = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
    return video, arr, (H0, W0), scale, (new_h, new_w)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------
def features_magnitude(tracks_xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Raw motion-magnitude features: mean speed, speed std, mean position.
    This is the 'naive' baseline — what optical-flow-based methods use.
    """
    T, N, _ = tracks_xy.shape
    velocity = np.diff(tracks_xy, axis=0)            # (T-1, N, 2)
    speed = np.linalg.norm(velocity, axis=2)         # (T-1, N)
    return np.stack([
        speed.mean(axis=0),
        speed.std(axis=0),
        tracks_xy[:, :, 0].mean(axis=0),  # mean x
        tracks_xy[:, :, 1].mean(axis=0),  # mean y
    ], axis=1)


def features_coherence(tracks_xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Coherence features: velocity time-series (zero-mean) + FFT magnitude
    profile. Captures the TEMPORAL PATTERN, invariant to absolute position
    and (after zero-meaning) to mean drift.
    """
    T, N, _ = tracks_xy.shape
    velocity = np.diff(tracks_xy, axis=0)            # (T-1, N, 2)

    # zero-mean per trajectory (kills constant drift offsets)
    v_zm = velocity - velocity.mean(axis=0, keepdims=True)

    # flatten to (N, 2*(T-1))
    raw = v_zm.transpose(1, 0, 2).reshape(N, -1)

    # FFT magnitude of speed (rotation-invariant pattern signature)
    speed = np.linalg.norm(velocity, axis=2)         # (T-1, N)
    speed_zm = speed - speed.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(speed_zm, axis=0)).T   # (N, T/2+1)

    # auto-correlation at small lags (captures motion smoothness)
    def autocorr(x, lag):
        x_zm = x - x.mean(axis=0, keepdims=True)
        denom = (x_zm ** 2).sum(axis=0) + 1e-8
        num = (x_zm[lag:] * x_zm[:-lag]).sum(axis=0)
        return num / denom
    ac = np.stack([autocorr(speed, l) for l in (1, 2, 4)], axis=1)  # (N, 3)

    # spatial position (down-weighted, for some locality bias)
    pos = tracks_xy.mean(axis=0) * 0.001  # very small weight

    # L2 normalize the velocity block so it doesn't dominate
    raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
    spec = spec / (np.linalg.norm(spec, axis=1, keepdims=True) + 1e-8)

    return np.hstack([raw, spec, ac, pos])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def cluster_to_region(tracks_xy: np.ndarray, cluster_mask: np.ndarray,
                      H: int, W: int, dilate_px: int = 21) -> np.ndarray:
    """Build a binary region mask from the trajectory points of one cluster:
    splat points across all frames, then dilate."""
    region = np.zeros((H, W), dtype=np.uint8)
    pts = tracks_xy[:, cluster_mask].reshape(-1, 2)  # (T*Nk, 2)
    xs = pts[:, 0].round().astype(int).clip(0, W - 1)
    ys = pts[:, 1].round().astype(int).clip(0, H - 1)
    region[ys, xs] = 1
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        region = cv2.dilate(region, k, iterations=1)
    return region.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return 0.0 if u == 0 else float((a & b).sum() / u)


def eval_clustering(tracks_xy: np.ndarray, labels: np.ndarray, gt: np.ndarray,
                    H: int, W: int) -> dict:
    """For each cluster, compute IoU of its dilated region vs GT, and report
    the best cluster."""
    K = labels.max() + 1
    rows = []
    for k in range(K):
        sel = labels == k
        if sel.sum() < 3:
            continue
        reg = cluster_to_region(tracks_xy, sel, H, W)
        rows.append({"cluster": int(k), "size": int(sel.sum()),
                     "iou": iou(reg, gt.astype(bool))})
    rows.sort(key=lambda r: -r["iou"])
    best = rows[0] if rows else {"cluster": -1, "size": 0, "iou": 0.0}
    return {"best": best, "all": rows}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def viz_trajectories(frame: np.ndarray, tracks_xy: np.ndarray, vis: np.ndarray,
                     labels: np.ndarray, out_path: Path,
                     title: str = "") -> None:
    """Draw colored trajectories over a frame. Color = cluster label."""
    canvas = frame.copy()
    K = int(labels.max()) + 1
    palette = (np.array([
        [255, 96, 96], [96, 255, 96], [96, 96, 255], [255, 255, 96],
        [255, 96, 255], [96, 255, 255], [255, 192, 96], [192, 96, 255],
    ], dtype=np.uint8))
    T, N, _ = tracks_xy.shape
    for n in range(N):
        if vis[:, n].mean() < 0.3:
            continue
        c = palette[labels[n] % len(palette)].tolist()
        # draw trajectory path
        pts = tracks_xy[:, n].round().astype(int)
        for t in range(1, T):
            if vis[t, n] and vis[t - 1, n]:
                cv2.line(canvas, tuple(pts[t - 1]), tuple(pts[t]),
                         c, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[0]), 2, c, -1)
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, lineType=cv2.LINE_AA)
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    Image.fromarray(canvas).save(str(out_path))


def viz_best_cluster_region(frame: np.ndarray, tracks_xy: np.ndarray,
                            cluster_mask: np.ndarray, gt: np.ndarray,
                            out_path: Path) -> None:
    H, W = frame.shape[:2]
    region = cluster_to_region(tracks_xy, cluster_mask, H, W)
    canvas = frame.copy()
    # GT in green, predicted region in red, overlap in yellow
    overlay = np.zeros_like(canvas)
    overlay[gt > 0] = (0, 255, 0)
    overlay[region] = (255, 0, 0)
    overlay[region & (gt > 0)] = (255, 255, 0)
    canvas = (canvas.astype(np.int32) * 0.5 + overlay * 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(canvas).save(str(out_path))


# ---------------------------------------------------------------------------
# Main per-video run
# ---------------------------------------------------------------------------
def run_video(model, name: str, imgs_dir: Path, gt_dir: Path,
              out_dir: Path, grid_size: int = 24) -> dict:
    print(f"\n=== {name} ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    video, frames_arr, (H0, W0), scale, (new_h, new_w) = load_video(imgs_dir)
    print(f"  loaded {video.shape[1]} frames  "
          f"({W0}x{H0} → {video.shape[4]}x{video.shape[3]})  "
          f"in {time.time()-t0:.1f}s")

    # Trajectory extraction
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            tracks, vis = model(video, grid_size=grid_size)
    tracks = tracks[0].float().cpu().numpy()  # (T, N, 2) — in resized coords
    vis_np = vis[0].float().cpu().numpy()      # (T, N)
    print(f"  CoTracker3 (grid={grid_size}, N={tracks.shape[1]}) → "
          f"{time.time()-t0:.1f}s   shape={tracks.shape}")

    # Drop trajectories rarely visible
    keep = vis_np.mean(axis=0) >= 0.2
    tracks_kept = tracks[:, keep]
    vis_kept = vis_np[:, keep]
    print(f"  visible ≥20% of frames: {keep.sum()} / {keep.size}")

    # Load GT first-frame mask, scale to resized coords
    gt0_path = gt_dir / f"{sorted(p.stem for p in imgs_dir.glob('*.jpg'))[0]}.png"
    gt0 = np.array(Image.open(gt0_path))
    if gt0.ndim == 3:
        gt0 = gt0[..., 0]
    gt_bin = (gt0 > 0).astype(np.uint8)
    # rescale GT to match the tracker's input frame
    gt_resized = cv2.resize(gt_bin, (new_w, new_h),
                            interpolation=cv2.INTER_NEAREST)
    pad_H, pad_W = video.shape[3], video.shape[4]   # (1, T, 3, H, W)
    gt_padded = np.zeros((pad_H, pad_W), dtype=np.uint8)
    gt_padded[:new_h, :new_w] = gt_resized
    H, W = gt_padded.shape

    # Sanity check on tracker: how many traj START inside GT, and how many
    # PASS THROUGH GT at any frame (we have a GT-per-5-frames dataset, but
    # we approximate by the frame-0 GT here; later we also load all GTs).
    pos0 = tracks_kept[0].round().astype(int)
    inside_gt_f0 = gt_padded[pos0[:, 1].clip(0, H - 1),
                             pos0[:, 0].clip(0, W - 1)].astype(bool).sum()

    # Build union-of-all-GTs to count "ever inside" trajectories (more lenient)
    gt_union = gt_padded.copy().astype(bool)
    for gp in sorted((gt_dir.glob("*.png"))):
        gt_full = np.array(Image.open(gp))
        if gt_full.ndim == 3:
            gt_full = gt_full[..., 0]
        gt_bin_t = (gt_full > 0).astype(np.uint8)
        gt_r = cv2.resize(gt_bin_t, (new_w, new_h),
                          interpolation=cv2.INTER_NEAREST)
        tmp = np.zeros((pad_H, pad_W), dtype=np.uint8)
        tmp[:new_h, :new_w] = gt_r
        gt_union |= tmp.astype(bool)

    ever_inside = 0
    for n in range(tracks_kept.shape[1]):
        pts = tracks_kept[:, n].round().astype(int)
        ys = pts[:, 1].clip(0, H - 1); xs = pts[:, 0].clip(0, W - 1)
        if gt_union[ys, xs].any():
            ever_inside += 1
    print(f"  traj starting in GT@frame0: {inside_gt_f0} / {tracks_kept.shape[1]}")
    print(f"  traj ever passing through any GT mask: {ever_inside}")

    # Build two feature sets and cluster
    Fmag = features_magnitude(tracks_kept, vis_kept)
    Fcoh = features_coherence(tracks_kept, vis_kept)

    K = 6
    km_mag = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Fmag)
    km_coh = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Fcoh)

    eval_mag = eval_clustering(tracks_kept, km_mag.labels_, gt_padded, H, W)
    eval_coh = eval_clustering(tracks_kept, km_coh.labels_, gt_padded, H, W)
    print(f"  magnitude features  →  best cluster IoU = {eval_mag['best']['iou']:.3f}  "
          f"size={eval_mag['best']['size']}")
    print(f"  coherence features  →  best cluster IoU = {eval_coh['best']['iou']:.3f}  "
          f"size={eval_coh['best']['size']}")

    # Visualizations
    frame0 = frames_arr[0]
    viz_trajectories(frame0, tracks_kept, vis_kept, km_mag.labels_,
                     out_dir / "traj_magnitude.jpg",
                     title=f"K-means on MAGNITUDE features (best IoU={eval_mag['best']['iou']:.2f})")
    viz_trajectories(frame0, tracks_kept, vis_kept, km_coh.labels_,
                     out_dir / "traj_coherence.jpg",
                     title=f"K-means on COHERENCE features (best IoU={eval_coh['best']['iou']:.2f})")
    viz_best_cluster_region(frame0, tracks_kept,
                            km_coh.labels_ == eval_coh['best']['cluster'],
                            gt_padded, out_dir / "best_cluster_vs_gt.jpg")

    # Plot trajectories colored by GT membership (sanity check on tracker)
    inside = np.zeros_like(km_mag.labels_)
    inside[gt_padded[pos0[:, 1].clip(0, H - 1),
                     pos0[:, 0].clip(0, W - 1)].astype(bool)] = 1
    viz_trajectories(frame0, tracks_kept, vis_kept, inside,
                     out_dir / "traj_gt_membership.jpg",
                     title="trajectories starting inside GT (yellow) vs outside (red)")

    return {
        "name": name,
        "n_frames": int(video.shape[1]),
        "n_trajectories": int(tracks_kept.shape[1]),
        "n_inside_gt_f0": int(inside_gt_f0),
        "n_ever_inside_gt": int(ever_inside),
        "magnitude_best_iou": eval_mag["best"]["iou"],
        "coherence_best_iou": eval_coh["best"]["iou"],
        "magnitude_all": eval_mag["all"],
        "coherence_all": eval_coh["all"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/VOSdataset"))
    ap.add_argument("--output",  type=Path,
                    default=Path("/root/autodl-tmp/VOScode/feasibility"))
    ap.add_argument("--videos",  nargs="*", default=None)
    ap.add_argument("--grid",    type=int, default=24,
                    help="CoTracker3 grid size (NxN seed points)")
    args = ap.parse_args()

    print("[init] loading CoTracker3 (offline) ...")
    model = torch.hub.load('facebookresearch/co-tracker',
                           'cotracker3_offline').cuda().eval()
    print("[init] ready.")

    args.output.mkdir(parents=True, exist_ok=True)
    video_dirs = sorted(d for d in args.dataset.iterdir() if d.is_dir())
    if args.videos:
        wanted = set(args.videos)
        video_dirs = [d for d in video_dirs if d.name in wanted]

    summary = []
    for vd in video_dirs:
        imgs = vd / "Imgs"; gt = vd / "GT"
        if not imgs.exists() or not gt.exists():
            print(f"[skip] {vd.name}")
            continue
        try:
            res = run_video(model, vd.name, imgs, gt, args.output / vd.name,
                            grid_size=args.grid)
            summary.append(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[error] {vd.name}: {e}")

    # Summary table
    print("\n========= FEASIBILITY SUMMARY =========")
    print(f"{'video':<22} {'#frm':>5} {'#traj':>6} {'inGT@0':>7} "
          f"{'ever-inGT':>10} {'IoU (mag)':>11} {'IoU (coh)':>11}")
    for r in summary:
        print(f"{r['name']:<22} {r['n_frames']:>5} {r['n_trajectories']:>6} "
              f"{r['n_inside_gt_f0']:>7} {r['n_ever_inside_gt']:>10} "
              f"{r['magnitude_best_iou']:>11.3f} "
              f"{r['coherence_best_iou']:>11.3f}")

    if summary:
        m = float(np.mean([r["magnitude_best_iou"] for r in summary]))
        c = float(np.mean([r["coherence_best_iou"] for r in summary]))
        print(f"{'MEAN':<22} {'':>8} {'':>6} {'':>6} "
              f"{m:>11.3f} {c:>11.3f}")

    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[written] {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
