"""
Fair comparison between feature representations for trajectory clustering.

All baselines share the same downstream pipeline (K-means with K=6, n_init=10,
fixed seed), so the ONLY thing that varies is the per-trajectory feature
representation. Each feature vector is L2-normalized per-trajectory before
clustering so that K-means' Euclidean distance is on a comparable scale.

Baselines (each tested independently):
  A. position-only      : mean (x, y)                              — 2-d
  B. raw-speed series   : (||v_1||, ..., ||v_{T-1}||)              — (T-1)-d
  C. raw-velocity series: (v_1, ..., v_{T-1}) flattened            — 2(T-1)-d
  D. coherence (ours)   : FFT magnitude of zero-mean speed
                          + autocorrelation at lags {1,2,4,8}
                          + local rigidity (pairwise-distance variance
                          to k nearest neighbors)                 — variable-d

Note: D is intentionally translation-and-magnitude invariant: it never sees
absolute position, and the FFT magnitude makes it invariant to constant
velocity offsets. Whether the FFT is the right invariance is the empirical
question this script tests.

If D beats both B and C, that supports the central thesis. If D only beats A,
then the win was just "richer features", and the paper's framing is wrong.
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
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Video loader (same as feasibility script)
# ---------------------------------------------------------------------------
def load_video(img_dir: Path, target_h: int = 384, target_w: int = 512):
    paths = sorted(img_dir.glob("*.jpg"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    frames_orig = [np.array(Image.open(p).convert("RGB")) for p in paths]
    H0, W0 = frames_orig[0].shape[:2]
    scale = min(target_h / H0, target_w / W0)
    new_h, new_w = int(H0 * scale), int(W0 * scale)
    resized = [cv2.resize(f, (new_w, new_h)) for f in frames_orig]
    padded = [cv2.copyMakeBorder(r, 0, target_h - new_h, 0, target_w - new_w,
                                 cv2.BORDER_CONSTANT, value=0) for r in resized]
    arr = np.stack(padded)
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
    return video, arr, (H0, W0), scale, (new_h, new_w)


# ---------------------------------------------------------------------------
# Feature extractors — all return (N, d_i) and are L2-normalized per row.
# ---------------------------------------------------------------------------
def feat_position(tracks: np.ndarray) -> np.ndarray:
    f = tracks.mean(axis=0)                     # (N, 2)
    return normalize(f, axis=1)


def feat_raw_speed(tracks: np.ndarray) -> np.ndarray:
    v = np.diff(tracks, axis=0)                 # (T-1, N, 2)
    s = np.linalg.norm(v, axis=2).T             # (N, T-1)
    return normalize(s, axis=1)


def feat_raw_velocity(tracks: np.ndarray) -> np.ndarray:
    v = np.diff(tracks, axis=0)                 # (T-1, N, 2)
    f = v.transpose(1, 0, 2).reshape(v.shape[1], -1)   # (N, 2(T-1))
    return normalize(f, axis=1)


def feat_coherence(tracks: np.ndarray) -> np.ndarray:
    """The coherence baseline — translation-and-magnitude invariant features.
    Concatenation of:
      - FFT magnitude of zero-meaned speed   (T/2+1 dims)
      - autocorrelation at lags {1,2,4,8}    (4 dims)
      - local rigidity descriptor            (8 dims)
    """
    T, N, _ = tracks.shape
    v = np.diff(tracks, axis=0)                  # (T-1, N, 2)
    s = np.linalg.norm(v, axis=2)                # (T-1, N)
    s_zm = s - s.mean(axis=0, keepdims=True)

    # 1) FFT magnitude
    spec = np.abs(np.fft.rfft(s_zm, axis=0)).T   # (N, T//2)

    # 2) autocorrelation at multiple lags
    def autocorr(x, lag):
        x_zm = x - x.mean(axis=0, keepdims=True)
        denom = (x_zm ** 2).sum(axis=0) + 1e-8
        return (x_zm[lag:] * x_zm[:-lag]).sum(axis=0) / denom
    ac = np.stack([autocorr(s, l) for l in (1, 2, 4, 8) if l < T - 1], axis=1)  # (N, ≤4)

    # 3) local rigidity: variance over time of pairwise distance to k nearest
    #    neighbors (NN computed on initial positions). For pixels on the same
    #    rigid body, these distances should be ~constant; for fluid background
    #    they fluctuate.
    init_pos = tracks[0]                         # (N, 2)
    diffs = init_pos[:, None, :] - init_pos[None, :, :]
    dists0 = np.linalg.norm(diffs, axis=2)       # (N, N) at frame 0
    np.fill_diagonal(dists0, np.inf)
    nn_idx = np.argsort(dists0, axis=1)[:, :8]   # (N, 8)
    rigidity = np.zeros((N, 8))
    for i in range(N):
        nn = nn_idx[i]
        # distance time series to each neighbor
        dij = np.linalg.norm(tracks[:, [i]] - tracks[:, nn], axis=2)  # (T, 8)
        rigidity[i] = dij.std(axis=0) / (dij.mean(axis=0) + 1e-6)
    rigidity_n = normalize(rigidity, axis=1)

    spec_n = normalize(spec, axis=1)
    ac_n = normalize(ac, axis=1)

    # equal-weight concatenation (each block is L2-normalized to unit norm)
    return np.hstack([spec_n, ac_n, rigidity_n])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def cluster_to_region(tracks: np.ndarray, mask: np.ndarray,
                      H: int, W: int, dilate_px: int = 21) -> np.ndarray:
    region = np.zeros((H, W), dtype=np.uint8)
    pts = tracks[:, mask].reshape(-1, 2)
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


def best_cluster_iou(tracks: np.ndarray, labels: np.ndarray, gt: np.ndarray,
                     H: int, W: int) -> float:
    K = int(labels.max()) + 1
    best = 0.0
    for k in range(K):
        sel = labels == k
        if sel.sum() < 3:
            continue
        reg = cluster_to_region(tracks, sel, H, W)
        best = max(best, iou(reg, gt.astype(bool)))
    return best


# ---------------------------------------------------------------------------
# Per-video runner
# ---------------------------------------------------------------------------
def run_video(model, name, imgs_dir, gt_dir, grid=48, K=6):
    video, frames_arr, (H0, W0), scale, (new_h, new_w) = load_video(imgs_dir)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            tr, vis = model(video, grid_size=grid)
    tr = tr[0].float().cpu().numpy()
    vis = vis[0].float().cpu().numpy()
    keep = vis.mean(axis=0) >= 0.2
    tr = tr[:, keep]
    pad_H, pad_W = video.shape[3], video.shape[4]

    # Use the union of all GT masks (lenient evaluation)
    gt_union = np.zeros((pad_H, pad_W), dtype=np.uint8)
    for gp in sorted(gt_dir.glob("*.png")):
        g = np.array(Image.open(gp))
        if g.ndim == 3:
            g = g[..., 0]
        g_r = cv2.resize((g > 0).astype(np.uint8), (new_w, new_h),
                         interpolation=cv2.INTER_NEAREST)
        gt_union[:new_h, :new_w] |= g_r

    feats = {
        "A_position":   feat_position(tr),
        "B_speed":      feat_raw_speed(tr),
        "C_velocity":   feat_raw_velocity(tr),
        "D_coherence":  feat_coherence(tr),
    }
    results = {"name": name, "n_traj": int(tr.shape[1]),
               "n_frames": int(video.shape[1]), "dims": {}}
    print(f"\n=== {name}  (T={video.shape[1]}, N={tr.shape[1]}) ===")
    for label, F in feats.items():
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(F)
        biou = best_cluster_iou(tr, km.labels_, gt_union, pad_H, pad_W)
        results[label] = biou
        results["dims"][label] = int(F.shape[1])
        print(f"  {label:<14} dim={F.shape[1]:>4}  best-cluster IoU = {biou:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/VOSdataset"))
    ap.add_argument("--output",  type=Path,
                    default=Path("/root/autodl-tmp/VOScode/feasibility_fair"))
    ap.add_argument("--videos",  nargs="*", default=None)
    ap.add_argument("--grid",    type=int, default=48)
    ap.add_argument("--K",       type=int, default=6)
    args = ap.parse_args()

    print("[init] loading CoTracker3 (offline) ...")
    model = torch.hub.load('facebookresearch/co-tracker',
                           'cotracker3_offline').cuda().eval()
    print("[init] ready.")

    args.output.mkdir(parents=True, exist_ok=True)
    vds = sorted(d for d in args.dataset.iterdir() if d.is_dir())
    if args.videos:
        want = set(args.videos)
        vds = [d for d in vds if d.name in want]

    rows = []
    for vd in vds:
        if not (vd / "Imgs").exists():
            continue
        try:
            rows.append(run_video(model, vd.name, vd / "Imgs", vd / "GT",
                                  grid=args.grid, K=args.K))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[error] {vd.name}: {e}")

    print("\n========= FAIR COMPARISON (best-cluster IoU) =========")
    keys = ["A_position", "B_speed", "C_velocity", "D_coherence"]
    print(f"{'video':<22} " + " ".join(f"{k:>14}" for k in keys))
    for r in rows:
        print(f"{r['name']:<22} " + " ".join(f"{r[k]:>14.3f}" for k in keys))
    print("-" * (22 + 4 * 15))
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print(f"{'MEAN':<22} " + " ".join(f"{means[k]:>14.3f}" for k in keys))

    print("\nFeature dimensionality (per video):")
    for r in rows:
        print(f"  {r['name']:<22} " +
              " ".join(f"{k}={r['dims'][k]}" for k in keys))

    with open(args.output / "summary.json", "w") as f:
        json.dump({"rows": rows, "means": means,
                   "grid": args.grid, "K": args.K}, f, indent=2)
    print(f"\n[written] {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
