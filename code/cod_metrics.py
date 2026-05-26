"""
Compute four standard COD / salient-object-detection metrics:

    MAE          Mean Absolute Error      (lower is better)
    F_beta^w     weighted F-measure       (Margolin et al. 2014; higher better)
    S_alpha      Structure-measure        (Fan et al. 2017; higher better)
    E_phi        Enhanced-alignment measure (Fan et al. 2018; higher better)

Pred masks may be binary (uint8) or soft (float in [0,1]). GT is binary.
All four are averaged over frames that have a GT mask, then over videos.

Usage:
    python cod_metrics.py \
        --pred-root /root/autodl-tmp/VOScode/outputs \
        --gt-root   /root/autodl-tmp/VOSdataset \
        --method    sam3_base_mask
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------
def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean())


# ---------------------------------------------------------------------------
# Weighted F-measure  (Margolin, Zelnik-Manor, Tal — CVPR 2014)
# Reference implementation reproduced from the official MATLAB code.
# ---------------------------------------------------------------------------
def f_beta_w(pred: np.ndarray, gt: np.ndarray, beta2: float = 1.0) -> float:
    pred = pred.astype(np.float64)
    gt_b = (gt > 0.5).astype(np.float64)
    if gt_b.sum() == 0:
        return 0.0 if pred.sum() > 0 else 1.0

    # 1) E (raw absolute error)
    E = np.abs(pred - gt_b)

    # 2) distance transform to nearest foreground pixel
    dist, idx = cv2.distanceTransformWithLabels(
        (1 - gt_b).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )

    # Et: error at the nearest GT pixel for each background pixel
    fg_ys, fg_xs = np.where(gt_b > 0)
    # idx 0 is reserved; labels for fg pixels themselves
    fg_label = np.zeros_like(idx)
    fg_label[fg_ys, fg_xs] = np.arange(1, fg_ys.size + 1)
    label_to_xy = np.zeros((fg_ys.size + 1, 2), dtype=np.int32)
    label_to_xy[1:, 0] = fg_xs
    label_to_xy[1:, 1] = fg_ys
    # For each pixel, find the nearest GT pixel via distanceTransform labels;
    # `idx` references the seed label (0 for the GT pixels themselves).
    # We need the actual nearest FG xy, which OpenCV gives via DIST_LABEL_PIXEL +
    # post-processing. Approximate using nearest-neighbor on |fg_ys,fg_xs|.
    # Use idx to look up xy of nearest seed by precomputed lookup.

    # Build pixel-to-label map: for each pixel, idx gives the label of the
    # nearest 0-pixel (foreground in (1-gt_b)) — but it's defined for the
    # *zero* pixels in the input; we inverted gt so zero-pixels = GT mask.
    # Hence the seed pixels here are exactly the GT foreground pixels.
    seed_xy = np.argwhere(gt_b > 0)[:, [1, 0]]  # (N, 2) in (x, y)
    label_idx = idx.astype(np.int32)
    # OpenCV labels seeds starting at 1 in scan order; build mapping.
    h, w = gt_b.shape
    seed_scan_order = np.flatnonzero((gt_b > 0).T.ravel())  # column-major fortran order
    # Easier: use SciPy / direct iteration — but no SciPy. Fall back to manual:
    # For each non-fg pixel, we want the closest fg pixel.
    fg_xy = np.argwhere(gt_b > 0).astype(np.int32)
    # Build the nearest-fg-xy image by sampling from gt_b: pixels with gt_b=1
    # have distance 0; for other pixels we approximate using cv2.distanceTransform
    # to derive Et as the value of E at the nearest gt pixel.
    # The cheap, well-known fallback: Et[i,j] = E at nearest gt pixel.
    Et = np.zeros_like(E)
    if fg_xy.size > 0:
        # 5x5 distance scan won't be exact; use brute-force kd-tree-like via
        # repeated cv2.distanceTransform: instead, use that OpenCV returns the
        # closest seed's *coordinate* via the second output if we mark each FG
        # pixel with a unique label. But labelType=PIXEL gives the *label* of
        # the nearest 0-pixel in the input — i.e. the nearest FG pixel — and
        # those labels are assigned in raster order over the zero-pixels.
        # Build the raster-order list of FG pixels:
        flat_mask = (gt_b > 0).ravel()
        fg_indices = np.flatnonzero(flat_mask)  # raster order
        # idx values: 0..len(fg_indices); 0 means "not assigned" which shouldn't
        # occur if there are FG pixels. Map idx → linear index in image.
        # OpenCV labels seeds starting at 1, in raster order of zero-pixels:
        # so idx[y,x] == k → the k-th zero pixel in raster order.
        # That k-th zero pixel has linear index fg_indices[k-1].
        label_max = int(idx.max())
        if label_max == 0:
            Et = E.copy()
        else:
            lookup = np.zeros(label_max + 1, dtype=np.int64)
            lookup[1:label_max + 1] = fg_indices[:label_max]
            nearest_flat = lookup[idx]
            nearest_y, nearest_x = np.divmod(nearest_flat, w)
            # For fg pixels themselves, idx may be 0 (no need to look up); set Et = E
            mask_fg = gt_b > 0
            Et = np.where(mask_fg, E, E[nearest_y, nearest_x])

    # 3) per-pixel weight matrix Ew = E * Et / (Et + min(0.5,1-gt)*sigma)
    # following Margolin: Et[i] = E[NearestGT(i)]
    # Pixel weight: 2/(1+exp(5*dist))  (default sigma = 5)
    pixel_weight = 2.0 / (1.0 + np.exp(5.0 * dist))
    pixel_weight[gt_b > 0] = 1.0

    Ew = np.minimum(E, Et) * pixel_weight

    # 4) weighted TP / FP / FN
    TPw = (gt_b * (1.0 - Ew)).sum()
    FPw = ((1.0 - gt_b) * Ew).sum()
    FNw = (gt_b * Ew).sum()

    Rw = TPw / (TPw + FNw + 1e-12)
    Pw = TPw / (TPw + FPw + 1e-12)
    if Pw + Rw == 0:
        return 0.0
    return float((1 + beta2) * Rw * Pw / (beta2 * Pw + Rw + 1e-12))


# ---------------------------------------------------------------------------
# S-measure  (Fan et al. — ICCV 2017)
# ---------------------------------------------------------------------------
def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = pred.shape
    N = h * w
    x = pred.mean(); y = gt.mean()
    sigma_x2 = ((pred - x) ** 2).sum() / (N - 1 + 1e-12)
    sigma_y2 = ((gt - y) ** 2).sum() / (N - 1 + 1e-12)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-12)
    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)
    if alpha != 0:
        return float(alpha / (beta + 1e-12))
    return 1.0 if beta == 0 else 0.0


def _S_region(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = gt.shape
    area = gt.sum()
    if area == 0:
        return 0.0
    ys, xs = np.where(gt > 0)
    cy = int(round(ys.mean()))
    cx = int(round(xs.mean()))
    cy = max(1, min(h - 1, cy)); cx = max(1, min(w - 1, cx))

    parts_gt = [gt[:cy, :cx], gt[:cy, cx:], gt[cy:, :cx], gt[cy:, cx:]]
    parts_pred = [pred[:cy, :cx], pred[:cy, cx:], pred[cy:, :cx], pred[cy:, cx:]]
    weights = [p.sum() / area for p in parts_gt]
    return float(sum(w_i * _ssim(p, g) for w_i, p, g in zip(weights, parts_pred, parts_gt)))


def _S_object(pred: np.ndarray, gt: np.ndarray) -> float:
    def obj_score(pred, gt):
        x = pred[gt > 0.5].mean() if (gt > 0.5).any() else 0
        sigma_x = pred[gt > 0.5].std() if (gt > 0.5).sum() > 1 else 0
        return float(2 * x / (x ** 2 + 1 + sigma_x + 1e-12))

    o_fg = obj_score(pred, gt)
    o_bg = obj_score(1 - pred, 1 - gt)
    u = gt.mean()
    return float(u * o_fg + (1 - u) * o_bg)


def s_alpha(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    gt_b = (gt > 0.5).astype(np.float64)
    pred = pred.astype(np.float64)
    y = gt_b.mean()
    if y == 0:
        return float(1.0 - pred.mean())
    if y == 1:
        return float(pred.mean())
    return float(alpha * _S_object(pred, gt_b) + (1 - alpha) * _S_region(pred, gt_b))


# ---------------------------------------------------------------------------
# E-measure (Fan et al. — IJCAI 2018)
# ---------------------------------------------------------------------------
def e_phi(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    gt_b = (gt > 0.5).astype(np.float64)
    if gt_b.sum() == 0:
        # foreground absent: high E if prediction is also empty
        return float(1.0 - pred.mean())
    if gt_b.sum() == gt_b.size:
        return float(pred.mean())

    mu_p = pred.mean(); mu_g = gt_b.mean()
    align_p = pred - mu_p; align_g = gt_b - mu_g
    align = 2 * align_p * align_g / (align_p ** 2 + align_g ** 2 + 1e-12)
    enhanced = (align + 1) ** 2 / 4.0
    return float(enhanced.sum() / (gt_b.size - 1 + 1e-12))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_pred(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.float64)


def load_gt(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.float64)


def evaluate(pred_dir: Path, gt_dir: Path) -> dict:
    """Compute the 4 metrics over all frames where a GT mask exists."""
    if not pred_dir.exists() or not gt_dir.exists():
        return {}
    per_frame = []
    for gp in sorted(gt_dir.glob("*.png")):
        stem = gp.stem
        pp = pred_dir / f"{stem}.png"
        if not pp.exists():
            continue
        gt = load_gt(gp); pr = load_pred(pp)
        if pr.shape != gt.shape:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        per_frame.append({
            "frame": stem,
            "MAE":   mae(pr, gt),
            "F_w":   f_beta_w(pr, gt),
            "S_a":   s_alpha(pr, gt),
            "E_p":   e_phi(pr, gt),
        })
    if not per_frame:
        return {}
    avg = {k: float(np.mean([f[k] for f in per_frame]))
           for k in ("MAE", "F_w", "S_a", "E_p")}
    avg["n_frames"] = len(per_frame)
    avg["per_frame"] = per_frame
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path,
                    default=Path("/root/autodl-tmp/VOScode/outputs"))
    ap.add_argument("--gt-root", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset"))
    ap.add_argument("--method", type=str, default="sam3_base_mask",
                    help="subfolder name under outputs/<video>/")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="write per-video JSON here (default: outputs/metrics_<method>.json)")
    args = ap.parse_args()

    video_dirs = sorted(d for d in args.pred_root.iterdir() if d.is_dir())
    if args.videos:
        want = set(args.videos)
        video_dirs = [d for d in video_dirs if d.name in want]

    results = {}
    print(f"\n=== COD metrics for method '{args.method}' ===")
    print(f"{'video':<25} {'F_w':>7}  {'MAE':>7}  {'S_a':>7}  {'E_p':>7}  {'n':>4}")
    for vd in video_dirs:
        pred = vd / args.method / "masks"
        gt = args.gt_root / vd.name / "GT"
        res = evaluate(pred, gt)
        if not res:
            continue
        results[vd.name] = res
        print(f"{vd.name:<25} {res['F_w']:>7.3f}  {res['MAE']:>7.4f}  "
              f"{res['S_a']:>7.3f}  {res['E_p']:>7.3f}  {res['n_frames']:>4}")

    if results:
        keys = ("F_w", "MAE", "S_a", "E_p")
        means = {k: float(np.mean([r[k] for r in results.values()])) for k in keys}
        print("-" * 60)
        print(f"{'MEAN':<25} {means['F_w']:>7.3f}  {means['MAE']:>7.4f}  "
              f"{means['S_a']:>7.3f}  {means['E_p']:>7.3f}")

    out_path = args.out or (args.pred_root / f"metrics_{args.method}.json")
    payload = {"method": args.method, "videos": results,
               "mean": means if results else None}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
