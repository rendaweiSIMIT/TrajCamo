"""
State builder for the TrajCamo agent.

Renders the MLLM's visual+textual state at each step:

  * Cluster overview image  — first frame with cluster trajectory points
    overlaid in distinct colors, each cluster numbered.
  * Thumbnail strip         — N frames from the video, evenly sampled, in a
    single horizontal panel.
  * Current-mask overlay    — same thumbnail strip but with the current
    predicted mask blended on top (shown only after the first SELECT).

The output is a list of PIL Images (we hand them to InternVL3's chat API as
multi-image inputs in the order [cluster_overview, thumbnails, mask_overlay]).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image


_PALETTE = np.array(
    [
        [255,  96,  96], [ 96, 255,  96], [ 96,  96, 255], [255, 255,  96],
        [255,  96, 255], [ 96, 255, 255], [255, 192,  96], [192,  96, 255],
        [128, 255, 128], [255, 128, 128], [128, 128, 255], [200, 200,  64],
        [200,  64, 200], [ 64, 200, 200], [255, 160,  80], [160,  80, 255],
    ],
    dtype=np.uint8,
)


def cluster_color(k: int) -> np.ndarray:
    return _PALETTE[k % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Cluster overview image
# ---------------------------------------------------------------------------
def render_cluster_overview(
    frame_bgr: np.ndarray,
    tracks_at_frame_t: np.ndarray,   # (Nk, 2) in PADDED coords
    visible_mask: np.ndarray,        # (Nk,) bool
    labels: np.ndarray,              # (Nk,) cluster ids
    target_h_w: tuple,               # (H_padded, W_padded)
    orig_h_w: tuple,                 # (H_orig, W_orig)
    new_h_w: tuple,                  # (H_resized, W_resized)
    scale: float,
    title: str = "Trajectory cluster overview",
) -> Image.Image:
    """
    Returns a PIL image: the original frame (in original resolution) with
    cluster trajectory points overlaid as colored dots, with cluster index
    labels in the centroid of each cluster.
    """
    H_orig, W_orig = orig_h_w
    canvas = frame_bgr.copy()

    cluster_centroids = {}
    valid = visible_mask
    pts = tracks_at_frame_t[valid] / scale     # padded -> original coords
    lbls = labels[valid]

    for k in np.unique(lbls):
        sel = lbls == k
        if sel.sum() < 3:
            continue
        cluster_pts = pts[sel]
        color = cluster_color(int(k))[::-1].tolist()    # PIL is RGB, cv2 is BGR
        for p in cluster_pts:
            x = int(np.clip(p[0], 0, W_orig - 1))
            y = int(np.clip(p[1], 0, H_orig - 1))
            cv2.circle(canvas, (x, y), 3, color, -1)
        cx, cy = int(cluster_pts[:, 0].mean()), int(cluster_pts[:, 1].mean())
        cluster_centroids[int(k)] = (cx, cy)

    # draw cluster index numbers
    for k, (cx, cy) in cluster_centroids.items():
        cv2.putText(canvas, str(k), (cx + 6, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, str(k), (cx + 6, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)

    # title
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    # cv2 → PIL is RGB
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Thumbnail strip
# ---------------------------------------------------------------------------
def sample_thumbnail_frames(
    imgs_dir: Path, frame_names: List[str], n: int = 4,
    each_h: int = 240,
) -> tuple:
    """
    Sample N evenly-spaced frames from frame_names, return:
      * PIL Image of horizontal strip
      * list of frame indices (in the imgs_dir order) that were used
    """
    if n >= len(frame_names):
        idxs = list(range(len(frame_names)))
    else:
        idxs = np.linspace(0, len(frame_names) - 1, n, dtype=int).tolist()
    frames = []
    for i in idxs:
        p = imgs_dir / f"{frame_names[i]}.jpg"
        if not p.exists():
            # fall back to whatever's at that index
            jpgs = sorted(imgs_dir.glob("*.jpg"))
            if i < len(jpgs):
                p = jpgs[i]
            else:
                continue
        im = Image.open(p).convert("RGB")
        w0, h0 = im.size
        new_w = int(w0 * each_h / h0)
        im = im.resize((new_w, each_h))
        frames.append(im)
    if not frames:
        return None, []
    total_w = sum(f.size[0] for f in frames)
    strip = Image.new("RGB", (total_w, each_h), (0, 0, 0))
    x = 0
    for i, f in zip(idxs, frames):
        strip.paste(f, (x, 0))
        # frame index label
        draw_text(strip, str(i), x + 4, 4)
        x += f.size[0]
    return strip, idxs


def draw_text(im: Image.Image, text: str, x: int, y: int) -> None:
    arr = np.array(im)
    cv2.putText(arr, text, (x, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(arr, text, (x, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    im.paste(Image.fromarray(arr))


# ---------------------------------------------------------------------------
# Current-mask overlay strip
# ---------------------------------------------------------------------------
def render_current_mask_strip(
    imgs_dir: Path, frame_names: List[str], idxs: List[int],
    per_frame_masks: dict,            # {frame_idx_in_imgs_dir: np.uint8 mask at (H_orig, W_orig)}
    each_h: int = 240,
    alpha: float = 0.55,
) -> Image.Image:
    """Same layout as sample_thumbnail_frames but with current mask overlaid in red."""
    frames = []
    for i in idxs:
        p = imgs_dir / f"{frame_names[i]}.jpg"
        if not p.exists():
            jpgs = sorted(imgs_dir.glob("*.jpg"))
            if i < len(jpgs):
                p = jpgs[i]
            else:
                continue
        im = np.array(Image.open(p).convert("RGB"))
        m = per_frame_masks.get(i)
        if m is not None and m.shape == im.shape[:2]:
            red = np.zeros_like(im)
            red[..., 0] = 255
            mask3 = (m > 0)[..., None]
            im = np.where(mask3,
                          (im.astype(np.int32) * (1 - alpha)
                           + red.astype(np.int32) * alpha).astype(np.uint8),
                          im)
        pim = Image.fromarray(im)
        w0, h0 = pim.size
        pim = pim.resize((int(w0 * each_h / h0), each_h))
        frames.append((i, pim))
    if not frames:
        return None
    total_w = sum(f.size[0] for _, f in frames)
    strip = Image.new("RGB", (total_w, each_h), (0, 0, 0))
    x = 0
    for i, f in frames:
        strip.paste(f, (x, 0))
        draw_text(strip, str(i), x + 4, 4)
        x += f.size[0]
    return strip
