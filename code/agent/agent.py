"""
TrajCamo agent loop.

For one video:
  1. Load CoTracker3 trajectory cache + cluster (raw_velocity, K=8, min_move=4).
  2. Build initial state (cluster overview + thumbnails).
  3. Step t=0..K_max-1:
       a. Render state.
       b. Call MLLM (InternVL3) → text → parse action.
       c. Execute action via SAM 3 base (re-init session + add prompts +
          propagate, taking the FRESH mask sequence as the new "current").
  4. On TERMINATE or K_max reached: emit the most recent mask sequence.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# our modules
sys.path.insert(0, str(Path(__file__).parent))
from actions import SYSTEM_PROMPT, Action, parse_action, format_history
from state_builder import (
    cluster_color,
    render_cluster_overview,
    render_current_mask_strip,
    sample_thumbnail_frames,
)


# ---------------------------------------------------------------------------
# Trajectory cache loader (small duplicate to keep this file self-contained)
# ---------------------------------------------------------------------------
def load_traj_cache(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return dict(
        tracks=d["tracks"], visibility=d["visibility"],
        frame_names=list(d["frame_names"]),
        grid_size=int(d["grid_size"]),
        orig_h=int(d["orig_h"]), orig_w=int(d["orig_w"]),
        new_h=int(d["new_h"]),   new_w=int(d["new_w"]),
        target_h=int(d["target_h"]), target_w=int(d["target_w"]),
        scale=float(d["scale"]),
    )


def feat_raw_velocity(tracks: np.ndarray) -> np.ndarray:
    v = np.diff(tracks, axis=0)
    f = v.transpose(1, 0, 2).reshape(v.shape[1], -1)
    return normalize(f, axis=1)


def cluster_trajectories(
    info: dict, K: int = 8, min_move_px: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (tracks_k, vis_k, labels, dyn_idx_in_visible_set)."""
    tracks_p = info["tracks"]; vis_p = info["visibility"]
    keep = vis_p.mean(axis=0) >= 0.2
    tracks_v = tracks_p[:, keep]; vis_v = vis_p[:, keep]
    pos_range = tracks_v.max(axis=0) - tracks_v.min(axis=0)
    movement = np.linalg.norm(pos_range, axis=1)
    dyn = movement > min_move_px
    if dyn.sum() < K * 3:
        dyn = np.ones_like(dyn, dtype=bool)
    tracks_k = tracks_v[:, dyn]; vis_k = vis_v[:, dyn]
    F = feat_raw_velocity(tracks_k)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(F)
    return tracks_k, vis_k, km.labels_, np.where(dyn)[0]


# ---------------------------------------------------------------------------
# SAM 3 executor (one persistent state per agent loop)
# ---------------------------------------------------------------------------
@dataclass
class PromptStream:
    """Accumulates positive and negative point prompts across agent actions."""
    points_per_frame: Dict[int, List[Tuple[float, float, int]]] = field(default_factory=dict)
    # frame_idx -> list of (x_rel, y_rel, label)  with label 1=pos, 0=neg

    def add(self, frame_idx: int, x_rel: float, y_rel: float, label: int) -> None:
        self.points_per_frame.setdefault(frame_idx, []).append((x_rel, y_rel, label))

    def total_pos(self) -> int:
        return sum(1 for pts in self.points_per_frame.values() for _,_,l in pts if l == 1)


def run_sam3_session(
    predictor, imgs_dir: Path, stream: PromptStream, T: int,
) -> Dict[int, np.ndarray]:
    """Init a fresh SAM 3 session, push all accumulated prompts, propagate.
    Returns per_frame_masks (binary, original-resolution)."""
    state = predictor.init_state(video_path=str(imgs_dir))
    first = True
    for f_idx, pts in stream.points_per_frame.items():
        pts_xy = torch.tensor([[p[0], p[1]] for p in pts], dtype=torch.float32)
        labels = torch.tensor([p[2] for p in pts], dtype=torch.int32)
        predictor.add_new_points(
            inference_state=state, frame_idx=int(f_idx), obj_id=1,
            points=pts_xy, labels=labels,
            clear_old_points=first,
        )
        first = False
    predictor.propagate_in_video_preflight(state, run_mem_encoder=True)
    per_frame_masks = {}
    for fi, oid, _, vrm, _ in predictor.propagate_in_video(
        state, start_frame_idx=0, max_frame_num_to_track=T, reverse=False,
    ):
        if len(oid) == 0:
            per_frame_masks[fi] = None
            continue
        m = (vrm[0] > 0).cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        per_frame_masks[fi] = m.astype(np.uint8)
    return per_frame_masks


# ---------------------------------------------------------------------------
# Cluster → SAM 3 prompts
# ---------------------------------------------------------------------------
def cluster_to_prompt_points(
    cluster_idx: int, info: dict, tracks_k: np.ndarray, vis_k: np.ndarray,
    labels: np.ndarray, n_prompt: int = 12,
) -> List[Tuple[int, float, float, int]]:
    """For the chosen cluster, sample tight centroid-neighborhood points at
    frame 0. Returns [(frame_idx, x_rel, y_rel, label=1), ...]."""
    sel = labels == cluster_idx
    f0_visible = vis_k[0] & sel
    pts = tracks_k[0, f0_visible]
    if len(pts) == 0:
        for t in range(tracks_k.shape[0]):
            ft = vis_k[t] & sel
            if ft.sum() > 0:
                pts = tracks_k[t, ft]
                break
    if len(pts) == 0:
        return []
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    order = np.argsort(dists)
    n_take = min(n_prompt, max(3, len(pts) // 4))
    sel_pts = pts[order[:n_take]]
    med_d = np.median(dists)
    keep = np.linalg.norm(sel_pts - centroid, axis=1) <= max(med_d * 2, 30.0)
    sel_pts = sel_pts[keep] if keep.any() else centroid[None, :]

    scale = info["scale"]; orig_h, orig_w = info["orig_h"], info["orig_w"]
    sel_orig = sel_pts / scale
    sel_orig[:, 0] = sel_orig[:, 0].clip(0, orig_w - 1)
    sel_orig[:, 1] = sel_orig[:, 1].clip(0, orig_h - 1)
    return [(0, float(p[0] / orig_w), float(p[1] / orig_h), 1) for p in sel_orig]


# ---------------------------------------------------------------------------
# MLLM call wrapper (InternVL3)
# ---------------------------------------------------------------------------
class InternVL3Agent:
    """Thin wrapper around InternVL3's `.chat(...)` API. Hides image preprocessing."""

    def __init__(self, model_path: str, dtype=torch.bfloat16, device="cuda"):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                        trust_remote_code=True,
                                                        use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self.device = device
        self.dtype = dtype
        print(f"[InternVL3Agent] loaded {model_path}", flush=True)

    def _preprocess(self, pil_image: Image.Image, image_size=448, max_num=6):
        """InternVL3's dynamic-tile preprocessing. Returns pixel_values tensor
        (num_patches, 3, image_size, image_size)."""
        try:
            from torchvision.transforms.functional import to_tensor, normalize as tv_normalize, resize
        except Exception:
            import torchvision.transforms.functional as F
            to_tensor = F.to_tensor
        # very simple: just one tile resized to image_size
        im = pil_image.convert("RGB").resize((image_size, image_size))
        arr = np.array(im).astype(np.float32) / 255.0
        # ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device, dtype=self.dtype)

    def query(self, system_text: str, user_text: str,
              images: List[Image.Image], max_new_tokens: int = 64) -> str:
        # build pixel_values for multiple images: concat along batch dim, model
        # interpolates positional embeddings if needed.
        pv_list = [self._preprocess(im) for im in images]
        num_patches_list = [pv.shape[0] for pv in pv_list]
        pixel_values = torch.cat(pv_list, dim=0)
        image_token_prefix = "".join(f"Image-{i+1}: <image>\n"
                                     for i in range(len(images)))
        question = system_text + "\n\n" + image_token_prefix + user_text
        gen_cfg = dict(
            do_sample=False, max_new_tokens=max_new_tokens,
            num_beams=1, temperature=1.0,
        )
        out = self.model.chat(
            self.tokenizer, pixel_values, question, gen_cfg,
            num_patches_list=num_patches_list,
            verbose=False,
        )
        return out.strip()


# ---------------------------------------------------------------------------
# Main agent loop on one video
# ---------------------------------------------------------------------------
def run_agent_on_video(
    mllm: InternVL3Agent,
    sam3_predictor,
    video_dir: Path,
    traj_cache_path: Path,
    K: int = 8,
    K_max: int = 5,
    n_thumbnails: int = 4,
    n_prompt_points: int = 12,
    verbose: bool = True,
) -> dict:
    name = video_dir.name
    imgs_dir = video_dir / "Imgs"
    info = load_traj_cache(traj_cache_path)
    tracks_k, vis_k, labels, _ = cluster_trajectories(info, K=K)
    K_actual = int(labels.max()) + 1
    T = int(info["tracks"].shape[0])
    frame_names = info["frame_names"]

    # Cluster overview image (uses frame 0 in ORIGINAL coords)
    first_frame_path = imgs_dir / f"{frame_names[0]}.jpg"
    first_frame_bgr = cv2.imread(str(first_frame_path))
    cluster_overview = render_cluster_overview(
        first_frame_bgr, tracks_k[0], vis_k[0], labels,
        target_h_w=(info["target_h"], info["target_w"]),
        orig_h_w=(info["orig_h"], info["orig_w"]),
        new_h_w=(info["new_h"], info["new_w"]),
        scale=info["scale"],
    )
    # Thumbnail strip
    thumbnail_strip, thumbnail_idxs = sample_thumbnail_frames(
        imgs_dir, frame_names, n=n_thumbnails,
    )

    stream = PromptStream()
    current_mask_strip: Optional[Image.Image] = None
    current_masks: Dict[int, np.ndarray] = {}
    history: List[Action] = []

    for step in range(K_max):
        # Compose user instruction for this step
        history_text = format_history(history)
        if step == 0:
            user_text = (
                f"This video has {T} frames. We have computed {K_actual} candidate "
                f"trajectory clusters (Image-1). Image-2 shows {n_thumbnails} sampled "
                f"frames. No mask predicted yet. Pick the cluster that is the "
                f"camouflaged animal.\n"
                f"Previous actions:\n{history_text}\n"
                f"Output ONE action."
            )
            images = [cluster_overview, thumbnail_strip]
        else:
            user_text = (
                f"This video has {T} frames, {K_actual} candidate clusters (Image-1). "
                f"Image-2 shows {n_thumbnails} sampled frames. Image-3 shows the "
                f"current predicted mask overlaid on those frames in red.\n"
                f"Previous actions:\n{history_text}\n"
                f"You may add positive/negative points to fix obvious errors, or "
                f"TERMINATE if the mask looks correct. Output ONE action."
            )
            images = [cluster_overview, thumbnail_strip, current_mask_strip]

        # MLLM call
        t0 = time.time()
        try:
            response = mllm.query(SYSTEM_PROMPT, user_text, images, max_new_tokens=48)
        except Exception as e:
            if verbose:
                print(f"  [step {step}] MLLM error: {e}", flush=True)
            break
        t_llm = time.time() - t0
        if verbose:
            print(f"  [step {step}] MLLM ({t_llm:.1f}s) → {response!r}", flush=True)

        try:
            action = parse_action(response)
        except ValueError as e:
            if verbose:
                print(f"  [step {step}] parse fail, defaulting to TERMINATE: {e}", flush=True)
            action = Action(type="TERMINATE", raw=response)

        history.append(action)

        if action.type == "TERMINATE":
            if verbose:
                print(f"  [step {step}] TERMINATE", flush=True)
            break

        if action.type == "SELECT":
            # Validate index
            if action.cluster_idx is None or action.cluster_idx < 0 \
               or action.cluster_idx >= K_actual:
                if verbose:
                    print(f"  [step {step}] invalid cluster {action.cluster_idx}, "
                          f"clipping to 0", flush=True)
                action.cluster_idx = 0
            # Convert cluster -> initial positive prompts at frame 0 (overrides
            # any previous SELECT but PRESERVES manual Add-Pos/Add-Neg later)
            stream.points_per_frame.clear()
            for (f_idx, x, y, lbl) in cluster_to_prompt_points(
                action.cluster_idx, info, tracks_k, vis_k, labels,
                n_prompt=n_prompt_points,
            ):
                stream.add(f_idx, x, y, lbl)

        elif action.type == "ADD_POS":
            stream.add(action.frame_idx, action.x, action.y, 1)
        elif action.type == "ADD_NEG":
            stream.add(action.frame_idx, action.x, action.y, 0)

        # Execute current prompt stream
        if stream.total_pos() == 0:
            if verbose:
                print(f"  [step {step}] no positive prompts yet; skipping SAM3", flush=True)
            continue
        t0 = time.time()
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                current_masks = run_sam3_session(sam3_predictor, imgs_dir,
                                                  stream, T)
        except Exception as e:
            if verbose:
                print(f"  [step {step}] SAM3 error: {e}", flush=True)
            continue
        t_sam = time.time() - t0
        if verbose:
            n_nonzero = sum(1 for m in current_masks.values()
                            if m is not None and m.sum() > 0)
            print(f"  [step {step}] SAM3 ({t_sam:.1f}s)  "
                  f"masks non-empty: {n_nonzero}/{len(current_masks)}", flush=True)

        # Update mask strip for next turn
        per_frame_orig = {}
        for fi, m in current_masks.items():
            if m is None: continue
            if m.shape != (info["orig_h"], info["orig_w"]):
                m = cv2.resize(m.astype(np.uint8),
                               (info["orig_w"], info["orig_h"]),
                               interpolation=cv2.INTER_NEAREST)
            per_frame_orig[fi] = m
        current_mask_strip = render_current_mask_strip(
            imgs_dir, frame_names, thumbnail_idxs, per_frame_orig,
        )

    return {
        "name": name,
        "n_steps": len(history),
        "history": [a.to_text() for a in history],
        "n_frames": T,
        "masks": current_masks,
        "frame_names": frame_names,
        "orig_h": info["orig_h"], "orig_w": info["orig_w"],
    }
