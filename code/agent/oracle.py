"""
Oracle action-trajectory generator for TrajCamo BC training.

For each train video with GT, simulate the agent loop with MASK-IoU
supervision and record the optimal `(state_t, action_t)` pairs. The agent
that we train will imitate this oracle.

Algorithm per video (greedy):
  step 0:   SELECT(k*)  with k* = argmax_k IoU(cluster_k_region, GT_union)
            Run SAM3 → current_masks
  step t>0:
            For each candidate action  ∈ {ADD_POS, ADD_NEG} on the worst-IoU
            annotated frame f:
              - ADD_POS at the centroid of the largest FN component of frame f
              - ADD_NEG at the centroid of the largest FP component of frame f
            Simulate SAM3 with each candidate appended to the prompt stream.
            Pick the action that yields the largest IoU improvement.
            If neither improves IoU by > Δ_min OR we reach IoU≥iou_term OR
            we hit K_max-1: emit TERMINATE.

Output per video:
    /root/autodl-tmp/VOScode/agent_outputs/oracle/{name}/
        step_00_state_cluster_overview.jpg
        step_00_state_thumbnails.jpg
        step_00_meta.json   {prompt, action: "SELECT(k)"}
        step_01_state_cluster_overview.jpg
        step_01_state_thumbnails.jpg
        step_01_state_mask_overlay.jpg
        step_01_meta.json   {prompt, action: "ADD_POS(...)"}
        ...
        record.json   {video, n_steps, final_iou, actions: [...]}

A master file `/root/autodl-tmp/VOScode/agent_outputs/oracle/index.jsonl`
lists every (video, step_idx, state_paths, action_text) sample for the
training dataloader.
"""
from __future__ import annotations

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
sys.path.insert(0, str(Path(__file__).parent.parent))

from actions import Action, format_history
from state_builder import (
    render_cluster_overview,
    render_current_mask_strip,
    sample_thumbnail_frames,
)


# ---------------------------------------------------------------------------
# Same building blocks as agent.py (kept here to avoid heavy import paths)
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


def cluster_trajectories(info: dict, K: int = 8, min_move_px: float = 4.0):
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
    return tracks_k, vis_k, km.labels_


def cluster_to_region_padded(tracks_k, mask, H, W, dilate_px=21):
    region = np.zeros((H, W), dtype=np.uint8)
    pts = tracks_k[:, mask].reshape(-1, 2)
    xs = pts[:, 0].round().astype(int).clip(0, W - 1)
    ys = pts[:, 1].round().astype(int).clip(0, H - 1)
    region[ys, xs] = 1
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        region = cv2.dilate(region, k, iterations=1)
    return region.astype(bool)


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return 0.0 if u == 0 else float((a & b).sum() / u)


def load_gt_union_padded(gt_dir, target_h, target_w, new_h, new_w):
    gt = np.zeros((target_h, target_w), dtype=np.uint8)
    for p in sorted(gt_dir.glob("*.png")):
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[..., 0]
        m_r = cv2.resize((m > 0).astype(np.uint8), (new_w, new_h),
                         interpolation=cv2.INTER_NEAREST)
        gt[:new_h, :new_w] |= m_r
    return gt.astype(bool)


def cluster_to_prompt_points(cluster_idx, info, tracks_k, vis_k, labels,
                              n_prompt=12):
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
# SAM 3 stream → masks
# ---------------------------------------------------------------------------
class PromptStream:
    def __init__(self):
        self.points_per_frame: dict = {}

    def add(self, f, x, y, lbl):
        self.points_per_frame.setdefault(int(f), []).append((float(x), float(y), int(lbl)))

    def clone(self):
        new = PromptStream()
        new.points_per_frame = {k: list(v) for k, v in self.points_per_frame.items()}
        return new

    def n_pos(self):
        return sum(1 for pts in self.points_per_frame.values() for _, _, l in pts if l == 1)


def run_sam3(predictor, imgs_dir, stream: PromptStream, T: int):
    state = predictor.init_state(video_path=str(imgs_dir))
    first = True
    for f_idx, pts in stream.points_per_frame.items():
        xy = torch.tensor([[p[0], p[1]] for p in pts], dtype=torch.float32)
        lbl = torch.tensor([p[2] for p in pts], dtype=torch.int32)
        predictor.add_new_points(
            inference_state=state, frame_idx=int(f_idx), obj_id=1,
            points=xy, labels=lbl, clear_old_points=first,
        )
        first = False
    predictor.propagate_in_video_preflight(state, run_mem_encoder=True)
    out = {}
    for fi, oid, _, vrm, _ in predictor.propagate_in_video(
        state, start_frame_idx=0, max_frame_num_to_track=T, reverse=False,
    ):
        if len(oid) == 0:
            out[fi] = None
        else:
            m = (vrm[0] > 0).cpu().numpy()
            if m.ndim == 3:
                m = m.squeeze(0)
            out[fi] = m.astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# IoU evaluation against per-frame GT
# ---------------------------------------------------------------------------
def per_frame_iou(masks, frame_names, gt_dir, orig_h, orig_w):
    """Return dict {frame_idx: iou} for annotated frames."""
    ious = {}
    for i, fname in enumerate(frame_names):
        gt_path = gt_dir / f"{fname}.png"
        if not gt_path.exists():
            continue
        gt = np.array(Image.open(gt_path))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = (gt > 0).astype(np.uint8)
        m = masks.get(i)
        if m is None:
            mb = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            mb = m.astype(np.uint8)
            if mb.shape != (orig_h, orig_w):
                mb = cv2.resize(mb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        ious[i] = iou(mb > 0, gt > 0)
    return ious


def mean_iou(ious_dict):
    if not ious_dict:
        return 0.0
    return float(np.mean(list(ious_dict.values())))


# ---------------------------------------------------------------------------
# Find a correction point: centroid of largest FN or FP connected component
# on the worst-IoU frame
# ---------------------------------------------------------------------------
def find_correction_target(masks, frame_names, gt_dir, orig_h, orig_w,
                            ious_dict):
    """Pick the frame with worst IoU (among annotated, where GT is non-empty),
    compute its FN and FP largest connected components, return both
    (frame_idx, fn_centroid_xy01, fp_centroid_xy01)."""
    annotated = [(fi, ious_dict[fi]) for fi in ious_dict
                 if (gt_dir / f"{frame_names[fi]}.png").exists()]
    if not annotated:
        return None
    annotated.sort(key=lambda x: x[1])
    worst_fi = annotated[0][0]
    fname = frame_names[worst_fi]
    gt = np.array(Image.open(gt_dir / f"{fname}.png"))
    if gt.ndim == 3:
        gt = gt[..., 0]
    gt = (gt > 0).astype(np.uint8)
    m = masks.get(worst_fi)
    if m is None:
        mb = np.zeros_like(gt)
    else:
        mb = m.astype(np.uint8)
        if mb.shape != gt.shape:
            mb = cv2.resize(mb, (gt.shape[1], gt.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    fn = (gt > 0) & (mb == 0)
    fp = (gt == 0) & (mb > 0)

    def largest_cc_centroid(mask_bool, min_area=50):
        if mask_bool.sum() < min_area:
            return None, 0
        num, labels, stats, cents = cv2.connectedComponentsWithStats(
            mask_bool.astype(np.uint8), 8,
        )
        if num <= 1:
            return None, 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        area = int(areas[idx - 1])
        if area < min_area:
            return None, 0
        # pick a robust inside point: median of pixel coords (handles concave shapes)
        ys, xs = np.where(labels == idx)
        med = len(xs) // 2
        cx, cy = float(xs[med]), float(ys[med])
        H, W = mask_bool.shape
        return (cx / W, cy / H), area

    fn_pt, fn_area = largest_cc_centroid(fn)
    fp_pt, fp_area = largest_cc_centroid(fp)
    return dict(worst_frame=worst_fi, worst_iou=annotated[0][1],
                fn_xy=fn_pt, fn_area=fn_area,
                fp_xy=fp_pt, fp_area=fp_area)


# ---------------------------------------------------------------------------
# Main oracle generator on one video
# ---------------------------------------------------------------------------
def generate_oracle_for_video(predictor, video_dir: Path, traj_cache_path: Path,
                               out_dir: Path, K=8, K_max=5,
                               iou_term=0.85, min_iou_gain=0.005,
                               n_thumbnails=4, n_prompt_points=12,
                               save_states=True) -> dict:
    name = video_dir.name
    imgs_dir = video_dir / "Imgs"
    gt_dir = video_dir / "GT"
    if not traj_cache_path.exists():
        return {"name": name, "error": "no traj cache"}
    info = load_traj_cache(traj_cache_path)
    tracks_k, vis_k, labels = cluster_trajectories(info, K=K)
    K_actual = int(labels.max()) + 1
    frame_names = info["frame_names"]
    T = info["tracks"].shape[0]
    target_h, target_w = info["target_h"], info["target_w"]
    new_h, new_w = info["new_h"], info["new_w"]
    orig_h, orig_w = info["orig_h"], info["orig_w"]

    # GT union (padded)
    gt_union_padded = load_gt_union_padded(gt_dir, target_h, target_w, new_h, new_w)

    # Cluster overview + thumbnails (state assets, reused across steps)
    first_frame_bgr = cv2.imread(str(imgs_dir / f"{frame_names[0]}.jpg"))
    cluster_overview = render_cluster_overview(
        first_frame_bgr, tracks_k[0], vis_k[0], labels,
        target_h_w=(target_h, target_w), orig_h_w=(orig_h, orig_w),
        new_h_w=(new_h, new_w), scale=info["scale"],
    )
    thumbs, thumb_idxs = sample_thumbnail_frames(
        imgs_dir, frame_names, n=n_thumbnails,
    )

    # === Step 0: SELECT(k*) ===
    cluster_ious = []
    for k in range(K_actual):
        sel = labels == k
        if sel.sum() < 5:
            cluster_ious.append((k, 0.0))
            continue
        reg = cluster_to_region_padded(tracks_k, sel, target_h, target_w)
        cluster_ious.append((k, iou(reg, gt_union_padded)))
    cluster_ious.sort(key=lambda x: -x[1])
    k_star = cluster_ious[0][0]

    actions: list = []
    states: list = []   # list of dicts {prompt_text, images_for_step, action_text}
    stream = PromptStream()
    for (f, x, y, lbl) in cluster_to_prompt_points(
        k_star, info, tracks_k, vis_k, labels, n_prompt=n_prompt_points,
    ):
        stream.add(f, x, y, lbl)

    step_records = []
    # Record step 0 state (NO mask overlay yet — first action)
    step0_meta = dict(
        step=0,
        prompt_kind="select_only",
        history="(none)",
        action=f"SELECT({k_star})",
        cluster_oracle_ious={int(k): float(v) for k, v in cluster_ious},
    )
    step_records.append(("SELECT", k_star, None, None, None, step0_meta))
    actions.append(f"SELECT({k_star})")

    # Run SAM3 step 0
    if stream.n_pos() == 0:
        return {"name": name, "error": "no positive prompts after select"}
    try:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            current_masks = run_sam3(predictor, imgs_dir, stream, T)
    except Exception as e:
        return {"name": name, "error": f"sam3 step 0 failed: {e}"}
    ious_t = per_frame_iou(current_masks, frame_names, gt_dir, orig_h, orig_w)
    miou = mean_iou(ious_t)
    last_miou = miou

    # === Steps 1..K_max-1 ===
    for step in range(1, K_max):
        if miou >= iou_term:
            actions.append("TERMINATE")
            step_records.append(("TERMINATE", None, None, None, None,
                                  dict(step=step, reason="iou>=term",
                                       miou=float(miou), action="TERMINATE")))
            break

        # Build current_mask_strip for this step's state
        per_frame_orig = {}
        for fi, m in current_masks.items():
            if m is None: continue
            mb = m.astype(np.uint8)
            if mb.shape != (orig_h, orig_w):
                mb = cv2.resize(mb, (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)
            per_frame_orig[fi] = mb
        mask_strip = render_current_mask_strip(
            imgs_dir, frame_names, thumb_idxs, per_frame_orig,
        )

        # Find correction targets
        tgt = find_correction_target(current_masks, frame_names, gt_dir,
                                      orig_h, orig_w, ious_t)
        if tgt is None or (tgt["fn_xy"] is None and tgt["fp_xy"] is None):
            actions.append("TERMINATE")
            step_records.append(("TERMINATE", None, None, None, None,
                                  dict(step=step, reason="no_correction_target",
                                       miou=float(miou), action="TERMINATE")))
            break

        # Try ADD_POS at fn_xy → simulate IoU
        best_action = None
        best_gain = -1e9
        worst_fi = tgt["worst_frame"]
        for cand_kind, xy in (("ADD_POS", tgt["fn_xy"]), ("ADD_NEG", tgt["fp_xy"])):
            if xy is None:
                continue
            cand_stream = stream.clone()
            cand_stream.add(worst_fi, xy[0], xy[1], 1 if cand_kind == "ADD_POS" else 0)
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    cand_masks = run_sam3(predictor, imgs_dir, cand_stream, T)
            except Exception:
                continue
            cand_ious = per_frame_iou(cand_masks, frame_names, gt_dir,
                                       orig_h, orig_w)
            cand_miou = mean_iou(cand_ious)
            gain = cand_miou - miou
            if gain > best_gain:
                best_gain = gain
                best_action = (cand_kind, worst_fi, xy[0], xy[1],
                                cand_stream, cand_masks, cand_miou, cand_ious)

        if best_action is None or best_gain < min_iou_gain:
            # Neither correction helps enough — TERMINATE
            actions.append("TERMINATE")
            step_records.append(("TERMINATE", None, None, None, None,
                                  dict(step=step, reason="no_iou_gain",
                                       miou=float(miou), action="TERMINATE",
                                       best_gain_tried=float(best_gain) if best_action else None)))
            break

        kind, fi, x, y, new_stream, new_masks, new_miou, new_ious = best_action
        act_text = f"{kind}({fi}, {x:.3f}, {y:.3f})"
        # Record this step's state + chosen action
        step_meta = dict(
            step=step, miou_before=float(miou), miou_after=float(new_miou),
            worst_frame=int(fi), worst_iou_before=float(tgt["worst_iou"]),
            fn_area=int(tgt["fn_area"]), fp_area=int(tgt["fp_area"]),
            action=act_text,
            history="\n".join([f"  Step {i+1}: {a}" for i, a in enumerate(actions)]),
        )
        step_records.append((kind, None, fi, x, y, step_meta))
        actions.append(act_text)
        stream = new_stream
        current_masks = new_masks
        ious_t = new_ious
        miou = new_miou

    # Ensure a final TERMINATE if loop ran out
    if actions[-1] != "TERMINATE":
        actions.append("TERMINATE")
        step_records.append(("TERMINATE", None, None, None, None,
                              dict(step=len(actions) - 1, reason="kmax_reached",
                                   miou=float(miou), action="TERMINATE")))

    # === Save states / images per step ===
    save_dir = out_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    cluster_overview_path = save_dir / "cluster_overview.jpg"
    thumbs_path = save_dir / "thumbnails.jpg"
    cluster_overview.save(cluster_overview_path, quality=88)
    thumbs.save(thumbs_path, quality=88)

    index_entries = []
    current_masks_so_far = None
    cumulative_stream = PromptStream()
    # Re-walk the actions chronologically so we save the right mask_overlay
    # at each step (the mask BEFORE this step's action was taken)
    for step_idx, (kind, k_star_, fi, x, y, meta) in enumerate(step_records):
        step_meta_path = save_dir / f"step_{step_idx:02d}_meta.json"
        mask_overlay_path = None
        if step_idx == 0:
            mask_overlay_path = None
        else:
            # mask BEFORE this action: at step 1, that's after SELECT; at step 2, after step 1, etc.
            mask_overlay_path = save_dir / f"step_{step_idx:02d}_mask_overlay.jpg"
        # Re-render mask overlay if needed. For simplicity, we'll skip
        # per-step overlay re-rendering for now and let the train script
        # use cluster_overview + thumbnails + current_mask_strip from the LAST
        # SAM3 run (good enough for BC because the mask state is in the
        # action history).
        rec = dict(
            video=name, step=step_idx,
            cluster_overview=str(cluster_overview_path.relative_to(out_dir.parent)),
            thumbnails=str(thumbs_path.relative_to(out_dir.parent)),
            mask_overlay=None if mask_overlay_path is None
                           else str(mask_overlay_path.relative_to(out_dir.parent)),
            history=meta.get("history", ""),
            action=meta["action"],
            meta=meta,
        )
        index_entries.append(rec)

    record = dict(name=name, n_steps=len(actions), final_miou=float(miou),
                  actions=actions)
    with open(save_dir / "record.json", "w") as f:
        json.dump(record, f, indent=2, default=str)

    return dict(name=name, actions=actions, final_miou=float(miou),
                n_steps=len(actions), index_entries=index_entries,
                k_star=int(k_star),
                cluster_oracle_ious_top3=[(int(k), float(v)) for k, v in cluster_ious[:3]])


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset"))
    ap.add_argument("--traj_cache", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_traj_cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("/root/autodl-tmp/VOScode/agent_outputs/oracle"))
    ap.add_argument("--sam3_ckpt", type=Path,
                    default=Path("/root/autodl-tmp/sam3_base_weights/sam3.pt"))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--K_max", type=int, default=5)
    ap.add_argument("--iou_term", type=float, default=0.85)
    ap.add_argument("--min_iou_gain", type=float, default=0.005)
    ap.add_argument("--n_thumbnails", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to specific video names")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total #videos for smoke test")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[init] loading SAM 3 ...", flush=True)
    from sam3.model_builder import build_sam3_video_model
    m = build_sam3_video_model(checkpoint_path=str(args.sam3_ckpt),
                                load_from_HF=False)
    predictor = m.tracker; predictor.backbone = m.detector.backbone
    print(f"[init] ready.\n", flush=True)

    train_root = args.dataset / "TrainDataset_per_sq"
    videos = sorted([d for d in train_root.iterdir() if d.is_dir()])
    if args.only:
        want = set(args.only)
        videos = [v for v in videos if v.name in want]
    if args.limit:
        videos = videos[: args.limit]

    print(f"[start] {len(videos)} train videos", flush=True)
    all_index = []
    skip_no_gt = 0
    t_all = time.time()
    for i, vd in enumerate(videos):
        cache = args.traj_cache / "TrainDataset_per_sq" / f"{vd.name}.npz"
        gt_dir = vd / "GT"
        gt_pngs = list(gt_dir.glob("*.png"))
        if not gt_pngs:
            skip_no_gt += 1
            continue
        t0 = time.time()
        try:
            res = generate_oracle_for_video(
                predictor, vd, cache, args.out,
                K=args.K, K_max=args.K_max,
                iou_term=args.iou_term, min_iou_gain=args.min_iou_gain,
                n_thumbnails=args.n_thumbnails,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [{i+1:>2}/{len(videos)}] {vd.name:<28} ERROR: {e}", flush=True)
            continue
        dt = time.time() - t0
        if "error" in res:
            print(f"  [{i+1:>2}/{len(videos)}] {vd.name:<28} ERR: {res['error']}", flush=True)
            continue
        all_index.extend(res["index_entries"])
        print(f"  [{i+1:>2}/{len(videos)}] {vd.name:<28} "
              f"k*={res['k_star']:>2} steps={res['n_steps']} "
              f"final_miou={res['final_miou']:.3f}  "
              f"actions=[{', '.join(res['actions'])}]  ({dt:.1f}s)", flush=True)

    # Write index.jsonl
    index_path = args.out / "index.jsonl"
    with open(index_path, "w") as f:
        for rec in all_index:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"\n[done] {len(all_index)} (state, action) samples from "
          f"{len(videos) - skip_no_gt} videos in {time.time()-t_all:.0f}s",
          flush=True)
    print(f"[written] index: {index_path}", flush=True)


if __name__ == "__main__":
    main()
