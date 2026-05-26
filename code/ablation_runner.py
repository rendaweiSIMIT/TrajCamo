"""
Unified ablation runner for TrajCamo phase-2 pipeline.

Single-script driver that takes (feature, K, n_prompt_points, min_move_px,
prompt_frames) as command-line knobs, loads the cached CoTracker3
trajectories, runs the K-means + SAM3 propagation pipeline on all 16
MoCA-Mask test videos, and writes a JSON of per-video metrics +
aggregates into the output directory.

Designed for fast iteration: SAM 3 model is loaded once per process and
amortized across all videos.

Usage (per ablation row, but a master shell driver exists too):
    python ablation_runner.py --tag feat_rawvel --feature raw_velocity
    python ablation_runner.py --tag K_16 --K 16
    python ablation_runner.py --tag mf_3 --prompt_frames 0,mid,last
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
# Feature extractors
# ---------------------------------------------------------------------------
def feat_position(tracks: np.ndarray) -> np.ndarray:
    f = tracks.mean(axis=0)                     # (N, 2)
    return normalize(f, axis=1)


def feat_raw_speed(tracks: np.ndarray) -> np.ndarray:
    v = np.diff(tracks, axis=0)
    s = np.linalg.norm(v, axis=2).T             # (N, T-1)
    return normalize(s, axis=1)


def feat_raw_velocity(tracks: np.ndarray) -> np.ndarray:
    v = np.diff(tracks, axis=0)
    f = v.transpose(1, 0, 2).reshape(v.shape[1], -1)
    return normalize(f, axis=1)


def feat_coherence(tracks: np.ndarray) -> np.ndarray:
    T, N, _ = tracks.shape
    v = np.diff(tracks, axis=0)
    s = np.linalg.norm(v, axis=2)
    s_zm = s - s.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(s_zm, axis=0)).T  # (N, T//2+1)

    def autocorr(x, lag):
        x_zm = x - x.mean(axis=0, keepdims=True)
        denom = (x_zm ** 2).sum(axis=0) + 1e-8
        return (x_zm[lag:] * x_zm[:-lag]).sum(axis=0) / denom
    ac = np.stack([autocorr(s, l) for l in (1, 2, 4, 8) if l < T - 1], axis=1)

    init_pos = tracks[0]
    diffs = init_pos[:, None, :] - init_pos[None, :, :]
    dists0 = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists0, np.inf)
    nn_idx = np.argsort(dists0, axis=1)[:, :8]
    rigidity = np.zeros((N, 8))
    for i in range(N):
        nn = nn_idx[i]
        dij = np.linalg.norm(tracks[:, [i]] - tracks[:, nn], axis=2)
        rigidity[i] = dij.std(axis=0) / (dij.mean(axis=0) + 1e-6)

    spec_n = normalize(spec, axis=1)
    ac_n = normalize(ac, axis=1)
    rig_n = normalize(rigidity, axis=1)
    return np.hstack([spec_n, ac_n, rig_n])


FEATURE_REGISTRY = {
    "position":      feat_position,
    "raw_speed":     feat_raw_speed,
    "raw_velocity":  feat_raw_velocity,
    "coherence":     feat_coherence,
}


# ---------------------------------------------------------------------------
# Utilities reused from phase2 (mask region, IoU, FPS, consistency)
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
    sel = labels == k
    if sel.sum() < 2:
        return 0.0
    fs = features[sel]
    mu = fs.mean(axis=0, keepdims=True)
    within = ((fs - mu) ** 2).sum(axis=1).mean()
    overall_mu = features.mean(axis=0, keepdims=True)
    total = ((features - overall_mu) ** 2).sum(axis=1).mean()
    return float(1.0 - within / (total + 1e-8))


def load_trajectory_cache(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return dict(
        tracks=data["tracks"], visibility=data["visibility"],
        frame_names=list(data["frame_names"]),
        grid_size=int(data["grid_size"]),
        orig_h=int(data["orig_h"]), orig_w=int(data["orig_w"]),
        new_h=int(data["new_h"]),   new_w=int(data["new_w"]),
        target_h=int(data["target_h"]), target_w=int(data["target_w"]),
        scale=float(data["scale"]),
    )


def load_gt_binary_at(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def load_gt_union(gt_dir: Path, target_h: int, target_w: int,
                  new_h: int, new_w: int) -> np.ndarray:
    gt = np.zeros((target_h, target_w), dtype=np.uint8)
    for p in sorted(gt_dir.glob("*.png")):
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[..., 0]
        m_r = cv2.resize((m > 0).astype(np.uint8), (new_w, new_h),
                         interpolation=cv2.INTER_NEAREST)
        gt[:new_h, :new_w] |= m_r
    return gt.astype(bool)


def build_sam3_tracker(ckpt_path: Path):
    from sam3.model_builder import build_sam3_video_model
    print(f"[init] building SAM 3 base ...", flush=True)
    m = build_sam3_video_model(checkpoint_path=str(ckpt_path), load_from_HF=False)
    p = m.tracker
    p.backbone = m.detector.backbone
    print(f"[init] SAM 3 ready.", flush=True)
    return p


def parse_prompt_frames_spec(spec: str, T: int):
    """Parse "0", "0,mid,last", "0,T/4,T/2,3T/4,last", ..."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in ("0", ""):
            out.append(0)
        elif tok == "mid":
            out.append(T // 2)
        elif tok == "last":
            out.append(T - 1)
        elif tok.startswith("T/"):
            d = int(tok.split("/")[1])
            out.append(T // d)
        elif tok.startswith("3T/"):
            d = int(tok.split("/")[1])
            out.append((3 * T) // d)
        else:
            out.append(int(tok))
    out = [max(0, min(T - 1, t)) for t in out]
    return sorted(set(out))


def sample_centroid_neighborhood(pts: np.ndarray, n_prompt: int):
    if len(pts) == 0:
        return pts
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    order = np.argsort(dists)
    n_take = min(n_prompt, max(3, len(pts) // 4))
    sel = pts[order[:n_take]]
    med_d = np.median(dists)
    keep = np.linalg.norm(sel - centroid, axis=1) <= max(med_d * 2, 30.0)
    sel = sel[keep]
    if len(sel) == 0:
        sel = centroid[None, :]
    return sel


# ---------------------------------------------------------------------------
# Per-video pipeline (configurable)
# ---------------------------------------------------------------------------
def evaluate_video(
    predictor,
    video_dir: Path,
    traj_cache_path: Path,
    out_dir: Path,
    feature: str,
    K: int,
    n_prompt_points: int,
    min_move_px: float,
    prompt_frames_spec: str,
    save_masks: bool,
):
    name = video_dir.name
    imgs_dir = video_dir / "Imgs"
    gt_dir = video_dir / "GT"
    if not traj_cache_path.exists():
        return {"name": name, "error": "no trajectory cache"}

    info = load_trajectory_cache(traj_cache_path)
    tracks_p = info["tracks"]; vis_p = info["visibility"]
    frame_names = info["frame_names"]
    target_h, target_w = info["target_h"], info["target_w"]
    new_h, new_w = info["new_h"], info["new_w"]
    orig_h, orig_w = info["orig_h"], info["orig_w"]
    scale = info["scale"]; T = tracks_p.shape[0]

    keep = vis_p.mean(axis=0) >= 0.2
    tracks_v = tracks_p[:, keep]; vis_v = vis_p[:, keep]

    pos_range = tracks_v.max(axis=0) - tracks_v.min(axis=0)
    movement = np.linalg.norm(pos_range, axis=1)
    dynamic = movement > min_move_px
    if dynamic.sum() < K * 3:
        dynamic = np.ones_like(dynamic, dtype=bool)
    tracks_k = tracks_v[:, dynamic]; vis_k = vis_v[:, dynamic]

    if tracks_k.shape[1] < K:
        return {"name": name, "error": f"too few trajectories ({tracks_k.shape[1]})"}

    F = FEATURE_REGISTRY[feature](tracks_k)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(F)
    labels = km.labels_

    gt_union_padded = load_gt_union(gt_dir, target_h, target_w, new_h, new_w)

    cluster_info = []
    for k in range(K):
        sel = labels == k
        if sel.sum() < 5:
            continue
        region = cluster_to_region_padded(tracks_k, sel, target_h, target_w)
        cluster_info.append(dict(
            k=int(k), size=int(sel.sum()),
            iou_oracle=iou(region, gt_union_padded),
            consistency=cluster_consistency(F, labels, k),
        ))
    if not cluster_info:
        return {"name": name, "error": "no valid clusters"}

    best_oracle = max(cluster_info, key=lambda c: c["iou_oracle"])
    best_heur = max(cluster_info, key=lambda c: c["consistency"])

    results = {
        "name": name, "n_frames": int(T),
        "n_traj_total": int(tracks_v.shape[1]),
        "n_traj_dynamic": int(tracks_k.shape[1]),
        "config": dict(feature=feature, K=K, n_prompt_points=n_prompt_points,
                       min_move_px=min_move_px,
                       prompt_frames_spec=prompt_frames_spec),
        "cluster_info": cluster_info,
    }

    prompt_frame_idxs = parse_prompt_frames_spec(prompt_frames_spec, T)

    for label_name, chosen in [("oracle", best_oracle), ("heuristic", best_heur)]:
        chosen_k = chosen["k"]; sel = labels == chosen_k

        # Sample prompts per requested frame
        prompts_per_frame = []
        for f_idx in prompt_frame_idxs:
            ft_visible = vis_k[f_idx] & sel
            pts_padded = tracks_k[f_idx, ft_visible]
            if len(pts_padded) < 1:
                continue
            sel_pts = sample_centroid_neighborhood(pts_padded, n_prompt_points)
            # padded -> orig -> [0,1]
            sel_orig = sel_pts / scale
            sel_orig[:, 0] = sel_orig[:, 0].clip(0, orig_w - 1)
            sel_orig[:, 1] = sel_orig[:, 1].clip(0, orig_h - 1)
            sel_rel = np.stack(
                [sel_orig[:, 0] / orig_w, sel_orig[:, 1] / orig_h], axis=1
            )
            prompts_per_frame.append((f_idx, sel_rel))

        if not prompts_per_frame:
            results[label_name] = {"error": "no visible cluster points"}
            continue

        # Run SAM 3 propagation
        try:
            from sam3.model_builder import build_sam3_video_model  # noqa
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = predictor.init_state(video_path=str(imgs_dir))
                for i_pf, (f_idx, sel_rel) in enumerate(prompts_per_frame):
                    pts_t = torch.tensor(sel_rel, dtype=torch.float32)
                    lbl_t = torch.ones(len(sel_rel), dtype=torch.int32)
                    predictor.add_new_points(
                        inference_state=state, frame_idx=int(f_idx), obj_id=1,
                        points=pts_t, labels=lbl_t,
                        clear_old_points=(i_pf == 0),
                    )
                predictor.propagate_in_video_preflight(state, run_mem_encoder=True)
                per_frame_masks = {}
                for fi, oid, _, vrm, _ in predictor.propagate_in_video(
                    state, start_frame_idx=0,
                    max_frame_num_to_track=T, reverse=False,
                ):
                    if len(oid) == 0:
                        per_frame_masks[fi] = None
                    else:
                        m = (vrm[0] > 0).cpu().numpy()
                        if m.ndim == 3:
                            m = m.squeeze(0)
                        per_frame_masks[fi] = m.astype(np.uint8)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[label_name] = {"error": str(e)}
            continue

        per_frame_metrics = []
        m_dir = out_dir / name / label_name / "masks" if save_masks else None
        if save_masks:
            m_dir.mkdir(parents=True, exist_ok=True)
        for i, fname in enumerate(frame_names):
            m = per_frame_masks.get(i)
            if m is None:
                m = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if m.shape != (orig_h, orig_w):
                m = cv2.resize(m.astype(np.uint8), (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
            if save_masks:
                Image.fromarray((m > 0).astype(np.uint8) * 255).save(
                    m_dir / f"{fname}.png"
                )
            gt_path = gt_dir / f"{fname}.png"
            if not gt_path.exists():
                continue
            gt = load_gt_binary_at(gt_path).astype(np.float64)
            pr = (m > 0).astype(np.float64)
            per_frame_metrics.append({
                "frame": fname,
                "F_w": f_beta_w(pr, gt), "MAE": mae(pr, gt),
                "S_a": s_alpha(pr, gt), "E_p": e_phi(pr, gt),
            })
        if per_frame_metrics:
            agg = {k: float(np.mean([m[k] for m in per_frame_metrics]))
                   for k in ("F_w", "MAE", "S_a", "E_p")}
            agg["n_frames"] = len(per_frame_metrics)
        else:
            agg = {"error": "no GT frames"}
        results[label_name] = {
            "selected_cluster": chosen,
            "n_prompts_per_frame": [len(pf[1]) for pf in prompts_per_frame],
            "prompt_frame_idxs": prompt_frame_idxs,
            "metrics": agg,
        }
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="short name used for output folder + report filename")
    ap.add_argument("--feature", default="raw_velocity",
                    choices=list(FEATURE_REGISTRY.keys()))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_prompt_points", type=int, default=12)
    ap.add_argument("--min_move_px", type=float, default=4.0)
    ap.add_argument("--prompt_frames", default="0",
                    help='comma list of "0|mid|last|<int>"')
    ap.add_argument("--save_masks", action="store_true",
                    help="save per-frame mask PNGs (off by default to save disk)")
    ap.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/VOSdataset"))
    ap.add_argument("--traj_cache", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_traj_cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("/root/autodl-tmp/VOScode/ablations"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/root/autodl-tmp/sam3_base_weights/sam3.pt"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    predictor = build_sam3_tracker(args.ckpt)

    test_root = args.dataset / "TestDataset_per_sq"
    videos = sorted([d for d in test_root.iterdir() if d.is_dir()])
    print(f"\n[start] tag={args.tag}  feat={args.feature}  K={args.K}  "
          f"n_prompt={args.n_prompt_points}  min_move={args.min_move_px}  "
          f"frames={args.prompt_frames}", flush=True)

    rows = []
    for v in videos:
        t0 = time.time()
        cache = args.traj_cache / "TestDataset_per_sq" / f"{v.name}.npz"
        out_dir = args.out / args.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        res = evaluate_video(
            predictor, v, cache, out_dir,
            feature=args.feature, K=args.K,
            n_prompt_points=args.n_prompt_points,
            min_move_px=args.min_move_px,
            prompt_frames_spec=args.prompt_frames,
            save_masks=args.save_masks,
        )
        res["seconds"] = round(time.time() - t0, 1)
        rows.append(res)
        def fmt(d):
            if not isinstance(d, dict) or "metrics" not in d:
                return "ERR"
            m = d["metrics"]
            if "error" in m: return "ERR"
            return f"F_w={m['F_w']:.3f} MAE={m['MAE']:.4f} S_a={m['S_a']:.3f} E_p={m['E_p']:.3f}"
        print(
            f"  {v.name:<28} ({res['seconds']:>5.1f}s)  "
            f"oracle:{fmt(res.get('oracle',{}))}   "
            f"heuristic:{fmt(res.get('heuristic',{}))}",
            flush=True,
        )

    # Aggregate + write
    aggregates = {}
    for sel in ("oracle", "heuristic"):
        vals = {k: [] for k in ("F_w", "MAE", "S_a", "E_p")}
        for r in rows:
            d = r.get(sel, {})
            m = d.get("metrics", {}) if isinstance(d, dict) else {}
            if "error" in m or not m: continue
            for k in vals: vals[k].append(m[k])
        if all(len(vals[k]) for k in vals):
            aggregates[sel] = {k: float(np.mean(v)) for k, v in vals.items()}
            aggregates[sel]["n_videos"] = len(vals["F_w"])

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    out_json = args.out / f"{args.tag}.json"
    with open(out_json, "w") as f:
        json.dump({"config": cfg, "aggregates": aggregates, "rows": rows},
                  f, indent=2, default=str)

    print(f"\n[aggregate]", flush=True)
    for sel, agg in aggregates.items():
        print(f"  {sel:<10} F_w={agg['F_w']:.3f}  MAE={agg['MAE']:.4f}  "
              f"S_a={agg['S_a']:.3f}  E_p={agg['E_p']:.3f}  "
              f"(n_videos={agg['n_videos']})", flush=True)
    print(f"\n[written] {out_json}", flush=True)


if __name__ == "__main__":
    main()
