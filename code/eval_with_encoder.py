"""
Evaluate TrajCamo using a trained Kinematic Signature Encoder (paper §3.2)
in place of the hand-crafted features. All other pipeline stages identical
to ablation_runner.py.

Reuses the same per-video pipeline (cluster on encoder embeddings → centroid
prompt → SAM 3 propagation → 4 COD metrics). Writes the same JSON shape.

Usage:
    python eval_with_encoder.py --ckpt /root/autodl-tmp/VOSdataset/_signature_encoder.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from cod_metrics import mae, f_beta_w, s_alpha, e_phi
from ablation_runner import (
    cluster_to_region_padded, iou, cluster_consistency,
    load_trajectory_cache, load_gt_binary_at, load_gt_union,
    build_sam3_tracker, parse_prompt_frames_spec, sample_centroid_neighborhood,
)
from phase4_train_signature_encoder import (
    KinematicSignatureEncoder, build_trajectory_inputs,
)


@torch.no_grad()
def encode_all(model: KinematicSignatureEncoder, feats: np.ndarray,
               device, batch: int = 1024) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(feats), batch):
        x = torch.from_numpy(feats[i : i + batch]).float().to(device)
        z = model(x)
        outs.append(z.cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_video(predictor, encoder, video_dir, traj_cache_path,
                   K: int, n_prompt_points: int, min_move_px: float,
                   prompt_frames_spec: str, device):
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

    # Encode through trained model
    feats = build_trajectory_inputs(tracks_k, L=encoder.pos_embed.size(1))
    Z = encode_all(encoder, feats, device)

    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Z)
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
            consistency=cluster_consistency(Z, labels, k),
        ))
    if not cluster_info:
        return {"name": name, "error": "no valid clusters"}

    best_oracle = max(cluster_info, key=lambda c: c["iou_oracle"])
    best_heur = max(cluster_info, key=lambda c: c["consistency"])

    results = {
        "name": name, "n_frames": int(T),
        "n_traj_dynamic": int(tracks_k.shape[1]),
        "config": dict(feature="learned_encoder", K=K,
                       n_prompt_points=n_prompt_points,
                       min_move_px=min_move_px,
                       prompt_frames_spec=prompt_frames_spec),
        "cluster_info": cluster_info,
    }
    prompt_frame_idxs = parse_prompt_frames_spec(prompt_frames_spec, T)

    for label_name, chosen in [("oracle", best_oracle), ("heuristic", best_heur)]:
        chosen_k = chosen["k"]; sel = labels == chosen_k
        prompts_per_frame = []
        for f_idx in prompt_frame_idxs:
            ft_visible = vis_k[f_idx] & sel
            pts_padded = tracks_k[f_idx, ft_visible]
            if len(pts_padded) < 1:
                continue
            sel_pts = sample_centroid_neighborhood(pts_padded, n_prompt_points)
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
        try:
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
        for i, fname in enumerate(frame_names):
            m = per_frame_masks.get(i)
            if m is None:
                m = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if m.shape != (orig_h, orig_w):
                m = cv2.resize(m.astype(np.uint8), (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
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
            "metrics": agg,
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_signature_encoder.pt"))
    ap.add_argument("--tag", default="learned_encoder")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_prompt_points", type=int, default=12)
    ap.add_argument("--min_move_px", type=float, default=4.0)
    ap.add_argument("--prompt_frames", default="0")
    ap.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/VOSdataset"))
    ap.add_argument("--traj_cache", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_traj_cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("/root/autodl-tmp/VOScode/ablations"))
    ap.add_argument("--sam3_ckpt", type=Path,
                    default=Path("/root/autodl-tmp/sam3_base_weights/sam3.pt"))
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"[error] encoder ckpt not found: {args.ckpt}", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # Load encoder
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder = KinematicSignatureEncoder(**cfg).to(device).eval()
    encoder.load_state_dict(ckpt["state_dict"])
    print(f"[init] loaded encoder ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M)",
          flush=True)

    # Load SAM 3 once
    predictor = build_sam3_tracker(args.sam3_ckpt)

    test_root = args.dataset / "TestDataset_per_sq"
    videos = sorted([d for d in test_root.iterdir() if d.is_dir()])

    print(f"\n[start] tag={args.tag} K={args.K} min_move={args.min_move_px}",
          flush=True)
    rows = []
    for v in videos:
        t0 = time.time()
        cache = args.traj_cache / "TestDataset_per_sq" / f"{v.name}.npz"
        res = evaluate_video(
            predictor, encoder, v, cache,
            K=args.K, n_prompt_points=args.n_prompt_points,
            min_move_px=args.min_move_px,
            prompt_frames_spec=args.prompt_frames, device=device,
        )
        res["seconds"] = round(time.time() - t0, 1)
        rows.append(res)
        def fmt(d):
            if not isinstance(d, dict) or "metrics" not in d: return "ERR"
            m = d["metrics"]
            if "error" in m: return "ERR"
            return f"F_w={m['F_w']:.3f} MAE={m['MAE']:.4f} S_a={m['S_a']:.3f} E_p={m['E_p']:.3f}"
        print(f"  {v.name:<28} ({res['seconds']:>5.1f}s)  "
              f"oracle:{fmt(res.get('oracle',{}))}   "
              f"heuristic:{fmt(res.get('heuristic',{}))}", flush=True)

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

    out_json = args.out / f"{args.tag}.json"
    cfg_dump = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()}
    with open(out_json, "w") as f:
        json.dump({"config": cfg_dump, "aggregates": aggregates, "rows": rows},
                  f, indent=2, default=str)
    print(f"\n[aggregate]", flush=True)
    for sel, agg in aggregates.items():
        print(f"  {sel:<10} F_w={agg['F_w']:.3f}  MAE={agg['MAE']:.4f}  "
              f"S_a={agg['S_a']:.3f}  E_p={agg['E_p']:.3f}  "
              f"(n_videos={agg['n_videos']})", flush=True)
    print(f"\n[written] {out_json}", flush=True)


if __name__ == "__main__":
    main()
