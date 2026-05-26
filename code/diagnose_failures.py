"""
For each test video, diagnose what stage of the trajectory pipeline succeeds
or fails. Specifically check:

  1. Tracker coverage: do ANY trajectories actually start inside the GT mask?
     If 0 → CoTracker3 grid is too sparse for this target.
  2. Dynamic-filter coverage: after the MIN_MOVE_PX filter, do any in-GT
     trajectories survive? If not → filter is too aggressive for this target.
  3. Cluster recoverability: does ANY of the K candidate clusters have
     >50% of its trajectory points inside GT? If yes → oracle CAN recover
     this video; if no → clustering itself fails.

The output is a per-video diagnostic JSON + markdown table.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from ablation_runner import (
    load_trajectory_cache, load_gt_union, feat_raw_velocity,
)


DATASET = Path("/root/autodl-tmp/VOSdataset")
TRAJ_CACHE = DATASET / "_traj_cache" / "TestDataset_per_sq"
OUT_DIR = Path("/root/autodl-tmp/VOScode/ablations")
OUT_JSON = OUT_DIR / "failure_diagnosis.json"
OUT_MD = OUT_DIR / "FAILURE_DIAGNOSIS.md"


def diagnose_one(video_dir: Path, K: int = 8, min_move_px: float = 4.0) -> dict:
    name = video_dir.name
    cache = TRAJ_CACHE / f"{name}.npz"
    if not cache.exists():
        return {"name": name, "error": "no traj cache"}

    info = load_trajectory_cache(cache)
    tracks_p = info["tracks"]; vis_p = info["visibility"]
    target_h, target_w = info["target_h"], info["target_w"]
    new_h, new_w = info["new_h"], info["new_w"]

    gt_union = load_gt_union(video_dir / "GT", target_h, target_w, new_h, new_w)

    # 1. Tracker coverage: trajectory starts inside GT?
    pos0 = tracks_p[0]
    xs = pos0[:, 0].round().astype(int).clip(0, target_w - 1)
    ys = pos0[:, 1].round().astype(int).clip(0, target_h - 1)
    inside_gt_f0 = gt_union[ys, xs].sum()
    n_traj_total = tracks_p.shape[1]

    # 2. Visible trajectories
    keep_vis = vis_p.mean(axis=0) >= 0.2
    tracks_v = tracks_p[:, keep_vis]
    vis_v = vis_p[:, keep_vis]

    # GT membership for visible trajectories (over all frames)
    in_gt_per_frame = np.zeros(tracks_v.shape[1], dtype=np.int32)
    visible_total = np.zeros(tracks_v.shape[1], dtype=np.int32)
    for t in range(tracks_v.shape[0]):
        pt = tracks_v[t]
        x = pt[:, 0].round().astype(int).clip(0, target_w - 1)
        y = pt[:, 1].round().astype(int).clip(0, target_h - 1)
        in_gt_per_frame += (vis_v[t] & (gt_union[y, x] > 0)).astype(np.int32)
        visible_total += vis_v[t].astype(np.int32)

    # Trajectory is "on target" if >=50% of visible frames are inside GT
    on_target_mask = (visible_total >= 3) & (in_gt_per_frame >= 0.5 * np.maximum(visible_total, 1))
    n_target_traj = int(on_target_mask.sum())

    # 3. Dynamic filter
    pos_range = tracks_v.max(axis=0) - tracks_v.min(axis=0)
    movement = np.linalg.norm(pos_range, axis=1)
    dynamic = movement > min_move_px
    n_dynamic = int(dynamic.sum())
    n_target_dynamic = int((on_target_mask & dynamic).sum())

    # 4. Cluster recoverability (on dynamic subset)
    if n_dynamic < K * 3:
        cluster_recoverability = "skip (too few dynamic)"
        best_cluster_purity = 0.0
    else:
        tracks_k = tracks_v[:, dynamic]
        F = feat_raw_velocity(tracks_k)
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(F)
        labels = km.labels_
        # For each cluster, what fraction of its points are on-target?
        on_target_d = on_target_mask[dynamic]
        best_purity = 0.0
        best_size_on_target = 0
        for k in range(K):
            sel = labels == k
            if sel.sum() < 2:
                continue
            ontarget_in_cluster = (on_target_d & sel).sum()
            purity = ontarget_in_cluster / sel.sum()
            if purity > best_purity and ontarget_in_cluster >= 3:
                best_purity = purity
                best_size_on_target = int(ontarget_in_cluster)
        cluster_recoverability = "yes" if best_purity >= 0.5 else "no"
        best_cluster_purity = float(best_purity)

    return dict(
        name=name,
        n_traj_total=int(n_traj_total),
        n_traj_visible=int(keep_vis.sum()),
        inside_gt_f0=int(inside_gt_f0),
        on_target_visible=int(n_target_traj),
        n_dynamic=int(n_dynamic),
        on_target_dynamic=int(n_target_dynamic),
        cluster_recoverable=cluster_recoverability,
        best_cluster_purity=round(best_cluster_purity, 3),
    )


def main():
    test_root = DATASET / "TestDataset_per_sq"
    videos = sorted([d for d in test_root.iterdir() if d.is_dir()])
    rows = []
    print(f"{'video':<28}  Ntot  Nvis  Ngt0  Ngt-vis  Ndyn  Ngt-dyn  recover  best-purity")
    for v in videos:
        r = diagnose_one(v)
        rows.append(r)
        if "error" in r:
            print(f"  {r['name']:<28}  ERROR: {r['error']}")
            continue
        print(f"  {r['name']:<28}  {r['n_traj_total']:>4}  "
              f"{r['n_traj_visible']:>4}  {r['inside_gt_f0']:>4}  "
              f"{r['on_target_visible']:>6}   {r['n_dynamic']:>4}  "
              f"{r['on_target_dynamic']:>6}   {r['cluster_recoverable']:<7}  "
              f"{r['best_cluster_purity']:.3f}")

    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2)

    # Markdown report
    lines = [
        "# Per-Video Failure Diagnosis (MoCA-Mask test, K=8, min_move=4.0)",
        "",
        "Goal: for each video, identify whether the trajectory pipeline has any",
        "chance of recovering the camouflaged target, and at which stage it",
        "fails when it does. Columns:",
        "",
        "* **N tot**: trajectories returned by CoTracker3 (grid=48 → 2304).",
        "* **N vis**: trajectories visible in ≥20% of frames.",
        "* **N start-in-GT**: trajectories whose frame-0 position falls inside any GT mask.",
        "* **N on-target**: visible trajectories whose ≥50% of visible positions",
        "  fall inside some GT mask (the trajectories that *should* form the",
        "  target cluster).",
        "* **N dynamic**: visible trajectories that survive `min_move_px=4` filter.",
        "* **N on-target ∩ dynamic**: target trajectories that also pass the",
        "  dynamic filter (these are the ones available to the clusterer).",
        "* **Cluster recoverable**: yes if K-means finds a cluster with >50% of",
        "  its points being on-target — i.e. the cluster IS a target cluster.",
        "* **Best purity**: max fraction of on-target points across all K clusters.",
        "",
        "| Video | N tot | N vis | start-in-GT | on-target | dynamic | on-target ∩ dynamic | Cluster recoverable | Best purity |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | ERR | | | | | | | |")
            continue
        lines.append(
            f"| {r['name']} | {r['n_traj_total']} | {r['n_traj_visible']} | "
            f"{r['inside_gt_f0']} | {r['on_target_visible']} | "
            f"{r['n_dynamic']} | {r['on_target_dynamic']} | "
            f"`{r['cluster_recoverable']}` | {r['best_cluster_purity']:.3f} |"
        )

    # Summary patterns
    failures = [r for r in rows if "error" not in r and r.get("best_cluster_purity", 0) < 0.5]
    failures_low_gt = [r for r in failures if r["on_target_visible"] < 10]
    failures_filter_kills = [r for r in failures
                              if r["on_target_visible"] >= 10 and r["on_target_dynamic"] < 5]
    failures_cluster = [r for r in failures
                         if r["on_target_dynamic"] >= 5 and r["best_cluster_purity"] < 0.5]

    lines += [
        "",
        "## Failure mode summary",
        "",
        f"* Total test videos: {len(rows)}",
        f"* Recoverable (best cluster purity ≥ 0.5): "
        f"**{len(rows) - len(failures)}**",
        f"* Failure mode A — *trajectory grid too sparse for tiny target* "
        f"(< 10 on-target trajectories from CoTracker3 grid): **{len(failures_low_gt)}**",
        f"* Failure mode B — *dynamic filter removes target* "
        f"(≥10 on-target visible but < 5 survive dynamic filter): "
        f"**{len(failures_filter_kills)}**",
        f"* Failure mode C — *clustering merges target with background* "
        f"(target trajectories present but not cleanly clustered): "
        f"**{len(failures_cluster)}**",
        "",
    ]
    if failures_low_gt:
        lines.append("**Failure A (sparse grid)**: " +
                     ", ".join(r["name"] for r in failures_low_gt))
    if failures_filter_kills:
        lines.append("\n**Failure B (filter)**: " +
                     ", ".join(r["name"] for r in failures_filter_kills))
    if failures_cluster:
        lines.append("\n**Failure C (clustering)**: " +
                     ", ".join(r["name"] for r in failures_cluster))

    OUT_MD.write_text("\n".join(lines))
    print(f"\n[written] {OUT_JSON}")
    print(f"[written] {OUT_MD}")


if __name__ == "__main__":
    main()
