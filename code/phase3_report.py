"""
Phase 3: Compile a Markdown report comparing TrajCamo (oracle + heuristic
cluster selection) against SLT-Net's published numbers on MoCA-Mask test.

Usage:
    python phase3_report.py
"""
import json
from pathlib import Path

import numpy as np


OUT = Path("/root/autodl-tmp/VOScode/main_outputs")
RESULTS = OUT / "main_results.json"
REPORT = OUT / "MAIN_REPORT.md"

# SLT-Net (CVPR 2022) published numbers on MoCA-Mask test split.
SLTNET = {"F_w": 0.357, "MAE": 0.0300, "S_a": 0.656, "E_p": 0.785}


def fmt(v, prec=3):
    return f"{v:.{prec}f}" if v is not None else "--"


def main():
    rows = json.load(open(RESULTS))

    # Per-video table for each selection rule
    lines = [
        "# TrajCamo Main Experiment Report (MoCA-Mask test, 16 videos)",
        "",
        "**Pipeline:** CoTracker3 trajectory pre-cache + raw-velocity K-means clustering "
        "(K=8) + centroid-neighborhood prompt sampling + SAM 3 base propagation. "
        "**Agentic MLLM loop is not yet applied in this run** — these numbers "
        "establish the trajectory-grouping pipeline's standalone strength (without "
        "language reasoning).",
        "",
        "Two cluster-selection rules are evaluated:",
        "",
        "* **Oracle**: pick the cluster with highest IoU against the GT union "
        "  (upper bound for what an MLLM agent could achieve on cluster selection alone).",
        "* **Heuristic**: pick the cluster with highest within-cluster consistency "
        "  score (no GT, simulates a no-language deployment).",
        "",
        "## Per-video metrics",
        "",
    ]

    for selection in ("oracle", "heuristic"):
        lines.append(f"### Selection rule: `{selection}`")
        lines.append("")
        lines.append("| Video | T | n_traj | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            d = r.get(selection, {})
            m = d.get("metrics", {}) if isinstance(d, dict) else {}
            if "error" in m or not m:
                lines.append(f"| {r['name']} | {r.get('n_frames','?')} | "
                             f"{r.get('n_trajectories','?')} | -- | -- | -- | -- |")
            else:
                lines.append(f"| {r['name']} | {r['n_frames']} | "
                             f"{r['n_trajectories']} | {fmt(m['F_w'])} | "
                             f"{fmt(m['MAE'], 4)} | {fmt(m['S_a'])} | {fmt(m['E_p'])} |")
        # Aggregate
        vals = {k: [] for k in ("F_w", "MAE", "S_a", "E_p")}
        for r in rows:
            d = r.get(selection, {})
            m = d.get("metrics", {}) if isinstance(d, dict) else {}
            if "error" in m or not m:
                continue
            for k in vals:
                vals[k].append(m[k])
        if all(len(vals[k]) for k in vals):
            means = {k: float(np.mean(v)) for k, v in vals.items()}
            lines.append(f"| **MEAN** | | | **{fmt(means['F_w'])}** | "
                         f"**{fmt(means['MAE'], 4)}** | **{fmt(means['S_a'])}** | "
                         f"**{fmt(means['E_p'])}** |")
        lines.append("")

    # Comparison table
    lines += [
        "## Comparison against SLT-Net (CVPR 2022)",
        "",
        "| Method | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ |",
        "|---|---:|---:|---:|---:|",
        f"| SLT-Net (paper) | {SLTNET['F_w']} | {SLTNET['MAE']} | "
        f"{SLTNET['S_a']} | {SLTNET['E_p']} |",
    ]
    for selection in ("oracle", "heuristic"):
        vals = {k: [] for k in ("F_w", "MAE", "S_a", "E_p")}
        for r in rows:
            d = r.get(selection, {})
            m = d.get("metrics", {}) if isinstance(d, dict) else {}
            if "error" in m or not m:
                continue
            for k in vals:
                vals[k].append(m[k])
        if not all(len(vals[k]) for k in vals):
            continue
        means = {k: float(np.mean(v)) for k, v in vals.items()}
        label = f"**TrajCamo — {selection}**"
        lines.append(
            f"| {label} | {fmt(means['F_w'])} | {fmt(means['MAE'], 4)} | "
            f"{fmt(means['S_a'])} | {fmt(means['E_p'])} |"
        )
        # Win/loss per metric
        gains = {
            "F_w": means["F_w"] - SLTNET["F_w"],
            "MAE": SLTNET["MAE"] - means["MAE"],     # lower MAE is better → gain = SLT − us
            "S_a": means["S_a"] - SLTNET["S_a"],
            "E_p": means["E_p"] - SLTNET["E_p"],
        }
        delta_str = " | ".join([f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in
                                [gains["F_w"], gains["MAE"], gains["S_a"], gains["E_p"]]])
        lines.append(f"| ↳ Δ vs SLT-Net | {delta_str} |")

    lines += [
        "",
        "## Reading the numbers",
        "",
        "* This is the **trajectory-grouping pipeline alone** — no MLLM agent, no",
        "  language input, no learned signature encoder. Just CoTracker3 + K-means",
        "  on raw velocity + centroid-sample to SAM 3.",
        "* The **oracle** row is an upper bound: it tells us the maximum accuracy",
        "  obtainable if the cluster-selection step were perfect.",
        "* The **heuristic** row is the realistic no-language number, comparable",
        "  to SLT-Net (which is also non-language).",
        "* Adding (i) a learned kinematic signature encoder, (ii) the agentic MLLM",
        "  loop with `Add-Pos` / `Add-Neg` corrections, and (iii) language-conditioned",
        "  cluster selection are the remaining sources of accuracy gain in the",
        "  full TrajCamo system.",
        "",
    ]

    REPORT.write_text("\n".join(lines))
    print(REPORT.read_text())
    print(f"\n[written] {REPORT}")


if __name__ == "__main__":
    main()
