"""
Collect all ablation_runner / eval_with_encoder JSON outputs in
/root/autodl-tmp/VOScode/ablations and produce a single Markdown
report grouping rows into:

  A. Feature ablation
  B. K-clusters sweep
  C. n_prompt_points sweep
  D. min_move_px sweep
  E. Multi-frame prompting
  F. Learned signature encoder (if available)

Each row reports the aggregate (oracle + heuristic) over 16 MoCA-Mask
test videos, with explicit deltas against SLT-Net's published numbers.
"""
import json
import re
from pathlib import Path

import numpy as np


ABLATIONS_DIR = Path("/root/autodl-tmp/VOScode/ablations")
OUT_MD = ABLATIONS_DIR / "ABLATIONS_REPORT.md"

SLTNET = {"F_w": 0.357, "MAE": 0.0300, "S_a": 0.656, "E_p": 0.785}


def load_all():
    rows = []
    for p in sorted(ABLATIONS_DIR.glob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
            rows.append((p.stem, d))
        except Exception:
            continue
    return rows


def fmt_row(tag, label, agg):
    if not agg:
        return None
    delta_f = agg["F_w"] - SLTNET["F_w"]
    delta_m = SLTNET["MAE"] - agg["MAE"]      # lower is better → positive means we beat SLT
    delta_s = agg["S_a"] - SLTNET["S_a"]
    delta_e = agg["E_p"] - SLTNET["E_p"]
    deltas = (
        f"{'+' if delta_f>=0 else ''}{delta_f:.3f} / "
        f"{'+' if delta_m>=0 else ''}{delta_m:.3f} / "
        f"{'+' if delta_s>=0 else ''}{delta_s:.3f} / "
        f"{'+' if delta_e>=0 else ''}{delta_e:.3f}"
    )
    return (
        f"| `{tag}` | {label} | {agg['F_w']:.3f} | {agg['MAE']:.4f} | "
        f"{agg['S_a']:.3f} | {agg['E_p']:.3f} | {deltas} |"
    )


def main():
    all_rows = load_all()
    print(f"[init] loaded {len(all_rows)} ablation runs", flush=True)

    sections = {
        "A": ("Feature ablation", [r for r in all_rows if r[0].startswith("feat_")]),
        "B": ("K (# clusters) sweep", [r for r in all_rows if r[0].startswith("K_")]),
        "C": ("n_prompt_points sweep", [r for r in all_rows if r[0].startswith("nprompt_")]),
        "D": ("min_move_px (dynamic-filter threshold) sweep",
              [r for r in all_rows if r[0].startswith("minmove_")]),
        "E": ("Multi-frame prompting", [r for r in all_rows if r[0].startswith("frames_")]),
        "F": ("Learned signature encoder",
              [r for r in all_rows if "learned_encoder" in r[0]]),
    }

    lines = [
        "# TrajCamo Ablation Studies — MoCA-Mask test split (16 videos)",
        "",
        "All ablations use the same Phase-2 pipeline (CoTracker3 trajectories + "
        "K-means + SAM 3 propagation), changing one knob at a time from the base "
        "configuration:  `feature=raw_velocity, K=8, n_prompt_points=12, "
        "min_move_px=4.0, prompt_frames=0`.",
        "",
        "Each cell reports the 16-video mean for that metric. The `Δ vs SLT` "
        "column shows `(F_w MAE S_α E_φ)` deltas vs SLT-Net's published "
        "numbers (`F_w=0.357 / MAE=0.030 / S_α=0.656 / E_φ=0.785`); positive "
        "delta means we beat SLT-Net (for MAE delta is sign-flipped so "
        "positive is still better).",
        "",
        f"**SLT-Net baseline (CVPR 2022):**  "
        f"F_w_β = {SLTNET['F_w']}, MAE = {SLTNET['MAE']}, "
        f"S_α = {SLTNET['S_a']}, E_φ = {SLTNET['E_p']}",
        "",
    ]

    def emit_section(letter: str, title: str, runs: list):
        if not runs:
            return
        lines.append(f"## {letter}. {title}")
        lines.append("")
        lines.append(
            "| Tag | Setting | F_w_β↑ | MAE↓ | S_α↑ | E_φ↑ | Δ vs SLT (F/M/S/E) |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for tag, d in runs:
            cfg = d.get("config", {})
            label = (
                f"{cfg.get('feature','?')} / K={cfg.get('K','?')} / "
                f"n={cfg.get('n_prompt_points','?')} / "
                f"mv={cfg.get('min_move_px','?')} / "
                f"frm={cfg.get('prompt_frames','0')}"
            )
            for sel in ("oracle", "heuristic"):
                agg = d.get("aggregates", {}).get(sel)
                row = fmt_row(tag, f"{label}  [{sel}]", agg)
                if row:
                    lines.append(row)
        lines.append("")

    for letter in ("A", "B", "C", "D", "E", "F"):
        title, runs = sections[letter]
        emit_section(letter, title, runs)

    # Per-video breakdown for the best oracle config (highest F_w)
    all_oracle = []
    for tag, d in all_rows:
        agg = d.get("aggregates", {}).get("oracle")
        if agg:
            all_oracle.append((agg["F_w"], tag, d))
    if all_oracle:
        all_oracle.sort(reverse=True)
        f_top, tag_top, d_top = all_oracle[0]
        lines.append(f"## Best oracle configuration: `{tag_top}` (F_w_β = {f_top:.3f})")
        lines.append("")
        lines.append("| Video | T | n_traj | F_w_β | MAE | S_α | E_φ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in d_top.get("rows", []):
            o = r.get("oracle", {})
            m = o.get("metrics", {}) if isinstance(o, dict) else {}
            if "error" in m or not m:
                lines.append(f"| {r['name']} | {r.get('n_frames','?')} | "
                             f"{r.get('n_traj_dynamic','?')} | -- | -- | -- | -- |")
            else:
                lines.append(f"| {r['name']} | {r.get('n_frames','?')} | "
                             f"{r.get('n_traj_dynamic','?')} | {m['F_w']:.3f} | "
                             f"{m['MAE']:.4f} | {m['S_a']:.3f} | {m['E_p']:.3f} |")
        lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"[written] {OUT_MD}", flush=True)
    print(f"[wc] {len(lines)} lines, {sum(len(s.splitlines()) for s in sections.values() if isinstance(s, tuple))} sections", flush=True)


if __name__ == "__main__":
    main()
