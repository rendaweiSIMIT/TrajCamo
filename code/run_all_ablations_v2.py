"""
Robust ablation driver. Runs each ablation as a separate Python subprocess,
captures stdout+stderr to a per-ablation log file, and continues to the
next one on any failure. Skips ablations whose result JSON already exists.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/root/autodl-tmp/VOScode")
OUT = ROOT / "ablations"
LOGS = ROOT / "main_outputs" / "_logs" / "ablations_v2"
LOGS.mkdir(parents=True, exist_ok=True)

# Filter to suppress noisy SAM3 / TIMM startup chatter
FILTER_PATTERNS = [
    "freqs_cis", "vision_backbone.trunk", "trunk.blocks", "patch_embed",
    "frame loading", "propagate in video", "trunk.pos_embed",
    "Missing keys", "Unexpected", "overflow encountered",
]


def run_one(tag: str, extra_args: list) -> bool:
    out_json = OUT / f"{tag}.json"
    if out_json.exists():
        print(f"  [skip-cached] {tag}", flush=True)
        return True
    log_file = LOGS / f"{tag}.log"
    cmd = ["python", "-u", str(ROOT / "ablation_runner.py"),
           "--tag", tag] + list(extra_args)
    print(f"\n{'='*60}\n  RUN {tag}\n  cmd: {' '.join(cmd)}\n{'='*60}", flush=True)
    t0 = time.time()
    try:
        with open(log_file, "wb") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=1800)  # 30-min per ablation hard cap
        ok = (proc.returncode == 0) and out_json.exists()
        elapsed = time.time() - t0
        # Print short summary from the log: oracle / heuristic aggregate lines
        with open(log_file) as f:
            lines = [ln for ln in f.readlines()
                     if any(s in ln for s in ("oracle ", "heuristic ", "ERROR", "Traceback"))]
        for ln in lines[-6:]:
            print(f"    {ln.rstrip()}", flush=True)
        print(f"  [{'ok' if ok else 'FAIL'}] {tag}   ({elapsed:.0f}s)", flush=True)
        return ok
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {tag}", flush=True)
        return False
    except Exception as e:
        print(f"  [EXC] {tag}: {e}", flush=True)
        return False


PLAN = []

# Feature ablation
for f in ["position", "raw_speed", "raw_velocity", "coherence"]:
    PLAN.append((f"feat_{f}", ["--feature", f]))

# K sweep
for K in [4, 6, 12, 16, 24]:
    PLAN.append((f"K_{K}", ["--K", str(K)]))

# Prompt count sweep
for n in [1, 4, 8, 16, 24]:
    PLAN.append((f"nprompt_{n}", ["--n_prompt_points", str(n)]))

# Dynamic-filter threshold sweep
for m in [0, 2, 8, 16, 32]:
    PLAN.append((f"minmove_{m}", ["--min_move_px", str(m)]))

# Multi-frame prompting
PLAN += [
    ("frames_0_only",       ["--prompt_frames", "0"]),
    ("frames_0_last",       ["--prompt_frames", "0,last"]),
    ("frames_0_mid_last",   ["--prompt_frames", "0,mid,last"]),
    ("frames_quartiles",    ["--prompt_frames", "0,T/4,T/2,3T/4,last"]),
]


def main():
    print(f"[plan] {len(PLAN)} ablations to run", flush=True)
    t_total = time.time()
    n_ok = 0
    for i, (tag, args) in enumerate(PLAN):
        print(f"\n>>> ({i+1}/{len(PLAN)}) {tag}", flush=True)
        if run_one(tag, args):
            n_ok += 1
    print(f"\n[done] {n_ok}/{len(PLAN)} ablations OK   "
          f"total {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
