"""
Smoke test for Approach A: SAM3 init_state reuse across rollouts.

Verifies:
  (1) Mask correctness — masks from fresh init_state are BIT-EXACT equal to
      masks from a reused state after clear_all_points_in_video.
  (2) Speedup — 4 sequential rollouts on a long video should be markedly
      faster on the reused-state path than the fresh-state path.

Run inside conda env `sam3`. No InternVL3 needed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import PromptStream, run_sam3_session


def build_sam3():
    from sam3.model_builder import build_sam3_video_model
    m = build_sam3_video_model(
        checkpoint_path="/root/autodl-tmp/sam3_base_weights/sam3.pt",
        load_from_HF=False,
    )
    p = m.tracker
    p.backbone = m.detector.backbone
    return p


def make_stream(seed: int) -> PromptStream:
    """Build a small deterministic PromptStream (3 positive points at frame 0)."""
    rng = np.random.default_rng(seed)
    s = PromptStream()
    for _ in range(3):
        x = float(rng.uniform(0.3, 0.7))
        y = float(rng.uniform(0.3, 0.7))
        s.add(0, x, y, 1)
    return s


def masks_equal(a: dict, b: dict) -> bool:
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        ma, mb = a.get(k), b.get(k)
        if (ma is None) != (mb is None):
            print(f"  frame {k}: None mismatch (a={ma is None}, b={mb is None})")
            return False
        if ma is None:
            continue
        if ma.shape != mb.shape:
            print(f"  frame {k}: shape mismatch {ma.shape} vs {mb.shape}")
            return False
        if not np.array_equal(ma, mb):
            diff = int(np.sum(ma != mb))
            tot = int(ma.size)
            print(f"  frame {k}: {diff}/{tot} pixels differ")
            return False
    return True


def main():
    # Pick a moderately long video to exercise the JPEG-decode savings.
    test_root = Path("/root/autodl-tmp/VOSdataset/TrainDataset_per_sq")
    # cuttlefish_4 was reported in the existing training log (~50s rollout); we
    # use it because it's a known-good MoCA-Mask video with non-empty masks.
    vd = test_root / "cuttlefish_4"
    if not vd.exists():
        # Fallback: pick the first valid train video
        vd = next(d for d in sorted(test_root.iterdir()) if d.is_dir())
    imgs_dir = vd / "Imgs"
    T = len(list(imgs_dir.glob("*.jpg")))
    print(f"[fixture] video={vd.name}  T={T} frames")

    print(f"[init] loading SAM 3")
    predictor = build_sam3()
    torch.cuda.synchronize()
    print(f"[init] done")

    # ---------- Correctness ----------
    streamA = make_stream(seed=0)
    streamB = make_stream(seed=1)

    print(f"\n[correctness] running OLD path (fresh init_state) on streamA")
    t0 = time.time()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        masks_old = run_sam3_session(predictor, imgs_dir, streamA, T, state=None)
    torch.cuda.synchronize()
    t_old_a = time.time() - t0
    print(f"  done in {t_old_a:.2f}s, non-empty masks: "
          f"{sum(1 for m in masks_old.values() if m is not None and m.sum() > 0)}/{len(masks_old)}")

    print(f"\n[correctness] preloading state ONCE, then streamA reuse")
    t0 = time.time()
    persistent = predictor.init_state(video_path=str(imgs_dir))
    torch.cuda.synchronize()
    t_init = time.time() - t0
    print(f"  init_state {t_init:.2f}s")

    t0 = time.time()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        masks_new = run_sam3_session(predictor, imgs_dir, streamA, T, state=persistent)
    torch.cuda.synchronize()
    t_new_a = time.time() - t0
    print(f"  done in {t_new_a:.2f}s")

    eq_first = masks_equal(masks_old, masks_new)
    print(f"\n[correctness] streamA fresh vs reused: {'PASS bit-exact' if eq_first else 'FAIL'}")

    # Reuse the same state for a DIFFERENT stream
    print(f"\n[correctness] reused state with DIFFERENT streamB")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        masks_b_reused = run_sam3_session(predictor, imgs_dir, streamB, T, state=persistent)

    # Run streamB fresh for comparison
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        masks_b_fresh = run_sam3_session(predictor, imgs_dir, streamB, T, state=None)

    eq_b = masks_equal(masks_b_fresh, masks_b_reused)
    print(f"\n[correctness] streamB fresh vs reused: {'PASS bit-exact' if eq_b else 'FAIL'}")

    # Run streamA AGAIN on the reused state (after streamB) to check for
    # memory-bank carryover.
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        masks_a_again = run_sam3_session(predictor, imgs_dir, streamA, T, state=persistent)
    eq_a_again = masks_equal(masks_old, masks_a_again)
    print(f"[correctness] streamA reused after streamB still matches fresh streamA: "
          f"{'PASS no carryover' if eq_a_again else 'FAIL carryover detected'}")

    # ---------- Speedup ----------
    print(f"\n[speedup] 4 sequential rollouts OLD path (fresh init each)")
    t0 = time.time()
    for g in range(4):
        s = make_stream(seed=g)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            run_sam3_session(predictor, imgs_dir, s, T, state=None)
        torch.cuda.synchronize()
    t_old = time.time() - t0
    print(f"  OLD total: {t_old:.2f}s ({t_old/4:.2f}s/rollout)")

    print(f"\n[speedup] 4 sequential rollouts NEW path (reuse pre-loaded state)")
    t0 = time.time()
    state_reuse = predictor.init_state(video_path=str(imgs_dir))
    torch.cuda.synchronize()
    t_init_only = time.time() - t0
    t0 = time.time()
    for g in range(4):
        s = make_stream(seed=g)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            run_sam3_session(predictor, imgs_dir, s, T, state=state_reuse)
        torch.cuda.synchronize()
    t_new = time.time() - t0
    print(f"  init_state once: {t_init_only:.2f}s, then 4 reused: {t_new:.2f}s "
          f"({t_new/4:.2f}s/rollout); GRAND TOTAL: {t_init_only + t_new:.2f}s")

    speedup = t_old / (t_init_only + t_new)
    print(f"\n[speedup] OLD {t_old:.1f}s vs NEW {t_init_only+t_new:.1f}s → {speedup:.2f}× faster")

    # Verdict
    print("\n" + "=" * 60)
    ok_correct = eq_first and eq_b and eq_a_again
    ok_fast = speedup >= 1.2  # demand at least 20% speedup
    print(f"  CORRECTNESS: {'PASS' if ok_correct else 'FAIL'}")
    print(f"  SPEEDUP:     {'PASS' if ok_fast else 'FAIL'} ({speedup:.2f}× ≥ 1.20×)")
    print(f"  OVERALL:     {'SHIP IT' if (ok_correct and ok_fast) else 'DO NOT SHIP'}")
    print("=" * 60)
    sys.exit(0 if (ok_correct and ok_fast) else 1)


if __name__ == "__main__":
    main()
