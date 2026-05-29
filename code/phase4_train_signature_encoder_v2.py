"""
Phase 4 (v2): Train the Kinematic Signature Encoder with cross-video
negatives + hard-negative mining.

What v1 (`phase4_train_signature_encoder.py`) did wrong:
  v1 sampled k target + k background trajectories from the SAME video and
  ran InfoNCE on that mini-pool. Same-video background trajectories all
  share global camera motion / scene context, so the negatives were
  trivially separable and the loss went flat. Final F_w = 0.281 vs raw
  velocity 0.507.

v2 changes:
  1. Cross-video pool: each minibatch concatenates trajectories from
     many train videos. SupCon loss treats target trajectories from any
     video as positives for a target anchor; bg trajectories (from any
     video) are negatives.
  2. Hard-negative mining: after each epoch, embed all trajectories in
     the pool, and pre-select for each anchor the top-K nearest
     differently-labeled trajectories. The next epoch uses those for
     each anchor's negative slot.
  3. Larger batches via on-the-fly pool aggregation, no per-video
     padding.

Saved to /root/autodl-tmp/VOSdataset/_signature_encoder_v2.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Reuse v1's data prep (loaders, label projection, feature build)
# ---------------------------------------------------------------------------
def load_npz(p: Path) -> dict:
    d = np.load(p, allow_pickle=True)
    return dict(
        tracks=d["tracks"], visibility=d["visibility"],
        frame_names=list(d["frame_names"]),
        target_h=int(d["target_h"]), target_w=int(d["target_w"]),
        new_h=int(d["new_h"]),       new_w=int(d["new_w"]),
        orig_h=int(d["orig_h"]),     orig_w=int(d["orig_w"]),
        scale=float(d["scale"]),
    )


def label_trajectories(info: dict, gt_dir: Path) -> np.ndarray:
    tracks = info["tracks"]; vis = info["visibility"]
    T, N, _ = tracks.shape
    target_h, target_w = info["target_h"], info["target_w"]
    new_h, new_w = info["new_h"], info["new_w"]
    inside = np.zeros(N, dtype=np.int32)
    seen = np.zeros(N, dtype=np.int32)
    name_to_idx = {n: i for i, n in enumerate(info["frame_names"])}
    for gp in sorted(gt_dir.glob("*.png")):
        if gp.stem not in name_to_idx:
            continue
        t = name_to_idx[gp.stem]
        m = np.array(Image.open(gp))
        if m.ndim == 3:
            m = m[..., 0]
        mr = cv2.resize((m > 0).astype(np.uint8), (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST)
        mp = np.zeros((target_h, target_w), dtype=np.uint8)
        mp[:new_h, :new_w] = mr
        pos = tracks[t]
        xs = pos[:, 0].round().astype(int).clip(0, target_w - 1)
        ys = pos[:, 1].round().astype(int).clip(0, target_h - 1)
        v = vis[t]
        seen += v.astype(np.int32)
        inside += (v & (mp[ys, xs] > 0)).astype(np.int32)
    labels = np.zeros(N, dtype=np.int32)
    is_target = (seen >= 3) & (inside >= 0.5 * np.maximum(seen, 1))
    labels[is_target] = 1
    return labels


def resample_to_L(arr: np.ndarray, L: int) -> np.ndarray:
    T = arr.shape[0]
    if T == L:
        return arr
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, L)
    flat = arr.reshape(T, -1)
    out = np.empty((L, flat.shape[1]), dtype=arr.dtype)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_new, t_old, flat[:, c])
    return out.reshape(L, *arr.shape[1:])


def build_trajectory_inputs(tracks: np.ndarray, L: int = 32) -> np.ndarray:
    """(N, L, 5) — velocity-xy + speed + sin/cos direction, resampled to L."""
    v = np.diff(tracks, axis=0)
    s = np.linalg.norm(v, axis=2, keepdims=True)
    eps = 1e-6
    sin_ = v[..., [1]] / (s + eps)
    cos_ = v[..., [0]] / (s + eps)
    feats = np.concatenate([v, s, sin_, cos_], axis=2).transpose(1, 0, 2)
    out = np.empty((feats.shape[0], L, feats.shape[2]), dtype=np.float32)
    for i in range(feats.shape[0]):
        out[i] = resample_to_L(feats[i], L)
    return out


# ---------------------------------------------------------------------------
# Same encoder architecture as v1 (so checkpoints are interchangeable)
# ---------------------------------------------------------------------------
class KinematicSignatureEncoder(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128,
                 n_layers: int = 4, n_heads: int = 4, L: int = 32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, L, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        z = self.input_proj(x) + self.pos_embed
        cls = self.cls_token.expand(z.size(0), 1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.encoder(z)
        return F.normalize(self.out_proj(z[:, 0]), dim=-1)


# ---------------------------------------------------------------------------
# Cross-video SupCon with hard-negative slots
# ---------------------------------------------------------------------------
def supcon_loss(z: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """Standard SupCon (Khosla et al. 2020). Anchors+positives+negatives all
    live in one batch; same-label pairs are positives."""
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(-1e4)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    same.fill_diagonal_(False)
    losses = []
    for i in range(z.size(0)):
        pos = same[i]
        if pos.sum() == 0:
            continue
        log_prob = sim[i] - torch.logsumexp(sim[i], dim=0)
        losses.append(-(log_prob * pos.float()).sum() / pos.sum())
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def compute_all_embeddings(model, pool_feats: np.ndarray,
                           batch: int = 4096, device="cuda") -> torch.Tensor:
    """Embed the entire pool. Returns (N, D) on GPU."""
    model.eval()
    outs = []
    N = pool_feats.shape[0]
    for i in range(0, N, batch):
        x = torch.from_numpy(pool_feats[i:i + batch]).float().to(device)
        outs.append(model(x))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def find_hard_negatives(emb: torch.Tensor, labels: np.ndarray,
                        K: int = 16) -> np.ndarray:
    """For each anchor, return indices of the K closest differently-labeled
    trajectories in the pool. Returns (N, K) int."""
    N = emb.size(0)
    lab = torch.from_numpy(labels).to(emb.device)
    sim = emb @ emb.T               # (N, N)
    diff = (lab.unsqueeze(0) != lab.unsqueeze(1))
    sim[~diff] = -1e4               # mask same-label
    sim.fill_diagonal_(-1e4)
    topk = torch.topk(sim, k=K, dim=1).indices    # (N, K)
    return topk.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_traj_cache/TrainDataset_per_sq"))
    ap.add_argument("--gt_root", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/TrainDataset_per_sq"))
    ap.add_argument("--out", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_signature_encoder_v2.pt"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_anchors", type=int, default=128,
                    help="anchors per batch (split half pos / half bg)")
    ap.add_argument("--hard_K", type=int, default=16,
                    help="hard negatives per anchor — refreshed each epoch")
    ap.add_argument("--rebuild_every", type=int, default=2,
                    help="rebuild hard-negative index every N epochs")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--L", type=int, default=32)
    args = ap.parse_args()

    # ---------- 1. Build the cross-video trajectory pool ----------
    npz_files = sorted(args.cache.glob("*.npz"))
    print(f"[init] loading + labeling {len(npz_files)} train videos ...", flush=True)
    t0 = time.time()
    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_video_ids: list[np.ndarray] = []
    n_kept = 0
    for vid, p in enumerate(npz_files):
        info = load_npz(p)
        gt_dir = args.gt_root / p.stem / "GT"
        if not gt_dir.exists():
            continue
        labels = label_trajectories(info, gt_dir)
        feats = build_trajectory_inputs(info["tracks"], L=args.L)
        keep = info["visibility"].mean(axis=0) >= 0.2
        feats = feats[keep]; labels = labels[keep]
        if (labels == 1).sum() < 3 or (labels == 0).sum() < 3:
            continue
        all_feats.append(feats)
        all_labels.append(labels.astype(np.int32))
        all_video_ids.append(np.full(len(labels), vid, dtype=np.int32))
        n_kept += 1
    if not all_feats:
        print("[error] no usable training videos", flush=True)
        return
    pool_feats = np.concatenate(all_feats, axis=0)          # (N_pool, L, 5)
    pool_labels = np.concatenate(all_labels, axis=0)        # (N_pool,)
    pool_videos = np.concatenate(all_video_ids, axis=0)     # (N_pool,)
    N_pool = pool_feats.shape[0]
    n_pos = int((pool_labels == 1).sum())
    n_neg = int((pool_labels == 0).sum())
    print(f"[init] {n_kept}/{len(npz_files)} usable videos -> "
          f"pool: {N_pool} trajectories ({n_pos} target, {n_neg} bg). "
          f"prep {time.time()-t0:.1f}s", flush=True)

    pos_pool = np.where(pool_labels == 1)[0]
    bg_pool = np.where(pool_labels == 0)[0]

    # ---------- 2. Model + optimizer ----------
    device = torch.device("cuda")
    model = KinematicSignatureEncoder(input_dim=5, hidden_dim=128,
                                       n_layers=4, n_heads=4, L=args.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"[init] encoder params: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    # ---------- 3. Training loop with hard-neg refresh ----------
    rng = np.random.RandomState(0)
    hard_neg_idx: np.ndarray = np.zeros((N_pool, args.hard_K), dtype=np.int64)
    steps_per_epoch = max(1, N_pool // args.batch_anchors)
    print(f"[train] {args.epochs} epochs × ~{steps_per_epoch} steps "
          f"(batch_anchors={args.batch_anchors}, hard_K={args.hard_K})",
          flush=True)
    t_start = time.time()

    for epoch in range(args.epochs):
        if epoch % args.rebuild_every == 0:
            emb = compute_all_embeddings(model, pool_feats, device=device)
            hard_neg_idx = find_hard_negatives(emb, pool_labels, K=args.hard_K)
            if epoch == 0:
                # first epoch: hardness is random, just use random negatives
                hard_neg_idx = rng.choice(N_pool, size=(N_pool, args.hard_K))
            del emb
            torch.cuda.empty_cache()

        model.train()
        epoch_loss = 0.0
        n_steps = 0
        order = rng.permutation(N_pool)
        for step in range(steps_per_epoch):
            # Pick balanced anchors: half from pos_pool, half from bg_pool
            k = args.batch_anchors // 2
            anchors_pos = rng.choice(pos_pool, size=min(k, len(pos_pool)),
                                      replace=False)
            anchors_bg = rng.choice(bg_pool, size=min(k, len(bg_pool)),
                                     replace=False)
            anchors = np.concatenate([anchors_pos, anchors_bg])
            # Add hard negatives for each anchor (deduplicated)
            negs = hard_neg_idx[anchors].flatten()
            batch_idx = np.unique(np.concatenate([anchors, negs]))
            x = torch.from_numpy(pool_feats[batch_idx]).float().to(device)
            y = torch.from_numpy(pool_labels[batch_idx]).long().to(device)
            opt.zero_grad()
            z = model(x)
            loss = supcon_loss(z, y, temperature=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1
        sched.step()
        avg = epoch_loss / max(1, n_steps)
        elapsed = time.time() - t_start
        print(f"  epoch {epoch+1:>2}/{args.epochs}  steps={n_steps:>3}  "
              f"loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
              f"({elapsed:.0f}s)", flush=True)

    # ---------- 4. Save ----------
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"input_dim": 5, "hidden_dim": 128, "n_layers": 4,
                   "n_heads": 4, "L": args.L},
        "training": {"epochs": args.epochs, "batch_anchors": args.batch_anchors,
                     "hard_K": args.hard_K, "rebuild_every": args.rebuild_every,
                     "pool_size": int(N_pool)},
    }, args.out)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
