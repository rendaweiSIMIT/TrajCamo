"""
Phase 4: Train the Kinematic Signature Encoder (paper §3.2).

A small temporal Transformer that maps each trajectory tau_i to an embedding
z_i in R^128. Trained with per-video trajectory-level contrastive (InfoNCE)
loss: trajectories that project onto the same GT target are pulled together,
trajectories from different targets / background are pushed apart.

Trajectories are labeled by projecting each frame's GT mask onto the
trajectory's visible position at that frame; a trajectory is "target" if a
majority of its visible positions fall inside any GT mask.

Input to the encoder per trajectory (interpolated to a fixed length L=32):
  - raw velocity time series         (L, 2)
  - per-step speed                   (L, 1)
  - per-step direction (sin, cos)    (L, 2)

Output: 128-dim L2-normalized embedding.

After training, the encoder weights are saved to:
  /root/autodl-tmp/VOSdataset/_signature_encoder.pt

Usage:
  python phase4_train_signature_encoder.py [--epochs 15 --batch_videos 4]
"""
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
# Data preparation
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
    """For each trajectory: 1 = "on target" (majority of visible positions fall in any GT),
       0 = "background". Returns (N,) int."""
    tracks = info["tracks"]; vis = info["visibility"]
    T, N, _ = tracks.shape
    target_h, target_w = info["target_h"], info["target_w"]
    new_h, new_w = info["new_h"], info["new_w"]

    inside_count = np.zeros(N, dtype=np.int32)
    seen_count = np.zeros(N, dtype=np.int32)

    name_to_idx = {n: i for i, n in enumerate(info["frame_names"])}
    for gp in sorted(gt_dir.glob("*.png")):
        stem = gp.stem
        if stem not in name_to_idx:
            continue
        t = name_to_idx[stem]
        m = np.array(Image.open(gp))
        if m.ndim == 3:
            m = m[..., 0]
        m_resized = cv2.resize((m > 0).astype(np.uint8), (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        # Pad to (target_h, target_w)
        mp = np.zeros((target_h, target_w), dtype=np.uint8)
        mp[:new_h, :new_w] = m_resized
        # Project trajectory positions at frame t
        pos = tracks[t]
        xs = pos[:, 0].round().astype(int).clip(0, target_w - 1)
        ys = pos[:, 1].round().astype(int).clip(0, target_h - 1)
        v = vis[t]
        seen_count += v.astype(np.int32)
        inside_count += (v & (mp[ys, xs] > 0)).astype(np.int32)

    labels = np.zeros(N, dtype=np.int32)
    enough_obs = seen_count >= 3
    is_target = enough_obs & (inside_count >= 0.5 * np.maximum(seen_count, 1))
    labels[is_target] = 1
    return labels


def resample_to_L(arr: np.ndarray, L: int) -> np.ndarray:
    """Linear-interpolate a (T, ...) array along axis 0 to length L."""
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
    """Returns (N, L, D) per-trajectory input tensor.
    D = 2 (velocity) + 1 (speed) + 2 (sin/cos direction) = 5
    """
    T = tracks.shape[0]
    v = np.diff(tracks, axis=0)                  # (T-1, N, 2)
    s = np.linalg.norm(v, axis=2, keepdims=True) # (T-1, N, 1)
    eps = 1e-6
    sin_ = v[..., [1]] / (s + eps)
    cos_ = v[..., [0]] / (s + eps)

    feats = np.concatenate([v, s, sin_, cos_], axis=2)  # (T-1, N, 5)
    feats = feats.transpose(1, 0, 2)                    # (N, T-1, 5)

    # Resample each trajectory's time series to length L
    out = np.empty((feats.shape[0], L, feats.shape[2]), dtype=np.float32)
    for i in range(feats.shape[0]):
        out[i] = resample_to_L(feats[i], L)
    return out


# ---------------------------------------------------------------------------
# Encoder model
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

    def forward(self, x):  # x: (B, L, D)
        z = self.input_proj(x) + self.pos_embed
        cls = self.cls_token.expand(z.size(0), 1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.encoder(z)
        return F.normalize(self.out_proj(z[:, 0]), dim=-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def info_nce_per_video(z_subset: torch.Tensor, labels: torch.Tensor,
                       temperature: float = 0.1) -> torch.Tensor:
    """SupCon-style loss. For each anchor, treat same-label trajectories
    as positives. Skip if there are no positives or no negatives."""
    z = F.normalize(z_subset, dim=-1)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(-1e4)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    same.fill_diagonal_(False)

    losses = []
    for i in range(z.size(0)):
        pos_mask = same[i]
        if pos_mask.sum() == 0:
            continue
        log_prob_i = sim[i] - torch.logsumexp(sim[i], dim=0)
        loss_i = -(log_prob_i * pos_mask.float()).sum() / pos_mask.sum()
        losses.append(loss_i)
    if not losses:
        return z.sum() * 0.0  # no-op grad
    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_traj_cache/TrainDataset_per_sq"))
    ap.add_argument("--gt_root", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/TrainDataset_per_sq"))
    ap.add_argument("--out", type=Path,
                    default=Path("/root/autodl-tmp/VOSdataset/_signature_encoder.pt"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_videos", type=int, default=4)
    ap.add_argument("--samples_per_video", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--L", type=int, default=32)
    args = ap.parse_args()

    # ------- 1. Pre-build everything in memory -------
    npz_files = sorted(args.cache.glob("*.npz"))
    print(f"[init] loading & labeling {len(npz_files)} train videos ...", flush=True)
    train_videos = []
    t_prep = time.time()
    for p in npz_files:
        info = load_npz(p)
        gt_dir = args.gt_root / p.stem / "GT"
        if not gt_dir.exists():
            continue
        labels = label_trajectories(info, gt_dir)
        feats = build_trajectory_inputs(info["tracks"], L=args.L)
        keep = info["visibility"].mean(axis=0) >= 0.2
        feats = feats[keep]
        labels = labels[keep]
        if (labels == 1).sum() < 3 or (labels == 0).sum() < 3:
            continue
        train_videos.append(dict(
            name=p.stem, feats=feats, labels=labels,
            n_target=int((labels == 1).sum()),
            n_bg=int((labels == 0).sum()),
        ))
    print(f"[init] kept {len(train_videos)} usable videos "
          f"(both target & bg trajectories) — prep time {time.time()-t_prep:.1f}s",
          flush=True)
    if not train_videos:
        print("[error] no usable training videos", flush=True)
        return

    # ------- 2. Build model + optimizer -------
    device = torch.device("cuda")
    model = KinematicSignatureEncoder(input_dim=5, hidden_dim=128,
                                       n_layers=4, n_heads=4, L=args.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"[init] encoder params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    # ------- 3. Train loop -------
    print(f"[train] {args.epochs} epochs, {len(train_videos)} videos, "
          f"batch_videos={args.batch_videos}, samples/video={args.samples_per_video}",
          flush=True)
    rng = np.random.RandomState(0)

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_videos))
        epoch_loss = 0.0
        n_steps = 0
        for vi in range(0, len(order), args.batch_videos):
            batch_idxs = order[vi : vi + args.batch_videos]
            opt.zero_grad()
            losses = []
            for ti in batch_idxs:
                tv = train_videos[ti]
                pos_idx = np.where(tv["labels"] == 1)[0]
                bg_idx  = np.where(tv["labels"] == 0)[0]
                # Balanced sample: half target, half background
                k = args.samples_per_video // 2
                k = min(k, len(pos_idx), len(bg_idx))
                if k < 2:
                    continue
                pi = rng.choice(pos_idx, size=k, replace=len(pos_idx) < k)
                bi = rng.choice(bg_idx,  size=k, replace=len(bg_idx)  < k)
                idxs = np.concatenate([pi, bi])
                x = torch.from_numpy(tv["feats"][idxs]).float().to(device)
                y = torch.from_numpy(tv["labels"][idxs]).long().to(device)
                z = model(x)
                losses.append(info_nce_per_video(z, y))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1
        avg = epoch_loss / max(1, n_steps)
        print(f"  epoch {epoch+1:>2}/{args.epochs}  steps={n_steps:>3}  loss={avg:.4f}",
              flush=True)

    # ------- 4. Save -------
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"input_dim": 5, "hidden_dim": 128, "n_layers": 4,
                   "n_heads": 4, "L": args.L},
    }, args.out)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
