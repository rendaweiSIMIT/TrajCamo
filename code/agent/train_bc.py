"""
Stage A.4: Behavior-cloning (BC) training of the TrajCamo agent.

Loads the oracle index from `agent_outputs/oracle/index.jsonl`, builds a
HuggingFace `Dataset` of (state_images, history_text, oracle_action_text)
samples, and fine-tunes InternVL3-{2B,8B} with a rank-16 LoRA adapter using
the auto-regressive next-token-prediction loss on the action tokens only
(history + question are masked out via the `labels` field).

Designed for single-GPU bf16 training on RTX PRO 6000 Blackwell 96GB.
For each sample we feed up to 3 images (cluster overview, thumbnails,
optional mask overlay) — chosen to fit InternVL3's standard multi-image
chat format.

Usage:
    python train_bc.py --model /root/autodl-tmp/models/InternVL3-2B  # smoke
    python train_bc.py --model /root/autodl-tmp/models/InternVL3-8B  # real
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from actions import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Image preprocessing (matches InternVL3's `.chat` API expectations)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(pil_image: Image.Image, image_size: int = 448) -> torch.Tensor:
    """Returns (1, 3, image_size, image_size) bf16 tensor."""
    im = pil_image.convert("RGB").resize((image_size, image_size))
    arr = np.array(im).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


# ---------------------------------------------------------------------------
# Dataset: load oracle (state, action) pairs
# ---------------------------------------------------------------------------
def build_user_text(step: int, n_clusters: int, T: int, history: str,
                    has_mask: bool) -> str:
    """Replicates the user-message template used in agent.py / infer.py."""
    if step == 0 or not has_mask:
        return (
            f"This video has {T} frames. We have computed {n_clusters} candidate "
            f"trajectory clusters (Image-1). Image-2 shows sampled frames. "
            f"No mask predicted yet. Pick the cluster that is the camouflaged "
            f"animal.\n"
            f"Previous actions:\n{history}\n"
            f"Output ONE action."
        )
    else:
        return (
            f"This video has {T} frames, {n_clusters} candidate clusters "
            f"(Image-1). Image-2 shows sampled frames. Image-3 shows the "
            f"current predicted mask overlaid on those frames in red.\n"
            f"Previous actions:\n{history}\n"
            f"You may add positive/negative points to fix obvious errors, or "
            f"TERMINATE if the mask looks correct. Output ONE action."
        )


class OracleBCDataset(Dataset):
    """Each item is a single (video, step) sample. Holds raw entry metadata;
    the collator does tokenization and image preprocessing.
    """
    def __init__(self, index_path: Path, oracle_root: Path,
                 n_clusters_default: int = 8):
        with open(index_path) as f:
            self.entries = [json.loads(ln) for ln in f if ln.strip()]
        self.oracle_root = oracle_root
        self.n_clusters_default = n_clusters_default

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        return dict(
            video=e["video"],
            step=int(e["step"]),
            cluster_overview_path=str(self.oracle_root / e["cluster_overview"]),
            thumbnails_path=str(self.oracle_root / e["thumbnails"]),
            mask_overlay_path=(str(self.oracle_root / e["mask_overlay"])
                                if e.get("mask_overlay") else None),
            history=e.get("history", "(none)"),
            action_text=e["action"],
            meta=e.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Tokenization + label masking
# ---------------------------------------------------------------------------
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class BCCollator:
    """Builds per-batch tensors for InternVL3 SFT on the action text.

    For each sample we produce:
      pixel_values:        (num_patches_per_sample_sum, 3, H, W)
      num_patches_list:    list of int per-sample
      input_ids / labels:  (B, L) padded with pad_token_id (labels=-100 on
                            prompt tokens, real ids on action tokens).
    """

    def __init__(self, tokenizer, model, image_size: int = 448,
                 max_seq_len: int = 4096):
        self.tok = tokenizer
        self.model = model
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        # Number of <IMG_CONTEXT> tokens per image — InternVL3 uses
        # (image_size // patch_size) ** 2 / (downsample_ratio ** 2)
        cfg = model.config
        ds = float(getattr(cfg, "downsample_ratio", 0.5))
        patch_size = int(getattr(getattr(cfg, "vision_config", cfg), "patch_size", 14))
        self.num_image_tokens = int((image_size // patch_size) ** 2 * (ds ** 2))
        # Reserve image-context token id for label masking
        self.img_context_token_id = self.tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def _format_user_prompt(self, sample: dict, n_clusters: int, T: int) -> str:
        has_mask = sample["mask_overlay_path"] is not None
        return build_user_text(
            sample["step"], n_clusters=n_clusters, T=T,
            history=sample["history"], has_mask=has_mask,
        )

    def _build_image_tokens(self, n_images: int) -> str:
        """Produces 'Image-1: <img><IMG_CONTEXT>...<IMG_CONTEXT></img>\\nImage-2: ...'"""
        parts = []
        for i in range(n_images):
            ctx = IMG_CONTEXT_TOKEN * self.num_image_tokens
            parts.append(f"Image-{i+1}: {IMG_START_TOKEN}{ctx}{IMG_END_TOKEN}\n")
        return "".join(parts)

    def __call__(self, samples: List[dict]) -> dict:
        all_pixel = []
        num_patches_list = []
        input_ids_list = []
        labels_list = []

        for s in samples:
            # Load images
            images = [Image.open(s["cluster_overview_path"]),
                      Image.open(s["thumbnails_path"])]
            if s["mask_overlay_path"] is not None and Path(s["mask_overlay_path"]).exists():
                images.append(Image.open(s["mask_overlay_path"]))
            n_imgs = len(images)
            for im in images:
                all_pixel.append(preprocess_image(im, self.image_size))
                num_patches_list.append(1)

            # Build prompt
            # InternVL3 uses internlm-style chat template; we approximate by
            # concatenating system + user + <|im_end|>... etc, mimicking the
            # template used by its .chat() method.
            n_clusters = int(s["meta"].get("cluster_oracle_ious", {}).keys().__len__()
                              or 8)
            T = int(s["meta"].get("n_frames", 0)) or 30  # fallback
            img_block = self._build_image_tokens(n_imgs)
            user_text = self._format_user_prompt(s, n_clusters=n_clusters, T=T)
            full_user = img_block + user_text

            prompt_text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{full_user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            action_text = s["action_text"]
            assistant_end = "<|im_end|>"

            prompt_ids = self.tok(prompt_text, add_special_tokens=False,
                                   return_tensors=None)["input_ids"]
            action_ids = self.tok(action_text + assistant_end,
                                   add_special_tokens=False,
                                   return_tensors=None)["input_ids"]

            input_ids = prompt_ids + action_ids
            # Mask prompt + image-context tokens with -100
            labels = [-100] * len(prompt_ids) + list(action_ids)
            # Truncate from left if too long
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[-self.max_seq_len:]
                labels = labels[-self.max_seq_len:]
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # Pad input_ids / labels to longest in batch
        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        max_len = max(len(x) for x in input_ids_list)
        B = len(samples)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
            L = len(ids)
            input_ids[i, :L] = ids
            labels[i, :L] = lbl
            attention_mask[i, :L] = 1

        pixel_values = torch.cat(all_pixel, dim=0)
        return dict(
            pixel_values=pixel_values.to(torch.bfloat16),
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            num_patches_list=num_patches_list,
        )


# ---------------------------------------------------------------------------
# Custom training loop (lightweight, no Trainer to keep multimodal in-control)
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda")

    print(f"[init] loading tokenizer + model from {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.gradient_checkpointing_enable()

    # Wrap LM (language_model) + vision tower with LoRA on attention/MLP
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2,
        target_modules=target_modules,
        task_type="CAUSAL_LM", lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # img_context_token_id is used by InternVL3 during forward to swap in image embeddings
    img_ctx_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    if hasattr(model, "base_model"):
        bm = model.base_model.model
    else:
        bm = model
    if hasattr(bm, "img_context_token_id"):
        bm.img_context_token_id = img_ctx_id

    # Dataset
    print(f"[data] loading oracle index from {args.oracle_index}", flush=True)
    ds = OracleBCDataset(Path(args.oracle_index),
                          oracle_root=Path(args.oracle_index).parent.parent)
    print(f"[data] {len(ds)} (video, step) samples", flush=True)
    collator = BCCollator(tok, bm, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=0, drop_last=True)
    if len(loader) == 0:
        print(f"[error] dataloader empty (need at least batch_size samples)", flush=True)
        return

    # Optimizer + scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01,
                             betas=(0.9, 0.95))
    total_steps = args.epochs * max(1, len(loader))
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=min(200, total_steps // 20),
        num_training_steps=total_steps,
    )

    print(f"[train] {args.epochs} epochs × {len(loader)} steps/epoch "
          f"= {total_steps} total", flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    flog = open(log_path, "w")

    global_step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps_epoch = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            # Forward through InternVL3 (custom multimodal forward)
            # InternVL3's forward signature: (pixel_values, input_ids, attention_mask, image_flags=..., labels)
            # We assemble image_flags=[1]*num_images, replicated to match
            n_imgs_batch = len(batch["num_patches_list"])
            image_flags = torch.ones(n_imgs_batch, dtype=torch.long, device=device)
            try:
                out = bm(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    labels=labels,
                    return_dict=True,
                )
                loss = out.loss
            except Exception as e:
                print(f"  [skip] forward error: {e}", flush=True)
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()

            epoch_loss += float(loss.item())
            n_steps_epoch += 1
            global_step += 1
            if global_step % 5 == 0 or global_step <= 3:
                elapsed = time.time() - t_start
                rec = dict(step=global_step, epoch=epoch+1,
                            loss=float(loss.item()),
                            lr=float(sched.get_last_lr()[0]),
                            elapsed_s=round(elapsed, 1))
                print(f"  step {global_step:>5}/{total_steps}  "
                      f"loss={loss.item():.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  "
                      f"({elapsed:.0f}s)", flush=True)
                flog.write(json.dumps(rec) + "\n"); flog.flush()

        avg = epoch_loss / max(1, n_steps_epoch)
        print(f"[epoch {epoch+1}/{args.epochs}] avg loss = {avg:.4f}  "
              f"({n_steps_epoch} steps)", flush=True)

    # Save final LoRA adapter
    model.save_pretrained(str(out_dir / "lora_final"))
    tok.save_pretrained(str(out_dir / "lora_final"))
    print(f"\n[saved] LoRA adapter at {out_dir / 'lora_final'}", flush=True)
    flog.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/InternVL3-2B",
                    help="path to InternVL3 base model dir")
    ap.add_argument("--oracle_index", type=str,
                    default="/root/autodl-tmp/VOScode/agent_outputs/oracle/index.jsonl")
    ap.add_argument("--out", type=str,
                    default="/root/autodl-tmp/VOScode/agent_outputs/bc_ckpt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=448)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
