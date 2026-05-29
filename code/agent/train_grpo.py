"""
Stage B.3 — GRPO RL fine-tuning of the BC-trained TrajCamo agent.

Starts from the BC LoRA adapter (Stage A.4) and further fine-tunes it with
mask-IoU reward and a group-relative-policy-optimization (GRPO) loss:

    For each train video v:
        Sample G=4 rollouts of the agent (sampling temp=0.7, top_p=0.9).
        Compute reward R_i = mean_F_w(M_final_i, GT) − λ · n_steps_i.
        Compute group baseline:  A_i = (R_i − mean(R)) / (std(R) + eps).
        Loss += −Σ_i A_i · Σ_t logπ(a^i_t | s^i_t).
    Optimizer step per video.

No KL, no PPO clip — keep it simple, low LR, group baseline for variance
reduction. Uses the existing `InternVL3Agent` + `run_sam3_session` plumbing
and just adds sampling + log-prob teacher-forcing.

Usage:
    python train_grpo.py --base /root/autodl-tmp/models/InternVL3-8B \
        --bc_lora /root/autodl-tmp/VOScode/agent_outputs/bc_8b/lora_final \
        --out /root/autodl-tmp/VOScode/agent_outputs/grpo_8b
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from actions import SYSTEM_PROMPT, Action, parse_action, format_history
from agent import (
    PromptStream, cluster_to_prompt_points, cluster_trajectories,
    load_traj_cache, run_sam3_session,
)
from state_builder import (
    render_cluster_overview, render_current_mask_strip, sample_thumbnail_frames,
)
from cod_metrics import f_beta_w


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def preprocess_image(pil_image: Image.Image, image_size: int = 448) -> torch.Tensor:
    im = pil_image.convert("RGB").resize((image_size, image_size))
    arr = np.array(im).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def build_user_text(step: int, n_clusters: int, T: int, history: str,
                    has_mask: bool) -> str:
    if step == 0 or not has_mask:
        return (
            f"This video has {T} frames. We have computed {n_clusters} candidate "
            f"trajectory clusters (Image-1). Image-2 shows sampled frames. "
            f"No mask predicted yet. Pick the cluster that is the camouflaged "
            f"animal.\n"
            f"Previous actions:\n{history}\n"
            f"Output ONE action."
        )
    return (
        f"This video has {T} frames, {n_clusters} candidate clusters "
        f"(Image-1). Image-2 shows sampled frames. Image-3 shows the "
        f"current predicted mask overlaid on those frames in red.\n"
        f"Previous actions:\n{history}\n"
        f"You may add positive/negative points to fix obvious errors, or "
        f"TERMINATE if the mask looks correct. Output ONE action."
    )


@dataclass
class StepRecord:
    """One agent-loop step's data — enough to recompute log P(action|state).

    response_token_ids is the EXACT token sequence the policy sampled
    (including the trailing <|im_end|>), captured directly from the
    generate() output. Storing this instead of re-tokenizing
    decode(response_text) eliminates the BPE round-trip bias that would
    otherwise make GRPO optimize log P(re-tokenized) instead of
    log P(actually-sampled).
    """
    question_text: str            # the user text (without image tokens)
    response_text: str            # decoded text (for logging only)
    response_token_ids: List[int] # actual sampled token ids incl. eos
    n_images: int                 # how many <image> placeholders in this turn
    pixel_values: torch.Tensor    # (n_images, 3, 448, 448) bf16 on GPU
    num_patches_list: List[int]


# ---------------------------------------------------------------------------
# Sampling InternVL3 wrapper that ALSO returns the step-record bundle
# ---------------------------------------------------------------------------
class SamplingInternVL3:
    def __init__(self, model, tokenizer, image_size: int = 448,
                 dtype=torch.bfloat16, device="cuda"):
        self.model = model
        self.tok = tokenizer
        self.image_size = image_size
        self.dtype = dtype
        self.device = device
        # detect base module for img_context_token_id / forward
        if hasattr(model, "base_model"):
            self.bm = model.base_model.model
        else:
            self.bm = model
        cfg = self.bm.config
        ds = float(getattr(cfg, "downsample_ratio", 0.5))
        patch_size = int(getattr(getattr(cfg, "vision_config", cfg), "patch_size", 14))
        self.num_image_tokens = int((image_size // patch_size) ** 2 * (ds ** 2))
        self.img_ctx_id = self.tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        if hasattr(self.bm, "img_context_token_id"):
            self.bm.img_context_token_id = self.img_ctx_id

    def _build_image_block(self, n_images: int) -> str:
        parts = []
        for i in range(n_images):
            ctx = IMG_CONTEXT_TOKEN * self.num_image_tokens
            parts.append(f"Image-{i+1}: {IMG_START_TOKEN}{ctx}{IMG_END_TOKEN}\n")
        return "".join(parts)

    def _build_prompt(self, user_text: str, n_images: int) -> str:
        img_block = self._build_image_block(n_images)
        full_user = img_block + user_text
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{full_user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @torch.no_grad()
    def sample(self, user_text: str, images: List[Image.Image],
               temperature: float = 0.7, top_p: float = 0.9,
               max_new_tokens: int = 48) -> Tuple[str, StepRecord]:
        pv_list = [preprocess_image(im, self.image_size) for im in images]
        pixel_values = torch.cat(pv_list, dim=0).to(self.device, self.dtype)
        n_imgs = len(images)
        num_patches_list = [1] * n_imgs
        prompt_text = self._build_prompt(user_text, n_imgs)
        input_ids = self.tok(prompt_text, add_special_tokens=False,
                             return_tensors="pt")["input_ids"].to(self.device)
        attn = torch.ones_like(input_ids)
        eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or eos_id < 0:
            eos_id = self.tok.eos_token_id
        gen = self.bm.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=temperature > 1e-4,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=self.tok.pad_token_id or eos_id,
        )
        # InternVL3.generate routes through language_model.generate with
        # inputs_embeds (not input_ids), so HuggingFace returns ONLY the newly
        # generated tokens — the prompt prefix is not included. Therefore gen[0]
        # IS the new-token sequence; do NOT slice off input_ids.shape[1].
        new_token_ids = gen[0].tolist()
        # Truncate at first occurrence of EOS, INCLUDING the EOS — we want the
        # model to learn to emit it. Anything after EOS is padding from the
        # generate() call's buffer and must NOT enter the log-prob.
        if eos_id in new_token_ids:
            i_eos = new_token_ids.index(eos_id)
            new_token_ids = new_token_ids[: i_eos + 1]
        response = self.tok.decode(new_token_ids, skip_special_tokens=True).strip()
        rec = StepRecord(
            question_text=user_text, response_text=response,
            response_token_ids=new_token_ids, n_images=n_imgs,
            pixel_values=pixel_values, num_patches_list=num_patches_list,
        )
        return response, rec


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------
def compute_video_reward(masks: Dict[int, np.ndarray],
                         frame_names: List[str],
                         gt_dir: Path,
                         orig_h: int, orig_w: int,
                         n_steps: int,
                         step_penalty: float = 0.01) -> Tuple[float, float]:
    """Returns (reward, raw_F_w). reward = F_w - step_penalty * n_steps."""
    per_frame_fw = []
    for i, fname in enumerate(frame_names):
        gp = gt_dir / f"{fname}.png"
        if not gp.exists():
            continue
        m = masks.get(i)
        if m is None:
            mb = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            mb = m
            if mb.shape != (orig_h, orig_w):
                mb = cv2.resize(mb.astype(np.uint8), (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)
        gt = np.array(Image.open(gp))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = (gt > 0).astype(np.float64)
        pr = (mb > 0).astype(np.float64)
        per_frame_fw.append(f_beta_w(pr, gt))
    if not per_frame_fw:
        return 0.0, 0.0
    fw = float(np.mean(per_frame_fw))
    return fw - step_penalty * n_steps, fw


# ---------------------------------------------------------------------------
# One agent rollout with sampling — adapted from agent.run_agent_on_video
# ---------------------------------------------------------------------------
def rollout_one(
    smpl: SamplingInternVL3, sam3, video_dir: Path, traj_cache_path: Path,
    gt_dir: Path,
    K: int = 8, K_max: int = 5, n_thumbnails: int = 4, n_prompt_points: int = 12,
    temperature: float = 0.7, top_p: float = 0.9,
    step_penalty: float = 0.01,
) -> dict:
    name = video_dir.name
    imgs_dir = video_dir / "Imgs"
    info = load_traj_cache(traj_cache_path)
    tracks_k, vis_k, labels, _ = cluster_trajectories(info, K=K)
    K_actual = int(labels.max()) + 1
    T = int(info["tracks"].shape[0])
    frame_names = info["frame_names"]

    first_frame_path = imgs_dir / f"{frame_names[0]}.jpg"
    first_frame_bgr = cv2.imread(str(first_frame_path))
    cluster_overview = render_cluster_overview(
        first_frame_bgr, tracks_k[0], vis_k[0], labels,
        target_h_w=(info["target_h"], info["target_w"]),
        orig_h_w=(info["orig_h"], info["orig_w"]),
        new_h_w=(info["new_h"], info["new_w"]),
        scale=info["scale"],
    )
    thumbnail_strip, thumbnail_idxs = sample_thumbnail_frames(
        imgs_dir, frame_names, n=n_thumbnails,
    )

    stream = PromptStream()
    current_mask_strip: Optional[Image.Image] = None
    current_masks: Dict[int, np.ndarray] = {}
    history: List[Action] = []
    step_records: List[StepRecord] = []

    for step in range(K_max):
        history_text = format_history(history)
        if step == 0 or current_mask_strip is None:
            user_text = build_user_text(step, K_actual, T, history_text, False)
            images = [cluster_overview, thumbnail_strip]
        else:
            user_text = build_user_text(step, K_actual, T, history_text, True)
            images = [cluster_overview, thumbnail_strip, current_mask_strip]

        try:
            response, rec = smpl.sample(user_text, images,
                                         temperature=temperature, top_p=top_p)
        except Exception as e:
            print(f"  [{name} step {step}] sample error: {e}", flush=True)
            break
        step_records.append(rec)

        try:
            action = parse_action(response)
        except ValueError:
            action = Action(type="TERMINATE", raw=response)
        history.append(action)

        if action.type == "TERMINATE":
            break

        if action.type == "SELECT":
            if action.cluster_idx is None or action.cluster_idx < 0 \
                    or action.cluster_idx >= K_actual:
                action.cluster_idx = 0
            stream.points_per_frame.clear()
            for (f_idx, x, y, lbl) in cluster_to_prompt_points(
                action.cluster_idx, info, tracks_k, vis_k, labels,
                n_prompt=n_prompt_points,
            ):
                stream.add(f_idx, x, y, lbl)
        elif action.type == "ADD_POS":
            f = max(0, min(int(action.frame_idx), T - 1))
            stream.add(f, action.x, action.y, 1)
        elif action.type == "ADD_NEG":
            f = max(0, min(int(action.frame_idx), T - 1))
            stream.add(f, action.x, action.y, 0)

        if stream.total_pos() == 0:
            continue
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                current_masks = run_sam3_session(sam3, imgs_dir, stream, T)
        except Exception as e:
            print(f"  [{name} step {step}] SAM3 error: {e}", flush=True)
            continue

        per_frame_orig = {}
        for fi, m in current_masks.items():
            if m is None:
                continue
            if m.shape != (info["orig_h"], info["orig_w"]):
                m = cv2.resize(m.astype(np.uint8),
                               (info["orig_w"], info["orig_h"]),
                               interpolation=cv2.INTER_NEAREST)
            per_frame_orig[fi] = m
        current_mask_strip = render_current_mask_strip(
            imgs_dir, frame_names, thumbnail_idxs, per_frame_orig,
        )

    reward, raw_fw = compute_video_reward(
        current_masks, frame_names, gt_dir,
        info["orig_h"], info["orig_w"], len(history),
        step_penalty=step_penalty,
    )
    return dict(
        name=name, n_steps=len(history),
        history=[a.to_text() for a in history],
        step_records=step_records, reward=reward, raw_fw=raw_fw,
    )


# ---------------------------------------------------------------------------
# Teacher-forcing log-prob computation (single + batched)
# ---------------------------------------------------------------------------
def _tokenize_record(smpl, rec: StepRecord):
    """Returns (full_ids:list[int], prompt_len:int, response_ids:list[int]).

    response_ids comes directly from rec.response_token_ids — the literal
    sampled tokens — so log P(response_ids) is genuinely the policy's log
    probability of the action it took, not a re-tokenized approximation.
    """
    prompt_text = smpl._build_prompt(rec.question_text, rec.n_images)
    prompt_ids = smpl.tok(prompt_text, add_special_tokens=False,
                          return_tensors=None)["input_ids"]
    response_ids = list(rec.response_token_ids)
    return list(prompt_ids), list(prompt_ids) + list(response_ids), \
           len(prompt_ids), list(response_ids)


def compute_step_logprob_batch(
    smpl: SamplingInternVL3, records: List[StepRecord],
) -> List[torch.Tensor]:
    """Batched teacher-forcing log-prob computation.

    Packs `len(records)` multimodal samples into ONE InternVL3 forward pass.
    Each record contributes a scalar tensor Σ_t logπ(response_t | ctx).
    The single forward holds all records' activations simultaneously — this
    is what drives steady-state GPU memory up to ~50-70 GB on Blackwell
    (vs ~20 GB when records are processed one-at-a-time).
    """
    if not records:
        return []
    B = len(records)
    device = smpl.device
    pad_id = smpl.tok.pad_token_id or smpl.tok.eos_token_id

    # 1. Tokenize each record
    full_ids_list: List[List[int]] = []
    prompt_lens: List[int] = []
    response_ids_list: List[List[int]] = []
    n_images_list: List[int] = []
    pixel_values_list: List[torch.Tensor] = []
    num_patches_concat: List[int] = []
    for rec in records:
        _, full_ids, prompt_len, response_ids = _tokenize_record(smpl, rec)
        full_ids_list.append(full_ids)
        prompt_lens.append(prompt_len)
        response_ids_list.append(response_ids)
        n_images_list.append(rec.n_images)
        pixel_values_list.append(rec.pixel_values)
        num_patches_concat.extend(rec.num_patches_list)

    max_len = max(len(x) for x in full_ids_list)

    # 2. Pad input_ids + build attention_mask. Pad on the RIGHT — InternVL3 is
    # causal, so right padding leaves the prompt+response positions intact.
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(full_ids_list):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, :L] = 1

    # 3. Concatenate pixel_values + build image_flags. Order matches the
    # left-to-right order of <IMG_CONTEXT> blocks across samples in input_ids
    # (we built input_ids/pixel_values in the same loop, so the ordering
    # matches by construction). image_flags is (total_images, 1) because the
    # model does image_flags.squeeze(-1) and the 1-D path collapses to a
    # 0-D bool when total_images==1.
    pixel_values = torch.cat(pixel_values_list, dim=0)
    total_images = sum(num_patches_concat)
    image_flags = torch.ones(total_images, 1, dtype=torch.long, device=device)

    # 4. Single forward pass
    out = smpl.bm(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attn,
        image_flags=image_flags,
        return_dict=True,
    )
    logits = out.logits  # (B, max_len, V)

    # 5. Per-sample log-prob extraction. Position-(prompt_len-1) predicts
    # response_ids[0]; position-(prompt_len-1+L_resp-1) predicts response_ids[-1].
    log_probs_per_rec: List[torch.Tensor] = []
    for i in range(B):
        L_p = prompt_lens[i]
        L_r = len(response_ids_list[i])
        if L_r == 0:
            # Graph-connected zero so .backward() doesn't crash on all-empty
            # chunks. Multiply a real logit by 0 to keep autograd happy.
            log_probs_per_rec.append((logits[i, 0, 0] * 0.0).float())
            continue
        tgt = logits[i, L_p - 1: L_p - 1 + L_r, :]
        lp = F.log_softmax(tgt.float(), dim=-1)
        resp_t = torch.tensor(response_ids_list[i], dtype=torch.long, device=device)
        gather = lp.gather(1, resp_t.unsqueeze(1)).squeeze(1)
        log_probs_per_rec.append(gather.sum())
    return log_probs_per_rec


def compute_step_logprob(
    smpl: SamplingInternVL3, rec: StepRecord,
) -> torch.Tensor:
    """Backwards-compat single-record wrapper for tests / one-off calls."""
    out = compute_step_logprob_batch(smpl, [rec])
    return out[0]


# ---------------------------------------------------------------------------
# GRPO update over G rollouts
# ---------------------------------------------------------------------------
def grpo_update(
    smpl: SamplingInternVL3, rollouts: List[dict],
    optimizer, max_grad_norm: float = 1.0,
    max_batch: int = 8,
) -> Tuple[float, float, float]:
    """Batched GRPO update.

    Builds a flat list of all step records across all G rollouts (weighted by
    their group-relative advantage), then chunks them into `max_batch`-sized
    batches. Each chunk → one InternVL3 forward → one .backward(). This
    fills the GPU with multimodal activations on every chunk forward, which
    is the whole point of using the 96 GB Blackwell.

    Gradients accumulate naturally across chunks (no zero_grad between
    chunks); after all records processed, we step once. This is
    mathematically equivalent to a single-shot loss = Σ w_i · logp_i
    followed by one .backward(), but the per-chunk forward holds only
    max_batch records' activations at once instead of all G·K_max.
    """
    rewards = np.array([r["reward"] for r in rollouts], dtype=np.float64)
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-4
    advs = (rewards - mean_r) / std_r

    n_used = sum(1 for ro, A in zip(rollouts, advs)
                  if abs(A) >= 1e-6 and len(ro["step_records"]) > 0)
    if n_used == 0:
        return 0.0, float(mean_r), float(std_r)

    # Flatten records into a work list of (weight, record).
    work: List[Tuple[float, StepRecord]] = []
    for ro, A in zip(rollouts, advs):
        if abs(A) < 1e-6 or len(ro["step_records"]) == 0:
            continue
        n_t = max(1, len(ro["step_records"]))
        w = -float(A) / (n_t * n_used)
        for rec in ro["step_records"]:
            work.append((w, rec))

    optimizer.zero_grad()
    total_loss_val = 0.0
    for start in range(0, len(work), max_batch):
        chunk = work[start:start + max_batch]
        weights = [w for w, _ in chunk]
        records = [r for _, r in chunk]
        logprobs = compute_step_logprob_batch(smpl, records)
        # Weighted sum -> scalar -> backward; chunk graph is freed afterwards.
        chunk_loss = sum(w * lp for w, lp in zip(weights, logprobs))
        chunk_loss.backward()
        total_loss_val += float(chunk_loss.detach().item())
        del chunk_loss, logprobs

    trainable = [p for p in smpl.model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()
    return total_loss_val, float(mean_r), float(std_r)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_sam3():
    from sam3.model_builder import build_sam3_video_model
    m = build_sam3_video_model(
        checkpoint_path="/root/autodl-tmp/sam3_base_weights/sam3.pt",
        load_from_HF=False,
    )
    p = m.tracker
    p.backbone = m.detector.backbone
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/InternVL3-8B")
    ap.add_argument("--bc_lora",
                    default="/root/autodl-tmp/VOScode/agent_outputs/bc_8b/lora_final")
    ap.add_argument("--out",
                    default="/root/autodl-tmp/VOScode/agent_outputs/grpo_8b")
    ap.add_argument("--train_root",
                    default="/root/autodl-tmp/VOSdataset/TrainDataset_per_sq")
    ap.add_argument("--traj_cache",
                    default="/root/autodl-tmp/VOSdataset/_traj_cache/TrainDataset_per_sq")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rollouts_per_video", type=int, default=8,
                    help="G — group size for GRPO. 8 gives a better baseline "
                         "and uses the GPU we paid for.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--reward_lambda", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--K_max", type=int, default=5)
    ap.add_argument("--max_videos_per_epoch", type=int, default=0,
                    help="0 means use all train videos")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="re-enable gradient checkpointing (slower, less memory)")
    ap.add_argument("--max_batch", type=int, default=8,
                    help="records batched per backward pass. Higher → more "
                         "GPU memory used and fewer .backward() calls.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    flog = open(log_path, "w")

    print(f"[init] base = {args.base}", flush=True)
    print(f"[init] bc_lora = {args.bc_lora}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True,
                                         use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModel.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(base_model, args.bc_lora, is_trainable=True)
    # NOTE: keep gradient checkpointing OFF — we have 96 GB to spend and the
    # speedup matters more than memory savings on a Blackwell.
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable params: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.2f}%)", flush=True)

    smpl = SamplingInternVL3(model, tok)
    print(f"[init] loading SAM 3", flush=True)
    sam3 = build_sam3()

    train_root = Path(args.train_root)
    traj_root = Path(args.traj_cache)
    all_videos = sorted([d for d in train_root.iterdir() if d.is_dir()])
    valid = [v for v in all_videos if (traj_root / f"{v.name}.npz").exists()]
    print(f"[data] {len(valid)} train videos with trajectory cache", flush=True)
    if args.max_videos_per_epoch > 0:
        valid = valid[:args.max_videos_per_epoch]
        print(f"[data] limiting to {len(valid)} videos/epoch", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01,
                             betas=(0.9, 0.95))
    total_steps = args.epochs * len(valid)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=min(20, total_steps // 20),
        num_training_steps=total_steps,
    )

    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    global_step = 0
    best_avg_fw = -1.0

    for epoch in range(args.epochs):
        order = rng.permutation(len(valid))
        epoch_fw = []
        epoch_loss = []
        for vi in order:
            vd = valid[int(vi)]
            cache = traj_root / f"{vd.name}.npz"
            gt_dir = vd / "GT"
            if not gt_dir.exists():
                continue
            t_v0 = time.time()
            rollouts = []
            model.eval()
            for g in range(args.rollouts_per_video):
                try:
                    ro = rollout_one(
                        smpl, sam3, vd, cache, gt_dir,
                        K_max=args.K_max,
                        temperature=args.temperature, top_p=args.top_p,
                        step_penalty=args.reward_lambda,
                    )
                except Exception as e:
                    print(f"  [{vd.name} g={g}] rollout error: {e}", flush=True)
                    continue
                rollouts.append(ro)
            if len(rollouts) < 2:
                print(f"  [{vd.name}] only {len(rollouts)} rollouts, skip", flush=True)
                continue
            t_rollout = time.time() - t_v0
            fws = [r["raw_fw"] for r in rollouts]
            rewards = [r["reward"] for r in rollouts]

            model.train()
            t_u0 = time.time()
            loss, mean_r, std_r = grpo_update(smpl, rollouts, opt,
                                                max_batch=args.max_batch)
            sched.step()
            t_update = time.time() - t_u0
            global_step += 1
            epoch_fw.append(float(np.mean(fws)))
            epoch_loss.append(loss)

            elapsed = time.time() - t_start
            rec = dict(
                step=global_step, epoch=epoch + 1, video=vd.name,
                loss=loss, mean_reward=mean_r, std_reward=std_r,
                fw_mean=float(np.mean(fws)), fw_max=float(np.max(fws)),
                fw_min=float(np.min(fws)),
                n_rollouts=len(rollouts),
                t_rollout=round(t_rollout, 1), t_update=round(t_update, 1),
                elapsed=round(elapsed, 1),
                lr=float(sched.get_last_lr()[0]),
            )
            print(f"  step {global_step:>4}/{total_steps} "
                  f"ep{epoch+1} {vd.name:>20}: "
                  f"F_w mean={np.mean(fws):.3f} [{np.min(fws):.3f},{np.max(fws):.3f}] "
                  f"loss={loss:+.3f} "
                  f"({t_rollout:.0f}s+{t_update:.0f}s)", flush=True)
            flog.write(json.dumps(rec) + "\n")
            flog.flush()

            # release rollout pixel_values to free memory
            for ro in rollouts:
                for rec_s in ro["step_records"]:
                    del rec_s.pixel_values
            torch.cuda.empty_cache()

        avg = float(np.mean(epoch_fw)) if epoch_fw else 0.0
        avg_loss = float(np.mean(epoch_loss)) if epoch_loss else 0.0
        print(f"\n[epoch {epoch+1}/{args.epochs}] avg F_w = {avg:.3f}  "
              f"avg loss = {avg_loss:+.3f}", flush=True)
        # Save checkpoint
        ckpt_dir = out / f"lora_epoch{epoch+1}"
        model.save_pretrained(str(ckpt_dir))
        tok.save_pretrained(str(ckpt_dir))
        print(f"[saved] {ckpt_dir}", flush=True)
        if avg > best_avg_fw:
            best_avg_fw = avg
            best_dir = out / "lora_best"
            model.save_pretrained(str(best_dir))
            tok.save_pretrained(str(best_dir))
            print(f"[saved] new best ({avg:.3f}) → {best_dir}", flush=True)

    flog.close()
    final_dir = out / "lora_final"
    model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"\n[done] final adapter → {final_dir}", flush=True)


if __name__ == "__main__":
    main()
