#!/bin/bash
# Master driver: run all non-MLLM ablations sequentially.
# Each individual run takes ~3 min on a single 4090.
# Total: ~70-80 min compute.

set -e
cd /root/autodl-tmp/VOScode
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3

LOG_DIR=/root/autodl-tmp/VOScode/main_outputs/_logs
mkdir -p "$LOG_DIR"

# Filter noisy SAM3 / TIMM / matplotlib / coTracker3 startup chatter.
FILTER='freqs_cis|vision_backbone\.trunk|trunk\.blocks|patch_embed|frame loading|propagate in video|trunk\.pos_embed|Missing keys|Unexpected|overflow encountered|^\[0m'

run_one () {
    local tag="$1"; shift
    echo ""
    echo "================================================"
    echo "  $tag    args: $*"
    echo "================================================"
    python -u ablation_runner.py --tag "$tag" "$@" 2>&1 \
        | grep -vE "$FILTER" \
        | tee -a "$LOG_DIR/ablations_all.log"
}

# === Ablation A: Feature type ===
# All other knobs default (K=8, n_prompt=12, min_move=4, frames=0)
for feat in position raw_speed raw_velocity coherence; do
    run_one "feat_${feat}" --feature "$feat"
done

# === Ablation B: K (# clusters) ===
for K in 4 6 12 16 24; do
    run_one "K_${K}" --K "$K"
done

# === Ablation C: # prompt points per frame ===
for np in 1 4 8 16 24; do
    run_one "nprompt_${np}" --n_prompt_points "$np"
done

# === Ablation D: dynamic-filter threshold ===
for mp in 0 2 8 16 32; do
    run_one "minmove_${mp}" --min_move_px "$mp"
done

# === Ablation E: multi-frame prompting (long-range memory test) ===
run_one "frames_0"            --prompt_frames "0"
run_one "frames_0_last"       --prompt_frames "0,last"
run_one "frames_0_mid_last"   --prompt_frames "0,mid,last"
run_one "frames_quartiles"    --prompt_frames "0,T/4,T/2,3T/4,last"

echo ""
echo "================ ALL ABLATIONS DONE ================"
ls -1 /root/autodl-tmp/VOScode/ablations/*.json | wc -l
