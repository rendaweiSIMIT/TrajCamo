#!/bin/bash
# One-shot environment setup for a NEW machine after rsync /root/autodl-tmp/.
#
# Tested on:
#   * RTX 4090 (sm_89, Ada)         → cu126 line works fine
#   * RTX PRO 6000 Blackwell (sm_120) / B100 / B200  → use cu128 line
#   * A100 80GB (sm_80, Ampere)     → either cu126 or cu128 works
#
# Wall-clock: ~10-15 minutes (mostly the pip install steps; weights are
# already on the data disk so nothing has to be re-downloaded).

set -e

ENV_NAME="${1:-sam3}"
echo "[setup] target conda env: $ENV_NAME"

# ---- 1. Make sure miniconda is bootstrapped ----
if ! command -v conda > /dev/null; then
    echo "[error] conda not found. Install miniconda first." >&2
    exit 1
fi
source /root/miniconda3/etc/profile.d/conda.sh

# ---- 2. Create / reuse env ----
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env $ENV_NAME already exists — reusing"
else
    echo "[setup] creating fresh env $ENV_NAME (Python 3.12)"
    conda create -n "$ENV_NAME" python=3.12 -y
fi
conda activate "$ENV_NAME"

# ---- 3. Pick the right PyTorch wheel for the GPU ----
# Detect the GPU compute capability:
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "[setup] detected compute capability sm_${CC}"

if [ "${CC:0:1}" = "1" ] && [ "$CC" -ge 100 ]; then
    # Blackwell (sm_100 = B200, sm_120 = RTX PRO 6000, sm_121 = B100) → cu128 wheel for native support
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    echo "[setup] Blackwell-class GPU → installing PyTorch with CUDA 12.8 wheels"
else
    # Ampere / Hopper / Ada (sm_80/86/89/90) → cu126 wheel is fine
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    echo "[setup] Ampere / Hopper / Ada GPU → installing PyTorch with CUDA 12.6 wheels"
fi

pip install --quiet "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" \
    --index-url "$TORCH_INDEX"

# ---- 4. Install the rest of the pinned dependencies ----
echo "[setup] installing pinned dependencies (~200 packages, takes ~5 min)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install --quiet -r "$SCRIPT_DIR/requirements_sam3_env.txt"

# ---- 5. Install SAM 3 in editable mode (source already on data disk) ----
if [ -d /root/autodl-tmp/sam3 ]; then
    echo "[setup] installing SAM 3 in editable mode"
    pip install --quiet -e /root/autodl-tmp/sam3
else
    echo "[warn] /root/autodl-tmp/sam3 not found — skip SAM3 install" >&2
fi

# ---- 6. Optional: install flash-attn v2.7+ for big speedups on training ----
echo "[setup] installing flash-attn v2.7+ (helps a lot on Blackwell / Hopper)"
pip install --quiet flash-attn>=2.7.0 --no-build-isolation || \
    echo "[warn] flash-attn install failed — not critical, training still works without it"

# ---- 7. Sanity ----
python - <<'PY'
import torch
print("torch         :", torch.__version__)
print("cuda built    :", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0      :", torch.cuda.get_device_name(0))
    print("compute cap   :", torch.cuda.get_device_capability(0))
try:
    import sam3
    print("sam3          : OK")
except ImportError as e:
    print("sam3          : NOT installed (", e, ")")
try:
    import flash_attn
    print("flash_attn    :", flash_attn.__version__)
except ImportError:
    print("flash_attn    : not installed (optional)")
PY

echo ""
echo "[setup] DONE. Activate with:  conda activate $ENV_NAME"
