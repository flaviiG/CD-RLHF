#!/bin/bash
# One-shot environment setup for running CD-RLHF on a vast.ai box with a single
# 32GB-VRAM GPU. Targets modern cards including Blackwell (RTX 5090, sm_120).
#
# Usage on the rented machine, after `git clone`:
#   bash setup_vastai.sh
#
# Works on any CUDA 12.4+ base image. The cu128 PyTorch wheel below is forward-
# compatible with Ampere/Hopper too, so this script is fine for A100/H100 boxes.
# Idempotent: safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 0. Environment fixes for vast.ai boxes.
# ---------------------------------------------------------------------------

# HuggingFace cache → local SSD. /workspace is a network volume that can
# produce "Stale file handle" errors when HF copies model blobs.
export HF_HOME=/root/.cache/huggingface
grep -qxF 'export HF_HOME=/root/.cache/huggingface' ~/.bashrc \
    || echo 'export HF_HOME=/root/.cache/huggingface' >> ~/.bashrc

# ---------------------------------------------------------------------------
# 1. PyTorch with CUDA 12.8 (Blackwell-capable; works on Ampere/Hopper too).
# ---------------------------------------------------------------------------
echo "[1/4] Installing PyTorch (cu128 wheels)..."
pip install --upgrade pip
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# 2. Python deps. requirements_vastai.txt drops the dozens of unused, Python
#    3.12-incompatible transitives from the original requirements.txt, and
#    relaxes the deepspeed pin so pip can pick a Blackwell-capable build.
# ---------------------------------------------------------------------------
echo "[2/4] Installing Python dependencies..."
pip install -r requirements_vastai.txt

# ---------------------------------------------------------------------------
# 3. Editable install of the dschat package.
# ---------------------------------------------------------------------------
echo "[3/4] Installing dschat in editable mode..."
pip install -e applications/DeepSpeed-Chat --no-deps

# ---------------------------------------------------------------------------
# 4. Sanity checks. Fail fast if the GPU isn't usable or dschat doesn't import.
# ---------------------------------------------------------------------------
echo "[4/4] Running sanity checks..."
python - <<'PY'
import sys

import torch
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("FAIL: no CUDA device visible.")

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
bf16 = torch.cuda.is_bf16_supported()
print(f"device=0  {name}  cc={cap[0]}.{cap[1]}  vram={vram_gb:.1f} GB  bf16_supported={bf16}")

# Fail loudly if the torch wheel doesn't have kernels for this GPU. On
# Blackwell, an undersized wheel produces a "sm_120 is not compatible" warning
# rather than an error, so check supported_archs explicitly.
sm = f"sm_{cap[0]}{cap[1]}"
supported = torch.cuda.get_arch_list()
if sm not in supported:
    sys.exit(f"FAIL: torch was built for {supported} but this GPU is {sm}. "
             f"Reinstall a torch wheel that includes {sm}.")

if vram_gb < 30:
    print("WARN: <30GB VRAM. The Qwen2.5-0.5B PPO config may OOM; lower "
          "per_device_*_batch_size in the step3 scripts if you hit OOM.")
if not bf16:
    print("WARN: bf16 not supported. Training scripts default to fp16; "
          "loss-scaling can be unstable on pre-Ampere cards.")

try:
    import dschat.utils.data.data_utils as m
    print(f"dschat OK: {m.__file__}")
except ModuleNotFoundError as e:
    sys.exit(f"FAIL: dschat package not importable.\n{e}")
PY

cat <<EOF

Setup complete. Run training in this order (each script writes to ./models/):

  cd applications/DeepSpeed-Chat/training/step1_supervised_finetuning && \\
    bash training_scripts/tldr/run_qwen2_5_0_5b.sh

  cd applications/DeepSpeed-Chat/training/step2_reward_model_finetuning && \\
    bash training_scripts/tldr/run_qwen2_5_0_5b.sh

  cd applications/DeepSpeed-Chat/training/step3_rlhf_finetuning && \\
    bash training_scripts/tldr/run_qwen2_5_0_5b.sh             # vanilla RLHF
  # or
  cd applications/DeepSpeed-Chat/training/step3_rlhf_finetuning && \\
    bash training_scripts/tldr/run_qwen2_5_0_5b_cdrlhf.sh      # CD-RLHF

If a training stage OOMs, halve --per_device_*_batch_size and double
--gradient_accumulation_steps in the corresponding script (effective batch
stays the same, activation peak halves).
EOF
