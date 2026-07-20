#!/usr/bin/env bash
# Wrapper that runs the ViT ADC pipeline with a working Python environment.
#
# Why this is needed: the project's default PYTHONPATH (/home/coder/project:...)
# shadows the installed torch and breaks `import torch`.  We unset PYTHONPATH and
# run from a neutral cwd so the conda torch/timm/torchvision stack loads cleanly.
#
# Usage:
#   ./run.sh --model vit_tiny_patch16_224 --configs fp int4_no_adc --smoke --val_portion 0.05
#   ./run.sh --model vit_tiny_patch16_224 --configs unsigned_ptq unsigned_ptq_lora --k 4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/vit_pipeline.py"

# ImageNet root: honour an already-exported IMAGENET_ROOT, else use the workspace default.
: "${IMAGENET_ROOT:=/home/coder/project/imagenet/data}"

cd /tmp
exec env -u PYTHONPATH IMAGENET_ROOT="${IMAGENET_ROOT}" python "${PIPELINE}" "$@"
