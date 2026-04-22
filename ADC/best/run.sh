#!/usr/bin/env bash
# Run unsigned PTQ configs and print a comparison table.
# Results are saved to outputs/results.json.
#
# Usage:
#   bash run.sh                                         # best_ptq + best_ptq_k + best_lora + best_lora_k (default)
#   bash run.sh --configs best_ptq best_ptq_k           # PTQ only, no LoRA
#   bash run.sh --configs best_ptq best_ptq_k_recal     # with FlatQuant recalibration
#   bash run.sh --configs fp int4_no_adc best_ptq       # custom selection
#   bash run.sh --output_dir /tmp/out                   # custom output directory

set -e
cd "$(dirname "$0")"

# Default: unsigned k=16, per-layer k search, LoRA correction, LoRA+k
if [[ "$*" != *"--configs"* ]]; then
    python pipeline.py --configs best_ptq best_ptq_k best_lora best_lora_k "$@"
else
    python pipeline.py "$@"
fi
