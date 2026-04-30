#!/usr/bin/env bash
# run_iter_k.sh — Iterative per-layer k search during LoRA training.
#
# Three variants, all starting from k=4:
#   iter_lora_k_pre  — k-search once before LoRA training (5 epochs)
#   iter_lora_k_i2   — k-search every 2 LoRA epochs (6 epochs, 3 searches)
#   iter_lora_k_i1   — k-search every LoRA epoch (5 epochs, 5 searches)
#
# Baseline for comparison (re-uses FlatQuant cache if available):
#   best_lora_k      — k-search once before LoRA, k_init=16 (existing best)
#
# Usage:
#   bash run_iter_k.sh                         # all 4 configs
#   bash run_iter_k.sh --configs iter_lora_k_i1  # single config
#   bash run_iter_k.sh --output_dir /data/out    # custom output dir

set -e
cd "$(dirname "$0")"

if [[ "$*" != *"--configs"* ]]; then
    python pipeline.py \
        --configs best_lora_k iter_lora_k_pre iter_lora_k_i2 iter_lora_k_i1 \
        "$@"
else
    python pipeline.py "$@"
fi
