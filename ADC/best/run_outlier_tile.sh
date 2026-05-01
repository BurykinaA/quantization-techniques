#!/usr/bin/env bash
# run_outlier_tile.sh — Outlier-aware tiling + per-tile k search experiments.
#
# Configs:
#   best_lora_k       — per-layer k + LoRA (baseline, re-uses FlatQuant cache)
#   best_ptq_k_tile   — per-tile k search (each QATLinearADC tile gets its own k)
#   best_lora_k_tile  — per-tile k + LoRA
#   outlier_tile_ptq  — outlier-aware tiling (channels grouped by |activation|)
#   outlier_tile_lora — outlier-aware tiling + LoRA
#
# Usage:
#   bash run_outlier_tile.sh                                # all 5 configs
#   bash run_outlier_tile.sh --configs outlier_tile_lora    # single config
#   bash run_outlier_tile.sh --output_dir /data/out         # custom output dir

set -e
cd "$(dirname "$0")"

if [[ "$*" != *"--configs"* ]]; then
    python pipeline.py \
        --configs best_lora_k best_ptq_k_tile best_lora_k_tile \
                  outlier_tile_ptq outlier_tile_lora \
        "$@"
else
    python pipeline.py "$@"
fi
