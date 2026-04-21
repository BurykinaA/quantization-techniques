#!/usr/bin/env bash
# Run unsigned PTQ configs and print a comparison table.
# Results are saved to outputs/results.json.
#
# Usage:
#   bash run.sh                                      # best_ptq + best_ptq_k (default)
#   bash run.sh --configs best_ptq best_ptq_k_recal  # with FlatQuant recalibration
#   bash run.sh --configs fp int4_no_adc best_ptq    # custom selection
#   bash run.sh --output_dir /tmp/out                # custom output directory

set -e
cd "$(dirname "$0")"

# Default: unsigned k=16 baseline + per-layer k search
if [[ "$*" != *"--configs"* ]]; then
    python pipeline.py --configs best_ptq best_ptq_k "$@"
else
    python pipeline.py "$@"
fi
