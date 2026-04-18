#!/usr/bin/env bash
# Run all four configurations and print a comparison table.
# Results are saved to outputs/results.json.
#
# Usage:
#   bash run.sh                        # run all four configs
#   bash run.sh --configs fp best_lora # run only selected configs
#   bash run.sh --output_dir /tmp/out  # custom output directory

set -e
cd "$(dirname "$0")"

python pipeline.py "$@"
