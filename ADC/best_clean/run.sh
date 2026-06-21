#!/usr/bin/env bash
# Run the ADC-aware INT4 configs and print a comparison table.
# Results are saved to outputs/results.json.
#
# Usage:
#   bash run.sh                                      # fp + int4_no_adc + best_ptq + best_lora
#   bash run.sh --configs best_ptq best_lora         # PTQ + LoRA only
#   bash run.sh --configs fp int4_no_adc best_ptq    # custom selection
#   bash run.sh --output_dir /tmp/out                # custom output directory

set -e
cd "$(dirname "$0")"

if [[ "$*" != *"--configs"* ]]; then
    python pipeline.py --configs fp int4_no_adc best_ptq best_lora "$@"
else
    python pipeline.py "$@"
fi
