#!/bin/bash
# run_slides.sh — Train loss-landscape checkpoints + generate all slide figures.
#
# Part 1: 3 PTQ experiments (no staged training, varying α)
#   landscape_alpha0   — α=0: pure FP loss, no propagation
#   landscape_alpha05  — α=0.5: mixed FP+ADC loss
#   landscape_alpha1   — α=1.0: full ADC propagation
#
# Part 2: Generate all figures in latex/slides/figs/
#   plot_all.py                 — 8 data-ready figures (hardcoded results)
#   plot_flatquant_intuition.py — synthetic histograms (no model needed)
#   plot_loss_landscape.py      — landscape from Part 1 checkpoints
#
# Usage: bash ADC/llama/run_slides.sh
# Results:
#   ADC/llama/checkpoints/landscape_DATE/  — 3 model_full.pt checkpoints
#   latex/slides/figs/                     — all slide figures

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SLIDES_DIR="$REPO_DIR/latex/slides"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/landscape_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_landscape_${DATE_TAG}"
CKPT_DIR="$SCRIPT_DIR/checkpoints/landscape_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CKPT_DIR"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo "Slides pipeline: loss landscape + all figures"
echo "Started: $(date)"
echo "Checkpoints: $CKPT_DIR"
echo "Figures:     $SLIDES_DIR/figs/"
echo "============================================================"

MODEL_NAME="meta-llama/Llama-3.2-1B"

# ── PTQ base config for landscape experiments ────────────────────────────────
# Simpler than run_test.sh: no staged training, no LoRA — clean α comparison.
# 256 calibration samples to keep runs ~1h each.
BASE_ARGS=(
    --model_name "$MODEL_NAME"
    --preprocess_method flat_quant
    --bx 4 --bw 4 --ba 8 --k 16
    --fq_w_bits 4 --fq_a_bits 4
    --fq_epochs 30
    --fq_nsamples 256
    --fq_cali_bsz 16
    --fq_lr 0.005
    --fq_lwc --fq_lac
    --fq_add_diag
    --mvm_limit 256
    --calibration_method percentile
    --calibration_max_length 512
    --eval_datasets wikitext2
    --run_no_adc_eval
    --results_json_path "$RESULTS_JSON"
    --disable_visualizations
    --wandb_project llama-flat-quant-adc-ptq-blocks
    --save_full_model_pt
    --seed 42
)

run_experiment() {
    local name="$1"
    shift
    local logfile="$LOG_DIR/${name}.log"
    local out_dir="$CKPT_DIR/${name}"

    echo ""
    echo "========================================"
    echo "START: $name"
    echo "Time:  $(date)"
    echo "Log:   $logfile"
    echo "========================================"

    python "$SCRIPT_DIR/runs/llama_smooth_quant_adc_ptq.py" \
        --output_dir "$out_dir" \
        --wandb_run_name "${name}_landscape" \
        "$@" \
        > "$logfile" 2>&1

    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "DONE:  $name  (exit 0)  $(date)"
        local wiki_adc wiki_bypass
        wiki_bypass=$(grep -oP "(?<=WITHOUT ADC -> WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | head -1)
        wiki_adc=$(grep -oP "(?<=WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | tail -1)
        echo "       wiki: bypass=${wiki_bypass:-n/a}  ADC=${wiki_adc:-n/a}"
        if [ -f "${out_dir}/model_full.pt" ]; then
            local sz
            sz=$(du -sh "${out_dir}/model_full.pt" | cut -f1)
            echo "       model_full.pt: $sz"
        fi
    else
        echo "FAIL:  $name  (exit $exit_code)  $(date)"
        echo "       Check log: $logfile"
    fi

    sleep 10
}

# ============================================================
# PART 1 — Train 3 checkpoints
# ============================================================

# α=0: standard FlatQuant, pure FP loss (no propagation)
run_experiment "landscape_alpha0" \
    "${BASE_ARGS[@]}"

# α=0.5: mixed FP + ADC loss — the sweet spot from v2 experiments
run_experiment "landscape_alpha05" \
    "${BASE_ARGS[@]}" \
    --fq_propagate_quant --fq_prop_alpha 0.5

# α=1.0: full ADC propagation — trains hard for the ADC noise distribution
run_experiment "landscape_alpha1" \
    "${BASE_ARGS[@]}" \
    --fq_propagate_quant --fq_prop_alpha 1.0

echo ""
echo "============================================================"
echo "PART 1 DONE — checkpoints saved to $CKPT_DIR"
echo "============================================================"

# ============================================================
# PART 2 — Generate all slide figures
# ============================================================

echo ""
echo "Generating figures in $SLIDES_DIR/figs/ ..."
cd "$SLIDES_DIR"

echo "[1/3] plot_all.py — 8 data-ready figures"
python scripts/plot_all.py

echo "[2/3] plot_flatquant_intuition.py — synthetic histogram"
python scripts/plot_flatquant_intuition.py

echo "[3/3] plot_loss_landscape.py — loss landscape"
# PTQ script appends a date/run suffix to the output_dir name — find robustly
alpha0_pt=$(find "$CKPT_DIR" -name "model_full.pt" -path "*alpha0*" ! -path "*alpha05*" | head -1)
alpha05_pt=$(find "$CKPT_DIR" -name "model_full.pt" -path "*alpha05*" | head -1)
alpha1_pt=$(find "$CKPT_DIR" -name "model_full.pt" -path "*alpha1*" ! -path "*alpha05*" | head -1)

echo "  alpha0  : ${alpha0_pt:-NOT FOUND}"
echo "  alpha05 : ${alpha05_pt:-NOT FOUND}"
echo "  alpha1  : ${alpha1_pt:-NOT FOUND}"

if [ -n "$alpha0_pt" ] && [ -n "$alpha05_pt" ] && [ -n "$alpha1_pt" ]; then
    python scripts/plot_loss_landscape.py \
        --checkpoint_alpha0  "$alpha0_pt" \
        --checkpoint_alpha05 "$alpha05_pt" \
        --checkpoint_alpha1  "$alpha1_pt" \
        --output             "$SLIDES_DIR/figs/plot_loss_landscape_alpha.pdf"
else
    echo "  [skip] one or more checkpoints missing — skipping landscape plot"
fi

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "============================================================"
echo ""
echo "Figures:"
ls -lh "$SLIDES_DIR/figs/"*.pdf 2>/dev/null || echo "  (no PDFs found)"
echo ""
echo "Compile slides:"
echo "  cd $SLIDES_DIR && pdflatex main.tex"
