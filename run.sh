#!/bin/bash

# Function to run experiments and log results
run_experiment() {
    local dataset=$1
    local quant_type=$2
    local bitwidth=$3
    local epochs=$4

    echo "Running experiment:"
    echo "Dataset: $dataset"
    echo "Quantization type: $quant_type"
    echo "Bitwidth: $bitwidth"
    echo "Epochs: $epochs"
    echo "----------------------------------------"

    python quant_resnet.py \
        --dataset $dataset \
        --quant_type $quant_type \
        --bitwidth $bitwidth \
        --epochs $epochs \
        --batch_size 128 \
        --lr 0.01

    echo "----------------------------------------"
    echo ""
}

# MNIST Experiments
echo "=== Starting MNIST Experiments ==="

# Regular training (no quantization)
run_experiment mnist none 8 10

# QAT experiments with different bitwidths
for bitwidth in 8 6 4; do
    run_experiment mnist qat $bitwidth 10
done

# PTQ experiment (fewer epochs needed as we're using pretrained model)
for bitwidth in 8 6 4; do
    run_experiment mnist ptq $bitwidth 1
done

# CIFAR Experiments
echo "=== Starting CIFAR Experiments ==="

# Regular training (no quantization)
run_experiment cifar none 8 10

# QAT experiments with different bitwidths
for bitwidth in 8 6 4; do
    run_experiment cifar qat $bitwidth 10
done

# PTQ experiment (fewer epochs needed as we're using pretrained model)
for bitwidth in 8 6 4; do
    run_experiment cifar ptq $bitwidth 1
done