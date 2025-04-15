#!/bin/bash

# Quick test script with minimal epochs
python quant_resnet.py --dataset mnist --quant_type none --epochs 1
python quant_resnet.py --dataset mnist --quant_type qat --bitwidth 8 --epochs 1
python quant_resnet.py --dataset mnist --quant_type ptq --bitwidth 8 --epochs 1 