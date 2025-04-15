#!/bin/bash

# Run only MNIST experiments
python quant_resnet.py --dataset mnist --quant_type none --epochs 10
python quant_resnet.py --dataset mnist --quant_type qat --bitwidth 8 --epochs 10
python quant_resnet.py --dataset mnist --quant_type qat --bitwidth 6 --epochs 10
python quant_resnet.py --dataset mnist --quant_type qat --bitwidth 4 --epochs 10
python quant_resnet.py --dataset mnist --quant_type ptq --bitwidth 8 --epochs 1 