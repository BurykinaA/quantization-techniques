#!/bin/bash

# Run only CIFAR experiments
#python quant_resnet.py --dataset cifar --quant_type none --epochs 50
python quant_resnet.py --dataset cifar --quant_type qat --bitwidth 8 --epochs 20
python quant_resnet.py --dataset cifar --quant_type qat --bitwidth 6 --epochs 20
python quant_resnet.py --dataset cifar --quant_type qat --bitwidth 4 --epochs 20
#python quant_resnet.py --dataset cifar --quant_type ptq --bitwidth 8 --epochs 1
#python quant_resnet.py --dataset cifar --quant_type ptq --bitwidth 6 --epochs 1
#python quant_resnet.py --dataset cifar --quant_type ptq --bitwidth 4 --epochs 1
