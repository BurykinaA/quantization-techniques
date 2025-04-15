import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
import argparse
import os
from model.resnet import resnet18
from trainers.mnist_trainer import MNISTTrainer
from trainers.cifar_trainer import CIFARTrainer


def check_pretrained_weights():
    """Check if pretrained weights exist"""
    pretrained_path = os.path.join('datasets', 'resnet18.pt')
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained weights not found at {pretrained_path}. "
            "Please ensure the weights file is in the datasets directory."
        )


def run_quantization_experiment(args):
    """Run quantization experiment based on provided arguments"""
    
    # Print experiment configuration
    print("\n=== Experiment Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Quantization: {args.quant_type}")
    print(f"Bitwidth: {args.bitwidth}")
    print(f"Epochs: {args.epochs}")
    print("=============================\n")
    
    # Check for pretrained weights if using PTQ
    if args.quant_type == 'ptq':
        check_pretrained_weights()
    
    # Select appropriate trainer based on dataset
    trainer_class = MNISTTrainer if args.dataset == 'mnist' else CIFARTrainer
    
    # Create and run trainer
    trainer = trainer_class(args)
    accuracy = trainer.train()
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Neural Network Quantization Experiments')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], required=True,
                      help='Dataset to use (mnist or cifar)')
    parser.add_argument('--quant_type', type=str, choices=['none', 'ptq', 'qat'], required=True,
                      help='Quantization type (none, post-training, or quantization-aware)')
    
    # Optional arguments with defaults
    parser.add_argument('--bitwidth', type=int, choices=[4, 6, 8], default=8,
                      help='Bitwidth for quantization (default: 8)')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='Learning rate (default: 0.01)')
    
    args = parser.parse_args()
    
    # Adjust epochs for PTQ
    if args.quant_type == 'ptq' and args.epochs > 1:
        print("Note: For PTQ, setting epochs to 1 as we're using a pretrained model")
        args.epochs = 1
    
    # Print warning for MNIST + PTQ combination
    if args.dataset == 'mnist' and args.quant_type == 'ptq':
        print("\nWarning: Using PTQ with MNIST might give suboptimal results")
        print("as we're using an ImageNet pretrained model on grayscale images.")
        print("Consider using QAT for MNIST or PTQ with CIFAR instead.\n")
    
    # Run the experiment
    accuracy = run_quantization_experiment(args)
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("===================\n")

if __name__ == '__main__':
    main()