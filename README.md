# Self-Pruning Neural Network

This project implements a self-pruning neural network in PyTorch on CIFAR-10.

Each weight has a learnable gate parameter. During training, L1 regularization on gates encourages sparsity, allowing the model to automatically prune unnecessary connections.

## Features
- Custom PrunableLinear layer
- Learnable sigmoid gates
- L1 sparsity regularization
- Accuracy vs sparsity comparison
- Gate distribution visualization

## How to Run
```bash
pip install -r requirements.txt
python self_pruning_cifar10.py