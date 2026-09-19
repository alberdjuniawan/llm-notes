# Experiment 001: Base Inference

## Objective

Run the pretrained base model locally and verify
that inference can successfully run on the NVIDIA GPU.

## Model

`HuggingFaceTB/SmolLM2-135M`

## Environment

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4 GB
- PyTorch: 2.12.1+cu129
- CUDA (PyTorch): 12.9
- Kernel: 6.19.14+kali-amd64

## Task

Generate text from a simple prompt using the pretrained
base model without any additional training or fine-tuning.

## Result

Inference successfully ran on the NVIDIA GPU.

## Observation

The pretrained base model can be loaded and executed
locally using GPU acceleration.

This experiment establishes the functional baseline before:
- CPT
- SFT
- Quantization
- Inference optimization