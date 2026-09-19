# Experiment 002: Continued Pre-Training

## Objective

Continue pre-training the pretrained model on a small educational text corpus and verify that the CPT pipeline can successfully update and save the model weights.

## Model

`HuggingFaceTB/SmolLM2-135M`

## Environment

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4 GB
- PyTorch: 2.12.1+cu129
- CUDA (PyTorch): 12.9
- Kernel: 6.19.14+kali-amd64

## Dataset

A small educational text corpus covering photosynthesis, the solar system, and fractions. The corpus contains 135 tokens, resulting in 2 training sequences with a block size of 64.

## Task

Continue training the pretrained causal language model using the next-token prediction objective on the educational corpus.

## Result

CPT training successfully ran on the NVIDIA GPU and produced a valid checkpoint. Training loss decreased from `1.6450` to `0.0044`, and the CPT model produced generations that were substantially closer to the training corpus than the original base model.

## Observation

The experiment demonstrates that CPT updates the existing model weights through backpropagation and optimizer steps. Weight analysis showed that 99.9984% of the model parameters changed during full-parameter training. The very small corpus caused the model to closely reproduce patterns from the training data, indicating strong overfitting.

## Limitations

The corpus contains only 135 tokens and 2 training sequences, so the experiment validates the CPT pipeline rather than meaningful educational domain adaptation.

## Report

[View detailed report](../../reports/002-cpt-135m.md)