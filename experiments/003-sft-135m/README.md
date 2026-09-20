# Experiment 003: Supervised Fine-Tuning

## Objective

Fine-tune the pretrained model on a small educational instruction-response dataset and verify that the SFT pipeline can successfully update and save the model weights.

## Model

`HuggingFaceTB/SmolLM2-135M`

## Environment

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4 GB
- PyTorch: 2.12.1+cu129
- CUDA (PyTorch): 12.9
- Kernel: 6.19.14+kali-amd64

## Dataset

A small educational instruction-response dataset covering photosynthesis, the solar system, fractions, the role of the Sun, and sunlight. The dataset contains 5 examples, with 219 total sequence tokens, 71 prompt tokens masked from the loss, and 148 response tokens.

## Task

Fine-tune the pretrained causal language model using supervised instruction-response examples while masking prompt tokens from the training loss.

## Result

SFT training successfully ran on the NVIDIA GPU and produced a valid checkpoint. Training loss decreased from `1.4784` to `0.0021`, and the SFT model produced direct responses to both training-set and unseen instructional prompts.

## Observation

The experiment demonstrates that SFT changes the existing model weights through supervised training and shifts the model's response behavior toward the instruction-response format used in the dataset. Weight analysis showed that 99.9955% of the model parameters changed during full-parameter training. The very small dataset and low final loss indicate strong fitting to the training examples.

## Limitations

The dataset contains only 5 instruction-response examples, so the experiment validates the SFT pipeline and response behavior rather than broad instruction-following or generalization.

## Report

[View detailed report](../../reports/003-sft-135m.md)
