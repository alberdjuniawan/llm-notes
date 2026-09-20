# Experiment 004: Quantization

## Objective

Study the fundamentals of model quantization and evaluate the effect of 4-bit NF4 weight quantization on local inference memory, latency, numerical outputs, and generated text.

## Model

`HuggingFaceTB/SmolLM2-135M`

## Environment

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4 GB
- PyTorch: 2.12.1+cu129
- CUDA (PyTorch): 12.9
- Kernel: 6.19.14+kali-amd64

## Task

Understand the mechanics of INT4 quantization, scale calculation, group quantization, dequantization, and INT4 packing before applying NF4 quantization to the pretrained model and comparing it with a BF16 baseline.

## Result

The toy INT4 experiments successfully demonstrated quantization, group scaling, dequantization, and packing. On a real `model.layers.0.self_attn.q_proj.weight` tensor, reducing the group size from 128 to 32 reduced mean absolute reconstruction error from `0.03210156` to `0.02429663`.

The theoretical raw weight storage for the 134,515,008-parameter model was approximately 513.13 MiB in FP32, 256.57 MiB in FP16, 128.28 MiB in INT8, and 64.14 MiB in INT4 before scale and metadata overhead.

The model was successfully loaded using 4-bit NF4 quantization and performed inference on the NVIDIA GPU. In the final evaluation, BF16 reached a peak GPU memory usage of `266.39 MiB`, while NF4 used `120.97 MiB`, corresponding to a measured peak-memory reduction of `54.59%`.

Across five evaluation prompts, NF4 had an average mean absolute logit error of `0.716722`, average maximum logit error of `3.838281`, `100.00%` next-token top-1 agreement, `64.00%` average top-5 overlap, and an average KL divergence of `0.212259` relative to BF16.

Average generation latency was `0.3383 s` for BF16 and `0.4665 s` for NF4 under the final evaluation setup.

## Observation

The experiment demonstrates that lower-precision weight representation can substantially reduce GPU memory usage, but quantization does not necessarily reduce inference latency. NF4 also changes the numerical outputs of the model, resulting in differences in logits and generated text compared with the BF16 baseline.

## Limitations

The model contains only 135 million parameters, and the final evaluation used five manually selected prompts. The latency benchmark was performed on a single RTX 3050 Laptop GPU and the local bitsandbytes environment reported that a CUDA 12.8 binary was used because no prebuilt CUDA 12.9 binary was available.

The experiment evaluates memory, latency, numerical fidelity, and short deterministic generations rather than broad language-model quality or perplexity.

## Report

[View detailed report](../../reports/004-quantization.md)