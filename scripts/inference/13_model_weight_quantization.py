import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def quantize_per_group(
    weights: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmin = -8
    qmax = 7

    flat = weights.flatten()
    quantized_groups = []
    scales = []

    for start in range(0, flat.numel(), group_size):
        group = flat[start : start + group_size]

        max_abs = group.abs().max()

        if max_abs == 0:
            scale = torch.tensor(1.0, device=weights.device)
        else:
            scale = max_abs / qmax

        quantized = torch.round(group / scale)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)

        quantized_groups.append(quantized)
        scales.append(scale)

    return torch.cat(quantized_groups), torch.stack(scales)


def dequantize_per_group(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    reconstructed_groups = []

    for index, start in enumerate(range(0, quantized.numel(), group_size)):
        group = quantized[start : start + group_size]
        reconstructed_groups.append(group.float() * scales[index])

    return torch.cat(reconstructed_groups)


def evaluate_quantization(
    weights: torch.Tensor,
    group_size: int,
) -> None:
    quantized, scales = quantize_per_group(weights, group_size)
    reconstructed = dequantize_per_group(
        quantized,
        scales,
        group_size,
    )

    error = (reconstructed - weights.flatten()).abs()

    print(f"\nGroup size: {group_size}")
    print(f"Number of groups: {scales.numel()}")
    print(f"Number of scales: {scales.numel()}")
    print(f"Mean abs error: {error.mean().item():.8f}")
    print(f"Max abs error: {error.max().item():.8f}")


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    parameter_name = "model.layers.0.self_attn.q_proj.weight"
    weights = model.get_parameter(parameter_name).detach()

    print(f"Parameter: {parameter_name}")
    print(f"Shape: {tuple(weights.shape)}")
    print(f"Dtype: {weights.dtype}")
    print(f"Number of elements: {weights.numel()}")
    print(f"Minimum: {weights.min().item():.6f}")
    print(f"Maximum: {weights.max().item():.6f}")
    print(f"Mean: {weights.mean().item():.6f}")
    print(f"Std: {weights.std().item():.6f}")

    evaluate_quantization(weights, group_size=128)
    evaluate_quantization(weights, group_size=64)
    evaluate_quantization(weights, group_size=32)


if __name__ == "__main__":
    main()
