import torch


def quantize_group(
    weights: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmin = -8
    qmax = 7

    quantized_groups = []
    scales = []

    for start in range(0, weights.numel(), group_size):
        group = weights[start : start + group_size]

        max_abs = group.abs().max()
        scale = max_abs / qmax

        quantized = torch.round(group / scale)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)

        quantized_groups.append(quantized)
        scales.append(scale)

    return torch.cat(quantized_groups), torch.stack(scales)


def dequantize_group(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    reconstructed_groups = []

    for index, start in enumerate(range(0, quantized.numel(), group_size)):
        group = quantized[start : start + group_size]
        reconstructed = group.float() * scales[index]
        reconstructed_groups.append(reconstructed)

    return torch.cat(reconstructed_groups)


def run_experiment(weights: torch.Tensor, group_size: int) -> None:
    quantized, scales = quantize_group(weights, group_size)
    reconstructed = dequantize_group(quantized, scales, group_size)

    absolute_error = (reconstructed - weights).abs()

    print(f"\nGroup size: {group_size}")
    print(f"Scales: {scales}")
    print(f"INT4 codes: {quantized}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Mean abs error: {absolute_error.mean():.6f}")
    print(f"Max abs error: {absolute_error.max():.6f}")


def main() -> None:
    weights = torch.tensor([-0.80, -0.31, 0.10, 0.01, 0.02, 0.04])

    print(f"Original weights: {weights}")

    run_experiment(weights, group_size=6)
    run_experiment(weights, group_size=3)


if __name__ == "__main__":
    main()
