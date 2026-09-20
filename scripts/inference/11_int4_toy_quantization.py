import torch


def quantize_int4(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qmin = -8
    qmax = 7

    max_abs = weights.abs().max()
    scale = max_abs / qmax

    quantized = torch.round(weights / scale)
    quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)

    return quantized, scale


def dequantize_int4(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantized.float() * scale


def main() -> None:
    weights = torch.tensor([-0.80, -0.31, 0.10, 0.12, 0.15, 0.80])

    quantized, scale = quantize_int4(weights)
    reconstructed = dequantize_int4(quantized, scale)

    error = reconstructed - weights
    abs_error = error.abs()

    print(f"Original weights : {weights}")
    print(f"Scale            : {scale:.6f}")
    print(f"INT4 codes       : {quantized}")
    print(f"Reconstructed    : {reconstructed}")
    print(f"Absolute error   : {abs_error}")
    print(f"Mean abs error   : {abs_error.mean():.6f}")
    print(f"Max abs error    : {abs_error.max():.6f}")


if __name__ == "__main__":
    main()
