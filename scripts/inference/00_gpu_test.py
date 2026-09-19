import torch


def main() -> None:
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version  : {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"GPU           : {props.name}")
    print(f"VRAM          : {props.total_memory / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
