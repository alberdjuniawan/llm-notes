from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Validate a saved CPT checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the saved CPT checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Photosynthesis is",
        help="Prompt used for a functional forward-pass test.",
    )

    return parser


def main() -> None:
    args = parse_args().parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.float32,
    )

    model.to(device)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Parameters: {parameter_count:,}")

    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    print("Checkpoint loaded successfully.")
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits dtype: {outputs.logits.dtype}")


if __name__ == "__main__":
    main()
