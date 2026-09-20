import gc
from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Compare base and SFT model generations.")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Hugging Face model ID or local base checkpoint.",
    )
    parser.add_argument(
        "--sft-model",
        type=Path,
        required=True,
        help="Path to the SFT checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens generated.",
    )

    return parser


def generate(
    model_path: str | Path,
    tokenizer_path: str | Path,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
    )

    model.to(device)
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    del model
    del tokenizer
    del inputs
    del outputs

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return generated_text


def main() -> None:
    args = parse_args().parse_args()

    if not args.sft_model.exists():
        raise FileNotFoundError(f"SFT checkpoint not found: {args.sft_model}")

    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be greater than 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompts = [
        "User: Explain photosynthesis for a fifth-grade student.\nAssistant:",
        "User: Explain the solar system in simple terms.\nAssistant:",
        "User: Explain what a fraction represents.\nAssistant:",
        "User: What is the role of the Sun in the solar system?\nAssistant:",
        "User: Explain why plants need sunlight.\nAssistant:",
    ]

    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"SFT model: {args.sft_model}")

    for prompt in prompts:
        print("\n" + "=" * 80)
        print(f"Prompt: {prompt}")
        print("=" * 80)

        base_output = generate(
            model_path=args.base_model,
            tokenizer_path=args.base_model,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        sft_output = generate(
            model_path=args.sft_model,
            tokenizer_path=args.sft_model,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        print("\nBase Model:")
        print(base_output)

        print("\nSFT Model:")
        print(sft_output)


if __name__ == "__main__":
    main()
