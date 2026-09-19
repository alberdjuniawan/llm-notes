from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare a raw text corpus for CPT.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model ID used to load the tokenizer.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw text corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Number of tokens in each training sequence.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.block_size <= 0:
        raise ValueError("block-size must be greater than 0")

    text = args.input.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("Input corpus is empty.")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Tokenizing: {args.input}")
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
    )

    input_ids = tokenized["input_ids"]

    print(f"Total tokens: {len(input_ids)}")

    sequences = [
        input_ids[start : start + args.block_size]
        for start in range(0, len(input_ids), args.block_size)
    ]

    dataset = Dataset.from_dict({"input_ids": sequences})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output))

    print(f"Sequences: {len(dataset)}")
    print(f"Block size: {args.block_size}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
