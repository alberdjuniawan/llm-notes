import json
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

    total_tokens = len(input_ids)
    usable_tokens = (total_tokens // args.block_size) * args.block_size
    dropped_tokens = total_tokens - usable_tokens

    input_ids = input_ids[:usable_tokens]

    sequences = [
        input_ids[start : start + args.block_size]
        for start in range(0, usable_tokens, args.block_size)
    ]

    dataset = Dataset.from_dict({"input_ids": sequences})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(args.output))

    metadata = {
        "model": args.model,
        "input": str(args.input),
        "block_size": args.block_size,
        "total_tokens": total_tokens,
        "usable_tokens": usable_tokens,
        "dropped_tokens": dropped_tokens,
        "num_sequences": len(dataset),
    }

    metadata_path = args.output.with_name(f"{args.output.name}_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Total tokens: {total_tokens}")
    print(f"Usable tokens: {usable_tokens}")
    print(f"Dropped tokens: {dropped_tokens}")
    print(f"Sequences: {len(dataset)}")
    print(f"Block size: {args.block_size}")
    print(f"Saved dataset to: {args.output}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
