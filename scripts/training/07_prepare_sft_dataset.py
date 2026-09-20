import json
from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Prepare instruction-response data for SFT."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw JSONL SFT dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where the processed dataset will be saved.",
    )
    return parser


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            if "instruction" not in row or "response" not in row:
                raise ValueError(
                    f"Line {line_number} must contain "
                    "'instruction' and 'response'."
                )

            if not isinstance(row["instruction"], str):
                raise TypeError(
                    f"Line {line_number}: instruction must be a string."
                )

            if not isinstance(row["response"], str):
                raise TypeError(
                    f"Line {line_number}: response must be a string."
                )

            if not row["instruction"].strip():
                raise ValueError(
                    f"Line {line_number}: instruction is empty."
                )

            if not row["response"].strip():
                raise ValueError(
                    f"Line {line_number}: response is empty."
                )

            rows.append(row)

    return rows


def build_example(
    tokenizer: AutoTokenizer,
    instruction: str,
    response: str,
) -> dict[str, list[int]]:
    prompt = f"User: {instruction.strip()}\nAssistant:"
    response_text = f" {response.strip()}{tokenizer.eos_token}"

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    response_ids = tokenizer(
        response_text,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def main() -> None:
    args = parse_args().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer does not define an EOS token.")

    print(f"Loading SFT dataset: {args.input}")
    rows = load_jsonl(args.input)

    examples = [
        build_example(
            tokenizer,
            row["instruction"],
            row["response"],
        )
        for row in rows
    ]

    dataset = Dataset.from_dict(
        {
            "input_ids": [example["input_ids"] for example in examples],
            "labels": [example["labels"] for example in examples],
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(args.output))

    input_lengths = [
        len(example["input_ids"])
        for example in examples
    ]

    response_lengths = [
        sum(label != -100 for label in example["labels"])
        for example in examples
    ]

    ignored_label_counts = [
        sum(label == -100 for label in example["labels"])
        for example in examples
    ]

    metadata = {
        "model": args.model,
        "input": str(args.input),
        "output": str(args.output),
        "num_examples": len(dataset),
        "prompt_format": "User: {instruction}\\nAssistant: {response}",
        "label_mask_value": -100,
        "response_eos": True,
        "total_sequence_tokens": sum(input_lengths),
        "total_response_tokens": sum(response_lengths),
        "total_prompt_labels": sum(ignored_label_counts),
        "min_sequence_length": min(input_lengths),
        "max_sequence_length": max(input_lengths),
        "avg_sequence_length": sum(input_lengths) / len(input_lengths),
    }

    metadata_path = args.output.with_name(
        f"{args.output.name}_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Examples: {len(dataset)}")
    print(f"Total sequence tokens: {sum(input_lengths)}")
    print(f"Total response tokens: {sum(response_lengths)}")
    print(f"Total prompt labels: {sum(ignored_label_counts)}")
    print(f"Saved dataset to: {args.output}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()