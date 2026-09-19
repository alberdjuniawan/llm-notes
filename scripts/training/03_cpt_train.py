import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run CPT on a causal language model.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the processed CPT dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where the CPT checkpoint will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [sample["input_ids"] for sample in batch],
        dtype=torch.long,
    )

    return {"input_ids": input_ids}


def calculate_loss(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids)

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    return loss


def evaluate_loss(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)

            loss = calculate_loss(model, input_ids)

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def main() -> None:
    args = parse_args().parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs must be greater than 0")

    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than 0")

    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be greater than 0")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading dataset: {args.dataset}")
    dataset = load_from_disk(str(args.dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    print(f"Sequences: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.config.use_cache = False
    model.to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    initial_loss = evaluate_loss(
        model,
        dataloader,
        device,
    )

    print(f"Initial loss: {initial_loss:.4f}")

    model.train()

    epoch_losses = []

    for epoch in range(args.epochs):
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)

            optimizer.zero_grad()

            loss = calculate_loss(
                model,
                input_ids,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.4f}")

    final_loss = evaluate_loss(
        model,
        dataloader,
        device,
    )

    print(f"Final loss: {final_loss:.4f}")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        args.output,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(args.output)

    metrics = {
        "model": args.model,
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "initial_loss": initial_loss,
        "epoch_losses": epoch_losses,
        "final_loss": final_loss,
        "device": str(device),
    }

    metrics_path = args.output / "metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Checkpoint saved to: {args.output}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
