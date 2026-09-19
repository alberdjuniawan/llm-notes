import json
from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoModelForCausalLM


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Analyze parameter changes between a base model and CPT checkpoint."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Hugging Face model ID or local base checkpoint.",
    )
    parser.add_argument(
        "--cpt-model",
        type=Path,
        required=True,
        help="Path to the CPT checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the JSON report.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-8,
        help="Minimum absolute parameter change considered significant.",
    )

    return parser


def main() -> None:
    args = parse_args().parse_args()

    if not args.cpt_model.exists():
        raise FileNotFoundError(f"CPT checkpoint not found: {args.cpt_model}")

    if args.threshold < 0:
        raise ValueError("threshold must be non-negative")

    print(f"Loading base model: {args.base_model}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
    )

    print(f"Loading CPT model: {args.cpt_model}")

    cpt_model = AutoModelForCausalLM.from_pretrained(
        args.cpt_model,
    )

    base_parameters = dict(base_model.named_parameters())
    cpt_parameters = dict(cpt_model.named_parameters())

    if base_parameters.keys() != cpt_parameters.keys():
        raise ValueError("Base and CPT models have different parameter names.")

    total_parameters = 0
    changed_parameters = 0
    total_absolute_delta = 0.0
    max_absolute_delta = 0.0

    parameter_deltas = []

    for name, base_parameter in base_parameters.items():
        cpt_parameter = cpt_parameters[name]

        if base_parameter.shape != cpt_parameter.shape:
            raise ValueError(f"Shape mismatch for parameter: {name}")

        base_values = base_parameter.detach().float()
        cpt_values = cpt_parameter.detach().float()

        delta = cpt_values - base_values
        absolute_delta = delta.abs()

        parameter_count = delta.numel()
        changed_count = (absolute_delta > args.threshold).sum().item()

        parameter_total_delta = absolute_delta.sum().item()
        parameter_mean_delta = parameter_total_delta / parameter_count
        parameter_max_delta = absolute_delta.max().item()

        total_parameters += parameter_count
        changed_parameters += changed_count
        total_absolute_delta += parameter_total_delta

        max_absolute_delta = max(
            max_absolute_delta,
            parameter_max_delta,
        )

        parameter_deltas.append(
            {
                "name": name,
                "parameter_count": parameter_count,
                "changed_parameters": changed_count,
                "mean_absolute_delta": parameter_mean_delta,
                "max_absolute_delta": parameter_max_delta,
            }
        )

    mean_absolute_delta = total_absolute_delta / total_parameters

    unchanged_parameters = total_parameters - changed_parameters

    changed_percentage = changed_parameters / total_parameters * 100

    parameter_deltas.sort(
        key=lambda item: item["mean_absolute_delta"],
        reverse=True,
    )

    metrics = {
        "base_model": args.base_model,
        "cpt_model": str(args.cpt_model),
        "threshold": args.threshold,
        "total_parameters": total_parameters,
        "changed_parameters": changed_parameters,
        "unchanged_parameters": unchanged_parameters,
        "changed_percentage": changed_percentage,
        "mean_absolute_delta": mean_absolute_delta,
        "max_absolute_delta": max_absolute_delta,
        "top_parameter_deltas": parameter_deltas[:10],
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nWeight Delta Analysis:")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Changed parameters: {changed_parameters:,}")
    print(f"Unchanged parameters: {unchanged_parameters:,}")
    print(f"Changed percentage: {changed_percentage:.4f}%")
    print(f"Mean |delta|: {mean_absolute_delta:.8f}")
    print(f"Max |delta|: {max_absolute_delta:.8f}")

    print("\nLargest Mean Parameter Changes:")

    for item in parameter_deltas[:5]:
        print(
            f"{item['name']}: "
            f"mean |delta|={item['mean_absolute_delta']:.8f}, "
            f"max |delta|={item['max_absolute_delta']:.8f}"
        )

    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
