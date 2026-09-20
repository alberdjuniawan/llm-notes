import gc
import json
import statistics
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Evaluate BF16 and NF4 model behavior on fixed prompts."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Maximum number of tokens generated.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed generation runs after warmup.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/004-quantization-evaluation.json"),
        help="Path for the JSON evaluation report.",
    )

    return parser


def clear_gpu_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_bf16_model(model_path: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )


def load_nf4_model(model_path: str) -> AutoModelForCausalLM:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )


def tokenize_prompt(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)


def get_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> torch.Tensor:
    inputs = tokenize_prompt(tokenizer, model, prompt)

    with torch.inference_mode():
        outputs = model(**inputs)

    return outputs.logits[:, -1, :].float()


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenize_prompt(tokenizer, model, prompt)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


def benchmark_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    runs: int,
) -> tuple[str, float]:
    generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
    )
    torch.cuda.synchronize()

    latencies = []
    generated_text = ""

    for _ in range(runs):
        start = time.perf_counter()

        generated_text = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
        )

        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

    return generated_text, statistics.median(latencies)


def compare_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | int]:
    difference = (candidate - reference).abs()

    reference_top1 = reference.argmax(dim=-1)
    candidate_top1 = candidate.argmax(dim=-1)

    top1_agreement = int(
        torch.equal(
            reference_top1,
            candidate_top1,
        )
    )

    reference_top5 = torch.topk(
        reference,
        5,
        dim=-1,
    ).indices

    candidate_top5 = torch.topk(
        candidate,
        5,
        dim=-1,
    ).indices

    top5_intersection = len(
        set(reference_top5[0].tolist())
        & set(candidate_top5[0].tolist())
    )

    top5_overlap = top5_intersection / 5

    reference_log_probs = F.log_softmax(
        reference,
        dim=-1,
    )

    candidate_probs = F.softmax(
        candidate,
        dim=-1,
    )

    kl_divergence = F.kl_div(
        reference_log_probs,
        candidate_probs,
        reduction="batchmean",
    ).item()

    return {
        "mean_abs_logit_error": difference.mean().item(),
        "max_abs_logit_error": difference.max().item(),
        "top1_agreement": top1_agreement,
        "top5_overlap": top5_overlap,
        "kl_divergence_nf4_vs_bf16": kl_divergence,
    }


def collect_results(
    model_name: str,
    loader,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    runs: int,
) -> dict:
    clear_gpu_memory()

    print(f"\n{model_name}:")

    model = loader()

    torch.cuda.synchronize()

    memory_after_load = (
        torch.cuda.memory_allocated() / (1024**2)
    )

    prompt_results = []

    for index, prompt in enumerate(prompts, start=1):
        logits = get_logits(
            model,
            tokenizer,
            prompt,
        )

        generated_text, latency = benchmark_generation(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            runs,
        )

        top_values, top_indices = torch.topk(
            logits[0],
            5,
        )

        top_tokens = [
            {
                "rank": rank,
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "logit": value,
            }
            for rank, (value, token_id) in enumerate(
                zip(
                    top_values.tolist(),
                    top_indices.tolist(),
                    strict=True,
                ),
                start=1,
            )
        ]

        prompt_results.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "latency_seconds": latency,
                "logits": logits.cpu(),
                "top_tokens": top_tokens,
            }
        )

        print(f"[{index}/{len(prompts)}] {prompt}")
        print(f"  Latency: {latency:.4f} s")
        print(f"  Generated: {generated_text}")

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**2)
    )

    print(f"Memory after load: {memory_after_load:.2f} MiB")
    print(f"Peak GPU memory: {peak_memory:.2f} MiB")

    del model
    clear_gpu_memory()

    return {
        "memory_after_load_mib": memory_after_load,
        "peak_gpu_memory_mib": peak_memory,
        "prompts": prompt_results,
    }


def build_comparison(
    bf16_results: dict,
    nf4_results: dict,
) -> dict:
    prompt_results = []

    for bf16_prompt, nf4_prompt in zip(
        bf16_results["prompts"],
        nf4_results["prompts"],
        strict=True,
    ):
        metrics = compare_logits(
            bf16_prompt["logits"],
            nf4_prompt["logits"],
        )

        metrics.update(
            {
                "prompt": bf16_prompt["prompt"],
                "bf16_text": bf16_prompt["generated_text"],
                "nf4_text": nf4_prompt["generated_text"],
                "bf16_latency_seconds": (
                    bf16_prompt["latency_seconds"]
                ),
                "nf4_latency_seconds": (
                    nf4_prompt["latency_seconds"]
                ),
                "bf16_top_tokens": bf16_prompt["top_tokens"],
                "nf4_top_tokens": nf4_prompt["top_tokens"],
            }
        )

        prompt_results.append(metrics)

    average_mean_error = statistics.mean(
        result["mean_abs_logit_error"]
        for result in prompt_results
    )

    average_max_error = statistics.mean(
        result["max_abs_logit_error"]
        for result in prompt_results
    )

    top1_agreement_rate = statistics.mean(
        result["top1_agreement"]
        for result in prompt_results
    )

    average_top5_overlap = statistics.mean(
        result["top5_overlap"]
        for result in prompt_results
    )

    average_kl_divergence = statistics.mean(
        result["kl_divergence_nf4_vs_bf16"]
        for result in prompt_results
    )

    average_bf16_latency = statistics.mean(
        result["bf16_latency_seconds"]
        for result in prompt_results
    )

    average_nf4_latency = statistics.mean(
        result["nf4_latency_seconds"]
        for result in prompt_results
    )

    bf16_peak_memory = bf16_results[
        "peak_gpu_memory_mib"
    ]

    nf4_peak_memory = nf4_results[
        "peak_gpu_memory_mib"
    ]

    return {
        "aggregate": {
            "average_mean_abs_logit_error": average_mean_error,
            "average_max_abs_logit_error": average_max_error,
            "top1_agreement_rate": top1_agreement_rate,
            "average_top5_overlap": average_top5_overlap,
            "average_kl_divergence_nf4_vs_bf16": (
                average_kl_divergence
            ),
            "average_bf16_latency_seconds": (
                average_bf16_latency
            ),
            "average_nf4_latency_seconds": (
                average_nf4_latency
            ),
            "latency_relative_change": (
                average_nf4_latency - average_bf16_latency
            )
            / average_bf16_latency,
            "bf16_peak_memory_mib": bf16_peak_memory,
            "nf4_peak_memory_mib": nf4_peak_memory,
            "memory_relative_reduction": (
                bf16_peak_memory - nf4_peak_memory
            )
            / bf16_peak_memory,
        },
        "prompts": prompt_results,
    }


def make_json_safe(results: dict) -> dict:
    return {
        "memory_after_load_mib": results[
            "memory_after_load_mib"
        ],
        "peak_gpu_memory_mib": results[
            "peak_gpu_memory_mib"
        ],
        "prompts": [
            {
                key: value
                for key, value in prompt.items()
                if key != "logits"
            }
            for prompt in results["prompts"]
        ],
    }


def main() -> None:
    args = parse_args().parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError(
            "max-new-tokens must be greater than 0"
        )

    if args.runs <= 0:
        raise ValueError(
            "runs must be greater than 0"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this evaluation."
        )

    prompts = [
        "Explain photosynthesis in simple terms:",
        "What is a fraction?",
        "Explain how rain forms:",
        "What is the difference between a CPU and a GPU?",
        "Why does the Moon have phases?",
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    print("Device: cuda")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Timed runs per prompt: {args.runs}")

    bf16_results = collect_results(
        "BF16",
        lambda: load_bf16_model(args.model),
        tokenizer,
        prompts,
        args.max_new_tokens,
        args.runs,
    )

    nf4_results = collect_results(
        "NF4",
        lambda: load_nf4_model(args.model),
        tokenizer,
        prompts,
        args.max_new_tokens,
        args.runs,
    )

    comparison = build_comparison(
        bf16_results,
        nf4_results,
    )

    report = {
        "model": args.model,
        "compute_dtype": "bfloat16",
        "quantization": {
            "type": "NF4",
            "bits": 4,
            "groupwise": True,
        },
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "timed_runs_per_prompt": args.runs,
        "bf16": make_json_safe(bf16_results),
        "nf4": make_json_safe(nf4_results),
        "comparison": comparison,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    aggregate = comparison["aggregate"]

    print("\nAggregate Results:")

    print(
        "Average mean abs logit error: "
        f"{aggregate['average_mean_abs_logit_error']:.6f}"
    )

    print(
        "Average max abs logit error: "
        f"{aggregate['average_max_abs_logit_error']:.6f}"
    )

    print(
        "Top-1 agreement rate: "
        f"{aggregate['top1_agreement_rate']:.2%}"
    )

    print(
        "Average top-5 overlap: "
        f"{aggregate['average_top5_overlap']:.2%}"
    )

    print(
        "Average KL divergence (NF4 || BF16): "
        f"{aggregate['average_kl_divergence_nf4_vs_bf16']:.6f}"
    )

    print(
        "Average BF16 latency: "
        f"{aggregate['average_bf16_latency_seconds']:.4f} s"
    )

    print(
        "Average NF4 latency: "
        f"{aggregate['average_nf4_latency_seconds']:.4f} s"
    )

    print(
        "Latency relative change: "
        f"{aggregate['latency_relative_change']:.2%}"
    )

    print(
        "BF16 peak memory: "
        f"{aggregate['bf16_peak_memory_mib']:.2f} MiB"
    )

    print(
        "NF4 peak memory: "
        f"{aggregate['nf4_peak_memory_mib']:.2f} MiB"
    )

    print(
        "Memory relative reduction: "
        f"{aggregate['memory_relative_reduction']:.2%}"
    )

    print(f"Evaluation report: {args.output}")


if __name__ == "__main__":
    main()
