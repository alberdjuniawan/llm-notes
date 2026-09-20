import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "Explain photosynthesis in simple terms:"
MAX_NEW_TOKENS = 30
TOP_K = 5


def clear_gpu_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_bf16_model() -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )


def load_nf4_model() -> AutoModelForCausalLM:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )


def get_inputs(tokenizer: AutoTokenizer, device: torch.device):
    return tokenizer(PROMPT, return_tensors="pt").to(device)


def get_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    inputs = get_inputs(tokenizer, model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    return outputs.logits[:, -1, :].float()


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> str:
    inputs = get_inputs(tokenizer, model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def print_top_tokens(
    logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    name: str,
) -> None:
    values, indices = torch.topk(logits[0], TOP_K)

    print(f"\nTop-{TOP_K} next-token candidates ({name}):")

    for rank, (value, index) in enumerate(
        zip(values.tolist(), indices.tolist(), strict=True),
        start=1,
    ):
        token = tokenizer.decode([index])
        print(f"{rank}. {token!r} | logit={value:.6f}")


def compare_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    difference = (candidate - reference).abs()

    return difference.mean().item(), difference.max().item()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}")

    print("\nBF16 Original:")
    bf16_model = load_bf16_model()

    bf16_logits = get_logits(bf16_model, tokenizer)
    bf16_text = generate_text(bf16_model, tokenizer)

    print(f"Generated: {bf16_text}")
    print_top_tokens(bf16_logits, tokenizer, "BF16")

    del bf16_model
    clear_gpu_memory()

    print("\nNF4:")
    nf4_model = load_nf4_model()

    nf4_logits = get_logits(nf4_model, tokenizer)
    nf4_text = generate_text(nf4_model, tokenizer)

    print(f"Generated: {nf4_text}")
    print_top_tokens(nf4_logits, tokenizer, "NF4")

    nf4_mean_diff, nf4_max_diff = compare_logits(
        bf16_logits,
        nf4_logits,
    )

    print(f"\nBF16 vs NF4 mean abs logit difference: {nf4_mean_diff:.8f}")
    print(f"BF16 vs NF4 max abs logit difference: {nf4_max_diff:.8f}")

    print("\nNF4 -> BF16 Dequantization:")
    nf4_model.dequantize()

    dequantized_logits = get_logits(nf4_model, tokenizer)
    dequantized_text = generate_text(nf4_model, tokenizer)

    print(f"Generated: {dequantized_text}")
    print_top_tokens(dequantized_logits, tokenizer, "Dequantized BF16")

    dequantized_mean_diff, dequantized_max_diff = compare_logits(
        bf16_logits,
        dequantized_logits,
    )

    print(
        f"\nBF16 vs dequantized BF16 mean abs logit difference: "
        f"{dequantized_mean_diff:.8f}"
    )
    print(
        f"BF16 vs dequantized BF16 max abs logit difference: {dequantized_max_diff:.8f}"
    )

    path_mean_diff, path_max_diff = compare_logits(
        nf4_logits,
        dequantized_logits,
    )

    print(f"\nNF4 vs dequantized BF16 mean abs logit difference: {path_mean_diff:.8f}")
    print(f"NF4 vs dequantized BF16 max abs logit difference: {path_max_diff:.8f}")

    del nf4_model
    clear_gpu_memory()


if __name__ == "__main__":
    main()
