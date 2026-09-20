import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "Explain photosynthesis in simple terms:"
MAX_NEW_TOKENS = 30


def clear_gpu_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_fp16_model() -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )


def load_nf4_model() -> AutoModelForCausalLM:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )


def get_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    return outputs.logits[:, -1, :].float()


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> str:
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compare_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    absolute_difference = (candidate - reference).abs()

    return (
        absolute_difference.mean().item(),
        absolute_difference.max().item(),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}")

    print("\nFP16:")
    fp16_model = load_fp16_model()

    fp16_logits = get_logits(fp16_model, tokenizer)
    fp16_text = generate_text(fp16_model, tokenizer)

    print(f"Generated: {fp16_text}")

    del fp16_model
    clear_gpu_memory()

    print("\nNF4:")
    nf4_model = load_nf4_model()

    nf4_logits = get_logits(nf4_model, tokenizer)
    nf4_text = generate_text(nf4_model, tokenizer)

    print(f"Generated: {nf4_text}")

    nf4_mean_diff, nf4_max_diff = compare_logits(fp16_logits, nf4_logits)

    print(f"Mean absolute logit difference: {nf4_mean_diff:.8f}")
    print(f"Max absolute logit difference: {nf4_max_diff:.8f}")

    print("\nNF4 -> dequantized:")
    nf4_model.dequantize()

    dequantized_logits = get_logits(nf4_model, tokenizer)
    dequantized_text = generate_text(nf4_model, tokenizer)

    print(f"Generated: {dequantized_text}")

    dequantized_mean_diff, dequantized_max_diff = compare_logits(
        fp16_logits,
        dequantized_logits,
    )

    print(f"Mean absolute logit difference: {dequantized_mean_diff:.8f}")
    print(f"Max absolute logit difference: {dequantized_max_diff:.8f}")

    print("\nNF4 vs dequantized NF4:")
    cross_mean_diff, cross_max_diff = compare_logits(
        nf4_logits,
        dequantized_logits,
    )

    print(f"Mean absolute logit difference: {cross_mean_diff:.8f}")
    print(f"Max absolute logit difference: {cross_max_diff:.8f}")

    del nf4_model
    clear_gpu_memory()


if __name__ == "__main__":
    main()
