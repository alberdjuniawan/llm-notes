import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "Explain photosynthesis in simple terms:"
MAX_NEW_TOKENS = 30
NUM_RUNS = 3


def clear_gpu_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


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


def benchmark(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> tuple[str, float]:
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text, elapsed


def run_experiment(
    name: str,
    loader,
    tokenizer: AutoTokenizer,
) -> None:
    clear_gpu_memory()

    print(f"\n{name}:")

    before_load = torch.cuda.memory_allocated()

    model = loader()

    torch.cuda.synchronize()

    after_load = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()

    generated_text, elapsed = benchmark(model, tokenizer)

    print(f"Memory before load: {before_load / (1024**2):.2f} MiB")
    print(f"Memory after load: {after_load / (1024**2):.2f} MiB")
    print(f"Peak GPU memory: {peak_memory / (1024**2):.2f} MiB")
    print(f"Generation latency: {elapsed:.4f} s")
    print(f"Generated: {generated_text}")

    del model
    clear_gpu_memory()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Runs: {NUM_RUNS}")

    run_experiment("FP16", load_fp16_model, tokenizer)
    run_experiment("NF4", load_nf4_model, tokenizer)


if __name__ == "__main__":
    main()
