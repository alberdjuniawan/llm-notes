import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def gpu_memory_mb() -> float:
    return torch.cuda.memory_allocated() / (1024**2)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )

    print(f"Model loaded on: {model.device}")
    print(f"CUDA memory allocated: {gpu_memory_mb():.2f} MB")

    quantized_parameters = 0

    for parameter in model.parameters():
        if parameter.__class__.__name__ == "Params4bit":
            quantized_parameters += parameter.numel()

    print(f"Parameters represented by Params4bit: {quantized_parameters:,}")

    prompt = "Explain photosynthesis in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
