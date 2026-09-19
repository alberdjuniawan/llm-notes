import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to("cuda")

    prompt = "Photosynthesis is"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nOUTPUT:")
    print(text)


if __name__ == "__main__":
    main()
