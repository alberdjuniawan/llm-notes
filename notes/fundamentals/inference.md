# Inference

Inference is the process of using a trained model to produce predictions or generated output for new input. For a causal language model, generation is autoregressive: the model predicts the next token based on the tokens that have already been processed.

Basic flow:

```text
input -> tokenization -> model forward pass -> next-token prediction -> next token -> repeat -> output
```