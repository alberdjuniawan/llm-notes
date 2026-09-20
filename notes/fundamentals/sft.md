# Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) is the process of training a pretrained language model on supervised instruction-response examples so that the model learns how to follow specific tasks and produce desired responses.

Basic flow:

```text
instruction + response -> tokenization -> model forward pass -> supervised loss -> backpropagation -> weight update
```