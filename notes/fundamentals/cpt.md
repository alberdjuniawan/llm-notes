# Continued Pre-Training

Continued Pre-Training (CPT) is the process of continuing the training of a pretrained language model on additional text data, typically to adapt the model toward a specific domain. For a causal language model, CPT keeps the next-token prediction objective: the model learns to predict the next token from the tokens that come before it.

Basic flow:

```text
raw text -> tokenization -> model forward pass -> next-token prediction -> loss -> backpropagation -> weight update 
```