# Experiment 004: Quantization

This report contains the results and analysis of the experiment described in [`experiments/004-quantization/README.md`](../experiments/004-quantization/README.md).

## INT4 Quantization

The first experiment implemented a simple symmetric INT4 quantization scheme on manually defined floating-point weights.

The quantization scale was calculated using:

$$
s = \frac{\max(|w|)}{7}
$$

The quantized values were calculated as:

$$
q = round\left(\frac{w}{s}\right)
$$

and reconstructed using:

$$
\hat{w}=q\times s
$$

For the weights:

```text
[-0.80, -0.31, 0.10, 0.12, 0.15, 0.80]
```

the calculated scale was approximately `0.114286` and the resulting INT4 codes were:

```text
[-7, -3, 1, 1, 1, 7]
```

The reconstructed values were:

```text
[-0.8000, -0.3429, 0.1143, 0.1143, 0.1143, 0.8000]
```

The mean absolute error was `0.014762` and the maximum absolute error was `0.035714`.

The result demonstrates that multiple original floating-point values can map to the same INT4 level, causing quantization error.

## Group Quantization

The second experiment examined the effect of group size on quantization error.

The test weights were:

```text
[-0.80, -0.31, 0.10, 0.01, 0.02, 0.04]
```

Using one group for all six values resulted in one scale determined by the largest magnitude weight, `0.80`. The resulting INT4 codes were:

```text
[-7, -3, 1, 0, 0, 0]
```

The smallest weights were mapped to zero because the scale was dominated by the much larger weight.

The mean absolute error was `0.019524`.

Using group size 3 divided the values into two groups:

```text
[-0.80, -0.31, 0.10] -> scale 1

[0.01, 0.02, 0.04] -> scale 2
```

The second group received a much smaller scale, allowing the smaller values to retain more numerical detail.

The mean absolute error decreased to `0.008571`.

This demonstrates why group quantization is useful: smaller groups provide more local scaling and generally reduce quantization error, but require more scales.

## Real Model Weight Quantization

A real weight tensor was extracted from the first Transformer layer:

`model.layers.0.self_attn.q_proj.weight`

The tensor had shape `(576, 576)` and contained `331,776` elements.

The observed statistics were:

```text
Minimum: -4.750000
Maximum:  5.218750
Mean:     0.000089
Std:      0.281865
```

The tensor was quantized using symmetric INT4 quantization with different group sizes.

```text
Group size    Mean abs error    Max abs error
128           0.03210156       0.36858261
64            0.02824641       0.36858261
32            0.02429663       0.36858261
```

The mean absolute error decreased as group size became smaller. However, the maximum absolute error remained unchanged for the three tested group sizes. This demonstrates that average reconstruction error and worst-case error do not necessarily change in the same way.

## Memory Analysis

The model contains `134,515,008` parameters.

Ignoring scales and other metadata, the theoretical raw weight storage is:

```text
FP32: 513.13 MiB
FP16: 256.57 MiB
INT8: 128.28 MiB
INT4:  64.14 MiB
```

INT4 therefore requires one eighth of the raw weight storage of FP32.

When FP16 scales are included, the estimated effective storage becomes:

```text
Group 128: 4.125 bits/param
Group  64: 4.250 bits/param
Group  32: 4.500 bits/param
```

Smaller groups require more scale values, increasing metadata overhead.

## INT4 Packing

The packing experiment demonstrated that two INT4 values can be stored in a single byte.

For example:

```text
-7 -> 1001
-3 -> 1101

packed -> 10011101
```

The six INT4 values:

```text
[-7, -3, 1, 1, 1, 7]
```

were packed into three bytes:

```text
[217, 17, 113]
```

with hexadecimal representation:

```text
['0xD9', '0x11', '0x71']
```

Unpacking reproduced the original INT4 values exactly.

This experiment also showed that storing INT4 codes inside a `torch.int8` tensor does not provide true 4-bit storage. The container still uses 8 bits per element, while actual packed INT4 storage combines two 4-bit values into one byte.

## NF4 Model Loading

The SmolLM2-135M model was successfully loaded using 4-bit NF4 quantization through bitsandbytes.

The model was configured with:

```text
Quantization: NF4
Weight representation: 4-bit
Compute dtype: BF16
```

The quantized model successfully performed inference on the NVIDIA RTX 3050 Laptop GPU.

The model contains 134,515,008 parameters in total. An inspection of the quantized model reported `53,084,160` parameters represented by the bitsandbytes `Params4bit` type. This value represents only the parameters using that quantized representation and should not be interpreted as the model's total parameter count.

## BF16 vs NF4

The final evaluation compared the pretrained model loaded in BF16 with the same pretrained model loaded using NF4 weight quantization and BF16 compute.

The evaluation used five fixed prompts and deterministic decoding.

Peak GPU memory usage was:

```text
BF16: 266.39 MiB
NF4:  120.97 MiB
```

The measured memory reduction was:

$$
\frac{266.39-120.97}{266.39}\times100\%
=54.59\%
$$

Therefore, NF4 reduced peak GPU memory usage by approximately `54.59%` under the evaluation setup.

## Generation Latency

Generation latency was measured using one warmup generation followed by three timed runs for each prompt. The median timing was used for each prompt.

The average latency was:

```text
BF16: 0.3383 s
NF4:  0.4665 s
```

The relative latency change was:

$$
\frac{0.4665-0.3383}{0.3383}\times100\%
=37.88\%
$$

NF4 was therefore approximately `37.88%` slower than BF16 in this local evaluation.

The result shows that lower weight precision does not automatically produce lower latency. Actual inference performance depends on the hardware, kernels, memory movement, dequantization behavior, and runtime implementation.

The bitsandbytes environment also reported:

```text
No prebuilt binary for CUDA 12.9, loading CUDA 12.8 instead.
```

This compatibility detail limits how broadly the latency measurement should be generalized.

## Logit Comparison

A logit is the raw score produced by the model for each possible next token before the scores are converted into probabilities.

The final evaluation compared the logits from BF16 and NF4 for five prompts.

The aggregate results were:

```text
Average mean abs logit error: 0.716722
Average max abs logit error : 3.838281
Top-1 agreement rate        : 100.00%
Average top-5 overlap       : 64.00%
Average KL divergence       : 0.212259
```

The `100.00%` top-1 agreement means that the highest-scoring next token after each of the five prompts was the same for BF16 and NF4.

This does not mean that the complete generated sequences were identical. Generation is autoregressive, so later predictions depend on previously generated tokens.

The `64.00%` average top-5 overlap shows that the broader ranking of likely next tokens differed even when the top-ranked token remained the same.

The non-zero logit error and KL divergence demonstrate that NF4 changes the model's output distribution relative to BF16.

## Generation Behavior

The BF16 and NF4 models produced different deterministic generations for the evaluation prompts.

For the photosynthesis prompt, BF16 produced a concise explanation beginning with the process of photosynthesis and the role of sunlight. NF4 produced a different explanation and showed repeated phrasing.

Other prompts also showed differences in wording and continuation structure.

These differences are consistent with changes introduced by weight quantization, because quantized weights are approximations of the original weights. However, the experiment does not establish that every observed repetition pattern was caused solely by NF4, because the model is small and base-model generation behavior can also be repetitive.

## NF4 Dequantization

A separate experiment investigated what happens when the NF4 model is dequantized.

The initial comparison between BF16 and NF4 produced:

```text
Mean absolute logit difference: 0.68603063
Max absolute logit difference:  3.68750000
```

After calling `dequantize()`, the environment reported that the modules were dequantized to BF16.

The comparison between the original BF16 model and the dequantized model produced:

```text
Mean absolute logit difference: 0.69329095
Max absolute logit difference:  3.78906250
```

The comparison between NF4 and the dequantized model produced:

```text
Mean absolute logit difference: 0.04242988
Max absolute logit difference:  0.23437500
```

The dequantized model therefore remained numerically much closer to the NF4 model than to the original BF16 model. This demonstrates that dequantization does not restore the original weights because information lost during quantization cannot be recovered.

## Final Evaluation

The final evaluation used five fixed prompts:

```text
Explain photosynthesis in simple terms:
What is a fraction?
Explain how rain forms:
What is the difference between a CPU and a GPU?
Why does the Moon have phases?
```

Each prompt was evaluated using BF16 and NF4 models with BF16 compute and deterministic decoding.

The final evaluation data was saved to:

`reports/004-quantization-evaluation.json`

The evaluation combined memory usage, generation latency, numerical logit differences, token ranking similarity, and generated text so that quantization was evaluated from both system and model-behavior perspectives.

## Interpretation

The experiments provide several pieces of evidence about the effect of quantization.

First, INT4 reduces raw weight storage substantially. The theoretical weight storage for the 134,515,008-parameter model decreases from approximately `513.13 MiB` in FP32 to `64.14 MiB` in INT4 before scale and metadata overhead.

Second, group size affects reconstruction error. The real model weight tensor showed lower mean absolute error with smaller groups because each group could use a more local scale.

Third, actual NF4 model loading substantially reduced measured peak GPU memory usage. The final evaluation showed a `54.59%` reduction compared with BF16.

Fourth, memory reduction did not translate into lower latency in this environment. NF4 was approximately `37.88%` slower than BF16 in the local benchmark.

Fifth, quantization changed the model's numerical outputs. The logits differed between BF16 and NF4, the average top-5 overlap was `64.00%`, and generated text differed across the evaluation prompts.

These observations illustrate the central quantization trade-off: lower-precision weight representation can reduce memory requirements substantially while introducing numerical approximation and potentially changing inference performance and model behavior.

## Limitations

This experiment was designed to study quantization mechanics and practical trade-offs rather than establish a general-purpose quantization benchmark.

The model has only 135 million parameters, and the final evaluation used five manually selected prompts. No perplexity measurement or task-specific benchmark accuracy was performed.

The latency results were collected on a single RTX 3050 Laptop GPU and were affected by the local bitsandbytes CUDA compatibility path.

The toy experiments used a simple symmetric INT4 quantizer, while the actual model experiment used NF4. These are different quantization schemes and the toy results should therefore be interpreted as conceptual demonstrations rather than direct measurements of NF4 behavior.

## Conclusion

The experiment successfully demonstrated the quantization pipeline from bit representation and symmetric INT4 quantization through scale calculation, group quantization, dequantization, INT4 packing, and real 4-bit NF4 LLM inference.

The toy experiments showed why scale and group size matter, while the real model experiment demonstrated that smaller groups reduced average reconstruction error on a Transformer weight tensor.

The model-level evaluation showed that NF4 substantially reduced GPU memory usage while preserving functional inference. However, the quantized model produced different logits and generated text and was slower than the BF16 baseline in the local benchmark.

The experiment therefore demonstrates the practical trade-off of low-bit inference: significant memory savings can be achieved by reducing weight precision, but numerical fidelity and runtime performance must be evaluated rather than assumed.