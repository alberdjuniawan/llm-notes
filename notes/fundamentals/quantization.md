# Quantization

Quantization is the process of representing numerical values using a lower-precision format so that a model requires less memory, storage, or computation, usually at the cost of some numerical precision.

Basic flow:

```text
FP weights -> grouping -> scale -> quantization -> packed representation -> dequantization
```

A bit is a binary digit that can contain either `0` or `1`. A representation using `n` bits provides $2^n$ possible bit patterns. For example, 4 bits provide $2^4 = 16$ possible patterns.

For floating-point representations, the bits are divided into sign, exponent, and fraction fields. FP16 uses 1 sign bit, 5 exponent bits, and 10 fraction bits, while BF16 uses 1 sign bit, 8 exponent bits, and 7 fraction bits. The exponent determines the scale of the number while the fraction provides numerical detail.

Integer quantization maps floating-point values to a smaller set of integer codes. For a simple symmetric INT4 quantization scheme, the quantization scale can be calculated as:

$$
s = \frac{\max(|w|)}{7}
$$

The quantized value is:

$$
q = round\left(\frac{w}{s}\right)
$$

and the reconstructed value is:

$$
\hat{w}=q\times s
$$

The reconstructed value is an approximation of the original floating-point weight, so quantization introduces quantization error.

Quantization can use groups of weights rather than one scale for the entire tensor. For example, with group size 128, every 128 weights share one scale. Smaller groups provide more local scaling and generally reduce quantization error, but require more scale values and therefore more metadata.

```text
larger group -> fewer scales -> lower metadata overhead -> potentially larger error

smaller group -> more scales -> higher metadata overhead -> potentially smaller error
```

INT4 codes require only 4 bits each. Two INT4 codes can therefore be packed into one byte:

```text
INT4 + INT4 -> 8 bits -> 1 byte
```

Packing only changes how the quantized codes are stored in memory. It does not change their numerical values.

NF4 is a 4-bit quantization format designed for neural network weights. Unlike the simple symmetric INT4 scheme used for the toy experiments, NF4 uses non-uniform quantization levels and was used through bitsandbytes for the model-level quantization experiment.

Weight dtype and compute dtype are separate concepts. A model can store or represent its weights using a quantized format such as NF4 while using a higher-precision dtype such as BF16 for computation.

Quantization reduces memory usage but can change intermediate values, logits, probabilities, and generated text. Therefore, model-level evaluation is required in addition to measuring weight reconstruction error.