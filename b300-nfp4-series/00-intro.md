# Unlocking FP4: The Future of Low Precision AI on NVIDIA Blackwell

**By Verda (formerly DataCrunch)** *Insights from EurIPS Copenhagen 2025*

The AI cloud landscape is shifting fundamentally. While legacy cloud infrastructure was built to optimize developer time, the AI-native cloud must optimize for performance, energy density, and compute costs. As models grow exponentially larger, the industry is racing to solve a critical bottleneck: how to increase compute efficiency without sacrificing accuracy.

At our recent event at EurIPS 2025 in Copenhagen, we explored the next leap in this evolution: **Low Precision AI and the arrival of FP4 on NVIDIA Blackwell**.

## The March Toward 1000x Performance
Over the last decade, single-chip inference performance has increased by roughly 1000x. This massive jump wasn’t achieved through a single breakthrough but a combination of factors:
* **Process Innovations:** The shrink from 28nm to 5nm, packing more transistors per $mm^2$.
* **Amortized Instructions:** The evolution of Tensor Cores from FMA to HMMA, WGMMA, and UMMA.
* **Number Representation:** The shift from FP32 to BF16, then FP8, and now FP4.

With the introduction of the NVIDIA Blackwell architecture (B200/B300), we are looking at a potential 3500x increase over 15 years, largely driven by the unlocking of 4-bit floating point (FP4) precision.

## Why Precision Matters: The Finite Reality
To understand why FP4 is revolutionary, we have to look at how computers represent numbers. The real number line allows for infinite precision, but silicon and memory are finite. We must sample the real line using "floating point" representations, which consist of three bit fields:
1.  **Sign:** Positive or negative.
2.  **Exponent:** The dynamic range (which power of 2 is sampled).
3.  **Mantissa:** The precision (samples between powers of two).

The mathematical representation is defined as:
$$ 
N_{10} = (-1)^{S} \times 1.M_{10} \times 2^{E_{10} - bias} 
$$

Historically, deep learning relied on FP32 (32-bit). Over time, the industry realized that deep learning models are surprisingly resilient to noise, allowing us to move to FP16 and BF16 (2016-2017). By 2022, the Hopper architecture introduced FP8. Now, Blackwell is pushing the boundary to **NVFP4**.

## Introducing NVFP4: Efficiency Without Compromise
NVFP4 is a 4-bit format consisting of **1 sign bit, 2 exponent bits, and 1 mantissa bit (E2M1)**.

The immediate challenge with 4-bit precision is the limited number of representable values—essentially 16 values to represent complex model weights. If you simply quantized a standard distribution to 4 bits, you would lose significant accuracy, particularly with outliers.

### The Secret Sauce: Block Scaling
To solve this, NVFP4 utilizes **Block Scaling**. Instead of scaling the entire tensor broadly, scaling factors are applied to small blocks of elements.

While the Open Compute Project (OCP) MXFP4 specification uses a block size of 32 elements, NVIDIA’s NVFP4 uses a finer granularity of **16 elements**.
* **Smaller Blocks:** A block size of 16 allows the format to better capture outliers.
* **Higher Accuracy:** This minimizes the number of values quantized to zero, maintaining model fidelity.

## The NVFP4 Training Recipe
Training in FP4 requires a sophisticated "recipe" to ensure convergence. The Blackwell architecture supports this through a specific pipeline:

1.  **2D Scaling:** Scaling factors are applied row-wise and column-wise to maximize dynamic range.
2.  **Stochastic Rounding (SR):** Unlike "nearest" rounding, SR rounds probabilistically. The formula preserves the cexpected value of the number: 
$$ 
\mathbb{E}[\text{Round}(x)] = x 
$$ .
$$
\text{Round}(x) = 
\begin{cases} 
    \lfloor x \rfloor, & \text{w/ prob. } 1-p \\ 
    \lceil x \rceil, & \text{w/ prob. } 
\end{cases}
$$

$$p = (x - \lfloor x \rfloor) / (\lceil x \rceil - \lfloor x \rfloor)$$

3.  **Random Hadamard Transforms:** This technique spreads outlier information across the vector, preventing specific weights from dominating the quantization error.

## From DeepSeek V3 to the Future
We are already seeing the industry move toward lower precision. Frontier labs like DeepSeek have successfully implemented FP8 mixed precision training, utilizing fine-grained quantization to mitigate errors caused by feature outliers.

NVFP4 on Blackwell represents the next step. By combining the E2M1 format with 16-element block scaling and stochastic rounding, we can achieve the speed and memory efficiency of 4-bit compute while retaining the necessary accuracy for frontier models.

At Verda, we are building the AI-native cloud to support these next-generation workloads, ensuring that infrastructure keeps pace with the rapid evolution of algorithmic efficiency.



---- 
