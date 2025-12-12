# Deep Dive: The Bridge to Low Precision – DeepSeek-V3’s FP8 Strategy

Before the industry fully transitions to the 4-bit precision offered by Blackwell, it is critical to understand the current state-of-the-art in low-precision training. A prime example discussed during our EurIPS event is the **DeepSeek-V3 Mixed Precision Framework**.

While NVIDIA Blackwell aims to unlock FP4, DeepSeek has successfully demonstrated the viability of **FP8 training** at scale by solving two fundamental challenges: handling feature outliers and maintaining accumulation precision.

## The FP8 Mixed Precision Workflow
DeepSeek’s approach deviates from standard BF16 training by aggressively quantizing the heavy computational workloads (GEMMs) while preserving high precision where it matters most for convergence.

As illustrated in their technical architecture:
* **Core Computation:** The forward propagation (Fprop), gradient with respect to inputs (Dgrad), and gradient with respect to weights (Wgrad) are all executed using **FP8 GEMM** operations.
* **Precision Preservation:** Despite the heavy compute happening in FP8, the **Master Weights** and **Optimizer States** are maintained in **FP32** to ensure stable updates.
* **Data Flow:** Activations and gradients flow in BF16 between layers but are quantized to FP8 immediately before entering the compute-intensive Linear operators.

## Solving the Precision Bottleneck
The risk with lowering precision from 16-bit to 8-bit (or 4-bit) is the loss of information, particularly when "outliers"—values significantly larger than the rest of the data—skew the quantization scale. DeepSeek-V3 addresses this with two specific innovations:

### 1. Fine-Grained Quantization for Outliers
Standard tensor-wise quantization often fails because a single large outlier forces the scaling factor to cover a huge range, squashing smaller, important values to zero.
DeepSeek implements **fine-grained quantization**. Instead of a single scale for a whole tensor, they apply scaling factors to smaller tiles or blocks ($1 \times N_c$). This isolates outliers to their specific blocks, allowing the rest of the tensor to maintain high fidelity.

### 2. Increasing Accumulation Precision
A subtle but killer issue in low-precision training is underflow during the accumulation of large matrix multiplications. If you sum thousands of small FP8 products, the precision loss accumulates.

DeepSeek mitigates this by using a hybrid accumulation strategy:
* **Tensor Core Speed:** The bulk of the multiplication happens on Tensor Cores.
* **CUDA Core Precision:** Crucially, they promote the accumulation to **CUDA Cores** at an interval of $N_c = 128$ elements.
* **The Result:** By periodically moving the partial sums to high-precision FP32 registers on CUDA cores, they prevent the quantization noise from drowning out the signal during the summation phase.

## Why This Matters for NVFP4
DeepSeek’s success with FP8 validates the "recipe" that NVIDIA is now pushing further with Blackwell and NVFP4. The move to block-based scaling (DeepSeek’s fine-grained quantization vs. NVFP4’s 16-element block scaling) and careful management of accumulation precision are the foundational techniques that allow us to push the boundaries of compute density without sacrificing model intelligence.