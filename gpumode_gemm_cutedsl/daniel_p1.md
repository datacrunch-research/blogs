# Huang's Law, not Moore's Law

Moore's Law is slowing its growth, so the performance curve of a single GPU chip will eventually reach saturation. However, the demand for computing power continues to grow, necessitating the development of more efficient GPUs. Jensen Huang, NVIDIA's CEO, notably observed that the rate of advancements in GPUs are growing much faster than the rate of CPUs. As the semiconductor industry pushes the physical limits of chips, they look on the inside and decrease the digital footprint of each bit. .

This is Part 1 of a two-part series on writing a GEMM kernel for Blackwell B200 using CuTeDSL. In Part 1, we’ll look at why the industry is pushing into ultra-low precision as AI usage climbs well past 100T tokens [1], and what Blackwell changed to make NVFP4 practical. Part 2 will be writing CuTeDSL to build a GEMM kernel on B200.


## Making Data Smaller

NVFP4 is an innovative 4-bit floating point format introduced with the NVIDIA Blackwell GPU architecture. NVFP4 builds on the concept of low-bit “micro” floating-point formats and grants greater flexibility to developers by providing an additional format to choose from.

One of the crucial points we have to keep in mind is that on a machine we have to deal with a fixed budget represented by the number of bits used. To understand the tradeoffs, we must distinguish between three concepts that depend on how we allocate the bits of the representation:

- **Dynamic range**, controlled by the **exponent** (E) bits, determines the scale of the number we are trying to represent, aka how large or how small a number can be (e.g., from $10^{-45}$ to $10^{38}$). With more E bits, we can represent a wider range, reducing the risk of overflow or underflow.
- **Precision**, controlled by the **mantissa** (M) bits, refers to the density of samples on the real number line $\mathbb{R}$.
- **Accuracy** measures the error between the stored number in the chosen representation and the actual real number.

![](figures/fp_00.png)
**Figure 1.** *The figure summarizes the different floating point formats discussed in this post. Chronologically: FP32, BF16, FP8 which usually uses just with a tensor-scale factor. The `MXFP*` formats use a 32-element block level `E8M0` (just the exponent of an FP32) scale factor, and finally NVFP4, which uses a combination of 16-element block-level fractional scaling `E4M3` and a full FP32 tensor-level scaling.*

> **Measuring Dynamic Range with Binades**
> 
> A **binade** is one power of 2 of dynamic range—essentially measuring how many "doublings" fit between the smallest and largest representable values:
> 
> $$\text{binades} = \log_2\left(\frac{\text{max representable}}{\text{min representable}}\right)$$
> 
> | Format | Exponent Bits | Binades |  |
> |--------|---------------|---------|-------------|
> | FP32 | 8 | ~277 | Good for most computations |
> | FP16 | 5 | ~40 | Good for most activations |
> | BF16 | 8 | ~261 | Same FP32 range, but limited precision |
> | FP8 E4M3 | 4 | ~18 | Suitable for forward pass |
> | FP8 E5M2 | 5 | ~32 | Needed for backward pass |
> | FP4 E2M1 | 2 | ~3.6 | Needs additional tricks to work |
> 
> FP4's 3.6 binades cannot represent typical tensor value distributions, which often span 10-20 binades. This is precisely why block scaling becomes essential at 4-bit precision.

As an example, if we want to represent $\pi$ with a fixed number of decimal digits we will end up with different approximations. We know that $\pi = 3.141592653\dots$, if we were limited with a budget of three digits we can have two choices:

1. $\hat{\pi}_1 = 3.141$, which has an absolute error of $|\pi - \hat{\pi}_1| \approx 0.00059$,
2. $\hat{\pi}_2 = 3.142$, which has an absolute error of $|\pi - \hat{\pi}_2| \approx 0.00041$.

Both approximations are using the same budget in terms of digits but they are achieving a different accuracy in representing the value we want to use in our computations.  
If we would like to use more digits, we could end up with more accurate representations but sometimes also less accurate even if more precise representations. Let's clarify with an example. If we use six digits we could for instance end up storing:

3. $\hat{\pi}_3 = 3.141543$, which has an absolute error of $|\pi - \hat{\pi}_3| \approx 0,00005$,
4. $\hat{\pi}_4 = 3.142738$, which has an absolute error of $|\pi - \hat{\pi}_4| \approx 0.00115$.

As we can see in Figure 2, the value we $\hat{\pi}_i$ try to capture a good approximation of the real value of $\pi$ and varying the budget for our representation we will end up with different solutions and tradeoffs.
This simple example shows clearly that the choice of numerical representation greatly affects the outcome of computations. 

![](figures/fp_01.png)
**Figure 2.** *Illustration of precision vs. accuracy using different approximations of $\pi$. A more precise representation (more decimal digits) does not necessarily mean higher accuracy (closer to the true value). The three examples show: `3.141` (low precision, moderate accuracy), `3.141543` (higher precision and accuracy), and `3.142738` (higher precision but lower accuracy than `3.141`).*



The real number line allows for infinite precision, but silicon and memory are finite. Using a floating point representation, we can sample the real line using three bit fields:

1. **Sign (S):** Positive or negative.
2. **Exponent (E):** The dynamic range (which power of 2 is used).
3. **Mantissa (M):** The precision (samples between powers of two).

The mathematical representation is defined as:

$$
\begin{equation}
N = (-1)^{S} \times 1.M \times 2^{E - \text{bias}}
\end{equation}
$$

Let's break down the formula:

- The sign bit (`S`) determines if the number is positive (`S = 0`) or negative (`S = 1`).
- The exponent (`E`) is an integer representing the power of 2, adjusted by the $\text{bias}$ term. The exponent gives us the dynamic range, meaning which slice of the real number line we are sampling.
- The mantissa (`M`) or significand is a binary number representing the precision; if the exponent gives us the scale, the mantissa tells us which sample we are taking from that slice of the real number line.

In normalized floating point representation, the significand always starts with an implicit leading `1` (this is why it's called "normalized"). The mantissa bits, e.g., `1001001000`, represent the fractional digits that come after this implicit `1`, forming the complete significand `1.1001001000` in binary. Each bit position corresponds to a negative power of 2: the first bit after the decimal point represents $2^{-1} = 0.5$, the second $2^{-2} = 0.25$, the third $2^{-3} = 0.125$, and so on.

### Microscaling (MX) Formats

While DeepSeek-V3 demonstrates that FP8 is viable with careful engineering, the desire for efficiency pushed AI workloads toward even smaller formats like 6-bit or 4-bit. At these precisions, standard per-tensor scaling breaks down. A single large outlier in a tensor of millions of parameters can skew the quantization scale, effectively pushing all smaller values to zero.

To solve this, a consortium of tech companies, including AMD, Arm, Intel, NVIDIA, and Qualcomm, aligned under the Open Compute Project (OCP) to introduce the specification of the Microscaling (MX) formats [2].

The core idea is moving from per-tensor to per-block scaling. Instead of assigning one scaling factor to an entire tensor, the tensor is divided into small blocks (e.g., 32 elements), each with its own shared 8-bit scale exponent.  
How it works:
1. **Block grouping:** Elements are grouped into blocks of $k$ elements (typically $k=32$).
2. **Shared per-block scale:** The hardware finds the maximum absolute value in each block to determine a shared 8-bit exponent.
3. **Local quantization:** Individual elements are quantized to 4 bits relative to their block's scale.

A simple example (real blocks use 32 elements): Consider a block of 4 values: `[0.001, 0.002, 100.0, 0.003]`. With per-tensor scaling, the scale would be dominated by `100.0`, and the small values would all round to zero. With per-block scaling, if this block gets its own scale, the outlier only affects these 4 neighbors; the rest of the tensor remains well-quantized. 
This compartmentalization of numerical noise is the key breakthrough enabling training at 4-bit precision.

### NVFP4
Building on the MX foundation, NVIDIA developed NVFP4 for their Blackwell architecture, adding hardware-specific refinements to push the limits of low-bit training.  

NVFP4 is a 4-bit floating point format (`E2M1`):
- Sign: 1 bit
- Exponent: 2 bits
- Mantissa: 1 bit (plus one implicit)

With only **16 unique values** available in a 4-bit representation, careful scaling becomes critical. To put this in perspective: FP32 can represent ~4 billion distinct positive values; NVFP4 can represent 8, approximately -6 to 6. For example, the values in the range could include 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6 (same for the negative range).

While the OCP MX specification suggests 32-element blocks, NVIDIA chose finer granularity: **16-element blocks**. By calculating the shared scale factor over fewer elements, NVFP4 confines outliers more tightly—a single spike distorts a smaller neighborhood, preserving fidelity in surrounding weights.

![](figures/nvfp4.png)
**Figure 5.** *A 16×32 matrix stored in NVFP4 format. Each block contains 16 contiguous FP4 elements (gray and green) with a shared FP8 scale factor (yellow). The largest magnitude element in each block (green) is scaled to the FP4 maximum representable value. A per tensor FP32 scale factor is also applied (not shown). Source [3].*

Hardware support is only half the story. Training a model in 4-bit precision without diverging into noise requires specific algorithmic interventions, as detailed in NVIDIA's paper "Pretraining Large Language Models with NVFP4" [3].

**1. 2D Block Scaling**  
Scaling is applied along both **row-wise** and **column-wise** dimensions for weight matrices (16×16 blocks). Why both? During forward pass, scaling happens along rows; during backward pass, tensors are transposed, so scaling happens along columns. Without 2D scaling, the same weight would have two different quantized representations, breaking the chain rule and degrading training quality.

**2. Random Hadamard Transform (RHT)**  
One of the biggest enemies of quantization is "outlier features"—specific neurons that consistently fire with massive values. These outliers can wreck the quantization scale for their entire block.  
The Random Hadamard Transform "smears" outlier information across the entire vector *before* quantization:
- **Before RHT:** One massive value, many small ones → hard to quantize
- **After RHT:** Many medium values → efficient quantization 

This mathematical operation redistributes energy so that no single element dominates the scale calculation.

**3. Stochastic Rounding (SR)**  
With very few bits, standard "round-to-nearest" creates systematic bias. Always rounding 0.4 down to 0 accumulates massive error over billions of operations.

NVFP4 uses **stochastic rounding**, which rounds probabilistically based on the distance to the nearest representable values:

$$\mathbb{E}[\text{Round}(x)] = x$$

$$
\text{Round}(x) = 
\begin{cases} 
    \lfloor x \rfloor & \text{with probability } 1-p \\ 
    \lceil x \rceil & \text{with probability } p
\end{cases}
$$

where $p = \frac{x - \lfloor x \rfloor}{\lceil x \rceil - \lfloor x \rfloor}$

This ensures that **on average**, the expected value of the rounded number equals the original. Over many operations, rounding errors cancel out rather than accumulate in one direction, allowing gradient descent to converge correctly despite the severe quantization.

![](figures/nvfp4_training.png)
**Figure 6.** *Illustration of the compute flow for an NVFP4 quantized linear layer. All GEMM operations quantize their inputs to NVFP4. Source [3].*


<!--
Now that we've described the amalgamation of mathematical tricks and floating points, we need to understand how these bits are supposed to move on the prescribed architecture. NVFP4 was introduced for the B100, B200 and B300 chips. 

History of NVDA GPU Architecture
* TensorCores 
* Shared Memory (SMEM) / Global Memory (GMEM)
* L1/L2 Cache
* DSM
* Tensor memoery through archictecute -> increase in bandwidth and inflith efficiency
```quote
"What I cannot create, I do not understand" Richard Feynman
```
-->

# GPU Architecture
A prerequisite for writing performant GPU kernels is to understand the hardware architecture inside the GPU and how data flows through it. A GPU is optimized for exactly two operations: arithmetic operations on data, and the movement/storage of that data between hierarchical memory pools. The efficiency of the former is bounded by transistor physics; the efficiency of the latter is bounded by the speed of light and wire capacitance. 

```Notice
Notice, the following information is based on the Blackwell architecture of the B200 GPU, and all further mentions of 'GPU' refer to the B200. An example of matrix multiplication is the following:
D = A * B or D = A * B + C
```




# Movement of Data: Physical Hierarchy
The memory architecture of modern GPUs is a spatially organized response to the inverse relationship between latency and density in CMOS design. If they could, NVIDIA would place every compute unit adjacent to register-speed memory. Instead, contemporary GPUs pack 10^4-10^5 threads across 148 Streaming Multiprocessors (SMs) on 2 distinct dies, connected by a 10TB/s NV-HBI interconnect. The GPU places the fastest and smallest memory closest to compute hardware, and iteratively hosts larger but slower memory further away, maximizing aggregate throughput by minimizing latency.

Memory in the GPU consists of the following:
1) Register File (RMEM): The fastest storage, placed adjacent to SMs. Each SM has 256KB of register files, summing to ~37.75MB across the GPU. Each SM has the ability to host up to 64 warps (2,048 threads); yielding 128 bytes of register storage per thread when fully populated.

2) Tensor Memory (TMEM): An update introduced to the Blackwell architecture, containing 256KB per SM of dedicated SRTAM accessible by Tensor Cores (more on these later). These play an important role in GEMM, so visualizing them is crucial. They are 2D matrices, 512 columns and 128 rows, or lanes, of 32-bit cells. TMEM functions as a loading dock for matrix multiply accumulate (MMA) tiles. Their introduction abstracts away from hardware cache prediction and gives the ability to manually control access patterns of tensor tiles. TMEM allows matrix A to be located in TMEM or SMEM, matrix B must be in SMEM, and the accumulator must be in TMEM.
![](figures/tensor-memory-layout.png)**Figure 3 - TMEM** Source [4].

3) Shared Memory (SMEM) and L1 Cache: Unified 256KB SRAM structure per SM. Percentages of how much data each structure has can be manually controlled.

4) L2 Cache: 192MB SRAM that is physically split across the dual-die boundary; each die maintains partitions local to its 74 SMs, with coherency traffic traversing the NV-HBI interconnect. 

5) Tensor Memory Accelerator (TMA): Introduced in Hopper to offload address-generation from the register file via descriptor-based async copy; refined in Blackwell with second-generation sub-byte quantization (FP4/FP6 unpacking) and CTA-pair semantics. This change in Blackwell architecture facilitates thread blocks sharing distributed SMEM and executing cooperative MMA across paired SMs.

6) VRAM (GMEM): The furthest and slowest memory, 192GB of HBM3e stacked on the same substrate as the GPU dies. The GPU implements 8 memory stacks (4 per die) delivering 8 TB/S of aggregate bandwidth. 

# Computation of Data: Visualizing the bits flow
All of the storage hardware mentioned above (excluding L2 and GMEM) is placed inside the SM. Inside each SM is more advanced hardware that allows us to do matmuls very fast.
![](figures/sm_breakdown.webp)
**Figure 1 - Look inside Blackwell SM** Source [5].

1) Tensor Cores: Tensor Cores are the fundamental parts of the GPU that facilitate MMA (matrix multiplication accumulation) instructions on small tiles. NVIDIA introduced these hardware cores in 2017 to rival Google's 2016 release of their systolic array TPUs. The earliest Tensor Core architecture, Volta, was only able to handle FP16 data types, matrix sizes of 4x4x4, was not sparse, and received data slowly from SMEM and register files (RF). Four architectural generations later, each iteration increased the computation-to-memory ratio and added support for smaller precision types. The 5th generation of Tensor Cores can handle low-precision FP4, use CTA pairs on a single SM, and include TMEM to reduce pressure on SMEM.

2) Warp Schedulers: A warp consists of 32 threads, and the scheduler issues instructions to all 32 per clock cycle.

3) LD/ST Units: Instructions that are responsible for loading and storing data. All active threads in a warp always issue the same type of instruction in the same clock cycle. If that instruction is a load or store, it gets issued to the LD/ST units. If a thread is inactive (due to looping or conditional execution), the corresponding LT/ST unit stays idle.
 

<!--![](figures/thread_block_cluster.webp)
**Figure 1. Thread blocks** Source [6].

blackwell tensorcores [7]

tensorcore architecture and big O notation [8]-->

## Links

1. https://openrouter.ai/state-of-ai
2. https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
3. https://arxiv.org/abs/2509.25149
4. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-addressing
5. https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
6. https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/
7. https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493056&idx=1&sn=1c6025f97df16a3b9576746b7944538e&chksm=f995f502cee27c145677e91761ebec0d59cde82d4562c463070f2c72efb3751567bac934c266&scene=178&cur_album_id=2538479717163761664&search_click_id=#rd
8. https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491424&idx=1&sn=0fc2110931b27714900e78d73b11a5b5&scene=21&poc_token=HOHnjWmj8OCvlO9eiSxuNJMDxVEsmgh7A4q_qNIq
