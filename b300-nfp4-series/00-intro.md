# Unlocking NVFP4: The Journey from 32-bit to 4-bit Precision
Over the past decade, we have seen a fascinating evolution and research going on in the field of low precision computing for AI models. The focus on this topic has become more and more important as following scaling laws ([1](https://arxiv.org/abs/2001.08361),[2](https://arxiv.org/abs/2203.15556)) models have been scaled up exponentially. This bet was also backed by another bet on the GPU, TPUs and other accelerators archiecture design. To fully understand and exploit the performances promised by the hardware vendors it is essential to understand the underlying hardare choices and the tradeoffs needed to improve the performences of our training or inference workloads.
Reducing the number of bits needed to represent the model weights, activations, and gradients impacts directly the design choices for the chip, but can be beneficial in terms of speed, energy efficiency and memory and communications bandwidth. Why so? Using less bits means that less memory is needed to store the model, but also the amount of data that gets tranferred over multiple GPUs is lower making communications faster. Computing is also faster as the data is smaller and the operations are less expensive. Adding all these together has some prices to pay however, since there is no free lunch and the researchers had to find some ways to mitigate the accuracy loss and the impact of the model performance at inference time or instabilities during the training phase. 
But before delving into that, let's first have a look at what is a floating point number that we will talk a lot about in this series of posts.

In simple terms, floating point numers is a way of representing real number on a computer using a fixed number of bits. This representation allows to represent a wide dynamic range of values. When referrring to numerical representations on a machine there are two concepts that allows to understand the tradeoffs we are making when choising that particular representation: 
- Precision -> refers to the sample density on the real number line for a given represntation. If the sampling is finer we have a higher precison
- Accuracy -> measures the error between the number stored in the machine representaiton and the actual real number.

An example taken from the GPU MODE lecture, if we want to represent $\pi$ we can for instance have several distinct representations:
1. `3.14` a very approximate representation of $\pi$.
2. `3.141543` is both more precise and more accurate of `3.14`.
3. `3.142738` which is more precise than `3.14` but at the same time less accurate than `3.14`.

This examples shows clearly that the choice of the numerical representation affects a lot the outcome of the mod
For more details on these concepts checkout the [GPU model lecture on numerics](https://youtu.be/ua2NhlenIKo?si=AG-ekf7DCkAkIJAa) by [Paulius Mickevicius](https://developer.nvidia.com/blog/author/pauliusm/). 

The real number line allows for infinite precision, but silicon and memory are finite. Using a floating p   oint representation we can sample the real line and represent it using three bit fields:
1.  Sign: Positive or negative.
2.  Exponent: The dynamic range (which power of 2 is sampled).
3.  Mantissa: The precision (samples between powers of two).

The mathematical representation is defined as:

$$N_{10} = (-1)^{S} \times 1.M_{10} \times 2^{E_{10} - bias}$$

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

$$\mathbb{E}\[\text{Round}(x)\] = x$$ 
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
