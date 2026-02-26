# What's difference between B200 and B300

The NVIDIA B200 and B300 are both part of the Blackwell architecture family. The B200 is the base model, while the B300 (also known as Blackwell Ultra) is a higher-performance variant.

**Performance numbers without sparsity** (divide by 2)

| **Technical Specifications** |              |                 |
| ---------------------------- | ------------ | --------------- |
|                              | **B200** | **B300**        |
| **FP4**                      | 9 PFLOPS     | **14 PFLOPS**    |
| **FP8/FP6**                  | 4.5 PFLOPS | 4.5 PFLOPS |
| **INT8**                     | 4.5 POPS | 0.15 POPS |
| **FP16/BF16**                | 2.25 PFLOPS | 2.25 PFLOPS |
| **TF32**                		 | 1.1 PFLOPS | 1.1 PFLOPS |
| **FP32**                		 | 0.075 PFLOPS | 0.075 PFLOPS |
| **GPU Memory**               | 180 GB HBM3E | **270 GB HBM3E** |
| **GPU Memory Bandwidth**     | 7.7 TB/s | 7.7 TB/s |
| **NVLink bandwidth per GPU** | 1.8 TB/s | 1.8 TB/s |
| **Max Thermal Design Power (TDP)** | Up to 1,000W | Up to **1,100W** |

Key Differences Between NVIDIA B200 and B300 GPUs

- **Compute Performance**: B300 dense **FP4** performance is **55.6% faster** (14 vs 9.0 petaFLOPS) compared to B200, due to higher clock speeds, optimized tensor cores, and the additional TDP headroom. Essentially no FP64 performance (1.25 TF on B300 vs. 37 TF on B200)
- **Memory and Bandwidth**: B300 has **50% more GPU memory** (270 GB HBM3E vs. 180 GB) for larger models and batches, with equal memory bandwidth.
- **Interconnect and Power**: NVLink bandwidth is the same, but B300 supports higher TDP (up to 1,100W vs. 1,000W), enabling the performance uplift but requiring better cooling (liquid-cooled systems recommended for DGX/HGX B300).
- **Compute capability:** B300 is **SM103** compare to B200 SM100.

# References

1. [White Paper, NVIDIA Blackwell Architecture Technical Brief, table 3. System Specifications, Per GPU specs](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief?ncid=no-ncid)
2. [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
3. [NVIDIA DGX B300 Datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b300-datasheet)
4. [NVIDIA DGX B200 Datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)
5. [NVIDIA B300, Glenn's Digital Garden](https://www.glennklockwood.com/garden/processors/B300#gb300-nvl72)
6. [NVIDIA Blackwell Datasheet: Individual B200 Specifications](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) 
7. [Triton [Blackwell] Revert to using inline asm for tcgen05 ops to unbreak sm103 support #8045](https://github.com/triton-lang/triton/pull/8045)
