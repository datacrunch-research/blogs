
# DeepSeek V3 SGLang Optimizations

Continuing our technical series on DeepSeek V3 integration with SGLang, we aim to give a comprehensive overview of the various optimization strategies available for enhancing performance and efficiency. The ultimate goal is to achieve competitive inference performance based on native optimizations for the DeepSeek V3 model family (Including R1) at the same time we aim to generate expertise in frontier improvements of LLMs. As an inference serving engine, SGLang interacts with multiple components of the ML infrastructure stack, providing opportunities for optimization at different levels. Most of the optimizations come in the form of flags for the `launch_server` CLI. These flags provide a convenient entry point into understanding the various performance enhancements that have been implemented over time in the SGLang ecosystem.

## Summary table of optimizations

| Optimization                              | Description / Benefit                                             | Related Flags / Notes                                              |
|------------------------------------------|-------------------------------------------------------------------|--------------------------------------------------------------------|
| **CUDA Graph Execution**                 | Reduces kernel launch overhead by replaying recorded CUDA ops     | `--cuda_graph_max_bs`, `--disable_cuda_graph`                      |
| **Torch Compile**                        | Applies kernel fusion, operator elimination, and graph optimizations | `--enable-torch-compile`, `--torch-compile-max-bs`             |
| **BF16 / FP8 BMM Kernels**               | Memory-efficient batch matmul with high accuracy                  | No flags (internal kernel optimization)                            |
| **NextN Speculative Decoding (EAGLE-2)** | Parallel speculative token generation with tree-based verification | `--speculative-algo`, `--speculative-draft`, `--speculative-*`     |
| **DP Attention for MLA**                | Enables data parallel attention for Multi-Head Latent Attention   | `--enable-dp-attention`                                            |
| **Overlap Scheduler**                    | Overlaps CPU scheduling with GPU execution to reduce idle time    | `--disable-overlap-schedule`                                      |
| **FlashInfer MLA Optimization**          | Fused MLA operations for faster prefill and decoding              | `--enable-flashinfer-mla`                                          |
| **FP8 Accuracy Improvements**            | Blockwise/tilewise scaling, FP32 accumulation to reduce overflow  | No flags (handled inside kernel)                                   |
| **FP8 GEMM Kernel Tuning**              | Selects optimal block shapes per GPU for the most optimal FP8 performance     | Script: `quantizationtuning_block_wise_fp8.py`                     |
| **FP8 GEMM (CUTLASS Kernel)**            | Efficient fused quantization and matrix multiplication            | No flags (kernel-level implementation)                             |
| **Fused MoE Kernel + Tuning**            | Faster Mixture of Experts with custom SGLang kernel tuning        | Script: `tuning_fused_moe_triton.py`                               |


## Kernel Execution Optimizations


#### Related flags:

```bash
--disable_cuda_graph: # Disable cuda graph.
--cuda_graph_bs: # The batch sizes to capture by `CudaGraphRunner`.
--cuda_graph_max_bs: # Adjust the maximum batchsize when using CUDA graph.
--enable-torch-compile: # Enable a torch.compile compilation of the CUDA graphs captured.
```

#### **Context:**

Both the [Cuda graph](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) and [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) flags deal with improving the efficiency of kernel operations. CUDA graphs significantly reduce kernel launch overhead by recording and replaying sequences of CUDA operations as a single unit, eliminating the per-kernel launch costs during inference. Meanwhile, `torch.compile` employs kernel fusion, operator elimination, and specialized kernel selection to optimize the computational graph. However, SGLang `torch.compile` can use either the PyTorch-generated graph or the CUDA graph to link the two optimizations.

#### **Commits:**

([Support CUDA graph for triton backend](https://github.com/sgl-project/sglang/pull/1401), [Support cuda graph for DP attention #2061](https://github.com/sgl-project/sglang/pull/2061))


#### **Benchmarks:**

```bash
$ python3 -m sglang.bench_one_batch --batch-size 1  --input 256
--output 32 --model deepseek-ai/DeepSeek-V3  --trust-remote-code  --tp 8
--torch-compile-max-bs 1 --disable-cuda-graph
--profile

$ python3 -m sglang.bench_one_batch --batch-size 1  --input 256
--output 32 --model deepseek-ai/DeepSeek-V3  --trust-remote-code  --tp 8
--torch-compile-max-bs 1 --cuda-graph-max-bs 1
--profile

$ python3 -m sglang.bench_one_batch --batch-size 1  --input 256
--output 32 --model deepseek-ai/DeepSeek-V3  --trust-remote-code  --tp 8
--enable-torch-compile --torch-compile-max-bs 1 --cuda-graph-max-bs 1
 
```

#### **Results:**

![compile-bench](imgs/cuda_graph_benchmark.png)

As expected, when stacking optimizations ( torch.compiler / cuda graphs + torch.compiler / torch.compiler(cuda graphs) + torch.compiler), we reduce total latency (`7.322 / 1.256 / 1.011 s`) and improve total throughput (`39.34 / 229.27 / 284.86 token/s`).

**Note:** We see a degradation in the prefill phase latency due to the initial increase in compute by torch.compiler compilations and CUDA graphs not capturing prefill phase operations (`0.21180 / 0.25809 / 0.26079 s`) and throughput (`1208.67 / 991.92 / 981.64 token/s`).

### bf16 Batch Matrix Multiplication (bmm)

#### **Context:**

Batch matrix multiplications are the main workload performed in LLMs. As DeepSeek-V3 uses different quantized fp8 dtypes (float8_e5m2 and float8_e4m3fn) for its training (thus, reducing the memory allocation), we test accuracy and latency for a random set of bmm with different combinations of dtypes for fp8 and base bf16. This optimization does not rely on flags.

#### **Commits:** 

([fix fp8 for MLA and support bmm fp8 for DeepSeek V2](https://github.com/sgl-project/sglang/pull/1285), [Enable DeepseekV3 on AMD GPUs](https://github.com/sgl-project/sglang/pull/2601), [integrate bmm_fp8 kernel into sgl-kernel](https://github.com/sgl-project/sglang/pull/3056))

#### **Benchmarks:**

```bash
$ pytest -s test_bmm_fp8.py
```

* Results obtained with a modified versiong of `test_bmm_fp8.py`  

#### **Results:**

![bmm-bench](imgs/fp8_latency_comparison.png)

The similarity between results is near identical (cosine similarity = 1 identical), which denotes no loss in accuracy, whereas latency for fp8 is worse than for bf16 due to casting compute. 

### Support nextn speculative decoding

#### **Related flags:**

```bash
--speculative-num-steps: # The number of steps sampled from a draft model in Speculative Decoding.
--speculative-eagle-topk: # The number of tokens sampled from the draft model in each step of EAGLE-2.
--speculative-num-draft-tokens: # The number of tokens sampled from the draft model in Speculative Decoding.
--speculative-draft: # The draft model to be used. It needs the same tokenizer as the verifier model (default: SGLang/DeepSeek-V3-NextN).
```

#### **Context:**

Speculative decoding accelerates inference by introducing a draft model (a smaller, faster model) that generates multiple tokens at once. A verification step then checks if these draft tokens match the predictions of a larger, more accurate LLM.

Its major flaw is that as naive speculative decoding generates a single linear sequence of draft tokens, all subsequent tokens are discarded, lowering the acceptance rate, even if a single token in the sequence is rejected.

SGLang's implementation of NextN is based on EAGLE-2 and SpecInfer:

![speculative_decoding.png](imgs/speculative_decoding.png)

With tree-based speculative decoding (SpecInfer and EAGLE-2), the predictions are organized as a tree, where each node represents a possible next token. With this approach, we generate multiple speculative branches that can be parallely verified by the verifier LLM, increasing the acceptance rate.

EAGLE-2's key improvements are dynamic draft trees based on the context and pruning of the nodes based on confidence score of the draft model.

#### **Commits:** 

([[Track] DeepSeek V3/R1 nextn progress #3472,](https://github.com/sgl-project/sglang/issues/3472), [Support NextN (MTP) speculative decoding for DeepSeek-V3/R1 #3582](https://github.com/sgl-project/sglang/pull/3582),  [Support Eagle2 for Triton backend #3466](https://github.com/sgl-project/sglang/pull/3466), [Eagle speculative decoding part 4: Add EAGLE2 worker #2150](https://github.com/sgl-project/sglang/pull/2150))

#### **Benchmarks:**

No flags.

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 256 --random-output 32 --random-range-ratio 1 --num-prompts 1 --host 127.0.0.1 --port 30000
```

`--speculative-algo NEXTN --speculative-draft SGLang/DeepSeek-V3-NextN --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 4`

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --speculative-algo NEXTN --speculative-draft SGLang/DeepSeek-V3-NextN --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 4 --tp 8 --trust-remote-code
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 256 --random-output 32 --random-range-ratio 1 --num-prompts 1 --host 127.0.0.1 --port 30000 
```

#### **Results:**

![spec-bench](imgs/speculative_serving.png)

We achieve an overall improvement of the general throughput (request, input, and output) and a significant (x6) reduction to the end-to-end latency.

## MLA

### TP+DP Attention

#### **Related flags:**

```bash
--enable-dp-attention: # Enable compatible MLA Data Parallelism.
```

#### **Context:**

Tensor Parallelism (TP) works with MHA by splitting the KV cache by TP devices (usually 8), so each device processes 1/TP of the KV cache. [1]

If we apply this to Multi-Head Latent Attention (MLA) and TP, each GPU splits the `kv cache` along the `head_num` dimension. However, MLA's `kvcache` has a `head_num` of `1`, making it impossible to split. Therefore, each GPU must maintain a complete `kvcache` → the `kvcache` gets duplicated per device.

When using DP (Data Parallelism) for MLA, it splits according to requests, and the latent states caches of different requests are stored in different GPUs. E.g., as we cannot divide the only KV cache, we divide the data into batches and paralelize them into different workers performing different tasks (prefill, decode).

After MLA, an all-gather operation is performed, allowing each GPU to acquire all sequences' `hidden_state`. Then, after **MOE (Mixture of Experts)**, each GPU extracts its corresponding sequences using **slice** operations.

![dp_attn.png](imgs/dp_attn.png)

#### **Commits:** 

([Support cuda graph for DP attention](https://github.com/sgl-project/sglang/pull/2061), [Support multinode DP Attention](https://github.com/sgl-project/sglang/pull/2925), [Multi-node Tensor Parallelism](https://github.com/sgl-project/sglang/pull/550), [Support DP MLA](https://github.com/sgl-project/sglang/pull/1970))

#### **Benchmarks:**

No flags

```bash
# Launch server with profiler env
export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/profiler_env_folders/ # Optional for profiling
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code

# Prefill
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 512 --random-output 1 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000 
# Decode
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 512 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000
```

`—enable-dp-attention`

```bash
# Launch server with profiler env
export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/profiler_env_folders/
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --enable-dp-attention

# Prefill
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 512 --random-output 1 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000
# Decode
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 512 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000
```

#### **Results:**

![dp-bench](imgs/dp_attention.png)

As it is a scheduler paradigm, it performs better when using large batch sizes. Otherwise, the added overhead is larger than the actual data parallelization.

For larger batch sizes (in this case 10,000), we see an overall improvement in both prefill and decode phases, from end-to-end latency to overall throughput and concurrency.

### Support overlap scheduler with DP Attention


#### **Related flags:**

```bash
--disable-overlap-schedule: # Disable the Overhead-Scheduler
```

#### **Context:**

We can overlap the CPU scheduling with the GPU computation. The scheduler runs one batch ahead and prepares all the metadata required for the next batch. By doing this, we can keep the GPUs busy for the entire duration and hide expensive overheads such as the radix cache operations. 

![overlap_scheduler.png](imgs/overlap_scheduler.png)

#### **Commits:** 

([Faster overlap scheduler](https://github.com/sgl-project/sglang/pull/1738), [Enable overlap by default](https://github.com/sgl-project/sglang/pull/2067),  [Enable overlap scheduler by default for the triton attention backend](https://github.com/sgl-project/sglang/pull/2105))

#### **Benchmarks:**

`--disable-overlap-schedule`

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --disable-overlap-schedule
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 256 --random-output 32 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000
```

No flags → enable overlap scheduler:

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 2500 --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1
```

#### **Results:**

![fig_overlap](imgs/ovelap_scheduler_latency.png)

We see a general reduction in latency: End to end (standard:`1080152.26s`| overlap: `1066166.84s`), Time per Output Token (standard:`348.10s`| overlap: `196.79s`) and inter-token latency (standard:`350.62s`| overlap: `197.96s`) although Time to first token presents a degradation result of the scheduling overhead (standard: `724050.93s`| overlap: `864850.926s`).

With larger request sizes for inputs and outputs, the effect of the overlap scheduler will be even more noticeable.

### FlashInfer prefill and MLA decoding

#### **Related flags:**

```bash
--enable-flashinfer-mla: # Enable FlashInfer MLA optimization
```

#### **Context:**

FlashInfer backend instead of triton.

#### **Commits:**
([Add fast decode plan for flashinfer mla,](https://github.com/sgl-project/sglang/pull/3987) [MLA prefill w/o weight absorption](https://github.com/sgl-project/sglang/pull/2349))

#### **Benchmarks:**

No flags:

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```

```bash
Accuracy: 0.951
Latency: 77.397 s
Output throughput: 1809.790 token/s
```

With `--enable-flashinfer-mla`

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --enable-flashinfer-mla
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```

```bash
Accuracy: 0.948
Latency: 71.480 s
Output throughput: 1920.021 token/s
```


#### **Results:**

![flashpng](imgs/flashinfer_mla.png)

Due to FlashInfer fused operation, we obtain less latency and more output throughput for similar accuracy.

## FP8

### Improve the accuracy for FP8

#### **Context:**

Numerical overflow occurs when a value exceeds the representable range of a given numerical format (like FP8), causing incorrect or infinite values. In the context of FP8 quantization on Tensor Cores, overflow happens because FP8 has a very limited dynamic range. To prevent numerical overflow, values are scaled down before being quantized using the max element of the matrix, although this makes it sensitive to outlier values. To avoid it, the DeepSeek team proposes a blockwise and tilewise scaling, in which each 128×128 submatrix of a weight matrix and each 1×128 subvector of an activation vector is scaled and quantized separately.

The FP8 GEMM accumulation on NVIDIA H800 Tensor Cores is limited to about `14 bits` of precision, which is significantly lower than FP32 accumulation precision. That is why DeepSeek uses a separate FP32 accumulator register using CUDA Cores, thus mitigating the loss of accuracy. The dequantizing scaling factor is also applied to this FP32 accumulator.

![fp8_deepseek.png](imgs/fp8_deepseek.png)

#### **Commits:** 
([support blockwise fp8 matmul kernel #3267](https://github.com/sgl-project/sglang/pull/3267), [add unit test for block wise fp8#3156](https://github.com/sgl-project/sglang/pull/3156), [integrate blockwise fp8 kernel#3529](https://github.com/sgl-project/sglang/pull/3529), [[Track] DeepSeek V3/R1 accuracy](https://github.com/sgl-project/sglang/issues/3486))

#### **Benchmarks:**

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --tp 8 --trust-remote-code
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```
#### **Results:**

![fig6.png](imgs/fp8_vs_bf16.png)

For the same accuracy (`0.955` vs `0.957` on gsm8k), we observe more output throughput and less latency.

### Tunning FP8 GEMM

#### **Context:**

SGLang employs FP8 blockwise quantization tuning to optimize performance for different GPUs. The implementation specifically benchmarks FP8 GEMM (General Matrix Multiplication) kernels across AMD and CUDA architectures, testing different block shapes to determine the most efficient configuration based on latency.

This approach ensures that blockwise quantization aligns with the optimal block size for GEMM operations, minimizing precision loss while maximizing computational efficiency. The computations are performed in FP8, but accumulation occurs in BF16 to maintain numerical stability before final output storage.

Key functions:

```bash
# fn: benchmark_config(A_fp8, B_fp8, As, Bs, block_size, config, out_dtype=torch.float16, num_iters=10)
A: torch.Tensor,     # Input matrix (FP8) - typically activations
B: torch.Tensor,     # Input matrix (FP8) - typically weights
As: torch.Tensor,    # Per-token-group scale factors for `A`
Bs: torch.Tensor,    # Per-block scale factors for `B`
block_size: List[int],  # Block size for quantization (e.g., [128, 128])
config: Dict[str, Any],  # Kernel configuration parameters
output_dtype: torch.dtype = torch.float16,  # Precision of output
```

```bash
# fn: tune(M, N, K, block_size, out_dtype, search_space):
M,N,K: int  # Shape of the matrix multiplication (M × K @ K × N → M × N)
block_size: int # Tuple defining blockwise quantization size ([block_n, block_k])
out_dtype: str # Output precision (e.g., float16, bfloat16)
search_space: List[dict{str,int}] # List of configurations to test (e.g., block sizes, number of warps).

# search_space example:
{
"BLOCK_SIZE_M": block_m,
"BLOCK_SIZE_N": block_n,
"BLOCK_SIZE_K": block_k,
"GROUP_SIZE_M": group_size,
"num_warps": num_warps,
"num_stages": num_stages,
}
```

#### **Commits:** 
([add tuning block wise fp8#3242](https://github.com/sgl-project/sglang/pull/3242))

#### **Benchmarks:**

```bash
$python3 benchmark/kernels/quantizationtuning_block_wise_fp8.py
```

#### **Results:**

Example of the optimal configuration for the kernel: `N=512,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]`

```bash
[...]
{
    "2048": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4
    },
    "3072": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3
    },
    "4096": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 64,
        "num_warps": 4,
        "num_stages": 3
    }
}
```

For all batch sizes to be tuned and a given FP8 dtype, the given script tests and compares different model weight dimensions (N and K) to optimize FP8 GEMM block-wise quantization based on the lowest latency. This obtains the most optimal configuration per batch size for the block tiling dimension(`BLOCK_SIZE_M/N/K`), the group size (`GROUP_SIZE_M`) for the number of tiles grouped together improving the L2 cache usage, the number of warps (`num_warps`) per thread block (i.e., per tile), and the number of stages (`num_stages`) for block loading into shared memory as a prefetching. In turn, this enables an autotuning of the compute parameters for different configurations.

### FP8 GEMM CUTLASS implementation

#### Context:

The quantization operation can be fused into the FP8 matmul operation for efficiency. In `sgl-kernel/src/sgl-kernel/csrc/int8_gemm_kernel.cu`, there is a CUDA-accelerated implementation of integer 8-bit (int8) scaled matrix multiplication fused with W8A8 quantization.

#### **Commits:** 
([support w8a8 fp8 kernel with CUTLASS #3047](https://github.com/sgl-project/sglang/pull/3047) , [Support cutlass Int8 gemm #2752](https://github.com/sgl-project/sglang/pull/2752), [Support sm90 Int8 gemm#3035](https://github.com/sgl-project/sglang/pull/3035), [Blockwise Scaling for FP8 #1932 from NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/pull/1932))

#### **Benchmarks:**

```bash
root@cluster-h200-02-f2:/sgl-workspace/sglang/sgl-kernel/benchmark# python3 bench_int8_gemm.py 
```
#### **Results:**

![fig6.png](imgs/int8_gemm_comparison.png)


Benchmarks measure GB/s per batch size (another measure of throughput). Comparing the vLLM kernel (int8 gemm) to the SGLang kernel, we get more throughput for different batch sizes for different configurations (N and K).

**Note**: We tested this benchmark using DeepSeek-Coder-V2-Lite-Instruct as the code for DeepSeek-V3 is not implemented in SGLang.

## MoE

### FusedMoE tuning for H200

The following is the implementation of the fused computation for a Mixture of Experts (MOE) using token and expert matrices with the multiplies `A @ B` (token × expert matmul) using top-k routing and supporting:

- `fp16`, `bfloat16`, `fp8`, `int8` formats
- Weight/activation scaling via `A_scale`, `B_scale`
- Block-wise quantization
- Expert-wise routing via `expert_ids`

#### **Context:**

The custom SGLang kernels for fusedMoE, using vLLM as reference and baseline, are composed of: 

`tuning_fused_moe_triton.py`: A tool for tuning the `fused_moe_triton` kernel. Adapted from [vllm's benchmark_moe.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py), with added support for various model architectures.

`benchmark_vllm_vs_sglang_fused_moe_triton.py`: A tool for comparing the performance of fused MoE kernels between vLLM and SGLang implementations. Supports various model architectures and data types.

`benchmark_torch_compile_fused_moe.py`: A tool for benchmarking the performance of the fused MoE kernel with `torch.compile` and the original fused MoE kernel.

#### **Commits:** 
([Add unitest for fused_moe](https://github.com/sgl-project/sglang/pull/2416), [MoE Expert Parallel Impl](https://github.com/sgl-project/sglang/pull/2203), [`benchmark/kernels/fused_moe_triton/README.md`](https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton))

#### **Benchmarks:**

```bash
$ python3 benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --model deepseek-ai/DeepSeek-V3 --tp-size 8  --dtype fp8_w8a8 --tune
                                                                          
Writing best config to E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json...                                                                       
Tuning took 5267.05 seconds
```

FusedMoE benchmarking sgl-kernel vs vllm:
```bash
python3 benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py
[...]
benchmark sglang_fused_moe_triton with batch_size=505
benchmark vllm_fused_moe_triton with batch_size=506
benchmark sglang_fused_moe_triton with batch_size=506
benchmark vllm_fused_moe_triton with batch_size=507
benchmark sglang_fused_moe_triton with batch_size=507
benchmark vllm_fused_moe_triton with batch_size=508
benchmark sglang_fused_moe_triton with batch_size=508
benchmark vllm_fused_moe_triton with batch_size=509
benchmark sglang_fused_moe_triton with batch_size=509
benchmark vllm_fused_moe_triton with batch_size=510
benchmark sglang_fused_moe_triton with batch_size=510
benchmark vllm_fused_moe_triton with batch_size=511
benchmark sglang_fused_moe_triton with batch_size=511
benchmark vllm_fused_moe_triton with batch_size=512
benchmark sglang_fused_moe_triton with batch_size=512

fused-moe-performance:
[...]
     batch_size  vllm_fused_moe_triton  sglang_fused_moe_triton
505       506.0               1.014688                 0.507488
506       507.0               1.011744                 0.509344
507       508.0               1.007200                 0.504288
508       509.0               1.007232                 0.505696
509       510.0               1.007792                 0.507712
510       511.0               1.011072                 0.507248
511       512.0               1.012992                 0.507840
````

#### Results:

We perform the tunning for the fused MoE kernel for DeepSeek-V3 with FP8 quantization, obtaining the optimal configuration for each batch size similar to when tunning FP8 GEMM:

> for the block tiling dimension(`BLOCK_SIZE_M/N/K`), the group size (`GROUP_SIZE_M`) for then number of tiles grouped together improves the L2 cache usage, the number of warps (`num_warps`) per thread block (i.e., per tile), and the number of stages (`num_stages`) for block loading into shared memory as a prefetching.

```bash
[...]
"2048": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3
    },
    "3072": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 4
    },
    "4096": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3
    }
```

We then compare the latency for the fused MoE kernel implementation of SGLang with the baseline implementation from vLLM, obtaining a more refined version with almost constant latency when incrementing the batch size.

![fig7](imgs/fused_moe_latency_comparison.png)

As closing notes, the versions used for this technical blog is sglang: v0.4.3.post2, sgl-kernel: 0.0.3.post6, torch: 2.5.1 and CUDA: 12.5. 

We strongly support the collaboration on sglang, which serves as the de facto open-source inference engine for the DeepSeek family of models. 

Future work plans to further explore these optimizations by analyzing the performance and incremental improvements of key components such as the FlashMLA kernel, FlashAttention, and the sglang Triton kernels. We also propose exploring new optimizations implemented by the sglang team, such as the splitting of the prefill and decode phases and the integration of DeepGemm for FusedMoE.

Thanks to the sglang team for their help, review of this blog, and their joint collaboration on the project.

## References

- [sglang kernels test](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/tests)

- [sglang kernels benchmarks](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/benchmark)

- [[Feature] DeepSeek V3 optimization #2591](https://github.com/sgl-project/sglang/issues/2591)

- [Blog deepseek v3 techniques behind 10x efficiency](https://dataturbo.medium.com/key-techniques-behind-deepseek-models-10x-efficiency-1-moe-9bd2534987c8)

- [AI Compiler Sglang optimizations work](https://carpedm30.notion.site/02-19-2024-2nd-meeting)

- [lmsys sglang 0.4 data parallellism](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models)

- [lmsys sglang 0.4 zero overhead batch scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#zero-overhead-batch-scheduler)

- [spaces.ac.cn: MQA, GQA, MLA blog](https://spaces.ac.cn/archives/10091)

- [Tree-based speculative decoding paper](https://arxiv.org/pdf/2305.09781)

- [EAGLE2 Speculative decoding paper](https://arxiv.org/pdf/2406.16858)

- [DeepSeek v3 paper](https://arxiv.org/pdf/2412.19437)

- [Zhihu blogs: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://zhuanlan.zhihu.com/p/687404563)

- [Zhihu blogs: MLA tp and dp](https://zhuanlan.zhihu.com/p/25573883266)

- [Zhihu blog: MLA tp and dp part 2](https://zhuanlan.zhihu.com/p/15280741714)[1]

- [Colfax deepseekv3 fp8 mixed precision training](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/)

  
