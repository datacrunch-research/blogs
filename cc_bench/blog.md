# Performance Cost of Confidential Computing on LLM Inference: B300 SXM6 Benchmarks

## Introduction

Confidential Computing (CC) enables computation on encrypted data without exposing it to the infrastructure operator. For LLM workloads, this means protecting model weights, activations, and user data from the cloud provider.

The security guarantees come at a cost. CC adds encryption at multiple system boundaries: CPU memory (AMD SEV-SNP), GPU VRAM (NVIDIA Confidential Computing), and GPU-to-GPU links (NVLink Encryption / NVLE). This blog quantifies that cost on NVIDIA B300 SXM6 GPUs across microbenchmarks and end-to-end LLM inference workloads.

We find that the hardware encryption overhead is moderate: ~12-15% on decode latency, with compute throughput unaffected. However, CC disables CUDA multicast, which breaks FlashInfer allreduce fusion and other collective communication optimizations in modern inference frameworks. This software optimization gap, rather than encryption itself, is the dominant source of overhead in tensor-parallel inference.

## CC Stack Under Test

Our test environment runs a full CC stack on 8x NVIDIA B300 SXM6 AC GPUs (275 GB VRAM each):

| Layer | Technology | Function |
|-------|-----------|----------|
| CPU | AMD SEV-SNP | Encrypts all VM memory with per-VM AES keys |
| GPU | NVIDIA CC | Encrypts GPU VRAM — 273,108 MiB of 275,040 MiB in protected pool (~99.3%) |
| Multi-GPU | NVLE (NVLink Encryption) | Encrypts all GPU-to-GPU NVLink traffic |

The VM runs Ubuntu 24.04 with NVIDIA driver 595.58.03 (open kernel module) and CUDA 13.2.

CC GPUs require explicit initialization before any CUDA workload:
```bash
nvidia-smi conf-compute -srs 1   # set GPU ready state
```
Without this, the CUDA runtime fails with error 802 (`cudaErrorSystemNotReady`) while `nvidia-smi` (which uses NVML, not CUDA) works normally.

## Microbenchmarks: First hypothesis

Before measuring inference workloads, we narrow the hardware-level CC overhead. These results apply to any GPU computation, including training.

We compare peak metrics between the B300 host (non-CC) and the CC VM using `microbench.py`:

### Compute Throughput

| Precision | Host | CC VM | Delta |
|-----------|------|-------|-------|
| Torch FP8 (E4M3) | 3308 TFLOPS | 3258 TFLOPS | -1.5% |
| TE FP8 (E4M3) | 3107 TFLOPS | 3010 TFLOPS | -3.1% |
| TE FP4 (E2M1) | 2592 TFLOPS | 2564 TFLOPS | -1.1% |
| BF16 | 1564 TFLOPS | 1560 TFLOPS | -0.2% |
| FP16 | 1482 TFLOPS | 1452 TFLOPS | -2.0% |

**Compute is unaffected.** Deltas range from -0.2% to -3.1%, within run-to-run variance. CC encryption operates at the memory interface, not inside the GPU's compute units. Once data is decrypted into registers and shared memory, ALU and tensor core operations proceed at full speed.

### Memory and Communication Bandwidth

| Metric | Host | CC VM | Delta |
|--------|------|-------|-------|
| D2D (intra-GPU) | 3327 GB/s | 3326 GB/s | **None** |
| H2D (CPU -> GPU) | 57.8 GB/s | 21.2 GB/s | **-63%** |
| D2H (GPU -> CPU) | 58.3 GB/s | 19.9 GB/s | **-66%** |
| P2P NVLink (GPU -> GPU) | 746 GB/s | 633 GB/s | **-15%** |

Three overhead profiles emerge:

1. **Intra-GPU memory (D2D): no overhead.** Data stays within the GPU's encrypted VRAM boundary: no trust boundary crossing, no encryption pipeline latency.

2. **CPU <-> GPU DMA (H2D/D2H): -63 to -66%.** Every transfer crosses the trust boundary between SEV-SNP encrypted host memory and CC-encrypted GPU VRAM.

3. **GPU <-> GPU NVLink (P2P): -15%.** NVLE encrypts all NVLink traffic between GPUs. The overhead is significantly lower than H2D, which favors architectures where data resides entirely on GPU memory and inter-GPU communication stays on the NVLink fabric without touching the host.

## End-to-End Inference: Qwen3.5-397B FP8

To measure how hardware-level overhead translates to real inference performance, we benchmark Qwen3.5-397B-A17B (FP8) using the InferenceX benchmarking framework with SGLang v0.5.10.post1.

- **Configuration:** TP=4, EP=1, SGLang, no speculative decoding. 
- **Baseline:** [InferenceX](https://github.com/SemiAnalysisAI/InferenceX/blob/8862360995f32b9ca1752b57f7cb1b774c69ee3b/benchmarks/single_node/qwen3.5_fp8_b300.sh) published B300 results (non-CC, 2026-04-17).

**Modification required for CC:** The `--enable-symm-mem` flag in the InferenceX benchmark script must be removed. This flag enables CUDA symmetric memory, which depends on multicast, a capability CC disables at the driver level. The baseline was run with this optimization enabled, so the measured delta includes both CC encryption overhead and the lost optimization, which overlaps allreduce communication with compute.

### ISL=1024, OSL=1024 (Short Context)

| CONC | Output tput/GPU (baseline) | Output tput/GPU (CC) | Delta | Median TPOT (baseline) | Median TPOT (CC) | Delta | Median TTFT (baseline) | Median TTFT (CC) | Delta |
|------|---------------------------|---------------------|-------|----------------------|-----------------|-------|----------------------|-----------------|-------|
| 4 | 109.6 tok/s | 92.4 tok/s | -15.6% | 8.70 ms | 10.16 ms | +16.9% | 193.8 ms | 280.0 ms | +44.5% |
| 8 | 196.9 tok/s | 148.4 tok/s | -24.6% | 9.79 ms | 12.74 ms | +30.1% | 192.4 ms | 287.4 ms | +49.4% |
| 16 | 320.8 tok/s | 220.9 tok/s | -31.1% | 12.11 ms | 17.62 ms | +45.5% | 197.7 ms | 305.8 ms | +54.6% |
| 32 | 504.3 tok/s | 389.6 tok/s | -22.8% | 15.38 ms | 19.86 ms | +29.1% | 211.5 ms | 289.6 ms | +36.9% |
| 64 | 778.4 tok/s | 445.6 tok/s | -42.8% | 20.17 ms | 35.70 ms | +77.0% | 234.3 ms | 373.6 ms | +59.4% |
| 128 | 1181.4 tok/s | 643.1 tok/s | -45.6% | 26.61 ms | 49.77 ms | +87.0% | 269.9 ms | 436.0 ms | +61.5% |
| 256 | 1743.9 tok/s | 1002.0 tok/s | -42.5% | 36.08 ms | 64.17 ms | +77.9% | 321.6 ms | 483.2 ms | +50.3% |

### ISL=8192, OSL=1024 (Long Context)

| CONC | Output tput/GPU (baseline) | Output tput/GPU (CC) | Delta | Median TPOT (baseline) | Median TPOT (CC) | Delta | Median TTFT (baseline) | Median TTFT (CC) | Delta |
|------|---------------------------|---------------------|-------|----------------------|-----------------|-------|----------------------|-----------------|-------|
| 4 | 107.5 tok/s | 92.7 tok/s | -13.8% | 8.79 ms | 10.04 ms | +14.2% | 295.7 ms | 344.4 ms | +16.5% |
| 8 | 177.6 tok/s | 141.4 tok/s | -20.4% | 10.78 ms | 12.99 ms | +20.5% | 301.2 ms | 343.4 ms | +14.0% |
| 16 | 280.2 tok/s | 204.9 tok/s | -26.9% | 13.55 ms | 18.72 ms | +38.2% | 346.7 ms | 389.0 ms | +12.2% |
| 32 | 410.8 tok/s | 337.3 tok/s | -17.9% | 18.74 ms | 22.42 ms | +19.6% | 412.0 ms | 405.2 ms | -1.6% |
| 64 | 576.0 tok/s | 372.3 tok/s | -35.4% | 26.98 ms | 41.90 ms | +55.3% | 481.9 ms | 532.2 ms | +10.4% |
| 128 | 753.5 tok/s | 486.4 tok/s | -35.4% | 41.41 ms | 64.69 ms | +56.2% | 608.0 ms | 686.6 ms | +12.9% |
| 256 | 937.4 tok/s | 648.7 tok/s | -30.8% | 66.82 ms | 97.24 ms | +45.5% | 846.4 ms | 862.8 ms | +1.9% |

### Key Observations

**Decode latency (TPOT) overhead scales with concurrency.** At low concurrency (CONC=4), TPOT overhead is +14-17% — consistent with the -15% P2P NVLink bandwidth reduction from microbenchmarks. At high concurrency (CONC=128), TPOT overhead reaches +56-87%. This is not a fixed per-token cost: it compounds under load as concurrent requests generate more allreduce operations competing for the encrypted NVLink fabric.

**Prefill latency (TTFT) overhead is context-length dependent.** At ISL=1024, TTFT overhead is 37-62% across concurrency levels. At ISL=8192, it drops to 2-16%. Longer prefill sequences mean compute dominates the TTFT measurement, making the fixed CC overhead (including the lost allreduce fusion) a smaller fraction. This strongly suggests the TTFT overhead at short context is dominated by the lost software optimization, not encryption latency.

**Throughput loss reaches 35-46% at high concurrency.** The CC VM saturates earlier than the baseline. At CONC=256 with 8k context, throughput loss is -31%, less severe than the -43% at 1k context, consistent with longer computation amortizing communication overhead.

## Isolating Pure CC Overhead: DeepSeek-V4-Pro FP4

The Qwen3.5 results combine CC encryption overhead with the lost `--enable-symm-mem` optimization. To isolate the pure hardware CC cost, we benchmark DeepSeek-V4-Pro (~680B MoE, FP4) using SGLang's custom B300-tuned image. Critically, the DSV4 non-DP code path runs with `enable_symm_mem=False` and `enable_flashinfer_allreduce_fusion=False` on **both** baseline and CC thus no software optimization difference.

- **Configuration:** FP4, SGLang (`lmsysorg/sglang:deepseek-v4-b300@sha256:2fec8d7958bb0d53b50d7bf04d6ae6a7de8a35503775826e0550a45dd8c3ee15`), non-DP attention, no MTP
- **Baseline:** [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) published B300 results (non-CC)

| TP | CONC | ISL/OSL | Tput/GPU (baseline) | Tput/GPU (CC) | Delta | Med TPOT (baseline) | Med TPOT (CC) | Delta |
|----|------|---------|--------------------|--------------| ------|--------------------|--------------| ------|
| 8 | 1 | 8k/1k | 9.9 tok/s | 7.3 tok/s | -25.9% | 11.08 ms | 12.38 ms | +11.7% |
| 4 | 32 | 8k/1k | 207.6 tok/s | 175.7 tok/s | -15.4% | 35.25 ms | 39.01 ms | +10.6% |
| 4 | 32 | 1k/1k | 275.5 tok/s | 191.1 tok/s | -30.6% | 26.53 ms | 33.35 ms | +25.7% |

**+11-26% TPOT with no software optimization mismatch.** The +11-12% at low concurrency aligns directly with the -15% P2P NVLink bandwidth from microbenchmarks. The higher +26% at TP=4/CONC=32 reflects increased allreduce contention under load. Similar scaling pattern observed in Qwen3.5, but at lower magnitude because allreduce fusion was never active on either side.

## Conclusion

1. **Compute is free, communication is not.** CC adds zero overhead to GPU compute (matmul, attention, MoE routing). The cost is entirely at data movement boundaries: -63-66% on CPU↔GPU DMA, -15% on GPU↔GPU NVLink. Intra-GPU memory is unaffected.

2. **Pure CC encryption overhead on decode is ~12-15%.** Measured on DeepSeek-V4-Pro where both baseline and CC run without FlashInfer allreduce fusion. This scales to ~25% under load as allreduce operations contend for the encrypted NVLink fabric.

3. **The bigger cost is software, not hardware.** CC disables CUDA multicast, which breaks FlashInfer allreduce fusion in SGLang. This lost optimization is the dominant source of overhead in tensor-parallel inference, particularly on prefill (TTFT) at short context lengths.

4. **Overhead scales with concurrency and shrinks with context length.** TPOT overhead ranges from +14% (CONC=4) to +87% (CONC=128) on Qwen3.5. Longer input sequences (8k vs 1k) amortize the fixed CC costs, reducing TTFT overhead from 44-62% down to 2-16%.

5. **CC favors GPU-resident architectures.** The 4x gap between P2P overhead (-15%) and DMA overhead (-63%) means workloads that keep data on GPU and minimize CPU involvement pay a much lower CC tax.

6. **Training overhead will be worse.** Inference is the forward pass, a subset of training. Training adds gradient synchronization, optimizer updates, and data loading, all of which hit the most penalized transfer paths (allreduce, H2D). The overheads measured here are a lower bound.

7. **Multi-node CC remains untested.** Our measurements are single-node (8 GPUs). Multi-node adds network-level encryption on top of per-node NVLE overhead, with cross-node collectives crossing additional trust boundaries.

## References

- [InferenceX Github Repo](https://github.com/SemiAnalysisAI/InferenceX) 
- [InferenceX Dashboard](https://inferencex.semianalysis.com)
- [SGLang Inference engine](https://github.com/sgl-project/sglang)
- [SGLang Optimization Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments#optimization/debug-options)
- [NVIDIA Whitepaper Confidential Computing Blackwell](https://docs.nvidia.com/nvidia-secure-ai-with-blackwell-and-hopper-gpus-whitepaper.pdf)

## Appendix: Reproducibility

### CC initialization (required after every VM reboot)

```bash
nvidia-smi conf-compute -srs 1        # set GPUs ready state
nvidia-smi conf-compute -grs          # verify: should show "ready"
nvidia-smi conf-compute -mgm          # verify: should show "NVLE"
```

### Verify CUDA multicast support

```python
python3 -c '
import ctypes
cuda = ctypes.cdll.LoadLibrary("libcuda.so")
cuda.cuInit(0)
val = ctypes.c_int(0)
cuda.cuDeviceGetAttribute(ctypes.byref(val), 132, 0)
print(f"MULTICAST_SUPPORTED = {val.value}")
'
# Returns 1 on non-CC B300, 0 on CC B300
```

### Qwen3.5 FP8 benchmark (single concurrency point)

```bash
docker run --gpus all \
  --ipc=host --network=host \
  -v /root/cc/InferenceX:/workspace \
  -v /root/cc/hf_cache:/root/.cache/huggingface \
  -v /root/cc/results:/workspace/results \
  -e MODEL=Qwen/Qwen3.5-397B-A17B-FP8 \
  -e TP=4 \
  -e EP_SIZE=1 \
  -e CONC=4 \
  -e ISL=1024 \
  -e OSL=1024 \
  -e RANDOM_RANGE_RATIO=0.5 \
  -e RESULT_FILENAME=qwen35_fp8_cc_tp4_isl1024_osl1024_conc4 \
  -w /workspace \
  lmsysorg/sglang:v0.5.10.post1-cu130 \
  bash -c "
    source /workspace/benchmarks/benchmark_lib.sh
    nvidia-smi
    start_gpu_monitor --output /workspace/results/gpu_metrics.csv
    CONTEXT_LENGTH=\$((ISL + OSL + 20))

    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server \
      --model-path=\$MODEL \
      --host=0.0.0.0 --port=8888 \
      --trust-remote-code \
      --tensor-parallel-size=\$TP \
      --data-parallel-size=1 \
      --expert-parallel-size=\$EP_SIZE \
      --disable-radix-cache \
      --quantization fp8 \
      --kv-cache-dtype fp8_e4m3 \
      --mamba-ssm-dtype bfloat16 \
      --attention-backend trtllm_mha \
      --moe-runner-backend flashinfer_trtllm \
      --cuda-graph-max-bs \$CONC \
      --max-running-requests \$CONC \
      --max-prefill-tokens 16384 \
      --chunked-prefill-size 16384 \
      --mem-fraction-static 0.8 \
      --stream-interval 50 \
      --scheduler-recv-interval 10 \
      --tokenizer-worker-num 6 \
      --tokenizer-path \$MODEL \
      --context-length \$CONTEXT_LENGTH > /workspace/results/server.log 2>&1 &

    SERVER_PID=\$!
    wait_for_server_ready --port 8888 --server-log /workspace/results/server.log --server-pid \$SERVER_PID
    pip install -q datasets pandas

    run_benchmark_serving \
      --model \$MODEL \
      --port 8888 \
      --backend vllm \
      --input-len \$ISL \
      --output-len \$OSL \
      --random-range-ratio \$RANDOM_RANGE_RATIO \
      --num-prompts \$((CONC * 10)) \
      --max-concurrency \$CONC \
      --result-filename \$RESULT_FILENAME \
      --result-dir /workspace/results/ \
      --use-chat-template

    stop_gpu_monitor
  "
```

**Difference from original InferenceX script:** `--enable-symm-mem` removed (crashes under CC).

To sweep concurrency, change `CONC` and `RESULT_FILENAME` accordingly. ISL/OSL can be changed to 8192/1024 for long context.

### DeepSeek-V4-Pro FP4 benchmark

```bash
docker run --gpus all \
  --ipc=host --network=host \
  -v /root/cc/InferenceX:/bench \
  -v /root/cc/hf_cache:/root/.cache/huggingface \
  -v /root/cc/results:/bench/results \
  -e MODEL=deepseek-ai/DeepSeek-V4-Pro \
  -e TP=8 \
  -e DP_ATTENTION=false \
  -e CONC=1 \
  -e ISL=8192 \
  -e OSL=1024 \
  -e RANDOM_RANGE_RATIO=0.5 \
  -e RESULT_FILENAME=dsv4_fp4_cc_tp8_isl8192_osl1024_conc1 \
  -w /bench \
  lmsysorg/sglang:deepseek-v4-b300@sha256:2fec8d7958bb0d53b50d7bf04d6ae6a7de8a35503775826e0550a45dd8c3ee15 \
  bash /bench/benchmarks/single_node/dsv4_fp4_b300_sglang.sh
```
