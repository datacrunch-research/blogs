# Deploy DeepSeek-R1 671B on 8x NVIDIA H200 with DataCrunch

## Inference engine: SGLang
SGLang is the recommended inference engine for deploying DeepSeek models, in particular [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf) and [Deep-Seek-R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf). SGLang currently supports MLA optimizations, DP Attention, FP8 (W8A8), FP8 KV Cache, and Torch Compile, enabling it to deliver state-of-the-art latency and throughput performance among other open-source frameworks.

Notably, SGLang v0.4.1 fully supports running DeepSeek-V3 on both NVIDIA and AMD GPUs, making it a highly versatile and robust solution. SGLang also supports multi-node tensor parallelism, enabling you to run this model on multiple network-connected machines.

Multi-Token Prediction (MTP) is in development, and progress can be tracked in the [optimization plan](https://github.com/sgl-project/sglang/issues/2591) (e.g. [FusedMoE H200 aware-tuning](https://github.com/sgl-project/sglang/issues/2471#event-15791112196)) and [custom kernels development](https://github.com/sgl-project/sglang/issues/2965).

We have been providing the SGLang team with GPU infrastructure targeting H200 aware-tunning for optimal performance. (see [H200 DeepSeek V3/R1 benchmarking](https://github.com/sgl-project/sglang/issues/2450)).

## Deploying DeepSeek-R1

1. The original sglang docker image is used as recommended:
```bash
docker pull lmsysorg/sglang:latest
```
2. The following command will create a valid docker container to host `DeepSeek R1`.
  - It is needed to mount a volume with the location of the huggingface cache folder. By default is: `~/.cache/huggingface`
  - It is needed to specify our `HF_TOKEN` to access the huggingface API. It can be exported as an environment variable as seen in `--env "HF_TOKEN=$HF_TOKEN"`

```bash
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name deepseek_r1 \
    -it \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    bash
```

3. This will create an interactive session inside the container where the following command should launch the server with `DeepSeek R1`.

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --tp 8 --trust-remote-code --enable-dp-attention
```

## Benchmarking DeepSeek-R1

4. The following command will perform a benchmark workload for 1 batch, 128 inputs token, and 256 outputs token:

```bash
python3 -m sglang.bench_one_batch --batch-size 1 --input 128 --output 256 --model deepseek-ai/DeepSeek-R1  --trust-remote-code --tp 8 --enable-torch-compile --torch-compile-max-bs 1
Prefill. latency: 1.91032 s, throughput:     67.00 token/s
Decode.  latency: 1.04900 s, throughput:      0.95 token/s
Decode.  latency: 0.02175 s, throughput:     45.99 token/s
Decode.  latency: 0.02097 s, throughput:     47.69 token/s
Decode.  latency: 0.02097 s, throughput:     47.68 token/s
Decode.  latency: 0.02080 s, throughput:     48.07 token/s
Decode.  median latency: 0.02097 s, median throughput:     47.68 token/s
Total. latency:  3.086 s, throughput:     44.07 token/s
Benchmark ...
Prefill. latency: 0.19635 s, throughput:    651.90 token/s
Decode.  latency: 0.02100 s, throughput:     47.62 token/s
Decode.  latency: 0.02078 s, throughput:     48.13 token/s
Decode.  latency: 0.02092 s, throughput:     47.80 token/s
Decode.  latency: 0.02086 s, throughput:     47.93 token/s
Decode.  latency: 0.02085 s, throughput:     47.97 token/s
Decode.  median latency: 0.02098 s, median throughput:     47.67 token/s
Total. latency:  5.537 s, throughput:     69.35 token/s

```

Two benchmarks are run as sanity checks. The final user will perceive the decode latency as shown below:
```bash
Decode.  median latency: 0.02098 s, median throughput:     47.67 token/s
```

## References
- [[Feature] DeepSeek V3 optimization #2591](https://github.com/sgl-project/sglang/issues/2591)

- [Deepseek-V3 Technical report](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)

- [Deepseek-R1 Technical report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)

- [FusedMoE H200 aware-tuning](https://github.com/sgl-project/sglang/issues/2471#event-15791112196)

- [Deepeek recommended inference engines](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended)

- [Sglang custom kernels development](https://github.com/sgl-project/sglang/issues/2965)
