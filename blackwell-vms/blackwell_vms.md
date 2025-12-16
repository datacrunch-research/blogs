# Blackwell VM series: B200 & B300

Verda GPU VM line adds the complete Blackwell architecture family, [B200](https://verda.com/b200) and [B300](https://verda.com/b300) (ultra). The main differences are:

- **Compute Performance:** B300 dense FP4 performance is 55.6% faster (14 vs 9.0 petaFLOPS) compared to B200, due to higher clock speeds, optimized tensor cores, and the additional TDP headroom. Essentially no FP64 performance (1.25 TF on B300 vs. 37 TF on B200).
- **Memory and Bandwidth:** B300 has 55.6% more GPU memory (288 GB HBM3E vs. 180 GB) for larger models and batches, with slightly higher bandwidth (8 TB/s vs. 7.7 TB/s) for better data movement.
- **Power:** B300 supports higher TDP (up to 1,100W vs. 1,000W), enabling the performance uplift but requiring better cooling.

(More on [B200 vs B300 performance comparison](https://verda.com/blog/nvidia-b300-vs-b200-complete-gpu-comparison-to-date) in our previous blog). 

Our VMs provide optimal CUDA performance and development environment. Latest ubuntu (24.04) with **CUDA 13.0/12.8** toolkit with the latest NVIDIA driver versions (**580**) at provisioning time, out of the box. These system level configuration is crucial in particular to support advance kernel DSL programming (e.g. [CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html), [Gluon](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/01-intro.py), [TVM](https://github.com/apache/tvm), [TK](https://github.com/HazyResearch/ThunderKittens)). If not for creating the kernels, to enable the most advanced tuned kernels (e.g. [FlashAttention 4 CuTe](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute), [flashinfer](https://github.com/flashinfer-ai/flashinfer)) which do also require these system CUDA settings to run efficiently. 

Docker can also be pre-installed when provisioning to also further facilitate a seamsealy development environment to run [cuDNN docker containers](https://hub.docker.com/layers/nvidia/cuda/12.8.1-cudnn-devel-ubuntu22.04/images/sha256-61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719) or [PyTorch](https://hub.docker.com/r/pytorch/pytorch) as base images. The VM instance storage can be solely local NVMe volumes, existing or new, or attach shared filesystem to be accessible by multiple instances.

> SXM6 means **SXM6** B300 on a **HGX** B300 board (with NVIDIA NVLink 5 + NVSwitch domain), which differs from the on-chip GB300s installed on the [trays of the NVL72 GB300](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html).

## NVIDIA’s Blackwell GPU architecture

Blackwell architecture includes new features such as 5th-generation tensor cores (fp8, fp6, fp4), Tensor Memory (TMEM) and CTA pairs (2 CTA):

- **5th-generation tensor cores (fp8, fp6, fp4):** The tensor cores in the B200 are notably larger and ~2–2.5x faster than the tensor cores in the H100, while B300 further increases the fp4 FLOPS. High tensor core utilization is critical for achieving major new-generation hardware speedups (see [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499)). In H100 using TMA was the main mechanism for achieved this high utilization, thus in B200/300 loading a lot more data at once becomes even more important.
- **TMEM:** Tensor memory is a new additional memory space, very similar to register memory. The main difference is that it requires user management and allocation, thus is reserved for advanced kernel programming. TMEM does not fundamentally change *how you* write CUDA kernels (the logical algorithm is the same) but adds new tools to optimize data flow (see [ThunderKittens Now Optimized for NVIDIA Blackwell GPUs](https://www.together.ai/blog/thunderkittens-nvidia-blackwell-gpus)). Hopper already introduced a new memory space, TMA. The following [memory spaces hierarchy](https://docs.jax.dev/en/latest/pallas/gpu/reference.html), being the top the fastest and smallest capacity:

![memory_spaces](./imgs/memory_spaces.svg)

- **2CTA (CTA Pairs) and Cluster Cooperation:** Blackwell PTX model allows [two CTAs to execute tensor core instructions that access each other’s TMEM](https://www.together.ai/blog/thunderkittens-nvidia-blackwell-gpus). This allows higher matrix operations efficiency.
- **New optimised data types:** FP8, FP6 and FP4 data types have the highest FLOPs and memory bandwidth utilization.

## Software Stack

 Ubuntu (24.04) with **CUDA 13.0/12.8** toolkit with the latest NVIDIA driver versions (**580**) can be selected during provisioning the VMs. Docker and NVIDIA container toolkit can also be installed when configuring the deployment of the VM:

```shell
docker run --rm --gpus all --pull=always ubuntu nvidia-smi
```

The CUDA environment is complete, including also system NCCL already. The following test command:

```shell
sudo apt install -y openmpi-bin libopenmpi-dev
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
git clone https://github.com/NVIDIA/nccl-tests.git /home/ubuntu/nccl-tests
cd /home/ubuntu/nccl-tests
make MPI=1 MPI_HOME=$MPI_HOME -j$(nproc)
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -np 4 all_reduce_perf -b 512M -e 8G -f 2 -g 1
```

As of now **(16-12-2025)**, current torch stable version [`torch==2.9.1+cu130`]([https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)) points to a Triton version which PTXAS is not compiled for `SM103` compute capabilites (B300). Thus a `venv` is provided with a solution. (See ***Annex I***). Nonetheless we have tested that Torch nightly version (`2.11.0.dev20251215+cu130` ) seems to fix the issue. Install it with:

```shell
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
```



## Profiling

Compared to other cloud providers we allow querying GPU HW counters with NVIDIA profilers like [ncu](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html), [Nsight systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), or other profilers that interact with CUPTI directly like native Triton profiler [Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton). We also configure the VM to not require sudo access when doing profiling which is a [well known issue](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) by the community. The following setting is specified in **/etc/modprobe.d/nvidia-profiling.conf:**

```bash
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

This is again a fundamental requirement for high technical teams doing both end-to-end performance work like accelerating training or inference, and low-level kernel benchmarking.

## Annex I

```shell
Instruct how we replace with sed pre-compiled PyTorch Triton bundle (pytorch-triton ) with Triton built from source with a given commit or nightly. Important with working in new GPU architectures (e.g. SM103 → B300) to enable torch.compile (Triton codegen) before official Torch stable release.
```

PyTorch comes with a pre-compiled Triton bundle (`pytorch-triton` package). For the latest GPU architectures (e.g. B300) when the SW support still experimental and early, we can bypass this limitation. We replace this pre-compiled Triton with one built from source with a given commit or nightly that we know can support the target GPU architecture. This for example a workaround that [SGLang has been using to have early support targeting GB300](https://github.com/sgl-project/sglang/blob/fca8e88f286867ccbfd6e484a5d6b1a1fb1599a4/docker/Dockerfile#L419).

If our  Triton wheels are do not  contemplate SM103 this error is raised when compiling:

```bash
raise NoTritonConfigsError(
torch._inductor.runtime.triton_heuristics.NoTritonConfigsError: No valid triton configs. PTXASError: PTXAS error: Internal Triton PTX codegen error
`ptxas` stderr:
ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'bas
```

This script automates the creation of a Torch compatible SM103 with the proper PTXAS location (Note `TRITON_PTXAS_PATH` var):

```bash
#!/bin/bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh
uv --version 
# create new .venv 
uv venv torch --python 3.12
# patch ptxas baked into the en activattion
echo "export TRITON_PTXAS_PATH="$(which ptxas)"">> ~/torch/bin/activate
# Activate the venv
source ~/torch/bin/activate
# install torch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```