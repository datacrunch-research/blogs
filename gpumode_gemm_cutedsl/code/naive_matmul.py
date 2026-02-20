import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings.driver import CUstream
from torch.cuda import current_stream


@cute.kernel
def naive_gemm_kernel(A, B, C, M, N, K):
    """Naive GEMM kernel: C = A @ B"""
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[m, k] * B[k, n]
            C[m, n] = acc


@cute.jit
def launch_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor,
                M: int, N: int, K: int, stream: CUstream):
    """Launch GEMM kernel with single thread for now"""
    naive_gemm_kernel(A, B, C, M, N, K).launch(
        grid=[1, 1, 1],
        block=[1, 1, 1],
        stream=stream
    )


def run():
    M = N = K = 1024

    A = torch.full((M, K), 1.0, dtype=torch.float32, device="cuda")
    B = torch.full((K, N), 2.0, dtype=torch.float32, device="cuda")
    C = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    # Convert to CuTe tensors
    A_cute, B_cute, C_cute = map(from_dlpack, (A, B, C))

    # Compile kernel
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(launch_gemm, A_cute, B_cute, C_cute, M, N, K, stream)

    # Time kernel execution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    compiled(A_cute, B_cute, C_cute, M, N, K, stream)
    end_event.record()

    # Wait for kernel to complete
    torch.cuda.synchronize()

    # Calculate performance
    time_ms = start_event.elapsed_time(end_event)
    flops = 2 * M * N * K  # Each output element: K multiplies + K adds
    tflops = (flops / (time_ms * 1e-3)) / 1e12

    print(f"\n{'='*50}")
    print(f"Naive GEMM Performance: {M}x{N}x{K}")
    print(f"Time: {time_ms:.2f} ms ({time_ms/1000:.3f} s)")
    print(f"Performance: {tflops:.4f} TFLOP/s")
    print(f"{'='*50}")

    # Verify result
    expected = K * 1.0 * 2.0
    is_correct = torch.allclose(C, torch.full_like(C, expected))
    print(f"\n✓ Result is correct: {is_correct}")


if __name__ == "__main__":
    run()
